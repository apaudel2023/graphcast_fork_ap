"""
GraphCast -> CorrDiff Zarr packaging pipeline
---------------------------------------------

Takes per-initialization GraphCast prediction NetCDFs (lat/lon/level/time,
GraphCast-native variable names such as `2m_temperature`, `temperature`,
`u_component_of_wind`, ...) and packages them into a CorrDiff-ready Zarr
store that matches the schema of a reference (training) Zarr:

    era5        (time, era5_channel, south_north, west_east)   float32
    era5_valid  (time, era5_channel)                           bool
    wrf         (time, wrf_channel,  south_north, west_east)   float32  (real or dummy)
    wrf_valid   (time,)                                        bool
    era5_center, era5_scale, wrf_center, wrf_scale   grafted from base zarr
    + static coords (XLAT, XLONG, etc.) from base zarr

Spatial regridding (GraphCast lat/lon -> WRF curvilinear grid) is done
in-memory with xESMF; no weight files are written.
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

try:
    import xesmf as xe
except ImportError:
    xe = None


# =============================================================
# Config
# =============================================================
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.graphcast_nc_dir:
        cfg["paths"]["graphcast_nc_dir"] = args.graphcast_nc_dir
    if args.base_zarr_path:
        cfg["paths"]["base_zarr_path"] = args.base_zarr_path
    if args.zarr_output_path:
        cfg["paths"]["zarr_output_path"] = args.zarr_output_path
    for k in ("graphcast_nc_dir", "base_zarr_path", "zarr_output_path"):
        cfg["paths"][k] = Path(cfg["paths"][k])
    return cfg


# =============================================================
# Logging
# =============================================================
def setup_logger(log_dir: Path, name: str = "graphcast_to_zarr") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(logfile); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler();     ch.setFormatter(fmt); logger.addHandler(ch)
    logger.info(f"Logging to file: {logfile}")
    return logger


# =============================================================
# File discovery + cleanup
# =============================================================
def collect_graphcast_files(nc_dir: Path, logger: logging.Logger) -> List[Path]:
    files = sorted(p for p in nc_dir.rglob("*.nc")
                   if not p.name.startswith("ground_truth_"))
    if not files:
        raise FileNotFoundError(f"No GraphCast .nc files found under {nc_dir}")
    logger.info(f"Discovered {len(files)} NetCDF files under {nc_dir}")
    return files


def squeeze_and_rename(ds: xr.Dataset) -> xr.Dataset:
    """Drop singleton dims (e.g. `batch`) and map GraphCast lat/lon to WRF dim names."""
    ds = ds.squeeze(drop=True)
    ren = {}
    if "lat" in ds.dims:       ren["lat"] = "south_north"
    if "lon" in ds.dims:       ren["lon"] = "west_east"
    # Legacy / alternative names, just in case
    if "latitude" in ds.dims:  ren["latitude"] = "south_north"
    if "longitude" in ds.dims: ren["longitude"] = "west_east"
    return ds.rename(ren)


def map_variable_names(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    rename = {k: v for k, v in cfg["var_map"].items() if k in ds}
    return ds.rename(rename)


def resize_spatial(ds: xr.Dataset, target_shape) -> xr.Dataset:
    ny, nx = target_shape
    return ds.interp(
        south_north=np.linspace(ds.south_north.min(), ds.south_north.max(), ny),
        west_east=np.linspace(ds.west_east.min(), ds.west_east.max(), nx),
        method="linear",
    )


# =============================================================
# In-memory xESMF regridding to WRF grid
# =============================================================
def regrid_spatial(ds: xr.Dataset, base_path: Path, method: str, logger: logging.Logger) -> xr.Dataset:
    if xe is None:
        raise ImportError("xesmf is required when regrid.enabled=true")
    logger.info(f"Regridding to WRF grid via xESMF (method={method})")

    base = xr.open_zarr(base_path, consolidated=False)
    if "XLAT" not in base or "XLONG" not in base:
        base.close()
        raise ValueError("Base Zarr missing XLAT/XLONG")
    lat = base["XLAT"]; lon = base["XLONG"]
    if "time" in lat.dims: lat = lat.isel(time=0)
    if "time" in lon.dims: lon = lon.isel(time=0)
    tgt = xr.Dataset({"lat": lat, "lon": lon})
    base.close()

    if not {"south_north", "west_east"} <= set(ds.dims):
        raise ValueError("Expected south_north / west_east dims after squeeze_and_rename()")

    src_lat = ds["south_north"]
    src_lon = ds["west_east"]
    src = xr.Dataset({"lat": src_lat, "lon": src_lon})

    regridder = xe.Regridder(src, tgt, method=method, reuse_weights=False)

    out_vars = {}
    for v in ds.data_vars:
        da = ds[v]
        if {"south_north", "west_east"} <= set(da.dims):
            tmp = da.rename({"south_north": "lat", "west_east": "lon"})
            res = regridder(tmp)
            res = res.rename({d: n for d, n in [("lat", "south_north"), ("lon", "west_east")] if d in res.dims})
            out_vars[v] = res
        else:
            out_vars[v] = da
    logger.info(f"Regridded {len(out_vars)} variables")
    return xr.Dataset(out_vars)


# =============================================================
# Variable extraction
# =============================================================
def select_pressure_levels(ds: xr.Dataset, cfg: dict, logger: logging.Logger) -> List[xr.DataArray]:
    out = []
    for v in cfg["var_era5_pressure"]:
        name = v["name"]
        if name not in ds:
            logger.warning(f"Pressure variable missing: {name}")
            continue
        da = ds[name]
        vdim = "level" if "level" in da.dims else ("bottom_top" if "bottom_top" in da.dims else None)
        if vdim is None:
            logger.warning(f"{name}: no vertical dim found, skipping")
            continue
        for lvl in v["pressure_levels"]:
            if lvl >= da.sizes[vdim]:
                logger.warning(f"{name}: level index {lvl} exceeds size {da.sizes[vdim]}")
                continue
            da_lvl = (da.isel({vdim: lvl})
                        .transpose("time", "south_north", "west_east")
                        .astype("float32"))
            da_lvl.name = f"{name}_{lvl}"
            out.append(da_lvl)
    return out


def select_surface_vars(ds: xr.Dataset, cfg: dict, logger: logging.Logger) -> List[xr.DataArray]:
    out = []
    for name in cfg["var_era5_surface"]:
        if name not in ds:
            logger.warning(f"Surface variable missing: {name}")
            continue
        da = ds[name].transpose("time", "south_north", "west_east").astype("float32")
        da.name = name
        out.append(da)
    return out


# =============================================================
# Stage 1: build the clean ERA5-style block from GraphCast NCs
# =============================================================
def build_clean_graphcast_dataset(cfg: dict, nc_files: List[Path], logger: logging.Logger) -> xr.Dataset:
    all_blocks, all_valid = [], []
    base_path = cfg["paths"]["base_zarr_path"]

    for f in tqdm(nc_files, desc="[Reading GraphCast outputs]"):
        ds = None
        try:
            ds = xr.open_dataset(f)
            ds = squeeze_and_rename(ds)
            ds = map_variable_names(ds, cfg)

            if cfg.get("regrid", {}).get("enabled", False):
                ds = regrid_spatial(ds, base_path, cfg["regrid"].get("method", "bilinear"), logger)
            elif cfg.get("resize", {}).get("enabled", False):
                ds = resize_spatial(ds, cfg["resize"]["target_shape"])

            chans = (select_pressure_levels(ds, cfg, logger)
                     + select_surface_vars(ds, cfg, logger))

            for i, da in enumerate(chans):
                drop = [c for c in da.coords if c not in ("time", "south_north", "west_east")]
                chans[i] = da.drop_vars(drop, errors="ignore")

            block = xr.concat(chans, dim="era5_channel",
                              coords="minimal", compat="override").astype("float32")
            block = block.transpose("time", "era5_channel", "south_north", "west_east")

            tvals = np.asarray(ds["time"].values).astype("datetime64[ns]")
            block = block.assign_coords(time=("time", tvals))

            valid = xr.DataArray(
                np.ones((block.sizes["time"], block.sizes["era5_channel"]), dtype=bool),
                dims=("time", "era5_channel"),
                coords={"time": ("time", tvals)},
            )

            all_blocks.append(block)
            all_valid.append(valid)
            logger.info(f"Processed {f.name} ({block.sizes['time']} timesteps)")
        finally:
            if ds is not None:
                ds.close()

    era5_all  = xr.concat(all_blocks, dim="time").sortby("time")
    valid_all = xr.concat(all_valid,  dim="time").sortby("time")

    # Drop duplicate times (overlapping rolling forecasts) — keep first occurrence
    _, unique_idx = np.unique(era5_all["time"].values, return_index=True)
    if len(unique_idx) != era5_all.sizes["time"]:
        logger.info(f"Deduplicating time: {era5_all.sizes['time']} -> {len(unique_idx)}")
        era5_all  = era5_all.isel(time=np.sort(unique_idx))
        valid_all = valid_all.isel(time=np.sort(unique_idx))

    ds_out = xr.Dataset(
        {"era5": era5_all, "era5_valid": valid_all},
        coords={
            "time": era5_all["time"],
            "era5_channel": np.arange(era5_all.sizes["era5_channel"], dtype=np.int64),
        },
    )
    ds_out["time"].attrs.clear()
    ds_out["time"].encoding.clear()
    logger.info(f"Assembled GraphCast dataset: {dict(ds_out.sizes)}")
    return ds_out


# =============================================================
# Stages 2–4: static graft, time encoding, WRF attach
# =============================================================
def graft_base_static(ds_gc: xr.Dataset, base_path: Path, logger: logging.Logger) -> xr.Dataset:
    logger.info(f"Grafting static metadata from base Zarr: {base_path}")
    base = xr.open_zarr(base_path, consolidated=False)
    static_coords = {c: base[c] for c in base.coords if c not in ("time", "era5", "wrf")}
    static_vars = {k: base[k] for k in base.data_vars
                   if k in ("era5_center", "era5_scale", "wrf_center", "wrf_scale")}
    out = xr.Dataset(
        data_vars={**static_vars, "era5": ds_gc["era5"], "era5_valid": ds_gc["era5_valid"]},
        coords={**static_coords, "time": ds_gc["time"]},
        attrs=base.attrs,
    )
    out["time"].attrs.clear()
    out["time"].encoding.clear()
    base.close()
    return out


def encode_time_to_origin(ds: xr.Dataset, logger: logging.Logger) -> xr.Dataset:
    times = ds["time"].values.astype("datetime64[ns]")
    origin = times.min()
    t_enc = ((times - origin) / np.timedelta64(1, "h")).astype("int64")
    ds = ds.assign_coords(time=("time", t_enc))
    ds["time"].attrs.update({
        "units": f"hours since {np.datetime_as_string(origin, unit='s')}",
        "calendar": "proleptic_gregorian",
    })
    ds["time"].encoding.clear()
    logger.info(f"Encoded time relative to origin {origin}")
    return ds


def _to_datetime64ns(da_time: xr.DataArray) -> np.ndarray:
    vals = da_time.values
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]")
    units = da_time.attrs.get("units", "")
    if isinstance(units, str) and units.lower().startswith("hours since "):
        origin = np.datetime64(units[len("hours since "):].strip(), "ns")
        return (origin + vals.astype("timedelta64[h]")).astype("datetime64[ns]")
    try:
        dec = xr.decode_cf(xr.Dataset({"_t": da_time}))._t.values
        if np.issubdtype(dec.dtype, np.datetime64):
            return dec.astype("datetime64[ns]")
    except Exception:
        pass
    raise ValueError("Unrecognized time encoding")


def add_wrf_fields(ds: xr.Dataset, base_path: Path, use_real: bool, logger: logging.Logger) -> xr.Dataset:
    logger.info(f"Adding {'real' if use_real else 'dummy'} WRF fields")
    base = xr.open_zarr(base_path, consolidated=False)

    if "wrf_channel" in base.coords:
        wrf_channels = base.sizes["wrf_channel"]
        wrf_channel_vals = base["wrf_channel"].values
    elif "wrf_variable" in base.coords:
        wrf_channels = base.sizes["wrf_variable"]
        wrf_channel_vals = np.arange(wrf_channels)
    else:
        base.close()
        raise ValueError("Base Zarr missing WRF channel metadata")

    time_len = ds.sizes["time"]
    ny = ds["era5"].sizes["south_north"]
    nx = ds["era5"].sizes["west_east"]

    if not use_real:
        wrf_data  = np.random.normal(0.0, 1.0, (time_len, wrf_channels, ny, nx)).astype("float32")
        wrf_valid = np.ones(time_len, dtype=bool)
        ds["wrf"] = xr.DataArray(
            wrf_data, dims=("time", "wrf_channel", "south_north", "west_east"),
            coords={"time": ds["time"], "wrf_channel": wrf_channel_vals},
        )
        ds["wrf_valid"] = xr.DataArray(wrf_valid, dims=("time",), coords={"time": ds["time"]})
        for c in ("south_north", "west_east"):
            if c in ds.coords: ds = ds.drop_vars(c)
        base.close()
        return ds

    if "wrf" not in base or "wrf_valid" not in base:
        base.close()
        raise ValueError("Base Zarr missing 'wrf' / 'wrf_valid'")

    gc_times   = _to_datetime64ns(ds["time"])
    base_times = _to_datetime64ns(base["wrf"]["time"])

    base_set = set(base_times.tolist())
    missing = [t for t in gc_times.tolist() if t not in base_set]
    if missing:
        base.close()
        raise ValueError(
            f"Base WRF Zarr missing {len(missing)} timesteps present in GraphCast Zarr. "
            f"First missing: {missing[0]}"
        )

    base_idx = {t: i for i, t in enumerate(base_times)}
    idx = np.array([base_idx[t] for t in gc_times], dtype=np.int64)

    wrf_sel       = base["wrf"].isel(time=idx).astype("float32")
    wrf_valid_sel = base["wrf_valid"].isel(time=idx).astype(bool)

    ds["wrf"] = xr.DataArray(
        wrf_sel.data, dims=base["wrf"].dims,
        coords={
            "time": ds["time"],
            "wrf_channel": wrf_channel_vals,
            "south_north": ds["era5"].coords.get("south_north", np.arange(ny)),
            "west_east":   ds["era5"].coords.get("west_east",   np.arange(nx)),
        },
    )
    ds["wrf_valid"] = xr.DataArray(wrf_valid_sel.data, dims=("time",), coords={"time": ds["time"]})
    for c in ("south_north", "west_east"):
        if c in ds.coords: ds = ds.drop_vars(c)
    base.close()
    logger.info(f"Attached real WRF: shape={tuple(ds['wrf'].shape)}")
    return ds


# =============================================================
# Main
# =============================================================
def run_pipeline(cfg: dict):
    logger = setup_logger(Path("./log"), "graphcast_to_zarr")
    try:
        logger.info("Starting GraphCast -> Zarr packaging pipeline")
        for k, v in cfg["paths"].items():
            logger.info(f"  {k}: {v}")

        nc_files = collect_graphcast_files(cfg["paths"]["graphcast_nc_dir"], logger)
        ds_gc = build_clean_graphcast_dataset(cfg, nc_files, logger)
        ds_out = graft_base_static(ds_gc, cfg["paths"]["base_zarr_path"], logger)
        ds_out = encode_time_to_origin(ds_out, logger)
        ds_out = add_wrf_fields(
            ds_out, cfg["paths"]["base_zarr_path"],
            use_real=cfg.get("wrf", {}).get("use_real", False),
            logger=logger,
        )

        out_path = cfg["paths"]["zarr_output_path"]
        if out_path.exists():
            logger.info(f"Removing existing Zarr store {out_path}")
            shutil.rmtree(out_path)
        ds_out.to_zarr(out_path, mode=cfg.get("zarr", {}).get("write_mode", "w"))
        logger.info(f"Write completed: {out_path}")
    except Exception:
        logger.exception("Pipeline failed.")
        raise
    finally:
        gc.collect()
        logger.info("Pipeline completed (cleanup done).")


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    default_cfg = here / "configs" / "default.yml"

    p = argparse.ArgumentParser(description="GraphCast -> CorrDiff Zarr packaging pipeline")
    p.add_argument("--config", type=str, default=str(default_cfg), help="YAML config path")
    p.add_argument("--graphcast_nc_dir", type=str, help="Override: GraphCast .nc directory")
    p.add_argument("--base_zarr_path",   type=str, help="Override: base CorrDiff Zarr")
    p.add_argument("--zarr_output_path", type=str, help="Override: output Zarr path")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    cfg = apply_cli_overrides(cfg, args)
    run_pipeline(cfg)
