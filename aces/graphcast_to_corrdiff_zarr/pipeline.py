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
# Variable extraction (layout-driven, label-based; asserts everything)
# =============================================================
# Variable names that are intrinsically single-level (no vertical dim).
# Source-of-truth for surface detection; the era5_pressure value alone is
# unreliable because the base Zarr may store it as an INDEX (with 0/NaN for
# the first index) rather than an actual hPa value.
SURFACE_VARS = {"t2m", "u10", "v10", "msl", "tp", "sp", "tcwv", "skt"}


def read_target_channel_layout(
    base_zarr_path: Path,
    gc_levels: List[int],
    logger: logging.Logger,
) -> List[dict]:
    """Derive the canonical channel layout from the base CorrDiff Zarr.

    Returns a list of length n_channels, where each entry is:
        {"idx": <int>, "name": <str>, "pressure": <float hPa> or None}

    Surface channels have pressure=None and are identified by variable name
    (SURFACE_VARS). Pressure-level channels carry an actual hPa value so
    downstream selection can use label-based .sel(level=p) and is immune to
    the order in which the GraphCast output stores its `level` coord.

    Handles two conventions for `era5_pressure`:
      1. Actual hPa values (50..1000).
      2. Indices 0..N-1 into the level dim of whatever array was used to
         build the base Zarr. Indices are interpreted as positions in a
         DESCENDING-pressure list (1000, 925, ..., 50) because that's the
         convention the existing base Zarr was built with — confirmed via
         the diagnostic compare_channel_stats.py run.

    The convention is auto-detected: if non-surface era5_pressure values are
    all < 20 they are treated as indices; otherwise as hPa.

    `gc_levels` is the GraphCast file's actual level coord values (e.g.
    [50, 100, ..., 1000]). It is reversed to drive the index→hPa mapping.
    """
    base = xr.open_zarr(base_zarr_path, consolidated=False)
    try:
        if "era5_variable" not in base.coords:
            raise ValueError(
                f"Base Zarr {base_zarr_path} is missing the 'era5_variable' coord — "
                f"cannot derive a canonical channel layout."
            )
        vars_per_ch = [str(v) for v in base["era5_variable"].values]

        if "era5_pressure" in base.coords:
            press_raw = np.asarray(base["era5_pressure"].values, dtype=float)
        else:
            logger.warning("Base Zarr has no 'era5_pressure' coord — assuming all surface.")
            press_raw = np.full(len(vars_per_ch), np.nan)
    finally:
        base.close()

    n = len(vars_per_ch)
    assert press_raw.size == n, (
        f"era5_variable ({n}) and era5_pressure ({press_raw.size}) length mismatch"
    )

    # Decide: indices or hPa?
    nonsurf_finite = [
        float(p) for v, p in zip(vars_per_ch, press_raw)
        if v not in SURFACE_VARS and np.isfinite(p)
    ]
    looks_like_indices = bool(nonsurf_finite) and max(nonsurf_finite) < 20.0

    desc_levels = sorted(int(v) for v in gc_levels)[::-1]  # e.g. [1000, 925, ..., 50]

    if looks_like_indices:
        logger.info(
            f"Base Zarr era5_pressure looks like LEVEL INDICES "
            f"(max non-surface value = {max(nonsurf_finite):.0f}); "
            f"mapping to DESCENDING GraphCast levels: {desc_levels}"
        )
    else:
        logger.info("Base Zarr era5_pressure looks like actual hPa values.")

    layout: List[dict] = []
    for i, (vname, p) in enumerate(zip(vars_per_ch, press_raw)):
        if vname in SURFACE_VARS:
            hpa = None
        else:
            assert np.isfinite(p), (
                f"Channel {i}: variable {vname!r} is pressure-level but "
                f"era5_pressure is NaN/missing."
            )
            if looks_like_indices:
                idx = int(round(float(p)))
                assert 0 <= idx < len(desc_levels), (
                    f"Channel {i} ({vname!r}): era5_pressure index {idx} out of range "
                    f"[0, {len(desc_levels)-1}] for {len(desc_levels)} GraphCast levels"
                )
                hpa = float(desc_levels[idx])
            else:
                hpa = float(p)
        layout.append({"idx": i, "name": vname, "pressure": hpa})

    logger.info(f"Target channel layout from base Zarr ({len(layout)} channels):")
    for ch in layout:
        if ch["pressure"] is None:
            logger.info(f"  ch {ch['idx']:>3}: {ch['name']!r:<8} (surface)")
        else:
            logger.info(f"  ch {ch['idx']:>3}: {ch['name']!r:<8} @ {ch['pressure']:>6.1f} hPa")
    return layout


def select_channels_by_layout(
    ds: xr.Dataset,
    layout: List[dict],
    logger: logging.Logger,
) -> List[xr.DataArray]:
    """Build per-channel DataArrays in the exact order of `layout`.

    For each channel:
      - Surface: requires `name` in ds, NO level dim. Assert both.
      - Pressure-level: requires `name` in ds, requires a `level` (or
        `bottom_top`) dim, and requires the target hPa value to be in
        ds[name].<vdim>.values. Selection is LABEL-based.

    All failures raise AssertionError with a clear message so the bug
    cannot pass silently the way the old positional `isel` did.
    """
    available = set(ds.data_vars)
    channels: List[xr.DataArray] = []

    for ch in layout:
        name = ch["name"]
        p    = ch["pressure"]
        idx  = ch["idx"]

        assert name in available, (
            f"Channel {idx} expects variable {name!r} but it is not in the GraphCast "
            f"dataset. Available variables: {sorted(available)}"
        )
        da = ds[name]

        # Spatial-dim sanity (applies to both surface and pressure-level)
        for d in ("south_north", "west_east"):
            assert d in da.dims, (
                f"Channel {idx} ({name!r}): missing spatial dim {d!r}. "
                f"Got dims: {da.dims}"
            )

        if p is None:
            # ------------------------- Surface -------------------------
            assert "level" not in da.dims and "bottom_top" not in da.dims, (
                f"Channel {idx} ({name!r}): base Zarr declares this as surface, "
                f"but GraphCast variable has a vertical dim. dims={da.dims}"
            )
            da_ch = da.transpose("time", "south_north", "west_east").astype("float32")
            da_ch.name = name
        else:
            # ---------------------- Pressure level ---------------------
            vdim = "level" if "level" in da.dims else ("bottom_top" if "bottom_top" in da.dims else None)
            assert vdim is not None, (
                f"Channel {idx} ({name!r} @ {p} hPa): no vertical dim found on variable. "
                f"dims={da.dims}"
            )
            levels = np.asarray(da.coords[vdim].values)
            target = float(p)
            # Use a tolerant exact-match: GraphCast levels are int32 hPa values,
            # base Zarr era5_pressure may be float. Compare with a tiny tolerance.
            matches = np.where(np.isclose(levels.astype(float), target, atol=1e-3))[0]
            assert matches.size == 1, (
                f"Channel {idx} ({name!r} @ {target} hPa): expected exactly one "
                f"matching level in dataset, found {matches.size}. "
                f"Available levels: {sorted(levels.tolist())}"
            )
            # Label-based selection — order of `level` coord in `ds` does NOT matter.
            da_ch = (
                da.sel({vdim: levels[matches[0]]})
                  .transpose("time", "south_north", "west_east")
                  .astype("float32")
            )
            da_ch.name = f"{name}_{int(target)}hPa"

        channels.append(da_ch)

    assert len(channels) == len(layout), (
        f"Built {len(channels)} channels but layout expects {len(layout)}"
    )
    logger.info(
        f"Selected {len(channels)} channels by layout "
        f"(label-based; first={channels[0].name!r}, last={channels[-1].name!r})"
    )
    return channels


# =============================================================
# Stage 1: build the clean ERA5-style block from GraphCast NCs
# =============================================================
def build_clean_graphcast_dataset(cfg: dict, nc_files: List[Path], logger: logging.Logger) -> xr.Dataset:
    all_blocks, all_valid = [], []
    base_path = cfg["paths"]["base_zarr_path"]

    # ----- Source of truth for channel layout: the base Zarr coords. -----
    # This eliminates positional-isel level-flip bugs: we ALWAYS select by
    # variable name + hPa label, never by index, and we always emit channels
    # in the exact order the base Zarr's era5_center / era5_scale expect.
    #
    # Need the GraphCast level coord up-front because the base Zarr may store
    # era5_pressure as level INDICES rather than actual hPa values; in that
    # case we map indices to hPa via the GraphCast level array (descending).
    logger.info(f"Reading level coord from first file: {nc_files[0].name}")
    with xr.open_dataset(nc_files[0]) as _ds0:
        if "level" not in _ds0.coords:
            raise ValueError(f"First nc file {nc_files[0]} has no 'level' coord")
        gc_levels = [int(v) for v in _ds0["level"].values]
    logger.info(f"GraphCast levels: {sorted(gc_levels)}")

    logger.info("Reading canonical channel layout from base Zarr ...")
    layout = read_target_channel_layout(base_path, gc_levels, logger)
    expected_n_channels = len(layout)

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

            chans = select_channels_by_layout(ds, layout, logger)

            for i, da in enumerate(chans):
                drop = [c for c in da.coords if c not in ("time", "south_north", "west_east")]
                chans[i] = da.drop_vars(drop, errors="ignore")

            block = xr.concat(chans, dim="era5_channel",
                              coords="minimal", compat="override").astype("float32")
            block = block.transpose("time", "era5_channel", "south_north", "west_east")

            assert block.sizes["era5_channel"] == expected_n_channels, (
                f"{f.name}: built {block.sizes['era5_channel']} channels but base "
                f"Zarr expects {expected_n_channels}"
            )

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

    assert ds_out.sizes["era5_channel"] == expected_n_channels, (
        f"Final dataset has {ds_out.sizes['era5_channel']} channels; "
        f"base Zarr layout expects {expected_n_channels}"
    )
    logger.info(
        f"Assembled GraphCast dataset: {dict(ds_out.sizes)} — channel layout "
        f"matches base Zarr ({expected_n_channels} channels)."
    )
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
    # Write logs next to the output Zarr store (alongside, not inside)
    out_path = cfg["paths"]["zarr_output_path"]
    log_dir = out_path.parent / f"{out_path.stem}_logs"
    logger = setup_logger(log_dir, "graphcast_to_zarr")
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
