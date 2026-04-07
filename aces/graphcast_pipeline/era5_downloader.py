"""Download ERA5 inputs (and optionally ground truth) for GraphCast via CDS API."""

import cdsapi
from datetime import timedelta
from pathlib import Path


def _download(client, label, path, dataset, request, logger):
    if path.exists():
        logger.info(f"  [SKIP] {label} -> {path}")
        return
    logger.info(f"  [DOWNLOAD] {label} -> {path}")
    client.retrieve(dataset, request, str(path))


def download_era5_inputs(cfg, input_times, output_dir, logger):
    """Download ERA5 static, surface, and pressure-level data for two input timesteps.

    Returns
    -------
    static_path, surface_path, pressure_path : Path
    """
    client = cdsapi.Client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    era5_cfg = cfg["era5"]
    t1, t2 = input_times

    dates = sorted({t1.strftime("%Y-%m-%d"), t2.strftime("%Y-%m-%d")})
    date_range = f"{dates[0]}/{dates[-1]}"
    times = sorted({t1.strftime("%H:%M"), t2.strftime("%H:%M")})

    common = {
        "product_type": "reanalysis",
        "area": [90, -180, -90, 180],
        "grid": [0.25, 0.25],
        "format": "netcdf",
        "date": date_range,
        "time": times,
    }

    static_path = output_dir / "static.nc"
    _download(client, "Static", static_path, "reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "area": [90, -180, -90, 180],
        "grid": [0.25, 0.25],
        "format": "netcdf",
        "variable": era5_cfg["variables"]["static"],
        "date": t2.strftime("%Y-%m-%d"),
        "time": [t2.strftime("%H:%M")],
    }, logger)

    surface_path = output_dir / "surface.nc"
    _download(client, "Surface", surface_path, "reanalysis-era5-single-levels", {
        **common,
        "variable": era5_cfg["variables"]["surface"],
    }, logger)

    pressure_path = output_dir / "pressure.nc"
    _download(client, "Pressure levels", pressure_path, "reanalysis-era5-pressure-levels", {
        **common,
        "variable": era5_cfg["variables"]["pressure"],
        "pressure_level": [str(p) for p in era5_cfg["pressure_levels"]],
    }, logger)

    logger.info("ERA5 input download complete.")
    return static_path, surface_path, pressure_path


def download_era5_ground_truth(cfg, init_time, rollout_steps, output_dir, logger):
    """Download ERA5 ground truth for the rollout period (for verification).

    Downloads pressure-level and surface variables for the forecast timesteps
    [init_time, init_time+6h, ..., init_time+(rollout_steps-1)*6h].

    Returns
    -------
    gt_pressure_path, gt_surface_path : Path
    """
    client = cdsapi.Client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    era5_cfg = cfg["era5"]

    rollout_times = [
        init_time + timedelta(hours=6 * i) for i in range(rollout_steps)
    ]
    rollout_dates = sorted({t.strftime("%Y-%m-%d") for t in rollout_times})
    rollout_hours = sorted({t.strftime("%H:%M") for t in rollout_times})

    common = {
        "product_type": "reanalysis",
        "area": [90, -180, -90, 180],
        "grid": [0.25, 0.25],
        "format": "netcdf",
        "date": f"{rollout_dates[0]}/{rollout_dates[-1]}",
        "time": rollout_hours,
    }

    gt_pressure_path = output_dir / "ground_truth_pressure.nc"
    _download(client, "Ground truth pressure", gt_pressure_path,
              "reanalysis-era5-pressure-levels", {
                  **common,
                  "variable": era5_cfg["variables"]["pressure"],
                  "pressure_level": [str(p) for p in era5_cfg["pressure_levels"]],
              }, logger)

    gt_surface_path = output_dir / "ground_truth_surface.nc"
    _download(client, "Ground truth surface", gt_surface_path,
              "reanalysis-era5-single-levels", {
                  **common,
                  "variable": era5_cfg["variables"]["surface"],
              }, logger)

    logger.info("ERA5 ground truth download complete.")
    return gt_pressure_path, gt_surface_path
