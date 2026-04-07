"""Crop GraphCast predictions to a regional domain."""

import numpy as np
import xarray as xr
from pathlib import Path


def crop_and_save_nc(input_nc, cfg, logger):
    """Crop a prediction .nc file to the configured regional window.

    Parameters
    ----------
    input_nc : str or Path
        Path to the full-resolution prediction .nc file.
    cfg : dict
        Must contain crop.center_lat, crop.center_lon, crop.window_size.
    logger : logging.Logger

    Returns
    -------
    out_path : Path
        Path to the cropped output file.
    """
    center_lat = cfg["crop"]["center_lat"]
    center_lon = cfg["crop"]["center_lon"] % 360  # GraphCast uses 0-360 longitude
    window_size = cfg["crop"]["window_size"]

    inp = Path(input_nc)
    out_path = inp.parent / f"{inp.stem}_cropped{inp.suffix}"

    ds = xr.open_dataset(inp)

    # Determine coordinate names (GraphCast uses lat/lon)
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    ilat = int(np.abs(lats - center_lat).argmin())
    ilon = int(np.abs(lons - center_lon).argmin())
    half = window_size // 2

    cropped = ds.isel({
        lat_name: slice(max(ilat - half, 0), min(ilat + half, len(lats))),
        lon_name: slice(max(ilon - half, 0), min(ilon + half, len(lons))),
    })

    cropped.to_netcdf(out_path)
    ds.close()
    logger.info(f"  Cropped output saved to {out_path}")
    return out_path


def crop_prediction(pred_path, cfg, logger):
    """Crop a single GraphCast prediction file."""
    return crop_and_save_nc(pred_path, cfg, logger)
