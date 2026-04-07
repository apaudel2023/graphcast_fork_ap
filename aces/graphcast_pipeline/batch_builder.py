"""Build xarray batch from ERA5 NetCDF files for GraphCast inference."""

import datetime
import numpy as np
import xarray


# CDS short name -> GraphCast internal variable name
CDS_TO_GRAPHCAST = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tisr": "toa_incident_solar_radiation",
    "lsm": "land_sea_mask",
}

COORD_RENAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "pressure_level": "level",
    "valid_time": "time",
}

DROP_VARS = ["number", "expver"]


def _prep_ds(ds, t_zero):
    """Rename CDS conventions to GraphCast conventions and fix coordinates."""
    ds = ds.rename({k: v for k, v in CDS_TO_GRAPHCAST.items() if k in ds.data_vars})
    ds = ds.rename({k: v for k, v in COORD_RENAMES.items() if k in ds.dims or k in ds.coords})

    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]") - t_zero)
    if "lat" in ds.dims and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    if "lon" in ds.dims and float(ds["lon"].min()) < 0:
        ds = ds.assign_coords(lon=ds["lon"] % 360).sortby("lon")

    ds = ds.assign_coords(
        lat=ds["lat"].astype(np.float32),
        lon=ds["lon"].astype(np.float32),
    )
    if "level" in ds.coords:
        ds = ds.assign_coords(level=ds["level"].astype(np.int32))
    return ds


def build_batch(static_path, surface_path, pressure_path, init_time, rollout_steps, logger):
    """Build the xarray batch for GraphCast from ERA5 files.

    Parameters
    ----------
    static_path, surface_path, pressure_path : Path
        ERA5 NetCDF files.
    init_time : datetime.datetime
        Forecast initialization time (00Z of the forecast day).
    rollout_steps : int
        Number of 6h rollout steps.
    logger : logging.Logger

    Returns
    -------
    batch : xarray.Dataset
        Shape (batch=1, time=2+rollout_steps, ...) ready for GraphCast.
    """
    t_zero = np.datetime64(init_time - datetime.timedelta(hours=12), "ns")

    def _load(path):
        return xarray.load_dataset(path, drop_variables=DROP_VARS).compute()

    # Input data (2 timesteps)
    ds_pl = _prep_ds(_load(pressure_path), t_zero)
    ds_sl = _prep_ds(_load(surface_path), t_zero)

    # NaN check on inputs
    for label, ds in [("pressure", ds_pl), ("surface", ds_sl)]:
        for var in ds.data_vars:
            n = int(np.isnan(ds[var].values).sum())
            if n > 0:
                raise ValueError(f"NaN in '{var}' ({label}): {n} values — check download")

    ds_input = xarray.merge([ds_pl, ds_sl], join="inner")
    assert ds_input.sizes["time"] == 2, f"Expected 2 input timesteps, got {ds_input.sizes['time']}"

    # NaN placeholders for future rollout steps
    future_times = np.array([
        np.timedelta64((i + 2) * 6, "h") for i in range(rollout_steps)
    ], dtype="timedelta64[ns]")

    future_data = {
        var: xarray.DataArray(
            np.full(
                tuple(
                    len(future_times) if d == "time" else ds_input.sizes[d]
                    for d in ds_input[var].dims
                ),
                np.nan, dtype=np.float32,
            ),
            dims=ds_input[var].dims,
            coords={
                d: (future_times if d == "time" else ds_input.coords[d])
                for d in ds_input[var].dims
                if d in ds_input.coords or d == "time"
            },
        )
        for var in ds_input.data_vars
    }
    ds = xarray.concat([ds_input, xarray.Dataset(future_data)], dim="time")
    assert ds.sizes["time"] == 2 + rollout_steps

    # Static fields (no time dim)
    ds_st = _prep_ds(_load(static_path), t_zero).isel(time=0, drop=True)
    if "geopotential" in ds_st.data_vars:
        ds_st = ds_st.rename({"geopotential": "geopotential_at_surface"})

    ds = ds.expand_dims("batch", axis=0)
    for var in ds_st.data_vars:
        ds[var] = ds_st[var]

    # Precipitation placeholder (target-only for operational model)
    if "total_precipitation_6hr" not in ds.data_vars:
        ds["total_precipitation_6hr"] = xarray.zeros_like(ds["2m_temperature"])

    # datetime coord required by data_utils.add_derived_vars
    abs_times = (t_zero + ds["time"].values).astype("datetime64[ns]")
    ds = ds.assign_coords(datetime=(("batch", "time"), abs_times[np.newaxis, :]))

    logger.info(f"  Batch shape: {dict(ds.sizes)}")
    logger.info(f"  Variables: {sorted(ds.data_vars)}")
    return ds


def load_ground_truth(gt_pressure_path, gt_surface_path, init_time, logger):
    """Load ERA5 ground truth and return with absolute datetime time coords.

    Returns an xarray.Dataset with the same variable names and grid as predictions,
    with absolute datetime time coords matching the saved predictions.nc format.
    """
    t_zero = np.datetime64(init_time - datetime.timedelta(hours=12), "ns")

    def _load(path):
        return xarray.load_dataset(path, drop_variables=DROP_VARS).compute()

    gt_pl = _prep_ds(_load(gt_pressure_path), t_zero)
    gt_sl = _prep_ds(_load(gt_surface_path), t_zero)
    gt = xarray.merge([gt_pl, gt_sl], join="inner")

    # Convert to absolute datetimes: timedelta from t_zero -> absolute
    # Ground truth times are at offsets [12h, 18h, 24h, ...] from t_zero
    # Predictions have absolute times [init_time+6h, init_time+12h, ...]
    # Since init_time = t_zero + 12h, absolute = t_zero + timedelta
    abs_times = (t_zero + gt["time"].values).astype("datetime64[ns]")
    gt = gt.assign_coords(time=abs_times)

    logger.info(f"  Ground truth shape: {dict(gt.sizes)}")
    logger.info(f"  Ground truth time: {gt.coords['time'].values[[0, -1]]}")
    logger.info(f"  Ground truth vars: {sorted(gt.data_vars)}")
    return gt
