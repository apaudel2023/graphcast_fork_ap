# GraphCast → CorrDiff Zarr Packaging Pipeline

Location: [`aces/graphcast_to_corrdiff_zarr/`](../graphcast_to_corrdiff_zarr/)

This document describes the pipeline that converts per-initialization GraphCast prediction NetCDFs, produced by [`aces/graphcast_pipeline/`](../graphcast_pipeline/), into a CorrDiff-ready Zarr store. The resulting Zarr matches the exact schema of the training Zarr used to train CorrDiff and can therefore be fed directly to a trained CorrDiff model for downscaling without any further transformation.

---

## 1. Motivation

CorrDiff was trained against a Zarr store with a fixed layout:

- `era5 (time, era5_channel, south_north, west_east)` — coarse driver fields on the WRF curvilinear grid.
- `wrf  (time, wrf_channel,  south_north, west_east)` — high-resolution targets.
- Normalization statics: `era5_center`, `era5_scale`, `wrf_center`, `wrf_scale`.
- Grid statics: `XLAT`, `XLONG`, `XLAT_U`, `XLAT_V`, `XLONG_U`, `XLONG_V`.
- Validity masks: `era5_valid`, `wrf_valid`.
- Channel metadata: `era5_variable`, `era5_pressure`, `wrf_variable`, `wrf_pressure`.

GraphCast produces global 0.25° NetCDFs with different variable names, a regular latitude/longitude grid, different dimension names, and a different time encoding. This pipeline converts those NetCDFs into the CorrDiff schema so that downstream inference requires no additional data handling.

A sibling pipeline was previously developed for Aurora outputs. The present pipeline follows the same stage layout, adapted to GraphCast's conventions.

---

## 2. Architecture

```
+-------------------------------------------------------------------+
|  Inputs                                                           |
|                                                                   |
|   GraphCast NetCDFs            Base CorrDiff Zarr                 |
|   (graphcast_*.nc)             (hampton_2007_2015_wpd_avg.zarr)   |
|   - lat/lon/level              - WRF grid (XLAT / XLONG)          |
|   - GraphCast var names        - era5_center/scale, wrf_*/scale   |
|                                - real WRF data + wrf_valid        |
+-----------------+------------------------------+------------------+
                  |                              |
                  v                              |
     +-----------------------------+             |
     | squeeze + rename dims       |             |
     |  lat/lon -> south_north/    |             |
     |  west_east; drop 'batch'    |             |
     +--------------+--------------+             |
                    |                            |
                    v                            |
     +-----------------------------+             |
     | rename variables            |             |
     | (var_map in config)         |             |
     | 2m_temperature -> t2m, ...  |             |
     +--------------+--------------+             |
                    |                            |
                    v                            |
     +-----------------------------+             |
     | in-memory regrid (xESMF)    |<---- XLAT/XLONG from base Zarr
     |  GraphCast lat/lon ->       |             |
     |  WRF curvilinear grid       |             |
     +--------------+--------------+             |
                    |                            |
                    v                            |
     +-----------------------------+             |
     | extract era5 channels       |             |
     |  pressure levels x vars     |             |
     |  + surface vars             |             |
     |  -> stack on era5_channel   |             |
     +--------------+--------------+             |
                    |                            |
                    v                            v
     +---------------------------------------------------+
     | graft_base_static                                 |
     |   copy XLAT/XLONG, era5_center/scale,             |
     |   wrf_*/scale, era5_variable/pressure             |
     |   from base Zarr                                  |
     +--------------+------------------------------------+
                    |
                    v
     +-----------------------------+
     | encode_time_to_origin       |
     |  -> "hours since <origin>"  |
     +--------------+--------------+
                    |
                    v
     +-----------------------------+
     | add_wrf_fields              |
     |  use_real=true:             |
     |   match datetimes exactly;  |
     |   copy wrf + wrf_valid rows |
     |  use_real=false:            |
     |   random-normal dummy + True|
     +--------------+--------------+
                    |
                    v
              output .zarr
```

A key design decision is that all processing occurs in memory. No xESMF weight files are written and no intermediate NetCDFs are produced. The only I/O consists of reading the input NetCDFs, reading the base Zarr, and writing the output Zarr.

---

## 3. Pipeline Stages

### 3.1 File Discovery — `collect_graphcast_files`

Recursively globs `*.nc` under `paths.graphcast_nc_dir` and excludes any file whose name begins with `ground_truth_` (verification-mode artifacts). A single file typically corresponds to one GraphCast initialization time and contains `rollout_steps` timesteps at 6-hour cadence.

### 3.2 Dimension Cleanup — `squeeze_and_rename`

GraphCast NetCDFs commonly include a singleton `batch` dimension inherited from the model; this is squeezed. Latitude and longitude dimensions are then renamed to `south_north` and `west_east` to match the CorrDiff Zarr convention.

| Source (GraphCast) | Output (CorrDiff schema) |
| --- | --- |
| `lat` | `south_north` |
| `lon` | `west_east` |
| `level` (pressure) | retained as `level` until extraction |
| `batch` | squeezed out |

### 3.3 Variable Renames — `map_variable_names`

Applies the `var_map` defined in the configuration. Defaults:

| GraphCast name | CorrDiff short name |
| --- | --- |
| `2m_temperature` | `t2m` |
| `10m_u_component_of_wind` | `u10` |
| `10m_v_component_of_wind` | `v10` |
| `mean_sea_level_pressure` | `msl` |
| `temperature` | `t` |
| `u_component_of_wind` | `u` |
| `v_component_of_wind` | `v` |
| `specific_humidity` | `q` |
| `geopotential` | `z_pl` |

Variables absent from the NetCDF are logged and skipped. Any mismatch between expected and available variables appears as a warning in the log.

### 3.4 Regridding — `regrid_spatial` (xESMF)

- **Source grid**: GraphCast's regular 0.25° latitude/longitude.
- **Target grid**: the WRF curvilinear `XLAT`/`XLONG` read from the base Zarr (or, if `regrid.wrf_ref_file` is set, from an external WRF `geo_em*.nc` file).

Implementation notes:
- A fresh `xesmf.Regridder` is constructed for each input file; weights are not cached.
- All variables carrying both `south_north` and `west_east` dimensions are regridded; other variables pass through unchanged.
- The interpolation method is configurable: `bilinear` (default), `nearest_s2d`, or `conservative`.

The `resize` option (`xr.interp` to a fixed `[ny, nx]`) is a non-georeferenced fallback, intended only for environments without xESMF. It should not be used for production runs.

### 3.5 Channel Extraction

- `select_pressure_levels` extracts `(time, south_north, west_east)` slices at each requested integer index along the `level` dimension and renames each slice to `<variable>_<idx>` (for example `t_0`, `t_1`, ..., `q_12`).
- `select_surface_vars` extracts each requested surface variable unchanged.
- Both are concatenated along a single `era5_channel` axis:

```
era5: (time, era5_channel, south_north, west_east)   float32
```

Pressure-level indices in the configuration are **indices into the GraphCast `level` dimension**, not pressure values in hPa. GraphCast Operational uses 13 levels in this order:

```
[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  hPa
```

Accordingly, `t_0` is temperature at 50 hPa and `t_12` is temperature at 1000 hPa.

### 3.6 Static Metadata Graft — `graft_base_static`

Copies the following from the base Zarr into the new dataset:
- Grid coordinates: `XLAT`, `XLONG`, `XLAT_U`, `XLAT_V`, `XLONG_U`, `XLONG_V`, and the staggered dimensions `south_north_stag`, `west_east_stag` when present.
- Channel-metadata coordinates: `era5_variable`, `era5_pressure`, `wrf_variable`, `wrf_pressure` (so the output Zarr is self-describing).
- Normalization variables: `era5_center`, `era5_scale`, `wrf_center`, `wrf_scale`.
- Dataset-level `attrs`.

The `era5_valid` mask is constructed fresh (all `True`) with shape `(time, era5_channel)`.

### 3.7 Time Encoding — `encode_time_to_origin`

Absolute `datetime64` timestamps are re-encoded as `int64` "hours since origin" values, where the origin is the earliest timestamp in the dataset. This matches the CF-style time encoding used by the base Zarr and allows downstream code to treat both datasets identically.

When multiple forecast initializations yield overlapping timesteps, duplicates are removed prior to encoding (first occurrence retained).

### 3.8 WRF Attachment — `add_wrf_fields`

Two modes are supported, selected by `wrf.use_real`.

**Real WRF (recommended when the base Zarr covers the forecast window):**
- Converts both `ds['time']` and `base['wrf']['time']` to `datetime64[ns]`, handling either native datetime encoding or CF-style `hours since ...` encoding.
- Requires every output timestep to exist in the base Zarr; otherwise a `ValueError` is raised with the list of missing timestamps.
- Copies `wrf` and `wrf_valid` rows by positional index. No interpolation or regridding is applied; the resulting values are bit-for-bit identical to the base Zarr.

**Dummy WRF:**
- Fills `wrf` with `np.random.normal(0, 1, ...)` values and sets `wrf_valid = True` everywhere.
- Useful when the base Zarr does not cover the forecast dates (for example, true future forecasts) but the CorrDiff runtime still expects the `wrf` variable to exist.

### 3.9 Write

If the target path already exists, it is removed with `shutil.rmtree` to prevent accidental mixing of old and new data, after which `ds.to_zarr(..., mode='w')` writes the final store.

---

## 4. Configuration

A single YAML file, [`configs/default.yml`](../graphcast_to_corrdiff_zarr/configs/default.yml), holds all paths, regridding options, WRF mode, and variable maps.

```yaml
paths:
  graphcast_nc_dir: /raid/apaudel/.../GRAPHCAST/RAW_OUTPUTS/2014-2015
  base_zarr_path:   /raid/apaudel/.../CORRDIFF/TRAINING/hampton_2007_2015_wpd_avg.zarr
  zarr_output_path: /raid/apaudel/.../GRAPHCAST/ZARR_FOR_CORRDIFF/graphcast_2014_2015.zarr

zarr:
  write_mode: w
  time_chunk: 1

regrid:
  enabled: true
  method: bilinear
  wrf_ref_file: null          # null -> take XLAT/XLONG from base_zarr

resize:
  enabled: false              # fallback; not used when regrid.enabled = true
  target_shape: [112, 112]

wrf:
  use_real: true              # true = copy real WRF; false = dummy

var_map:
  "2m_temperature":            t2m
  "10m_u_component_of_wind":   u10
  "10m_v_component_of_wind":   v10
  "mean_sea_level_pressure":   msl
  "temperature":               t
  "u_component_of_wind":       u
  "v_component_of_wind":       v
  "specific_humidity":         q
  "geopotential":              z_pl

var_era5_pressure:
  - {name: t,    pressure_levels: [0,1,2,3,4,5,6,7,8,9,10,11,12]}
  - {name: u,    pressure_levels: [0,1,2,3,4,5,6,7,8,9,10,11,12]}
  - {name: v,    pressure_levels: [0,1,2,3,4,5,6,7,8,9,10,11,12]}
  - {name: q,    pressure_levels: [0,1,2,3,4,5,6,7,8,9,10,11,12]}
  - {name: z_pl, pressure_levels: [0,1,2,3,4,5,6,7,8,9,10,11,12]}

var_era5_surface: [t2m, u10, v10]
```

CLI flags override the three path values from the YAML:

```bash
python pipeline.py --config configs/default.yml \
  --graphcast_nc_dir /new/nc/dir \
  --base_zarr_path   /new/base.zarr \
  --zarr_output_path /new/out.zarr
```

---

## 5. Running the Pipeline

### SLURM

```bash
cd aces/graphcast_to_corrdiff_zarr
sbatch submit.sh                                       # uses configs/default.yml
sbatch submit.sh --config configs/default.yml
sbatch submit.sh --graphcast-nc-dir /new/path --zarr-output-path /new/out.zarr
```

`submit.sh` loads the same `wrf-python` environment used by the Aurora packaging job (including the `LD_PRELOAD` directive for `libstdc++`). When invoked without arguments, the pipeline runs end-to-end against the YAML defaults.

### Local Execution

```bash
python pipeline.py --config configs/default.yml
```

### Log Location

Logs are written alongside the output Zarr rather than next to the code:

```
<zarr_output_path>.parent/
├── graphcast_2014_2015.zarr/
└── graphcast_2014_2015_logs/
    └── graphcast_to_zarr_<timestamp>.log
```

---

## 6. Output Zarr Schema

For a 112×112 domain with 12 timesteps, the resulting store has the following structure.

```
Dimensions:
  time                 = 12
  era5_channel         = 68
  wrf_channel          = 49
  south_north          = 112
  west_east            = 112
  south_north_stag     = 112
  west_east_stag       = 112

Coordinates:
  time                 (time,)                        int64    "hours since <origin>"
  era5_channel         (era5_channel,)                int64
  era5_variable        (era5_channel,)                <U26
  era5_pressure        (era5_channel,)                float64
  wrf_channel          (wrf_channel,)                 int64
  wrf_variable         (wrf_channel,)                 <U26
  wrf_pressure         (wrf_channel,)                 float64
  XLAT, XLONG          (south_north, west_east)       float32
  XLAT_U, XLONG_U      (south_north, west_east_stag)  float32
  XLAT_V, XLONG_V      (south_north_stag, west_east)  float32

Data variables:
  era5                 (time, era5_channel, south_north, west_east)  float32
  era5_valid           (time, era5_channel)                          bool
  era5_center          (era5_channel,)                               float32
  era5_scale           (era5_channel,)                               float32
  wrf                  (time, wrf_channel, south_north, west_east)   float32
  wrf_valid            (time,)                                       bool
  wrf_center           (wrf_channel,)                                float32
  wrf_scale            (wrf_channel,)                                float32
```

The channel counts shown reflect one particular training configuration: five pressure variables × 13 levels + three surface variables = 68 ERA5 channels; the WRF channel count is inherited from the base Zarr.

---

## 7. Verifying the Packaged Output

The notebook [`inspect_zarr.ipynb`](../graphcast_to_corrdiff_zarr/inspect_zarr.ipynb) provides an end-to-end inspection of the output Zarr. Set the three paths at the top of the notebook (`ZARR_PATH`, `BASE_ZARR_PATH`, `GRAPHCAST_NC_DIR`) and execute the cells in order.

Sections:

1. **Top-level summary** — dimensions, coordinates, data variables, global attributes.
2. **Time axis** — decodes the `hours since <origin>` encoding and reports `unique dt(h)` to confirm the 6-hour cadence. Multiple unique deltas indicate gaps between independent forecast initializations.
3. **Group summaries** — tabular listings of the `era5` and `wrf` channels with index, variable name, and pressure.
4. **Validity masks** — counts of `True` and `False` values in `era5_valid` and `wrf_valid`.
5. **Normalization statistics** — full per-channel listing of `center` and `scale` with variable name and pressure.
6. **Field plots** — two-dimensional maps of a chosen `(time_idx, channel_idx)` for both `era5` and `wrf`.
7. **Verification** — two independent correctness checks:
   - **7a. WRF round-trip.** With `wrf.use_real = true`, the output `wrf` must be bit-for-bit identical to the base Zarr's `wrf` at the matching timestamp. The notebook performs `np.array_equal`, reports `|diff|.max` and `|diff|.mean`, and renders a three-panel figure (output, base, absolute difference). Expected result: `array_equal = True` with zero difference.
   - **7b. ERA5 bracket check.** The output `era5` has been regridded from the GraphCast NetCDF onto the WRF grid, so pixel equality is not attainable. Instead, the notebook selects an `era5` channel, locates the matching `(time, variable)` in a source NetCDF, and verifies that the regridded minimum and maximum lie within the raw field's minimum and maximum (a bilinear regrid preserves this bracket). A side-by-side figure shows the raw lat/lon field and the regridded WRF-grid field.

---

## 8. Common Pitfalls

- **Missing timesteps when `wrf.use_real = true`.** Every output timestep must be present in the base Zarr; otherwise the pipeline raises an error. Remedies include widening the base Zarr's coverage, switching to dummy WRF, or restricting the GraphCast inputs.
- **Overlapping forecast initializations.** Multiple GraphCast NetCDFs may cover the same timestamps (for example, consecutive daily rollouts). The pipeline deduplicates by retaining the first occurrence; the log line `Deduplicating time: N -> M` reports how many timesteps were removed.
- **xESMF not installed.** The `regrid.enabled = true` path requires `xesmf`. Without it, the only option is the non-georeferenced `resize` fallback, which is unsuitable for CorrDiff. Install xESMF or rebuild the environment.
- **Variable or channel-count mismatch with the trained model.** Adding or removing variables from `var_era5_pressure` or `var_era5_surface` changes the `era5_channel` count. CorrDiff expects the exact channel layout it was trained on; the grafted `era5_center` and `era5_scale` from the base Zarr implicitly encode this layout. If the channel lists are modified, the normalization statistics must be regenerated accordingly.
- **Time encoding after read-back.** `xr.open_zarr` automatically decodes `"hours since ..."` encoding to `datetime64`, so `ds.time.attrs` may appear empty on read. This is expected behavior; the CF units remain on disk.
- **GraphCast `level` indices vs. hPa values.** The `pressure_levels` values in the configuration are indices, not pressure values. Index 7 corresponds to 500 hPa, not to 500.

---

## 9. Relationship to the Aurora Pipeline

The Aurora pipeline (`aurora_to_cordiff_zarr_real_wrf.py`) and this pipeline solve the same problem for two different weather models. The principal differences are summarized below.

| Aspect | Aurora pipeline | GraphCast pipeline |
| --- | --- | --- |
| Source dimension names | `latitude` / `longitude` | `lat` / `lon` |
| Vertical dimension | `level` or `bottom_top` | `level` |
| Surface 2m-temperature variable | `surf_2t` → `t2m` | `2m_temperature` → `t2m` |
| Handles `batch` dimension | No | Yes (squeezed) |
| Skips `ground_truth_*.nc` | Not applicable | Yes |
| Deduplicates overlapping times | No | Yes |
| Configuration format | Inline `CONFIG` dictionary | External YAML |
| Log location | `./log/` | Alongside the output Zarr |

To package outputs from another weather model (for example Pangu, FourCastNet, or GenCast), the simplest approach is to copy this pipeline and adjust `squeeze_and_rename`, the `var_map`, and the vertical-dimension handling. The remaining stages (regrid, graft, WRF attachment, write) are model-agnostic.
