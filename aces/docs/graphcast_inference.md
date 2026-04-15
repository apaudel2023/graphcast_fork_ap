# GraphCast Inference Pipeline

Location: [`aces/graphcast_pipeline/`](../graphcast_pipeline/)

This document describes the end-to-end pipeline for running GraphCast (Operational) weather forecasts on HPC. The pipeline supports two operating modes: rolling production forecasts over multi-day periods, and single-shot verification runs with ERA5 ground-truth comparison.

---

## 1. Model Background

The pipeline targets the **GraphCast Operational** variant from DeepMind:

> `GraphCast_operational — ERA5-HRES 1979-2021 — resolution 0.25° — pressure levels 13 — mesh 2-to-6 — precipitation output only`

Adaptation to other GraphCast variants (`GraphCast_small`, `GenCast`) requires swapping the checkpoint, the normalization statistics, and the variable lists. The high-level pipeline layout remains unchanged.

### 1.1 Model Inputs

GraphCast requires two consecutive ERA5 timesteps separated by six hours (`t−12h` and `t−6h`) as the initial condition.

| Input category | Variables |
| --- | --- |
| Pressure-level (×2 timesteps, 13 levels) | `geopotential`, `specific_humidity`, `temperature`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity` |
| Surface (×2 timesteps) | `2m_temperature`, `mean_sea_level_pressure`, `10m_u_component_of_wind`, `10m_v_component_of_wind` |
| Static (×1 snapshot) | `geopotential_at_surface` (orography), `land_sea_mask` |
| Forcing (time-dependent) | `toa_incident_solar_radiation` |

For `init_time = 2025-01-01 00Z`, the two input timesteps are `2024-12-31 12Z` and `2024-12-31 18Z`. All ERA5 inputs (surface, pressure, and static) are retrieved at runtime from the Copernicus Climate Data Store (CDS).

### 1.2 Model Outputs

Each rollout step advances the atmospheric state by six hours. Setting `rollout_steps = N` therefore produces `N × 6h` of forecast output.

| Variable type | Variables | Dimensions |
| --- | --- | --- |
| Pressure-level | `geopotential`, `temperature`, `specific_humidity`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity` | `(time, level, lat, lon)` |
| Surface | `2m_temperature`, `mean_sea_level_pressure`, `10m_u_component_of_wind`, `10m_v_component_of_wind` | `(time, lat, lon)` |
| Precipitation | `total_precipitation_6hr` | `(time, lat, lon)` |

On the native 0.25° global grid the spatial dimensions are `(lat, lon) = (721, 1440)`.

### 1.3 Required Artifacts

- **Model checkpoint** (`.npz`) from `gs://dm_graphcast/`
- **Normalization statistics**: `diffs_stddev_by_level.nc`, `mean_by_level.nc`, `stddev_by_level.nc` from the same bucket
- **CDS API credentials** in `~/.cdsapirc`

---

## 2. Pipeline Architecture

```
+----------------------------+
|        YAML config         |
|   (base + period/verify)   |
+-------------+--------------+
              |
              v
+----------------------------+
|          main.py           |
|      (pipeline driver)     |
+-------------+--------------+
              |
              v
+----------------------------+
|   Load GraphCast model     |
|         (model.py)         |
|  checkpoint + stats + JIT  |
+-------------+--------------+
              |
              v
+--------------------------------------------------+
|   For each init_time in the schedule:            |
|                                                  |
|   1. Download ERA5 inputs       era5_downloader  |
|      (and ground truth in                        |
|       verify mode)                               |
|                                                  |
|   2. Build xarray batch         batch_builder    |
|      (2 input steps + NaN                        |
|       target placeholders)                       |
|                                                  |
|   3. Run GraphCast rollout      model            |
|                                                  |
|   4. Save prediction .nc                         |
|      (absolute datetime coords)                  |
|                                                  |
|   5. Crop to region             crop             |
|      (optional)                                  |
|                                                  |
|   6. Run analysis               analysis         |
|      (verify mode only)                          |
|                                                  |
|   7. Clean up temporary                          |
|      ERA5 files                                  |
+--------------------------------------------------+
```

### 2.1 Module Responsibilities

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Parses configuration, selects mode, iterates the schedule, and invokes `run_single_forecast`. |
| `model.py` | Loads the GraphCast checkpoint, constructs the JIT-compiled `predict` function, and performs the autoregressive rollout. |
| `batch_builder.py` | Reads the three ERA5 NetCDFs and assembles the `xarray` batch consumed by GraphCast. Handles CDS-to-GraphCast variable renames (`t2m → 2m_temperature`, `u → u_component_of_wind`, etc.) and lat/lon orientation. |
| `era5_downloader.py` | CDS API wrappers for pressure-level, single-level, and static fields; also retrieves ground truth for verification runs. |
| `crop.py` | Regional cropping around a `(lat, lon)` center using a window expressed in grid points. |
| `analysis.py` | Verification utilities: RMSE and MAE tables, three-panel comparison plots, error-vs-lead-time curves, GIF animations, and crop-preview figures. |
| `utils.py` | Configuration loading and merging, logging, scheduling helpers, validation. |
| `submit.sh` | SLURM submission wrapper. |

---

## 3. Operating Modes

### 3.1 Forecast Mode (Production)

Forecast mode performs rolling re-initialization across a date range. The pipeline advances from `start` to `end` in steps of `reinit_interval_hours` (default: 24 hours) and runs one forecast per step. With the default interval, `rollout_steps = 24h / 6h = 4` per initialization.

Per-iteration workflow:
1. Download the required ERA5 inputs.
2. Execute the four-step rollout.
3. Save and optionally crop the prediction.
4. When cropping is enabled, remove the full-resolution NetCDF and retain only the cropped file to conserve disk space.
5. Delete temporary ERA5 files.

Previously completed initializations are skipped automatically upon resubmission: the pipeline checks for the expected output filename before running each step.

```bash
sbatch submit.sh --config periods_job1.yml
```

### 3.2 Verification Mode

Verification mode performs a single forecast with ground-truth comparison and generates an analysis report.

- Downloads ERA5 inputs *and* ground truth for the rollout window.
- Writes both the prediction and ground-truth NetCDFs at full resolution, plus the cropped prediction if `crop.enabled` is `true`.
- Computes RMSE and MAE per variable, per pressure level, per rollout step.
- Produces three-panel comparison plots, error-vs-lead-time curves, GIF animations, and crop-preview figures.

```bash
sbatch submit.sh --config verify_test.yml
```

---

## 4. Configuration

Configurations are layered. A **base** configuration supplies defaults (paths, variable lists, crop settings, analysis defaults); an **overlay** configuration specifies what is unique to a particular run (mode, period, rollout length).

```
configs/
├── base/graphcast.yml       # defaults (model paths, variables, crop, analysis)
├── periods_job1.yml         # forecast overlay (run_periods)
├── periods_job2.yml
└── verify_test.yml          # verification overlay (mode, verification, analysis)
```

### 4.1 Forecast Overlay

```yaml
run_periods:
  period1:
    start: "2024-01-01"
    end:   "2024-01-15"
  period2:
    start: "2024-02-01"
    end:   "2024-02-15"
```

Each period is written to its own output subdirectory. Multiple periods may be packed into a single SLURM job.

### 4.2 Verification Overlay

```yaml
run:
  mode: "verify"

verification:
  init_time: "2024-01-01"     # date-only OR "YYYY-MM-DD HH:MM"
  rollout_steps: 4            # 4 × 6h = 24h

analysis:
  enabled: true
  crop_preview: true
  metrics: true
  plots: true
  animations: true
  levels: null                # null = all 13 pressure levels
```

### 4.3 Semantics of `init_time`

Two string formats are accepted, each with a distinct meaning.

| Format | Interpretation | ERA5 inputs used | Predicted times (4 steps) |
| --- | --- | --- | --- |
| `"2024-01-01"` | Predictions for that calendar day, starting at 00Z | Dec 31 12Z, 18Z | Jan 1 **00Z**, 06Z, 12Z, 18Z |
| `"2024-01-01 00:00"` | The given hour is the initialization point (last model input) | Dec 31 18Z, Jan 1 00Z | Jan 1 **06Z**, 12Z, 18Z, Jan 2 00Z |

- **Date-only** format specifies the start of the prediction period.
- **Date + hour** format specifies the final model input timestamp; predictions commence six hours later.

### 4.4 Selected Base-Configuration Options

| Key | Default | Description |
| --- | --- | --- |
| `run.mode` | `forecast` | `forecast` or `verify` |
| `run.reinit_interval_hours` | `24` | Must be a multiple of 6 |
| `run.cleanup_temp` | `true` | Delete temporary ERA5 files per iteration |
| `crop.enabled` | `true` | Apply regional cropping |
| `crop.center_lat` / `crop.center_lon` | `39.2 / -76.3` | Center of the crop window |
| `crop.window_size` | `60` | Crop window size in grid points |
| `analysis.enabled` | `false` | Enable analysis (verification only) |
| `analysis.crop_preview` | `false` | Generate global + cropped side-by-side figures |
| `analysis.levels` | `null` | Pressure levels to plot (`null` = all) |

---

## 5. Output Layout

### 5.1 Forecast Mode

```
GRAPHCAST_OUTPUTS/
└── 2024_01_01_2024_01_15/                  <- one directory per period
    ├── logs/
    │   └── slurm_run_12345_period1.log
    ├── graphcast_2024_01_01_cropped.nc     <- full-resolution removed after cropping
    ├── graphcast_2024_01_02_cropped.nc
    └── ...
```

### 5.2 Verification Mode

```
GRAPHCAST_OUTPUTS/
└── verify_20240101/
    ├── logs/
    ├── graphcast_2024_01_01.nc             <- full-resolution prediction (retained)
    ├── graphcast_2024_01_01_cropped.nc
    ├── ground_truth_2024_01_01.nc          <- full-resolution ERA5 ground truth
    └── analysis/
        ├── metrics/metrics_summary.csv     <- RMSE + MAE per variable × level × step
        ├── figures/
        │   ├── comparisons/                <- truth | prediction | residual
        │   │   ├── 2m_temperature/step_01_+006h.png
        │   │   └── geopotential_500hPa/step_01_+006h.png
        │   ├── error_evolution/            <- error versus lead time
        │   │   ├── pressure_geopotential_rmse.png
        │   │   ├── pressure_temperature_mae.png
        │   │   ├── surface_temperature_rmse.png
        │   │   └── surface_wind_rmse.png
        │   └── crop_preview/               <- global + red box + zoomed crop
        │       └── 2m_temperature_crop_preview.png
        └── animations/
            ├── 2m_temperature.gif
            └── geopotential_500hPa.gif
```

### 5.3 Prediction NetCDF Schema

- The time coordinate holds **absolute** `datetime64[ns]` values rather than lead times. For `init_time = 2024-01-01 00Z` and `rollout_steps = 4`, the times are `[00Z, 06Z, 12Z, 18Z]`. Internally, GraphCast emits lead times `[6h, 12h, ...]` relative to the last input (`init_time − 6h`); `save_predictions` converts these to absolute timestamps.
- The `batch` dimension is squeezed prior to writing.
- Pressure-level coordinate values: `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` hPa.

---

## 6. Running the Pipeline

### SLURM (recommended on HPC)

```bash
cd aces/graphcast_pipeline

# Forecast
sbatch submit.sh --config periods_job1.yml
sbatch submit.sh --config periods_job1.yml --output-dir /raid/apaudel/test_run
sbatch submit.sh --config periods_job2.yml --tag jan_run

# Verification
sbatch submit.sh --config verify_test.yml
```

### Local Execution

```bash
python main.py --config verify_test.yml
python main.py --config periods_job1.yml --output-dir ./local_test
```

### Resuming After Failure

Resubmit the original command. The pipeline verifies the expected output filename for each `init_time` and skips initializations that are already complete.

---

## 7. Analysis (Verification Mode)

### Metrics

Metrics are computed **per variable × pressure level × rollout step** with no averaging across levels and written to CSV.

- `RMSE = sqrt(mean((pred − truth)^2))` over `(lat, lon)`
- `MAE  = mean(|pred − truth|)` over `(lat, lon)`

### Comparison Plots

Three-panel figures per `(variable, level, timestep)`: ground truth, prediction, and signed residual (diverging colormap). The truth and prediction panels share a **fixed colorbar** (global minimum and maximum across all timesteps) to produce stable color scales in animations.

### Error-Evolution Plots

- Pressure-level variables: one figure per variable, one line per pressure level.
- Surface variables: grouped by physical similarity (temperature alone; `u10` and `v10` together; MSLP alone).

### Crop Preview

A side-by-side figure showing the global prediction with a red rectangle marking the crop domain alongside the corresponding zoomed crop view. Generated for three surface variables and one representative pressure level (500 hPa) per pressure variable.

---

## 8. Dependencies

The full environment specification is provided in [`aces/requirements.yaml`](../requirements.yaml).

Key packages:
- Python 3.10+
- `jax[cuda12]` (GPU) or CPU JAX
- GraphCast dependencies: `dm-haiku`, `chex`, `jraph`, `trimesh` (see repository-root `setup.py`)
- `cdsapi` (with `~/.cdsapirc` configured), `xarray`, `netCDF4`, `PyYAML`
- `matplotlib`, `pandas`, `Pillow` (for animations)

HPC environment creation:

```bash
conda env create -f aces/requirements.yaml -n env_graphcast
```

---

## 9. Common Pitfalls

- **CDS throttling and queue times.** Initial downloads for a new period can stall. The skip-on-exists logic allows safe resubmission.
- **`init_time` interpretation.** The date-only versus date + hour formats differ by six hours. Confirm the intended convention before constructing a long schedule.
- **Disk usage in forecast mode.** With cropping disabled, every initialization writes a full-resolution NetCDF (several hundred MB). With cropping enabled, only the cropped file is retained.
- **Checkpoint and statistics compatibility.** The operational checkpoint is bound to the specific 13 pressure levels listed above. Changing `levels` requires a corresponding checkpoint.
- **GPU memory and JIT compilation.** The JIT-compiled rollout holds a single chunk of state in memory; long rollouts fit on a single GPU, but the first compilation is expensive. The pipeline keeps the model loaded across iterations to amortize this cost.
