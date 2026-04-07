# GraphCast Operational — Inference Pipeline

Modular pipeline for running [GraphCast](https://github.com/google-deepmind/graphcast) weather forecasts on HPC, with optional verification against ERA5 reanalysis.

> **Note:** This pipeline is built for the **GraphCast Operational** model variant:
> `GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only`
>
> Other GraphCast variants (e.g., GraphCast_small, GenCast) use different checkpoints, variable sets, and mesh configurations. Adapting this pipeline for other variants would require changes to the variable lists, checkpoint loading, and potentially the batch construction.

## What is GraphCast?

GraphCast is a machine learning weather prediction model developed by DeepMind. It operates on a **0.25-degree global grid** with **13 pressure levels** and produces **6-hourly forecasts** autoregressively.

### What it expects as input

GraphCast requires **two consecutive ERA5 reanalysis timesteps** separated by 6 hours as initial conditions:

| Input | Description |
|-------|-------------|
| **Pressure-level variables** (2 timesteps) | geopotential, specific_humidity, temperature, u/v wind, vertical_velocity at 13 levels |
| **Surface variables** (2 timesteps) | 2m_temperature, mean_sea_level_pressure, 10m u/v wind |
| **Static fields** (1 snapshot) | surface geopotential (orography), land-sea mask |

For a forecast initialized at `2025-01-01 00Z`, the two input timesteps are:
- `2024-12-31 12:00Z` (t-12h)
- `2024-12-31 18:00Z` (t-6h)

These are downloaded automatically from the [CDS API](https://cds.climate.copernicus.eu/).

### What it produces

Each rollout step predicts **6 hours ahead**. For N rollout steps, the output covers N x 6 hours:

| Variable type | Variables | Dimensions |
|---------------|-----------|------------|
| Pressure-level | geopotential, temperature, specific_humidity, u/v wind, vertical_velocity | (time, level, lat, lon) |
| Surface | 2m_temperature, mean_sea_level_pressure, 10m u/v wind | (time, lat, lon) |
| Precipitation | total_precipitation_6hr | (time, lat, lon) |

### What else is needed

- **Model checkpoint** (`*.npz`) — downloaded from `gs://dm_graphcast/`
- **Normalization statistics** — three `.nc` files (`diffs_stddev_by_level.nc`, `mean_by_level.nc`, `stddev_by_level.nc`) from the same bucket

---

## Pipeline overview

```
                    ┌─────────────────┐
                    │   YAML Config   │
                    │  (base + period)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    main.py      │
                    │  (entry point)  │
                    └────────┬────────┘
                             │
                ┌────────────▼────────────┐
                │  Load model once        │
                │  (model.py)             │
                │  checkpoint + stats     │
                │  + JIT compile          │
                └────────────┬────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  For each day in schedule:  │
              │                             │
              │  1. Download ERA5 inputs    │ ◄── era5_downloader.py
              │     (+ ground truth if      │
              │      verify mode)           │
              │                             │
              │  2. Build xarray batch      │ ◄── batch_builder.py
              │     (2 input steps +        │
              │      NaN placeholders)      │
              │                             │
              │  3. Run GraphCast rollout   │ ◄── model.py
              │     (autoregressive,        │
              │      chunked prediction)    │
              │                             │
              │  4. Save predictions.nc     │
              │     (absolute time coords)  │
              │                             │
              │  5. Crop to region          │ ◄── crop.py
              │     (optional)              │
              │                             │
              │  6. Run analysis            │ ◄── analysis.py
              │     (verify mode only)      │
              │                             │
              │  7. Cleanup temp ERA5 files │
              └─────────────────────────────┘
```

---

## Two operating modes

### Forecast mode (production)

Rolling 24-hour re-initialization over multi-day periods. Each day:
- Downloads 2 ERA5 input timesteps
- Runs 4 rollout steps (24h)
- Saves prediction `.nc` (+ cropped version)
- Cleans up temp ERA5 files

```bash
sbatch submit.sh --config periods_job1.yml
```

### Verify mode (validation)

Single forecast with ground truth comparison and optional analysis:
- Downloads ERA5 inputs + ground truth for rollout period
- Runs inference and saves both prediction and ground truth `.nc`
- Computes RMSE/MAE metrics, generates comparison plots, GIF animations
- Generates crop preview showing global vs regional domain

```bash
sbatch submit.sh --config verify_test.yml
```

---

## Directory structure

```
aces/graphcast_pipeline/
├── main.py                  # Entry point
├── model.py                 # GraphCast model loading + inference
├── batch_builder.py         # Build xarray batch from ERA5 files
├── era5_downloader.py       # ERA5 download via CDS API
├── crop.py                  # Regional cropping
├── analysis.py              # Metrics, plots, GIFs (verify mode)
├── utils.py                 # Config, logging, scheduling, validation
├── submit.sh                # SLURM submission script
├── README.md
└── configs/
    ├── base/
    │   └── graphcast.yml    # Base configuration (paths, variables, crop, etc.)
    ├── periods_job1.yml     # Forecast periods for job 1
    ├── periods_job2.yml     # Forecast periods for job 2
    └── verify_test.yml      # Verification config
```

---

## Configuration

### Base config (`configs/base/graphcast.yml`)

Contains model paths, ERA5 variables, crop settings, and analysis options. Loaded automatically by `main.py`.

### Period configs (forecast mode)

Override `run_periods` to specify date ranges:

```yaml
# periods_job1.yml
run_periods:
  period1:
    start: "2024-01-01"
    end: "2024-01-15"
  period2:
    start: "2024-02-01"
    end: "2024-02-15"
```

### Verification config (verify mode)

Override `run.mode`, `verification`, and `analysis`:

```yaml
# verify_test.yml
run:
  mode: "verify"

verification:
  init_time: "2024-01-01 00:00"
  rollout_steps: 4               # 4 x 6h = 24h

analysis:
  enabled: true
  crop_preview: true
```

### Key config options

| Key | Default | Description |
|-----|---------|-------------|
| `run.mode` | `"forecast"` | `"forecast"` or `"verify"` |
| `run.reinit_interval_hours` | `24` | Re-initialization interval (must be multiple of 6) |
| `run.cleanup_temp` | `true` | Delete temp ERA5 files after each day |
| `crop.enabled` | `true` | Crop predictions to regional domain |
| `crop.center_lat` | `39.2` | Crop center latitude |
| `crop.center_lon` | `-76.3` | Crop center longitude |
| `crop.window_size` | `60` | Crop window size (grid points) |
| `analysis.enabled` | `false` | Run analysis (only in verify mode) |
| `analysis.crop_preview` | `false` | Generate global vs crop side-by-side plots |
| `analysis.levels` | `null` | Pressure levels for plots (`null` = all 13) |

---

## Output structure

### Forecast mode

In forecast mode with cropping enabled, the full-resolution `.nc` is **deleted** after cropping to save disk space. Only the cropped file is kept.

```
GRAPHCAST_OUTPUTS/
  2024_01_01_2024_01_15/                    # one dir per period
    logs/
      slurm_run_12345_period1.log
    graphcast_2024_01_01_cropped.nc         # cropped (full-res deleted after crop)
    graphcast_2024_01_02_cropped.nc
    ...
```

### Verify mode

In verify mode, **both** full-resolution and cropped prediction files are kept. Ground truth is always saved at full resolution.

```
GRAPHCAST_OUTPUTS/
  verify_20240101_00/
    logs/
    graphcast_2024_01_01.nc                 # full resolution prediction (kept)
    graphcast_2024_01_01_cropped.nc         # cropped prediction
    ground_truth_2024_01_01.nc              # full resolution ERA5 ground truth
    analysis/
      metrics/
        metrics_summary.csv                 # RMSE + MAE per variable x level x step
      figures/
        comparisons/
          2m_temperature/
            step_01_+006h.png               # truth | prediction | residual
            step_02_+012h.png
            ...
          geopotential_500hPa/
            step_01_+006h.png
            ...
        error_evolution/
          pressure_geopotential_rmse.png    # RMSE vs lead time, one line per level
          pressure_temperature_mae.png
          surface_temperature_rmse.png      # surface vars grouped by type
          surface_wind_rmse.png
          ...
        crop_preview/
          2m_temperature_crop_preview.png   # global with red box + zoomed crop
          geopotential_500hPa_crop_preview.png
          ...
      animations/
        2m_temperature.gif                  # rollout animation
        geopotential_500hPa.gif
        ...
```

---

## Usage

### Submit a forecast job

```bash
cd aces/graphcast_pipeline
sbatch submit.sh --config periods_job1.yml
```

### Submit with custom output directory

```bash
sbatch submit.sh --config periods_job1.yml --output-dir /raid/apaudel/test_run
```

### Run verification

```bash
sbatch submit.sh --config verify_test.yml
```

### Run locally (no SLURM)

```bash
python main.py --config verify_test.yml
python main.py --config periods_job1.yml --output-dir ./local_test
```

### Resume after failure

Re-submit the same command. The pipeline checks for existing output files and skips completed days automatically.

---

## Analysis details (verify mode)

### Metrics

- **RMSE** = `sqrt(mean((prediction - truth)^2))` over lat, lon
- **MAE** = `mean(|prediction - truth|)` over lat, lon

Computed per variable, per pressure level, per rollout step. No averaging across levels. Saved as CSV for easy inspection.

### Comparison plots

Three-panel plots for each (variable, level, timestep):
- **ERA5 Truth** — ground truth from reanalysis
- **GraphCast Prediction** — model output
- **Prediction - Truth** — signed residual with diverging colormap (blue = underprediction, red = overprediction)

Truth and prediction panels share a **fixed colorbar** (global min/max across all timesteps) so GIF animations show stable colors.

### Error evolution plots

Line plots of error metric vs lead time:
- **Pressure variables**: one plot per variable, one line per pressure level
- **Surface variables**: grouped by physical similarity (temperature alone, u10+v10 together, MSLP alone)

### Crop preview

Side-by-side plot showing the global prediction with a **red rectangle** marking the crop region, alongside the zoomed cropped view. Generated for 3 surface variables + 1 representative pressure level (500 hPa) per pressure variable.

---

## Requirements

See [`aces/requirements.yaml`](../requirements.yaml) for the full conda environment specification.

Key dependencies:
- Python 3.10+
- JAX with GPU support (`jax[cuda12]`)
- GraphCast dependencies (`dm-haiku`, `chex`, `jraph`, etc. — see `setup.py` in repo root)
- CDS API credentials configured (`~/.cdsapirc`)
- `matplotlib`, `pandas`, `Pillow` (PIL), `pyyaml`, `xarray`, `netcdf4`

### Environment setup (HPC)

```bash
conda env create -f aces/requirements.yaml -n env_graphcast
```
