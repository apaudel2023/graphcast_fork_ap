"""GraphCast rolling-window inference pipeline.

Modes:
    forecast — rolling 24h re-initialization over date ranges (multi-period)
    verify   — single init_time + rollout_steps with ground truth and analysis

Usage:
    # Forecast (production)
    python main.py --config periods_job1.yml

    # Verification
    python main.py --config verify_test.yml
"""

import argparse
import copy
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray

# Add repo root to path so `from graphcast import ...` works
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from utils import (
    load_config,
    setup_logger,
    log_config,
    ensure_dir,
    cleanup_dir,
    validate_cfg,
    compute_rollout_steps,
    compute_period_bounds,
    build_reinit_schedule,
    get_era5_input_times,
    get_expected_output_file,
)
from era5_downloader import download_era5_inputs, download_era5_ground_truth
from batch_builder import build_batch, load_ground_truth
from model import GraphCastModel
from crop import crop_prediction
from analysis import GraphCastAnalysis


def save_predictions(predictions, init_time, output_path, logger):
    """Save predictions with absolute datetime time coordinates.

    Prediction lead times [6h, 12h, ...] are relative to the last input timestep
    (init_time - 6h). So absolute = (init_time - 6h) + lead_time.
    For init_time=2024-01-01 00Z with 4 steps: [00Z, 06Z, 12Z, 18Z].
    """
    last_input_time = init_time - timedelta(hours=6)
    abs_times = np.array([
        np.datetime64(last_input_time, "ns") + td
        for td in predictions.coords["time"].values
    ])
    ds = predictions.copy()
    ds = ds.assign_coords(time=abs_times)

    if "batch" in ds.dims:
        ds = ds.squeeze("batch", drop=True)

    ds.to_netcdf(output_path)
    logger.info(f"  Saved predictions -> {output_path}")
    logger.info(f"    time range: {ds.coords['time'].values[0]} to {ds.coords['time'].values[-1]}")
    logger.info(f"    variables:  {sorted(ds.data_vars)}")
    logger.info(f"    shape:      {dict(ds.sizes)}")


def run_single_forecast(gc_model, cfg, current_time, rollout_steps, output_root, logger):
    """Run one forecast window: download, build, predict, save, crop, analyze.

    Used by both forecast mode (called per day) and verify mode (called once).
    Returns (pred_path, gt_path_or_None).
    """
    mode = cfg["run"]["mode"]
    crop_enabled = cfg["crop"]["enabled"]

    era5_input_times = get_era5_input_times(current_time)
    last_input_time = current_time - timedelta(hours=6)
    first_pred_time = current_time  # last_input + 6h = init_time
    last_pred_time = current_time + timedelta(hours=(rollout_steps - 1) * 6)

    logger.info(f"  init_time:       {current_time}")
    logger.info(f"  ERA5 input 1:    {era5_input_times[0]}")
    logger.info(f"  ERA5 input 2:    {era5_input_times[1]}")
    logger.info(f"  rollout_steps:   {rollout_steps} x 6h = {rollout_steps * 6}h")
    logger.info(f"  prediction range: {first_pred_time} to {last_pred_time}")
    logger.info(f"  mode:            {mode}")
    logger.info(f"  crop:            {crop_enabled}")

    ts_tag = current_time.strftime("%Y%m%d_%H")
    era5_dir = os.path.join(output_root, f"era5_tmp_{ts_tag}")
    ensure_dir(era5_dir)

    # Step 1: Download ERA5 inputs
    logger.info("  Downloading ERA5 inputs ...")
    static_path, surface_path, pressure_path = download_era5_inputs(
        cfg, era5_input_times, era5_dir, logger
    )

    # Step 1b: Download ground truth (verify mode)
    gt_pressure_path = None
    gt_surface_path = None
    if mode == "verify":
        logger.info("  Downloading ERA5 ground truth ...")
        gt_pressure_path, gt_surface_path = download_era5_ground_truth(
            cfg, current_time, rollout_steps, era5_dir, logger
        )

    # Step 2: Build batch
    logger.info("  Building batch ...")
    batch = build_batch(
        static_path, surface_path, pressure_path,
        current_time, rollout_steps, logger
    )

    # Step 3: Run inference
    logger.info(f"  Running GraphCast rollout ({rollout_steps} steps) ...")
    predictions = gc_model.predict(batch, rollout_steps)

    # Step 4: Save predictions (full resolution)
    date_tag = current_time.strftime("%Y_%m_%d")
    pred_path = os.path.join(output_root, f"graphcast_{date_tag}.nc")
    save_predictions(predictions, current_time, pred_path, logger)

    # Step 4b: Save ground truth (verify mode, full resolution)
    gt_path = None
    if mode == "verify" and gt_pressure_path is not None:
        logger.info("  Loading and saving ground truth ...")
        gt_ds = load_ground_truth(gt_pressure_path, gt_surface_path, current_time, logger)
        gt_path = os.path.join(output_root, f"ground_truth_{date_tag}.nc")
        gt_ds.to_netcdf(gt_path)
        logger.info(f"  Saved ground truth -> {gt_path}")

    # Step 5: Crop
    if crop_enabled:
        logger.info("  Cropping prediction ...")
        crop_prediction(pred_path, cfg, logger)

        if mode == "forecast":
            # Forecast mode: delete full-resolution file, keep only cropped
            os.remove(pred_path)
            logger.info(f"  Deleted full-resolution prediction: {pred_path}")
        # Verify mode: keep both full-resolution and cropped

    # Step 6: Cleanup temp ERA5 files
    if cfg["run"]["cleanup_temp"]:
        cleanup_dir(era5_dir, logger)

    return pred_path, gt_path


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def parse_init_time(init_time_str):
    """Parse init_time from config.

    Two formats with different semantics:

    - "YYYY-MM-DD" (date only):
        Predictions START at 00Z of this day.
        ERA5 inputs: previous day 12Z and 18Z.
        Example: "2024-01-01" with 4 steps → predictions [00Z, 06Z, 12Z, 18Z]

    - "YYYY-MM-DD HH:MM" (date + hour):
        The specified hour is the INITIALIZATION POINT (last model input).
        Predictions start 6h AFTER the specified hour.
        ERA5 inputs: specified_hour - 6h and specified_hour.
        Example: "2024-01-01 00:00" with 4 steps → predictions [06Z, 12Z, 18Z, Jan 2 00Z]

    Returns the internal init_time used by the pipeline.
    """
    # Try date+hour format first
    try:
        user_time = datetime.strptime(init_time_str, "%Y-%m-%d %H:%M")
        # User specified an hour = the initialization point (last input).
        # Shift forward by 6h so the pipeline produces predictions starting
        # 6h after the user's specified hour.
        return user_time + timedelta(hours=6)
    except ValueError:
        pass

    # Date-only format
    try:
        return datetime.strptime(init_time_str, "%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(
        f"Invalid init_time '{init_time_str}'. "
        f"Expected format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'"
    )


def run_verify(gc_model, cfg, tag, output_dir):
    """Single-forecast verification with ground truth comparison and analysis."""
    v_cfg = cfg["verification"]
    init_time_str = v_cfg["init_time"]
    init_time = parse_init_time(init_time_str)
    rollout_steps = v_cfg["rollout_steps"]

    # Use user-provided string for directory naming (not the shifted internal time)
    dir_tag = init_time_str.replace(" ", "_").replace(":", "").replace("-", "")
    verify_dir = os.path.join(output_dir, f"verify_{dir_tag}")
    ensure_dir(verify_dir)

    log_dir = os.path.join(verify_dir, "logs")
    ensure_dir(log_dir)
    logger = setup_logger(log_dir, tag=f"{tag}_verify")
    log_config(logger, cfg, header="Verification configuration")

    last_input_time = init_time - timedelta(hours=6)
    first_pred_time = last_input_time + timedelta(hours=6)
    last_pred_time = last_input_time + timedelta(hours=rollout_steps * 6)

    logger.info("=" * 80)
    logger.info(f"Verification mode")
    logger.info(f"  config init_time:  {init_time_str}")
    logger.info(f"  ERA5 input 1:      {init_time - timedelta(hours=12)}")
    logger.info(f"  ERA5 input 2:      {last_input_time}")
    logger.info(f"  rollout_steps:     {rollout_steps} ({rollout_steps * 6}h)")
    logger.info(f"  prediction range:  {first_pred_time} to {last_pred_time}")
    logger.info(f"  crop:              {cfg['crop']['enabled']}")
    logger.info("=" * 80)

    # Run the forecast
    pred_path, gt_path = run_single_forecast(
        gc_model, cfg, init_time, rollout_steps, verify_dir, logger
    )

    # Run analysis
    analysis_cfg = cfg.get("analysis", {})
    if analysis_cfg.get("enabled", False) and gt_path is not None:
        logger.info("Running analysis ...")
        analysis_dir = os.path.join(verify_dir, "analysis")

        preds_ds = xarray.load_dataset(pred_path)
        gt_ds = xarray.load_dataset(gt_path)

        analyzer = GraphCastAnalysis(
            predictions=preds_ds,
            ground_truth=gt_ds,
            analysis_dir=analysis_dir,
            cmap=analysis_cfg.get("cmap", "viridis"),
            residual_cmap=analysis_cfg.get("residual_cmap", "RdBu_r"),
            logger=logger,
        )
        analyzer.run(
            levels=analysis_cfg.get("levels", None),
            fps=analysis_cfg.get("fps", 2),
            metrics=analysis_cfg.get("metrics", True),
            plots=analysis_cfg.get("plots", True),
            animations=analysis_cfg.get("animations", True),
        )

        # Crop preview: global vs cropped side-by-side
        if analysis_cfg.get("crop_preview", False) and cfg["crop"]["enabled"]:
            logger.info("Generating crop preview plots ...")
            analyzer.plot_crop_preview(cfg["crop"])

    logger.info("=" * 80)
    logger.info(f"Verification complete. Outputs in: {verify_dir}")


# ---------------------------------------------------------------------------
# Forecast mode
# ---------------------------------------------------------------------------

def run_forecast(gc_model, cfg, tag, output_dir):
    """Rolling-window forecast over multiple periods."""
    reinit_interval_hours = cfg["run"]["reinit_interval_hours"]
    rollout_steps = compute_rollout_steps(reinit_interval_hours)
    crop_enabled = cfg["crop"]["enabled"]

    for pname, pvals in cfg["run_periods"].items():
        start_str = pvals["start"]
        end_str = pvals["end"]

        start_dt, end_dt = compute_period_bounds(
            start_str, end_str, reinit_interval_hours
        )

        period_dir = os.path.join(
            output_dir,
            f"{datetime.strptime(start_str, '%Y-%m-%d').strftime('%Y_%m_%d')}_"
            f"{datetime.strptime(end_str, '%Y-%m-%d').strftime('%Y_%m_%d')}"
        )
        ensure_dir(period_dir)

        log_dir = os.path.join(period_dir, "logs")
        ensure_dir(log_dir)
        logger = setup_logger(log_dir, tag=f"{tag}_{pname}")
        log_config(logger, cfg, header=f"[{pname}] Resolved configuration")

        schedule = build_reinit_schedule(start_dt, end_dt, reinit_interval_hours)

        logger.info("=" * 80)
        logger.info(f"[{pname}] Period: {start_dt} -> {end_dt}")
        logger.info(f"[{pname}] Rollout steps: {rollout_steps} (x 6h)")
        logger.info(f"[{pname}] Forecast starts: {len(schedule)}")

        n_total = 0
        n_skipped = 0
        n_ran = 0

        for current_time in schedule:
            n_total += 1
            logger.info("=" * 80)

            expected_output = get_expected_output_file(period_dir, current_time, crop_enabled)
            if os.path.exists(expected_output):
                logger.info(f"[{pname}] Exists, skipping: {expected_output}")
                n_skipped += 1
                continue

            logger.info(f"[{pname}] Forecast: {current_time}")

            try:
                run_single_forecast(
                    gc_model, cfg, current_time, rollout_steps, period_dir, logger
                )
                n_ran += 1
            except Exception as e:
                logger.exception(f"[{pname}] Failed for {current_time}: {e}")
                raise

        logger.info("=" * 80)
        logger.info(f"[{pname}] Complete. total={n_total}, skipped={n_skipped}, ran={n_ran}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg, tag, output_dir):
    if output_dir is not None:
        cfg["output_dir"] = output_dir

    validate_cfg(cfg)
    base_output = cfg["output_dir"]
    ensure_dir(base_output)

    # Load model once
    model_logger = setup_logger(base_output, tag=f"{tag}_model_load")
    gc_model = GraphCastModel(cfg, model_logger)

    mode = cfg["run"]["mode"]
    if mode == "verify":
        run_verify(gc_model, cfg, tag, base_output)
    elif mode == "forecast":
        run_forecast(gc_model, cfg, tag, base_output)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'forecast' or 'verify'.")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    default_base_config = here / "configs" / "base" / "graphcast.yml"

    parser = argparse.ArgumentParser(
        description="GraphCast inference pipeline (forecast or verify mode)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Config YAML (resolved under ./configs/ if not absolute)",
    )
    parser.add_argument(
        "--tag", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run tag for log file naming",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", type=str, default=None,
        help="Root output directory (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(default_base_config, args.config, validate=True)
    main(cfg, args.tag, args.output_dir)
