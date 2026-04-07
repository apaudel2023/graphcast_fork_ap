"""Utilities: config loading, logging, scheduling, validation."""

import os
import shutil
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# YAML / Config
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_config_path(path_str, default_dir=None):
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p.resolve()
    if default_dir is not None:
        candidate = Path(default_dir) / path_str
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find config file '{path_str}'. "
        f"Tried as given and under '{default_dir}'."
    )


def deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(base_config_path, override_config_path=None, validate=True):
    here = Path(__file__).resolve().parent
    configs_dir = here / "configs"
    base_dir = configs_dir / "base"

    base_config_path = resolve_config_path(base_config_path, default_dir=base_dir)
    cfg = load_yaml(base_config_path)

    if override_config_path is not None:
        override_config_path = resolve_config_path(
            override_config_path, default_dir=configs_dir
        )
        override_cfg = load_yaml(override_config_path)
        cfg = deep_update(cfg, override_cfg)

    if validate:
        validate_cfg(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir, tag="graphcast_run"):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{tag}.log")

    logger = logging.getLogger(f"GraphCastPipeline.{tag}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    fh = logging.FileHandler(logfile, mode="a")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to: {logfile}")
    return logger


def log_config(logger, cfg, header="Resolved configuration"):
    logger.info("=" * 80)
    logger.info(header)
    logger.info("\n" + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def compute_rollout_steps(reinit_interval_hours):
    """GraphCast time step is always 6h."""
    if reinit_interval_hours <= 0:
        raise ValueError("reinit_interval_hours must be > 0")
    if reinit_interval_hours % 6 != 0:
        raise ValueError(
            f"reinit_interval_hours ({reinit_interval_hours}) must be a multiple of 6"
        )
    return reinit_interval_hours // 6


def get_era5_input_times(init_time):
    """GraphCast needs two ERA5 timesteps at t-12h and t-6h."""
    t1 = init_time - timedelta(hours=12)
    t2 = init_time - timedelta(hours=6)
    return [t1, t2]


def compute_period_bounds(start_date_str, end_date_str, reinit_interval_hours):
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid start date '{start_date_str}'. Expected YYYY-MM-DD.") from e

    try:
        end_date_midnight = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid end date '{end_date_str}'. Expected YYYY-MM-DD.") from e

    if reinit_interval_hours <= 24:
        if 24 % reinit_interval_hours != 0:
            raise ValueError(
                f"reinit_interval_hours={reinit_interval_hours} must divide 24 exactly."
            )
        offset_hours = 24 - reinit_interval_hours
    else:
        offset_hours = 0

    end_dt = end_date_midnight + timedelta(hours=offset_hours)

    if end_dt < start_dt:
        raise ValueError(f"Invalid period: start {start_dt} > end {end_dt}")

    return start_dt, end_dt


def build_reinit_schedule(start_dt, end_dt, reinit_interval_hours):
    schedule = []
    current = start_dt
    while current <= end_dt:
        schedule.append(current)
        current += timedelta(hours=reinit_interval_hours)

    if not schedule:
        raise ValueError(
            f"No reinit times for {start_dt} -> {end_dt} "
            f"with interval {reinit_interval_hours}h"
        )
    return schedule


def get_expected_output_file(output_root, current_time, crop_enabled=True):
    date_tag = current_time.strftime("%Y_%m_%d")
    if crop_enabled:
        return os.path.join(output_root, f"graphcast_{date_tag}_cropped.nc")
    return os.path.join(output_root, f"graphcast_{date_tag}.nc")


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cleanup_dir(path, logger=None):
    if os.path.exists(path):
        shutil.rmtree(path)
        if logger:
            logger.info(f"Deleted temporary directory: {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cfg(cfg):
    required_top = ["run", "model", "era5", "crop", "output_dir"]
    for key in required_top:
        if key not in cfg:
            raise KeyError(f"Missing required top-level config key: '{key}'")

    r = cfg["run"]
    mode = r.get("mode", "forecast")
    if mode not in ("forecast", "verify"):
        raise ValueError(f"run.mode must be 'forecast' or 'verify', got '{mode}'")
    if not isinstance(r.get("reinit_interval_hours"), int):
        raise TypeError("run.reinit_interval_hours must be an int")
    if not isinstance(r.get("cleanup_temp"), bool):
        raise TypeError("run.cleanup_temp must be a bool")
    compute_rollout_steps(r["reinit_interval_hours"])

    m = cfg["model"]
    if not isinstance(m.get("ckpt_path"), str) or not m["ckpt_path"]:
        raise TypeError("model.ckpt_path must be a non-empty string")
    if not isinstance(m.get("stats_dir"), str) or not m["stats_dir"]:
        raise TypeError("model.stats_dir must be a non-empty string")

    e = cfg["era5"]
    if "variables" not in e:
        raise KeyError("Missing era5.variables")
    for k in ("surface", "pressure", "static"):
        if k not in e["variables"]:
            raise KeyError(f"Missing era5.variables.{k}")
    if "pressure_levels" not in e:
        raise KeyError("Missing era5.pressure_levels")

    c = cfg["crop"]
    for key in ("enabled", "center_lat", "center_lon", "window_size"):
        if key not in c:
            raise KeyError(f"Missing crop.{key}")

    if not isinstance(cfg["output_dir"], str) or not cfg["output_dir"]:
        raise TypeError("output_dir must be a non-empty string")

    # Mode-specific validation
    if mode == "verify":
        if "verification" not in cfg:
            raise KeyError("verify mode requires a 'verification' section with init_time and rollout_steps")
        v = cfg["verification"]
        if "init_time" not in v:
            raise KeyError("verification.init_time is required")
        if "rollout_steps" not in v or not isinstance(v["rollout_steps"], int):
            raise KeyError("verification.rollout_steps must be an int")
    elif mode == "forecast":
        if "run_periods" not in cfg:
            raise KeyError("forecast mode requires 'run_periods'")
        for pname, pvals in cfg["run_periods"].items():
            if "start" not in pvals or "end" not in pvals:
                raise KeyError(f"run_periods.{pname} must define 'start' and 'end'")
            compute_period_bounds(pvals["start"], pvals["end"], r["reinit_interval_hours"])
