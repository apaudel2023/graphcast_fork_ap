"""
Diagnose the GraphCast -> CorrDiff Zarr channel/level alignment.

Hypothesis: pressure-level channels in the packaged Zarr are paired with the
wrong era5_center / era5_scale entries because:
  (a) GraphCast's `level` coord order may not match what the base Zarr assumes, and/or
  (b) the channel layout `[t*13, u*13, v*13, q*13, z_pl*13, t2m, u10, v10]` written
      by pipeline.py does not match the channel layout used to build the base Zarr's
      era5_center / era5_scale.

Run:
    python check_channel_alignment.py \
        --gc-nc /raid/apaudel/.../graphcast_2024_01_01.nc \
        --base-zarr /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/CORRDIFF/TRAINING/hampton_2007_2015_wpd_avg.zarr \
        --packaged-zarr /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/GRAPHCAST/ZARR_FOR_CORRDIFF/graphcast_2014_2015.zarr

The --packaged-zarr arg is optional; if given, the script compares the
data range of each channel to the era5_center to spot obvious mismatches
(e.g. temperature data slotted into a wind-stats channel).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


class _Tee:
    """Mirror writes to multiple streams (stdout + a file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# Expected physical ranges per variable (rough sanity bounds, broad)
# Used to flag suspicious (channel data, era5_center) pairings.
PHYSICAL_HINT = {
    # var_name : (typical_mean_low, typical_mean_high, units_hint)
    "t":    (180.0, 320.0, "K (air temp, all levels)"),
    "u":    (-30.0, 30.0,  "m/s (zonal wind)"),
    "v":    (-30.0, 30.0,  "m/s (meridional wind)"),
    "q":    (1e-7,  3e-2,  "kg/kg (specific humidity)"),
    "z_pl": (0.0,   2.0e5, "m^2/s^2 (geopotential)"),
    "t2m":  (220.0, 320.0, "K"),
    "u10":  (-20.0, 20.0,  "m/s"),
    "v10":  (-20.0, 20.0,  "m/s"),
    "msl":  (95000.0, 105000.0, "Pa"),
}


# Expected pipeline.py channel layout (must mirror configs/default.yml)
EXPECTED_PRESSURE_VARS = ["t", "u", "v", "q", "z_pl"]
EXPECTED_LEVEL_INDICES = list(range(13))
EXPECTED_SURFACE_VARS  = ["t2m", "u10", "v10"]


def header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# Check 1: GraphCast level coord order
# -----------------------------------------------------------------------------
def check_graphcast_level_order(gc_path: Path):
    header(f"[1] GraphCast NetCDF level ordering — {gc_path}")
    ds = xr.open_dataset(gc_path)

    print(f"  dims:        {dict(ds.sizes)}")
    print(f"  data_vars:   {sorted(ds.data_vars)}")

    if "level" not in ds.coords:
        print("  ⚠ no 'level' coord found.")
        ds.close()
        return None

    levels = ds["level"].values
    print(f"  level dtype: {levels.dtype}")
    print(f"  level values (in stored order): {levels.tolist()}")

    is_ascending  = bool(np.all(np.diff(levels) > 0))
    is_descending = bool(np.all(np.diff(levels) < 0))
    if is_ascending:
        order = "ASCENDING (low pressure / high altitude first)"
    elif is_descending:
        order = "DESCENDING (high pressure / low altitude first)"
    else:
        order = "NON-MONOTONIC — unusual"
    print(f"  order:       {order}")

    expected_set = {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}
    if set(levels.tolist()) == expected_set:
        print(f"  set match:   ✓ all 13 GraphCast-operational levels present")
    else:
        missing = expected_set - set(levels.tolist())
        extra   = set(levels.tolist()) - expected_set
        print(f"  set match:   ✗ missing={sorted(missing)} extra={sorted(extra)}")

    # Show what isel(level=0) and isel(level=12) actually resolve to —
    # this is what the packaging pipeline does positionally.
    if levels.size >= 13:
        print(f"\n  pipeline.py uses ISEL (positional). With the current ordering:")
        print(f"    isel(level= 0)  ->  {int(levels[0])} hPa")
        print(f"    isel(level= 6)  ->  {int(levels[6])} hPa")
        print(f"    isel(level=12)  ->  {int(levels[12])} hPa")

    ds.close()
    return levels.tolist()


# -----------------------------------------------------------------------------
# Check 2: Base Zarr channel layout + era5_center semantics
# -----------------------------------------------------------------------------
def check_base_zarr_layout(base_zarr: Path):
    header(f"[2] Base Zarr layout — {base_zarr}")
    base = xr.open_zarr(base_zarr, consolidated=False)

    print(f"  dims:        {dict(base.sizes)}")
    print(f"  coords:      {list(base.coords)}")
    print(f"  data_vars:   {list(base.data_vars)}")
    print(f"  attrs keys:  {list(base.attrs)}")
    if base.attrs:
        for k, v in base.attrs.items():
            sval = str(v)
            if len(sval) > 200:
                sval = sval[:200] + " ..."
            print(f"    attr {k}: {sval}")

    # Look for any channel-naming coord
    name_candidates = [c for c in base.coords
                       if "channel" in c.lower() or "var" in c.lower() or "name" in c.lower()]
    print(f"\n  channel-name candidates among coords: {name_candidates}")
    for c in name_candidates:
        vals = base[c].values
        print(f"    {c} (shape={base[c].shape}, dtype={base[c].dtype}):")
        try:
            print(f"      {vals.tolist()[:80]}")
        except Exception:
            print(f"      <not listable: {vals}>")

    # era5_center / era5_scale
    if "era5_center" not in base.data_vars:
        print("\n  ⚠ 'era5_center' missing from base Zarr — cannot diagnose channel pairing.")
        base.close()
        return None

    centers = base["era5_center"].values
    scales  = base["era5_scale"].values
    print(f"\n  era5_center shape: {centers.shape}, dtype: {centers.dtype}")
    print(f"  era5_scale  shape: {scales.shape}, dtype: {scales.dtype}")

    # If shape is (era5_channel,) or (era5_channel, 1, 1) etc., flatten to 1D
    centers_1d = np.asarray(centers).reshape(centers.shape[0], -1).mean(axis=1)
    scales_1d  = np.asarray(scales).reshape(scales.shape[0], -1).mean(axis=1)

    print(f"\n  Per-channel center / scale (flattened to scalar per channel):")
    print(f"  {'idx':>3}  {'center':>14}  {'scale':>14}  guess-from-magnitude")
    for i, (m, s) in enumerate(zip(centers_1d, scales_1d)):
        guess = guess_variable_from_stats(m, s)
        print(f"  {i:>3}  {m:>14.4g}  {s:>14.4g}  {guess}")

    base.close()
    return centers_1d, scales_1d


def guess_variable_from_stats(mean: float, scale: float) -> str:
    """Heuristic: given a per-channel mean and std, guess what kind of variable it is."""
    m, s = float(mean), float(scale)
    candidates = []
    for name, (lo, hi, hint) in PHYSICAL_HINT.items():
        if lo <= m <= hi:
            candidates.append(f"{name} ({hint})")
    if not candidates:
        return f"?? mean={m:.3g}, std={s:.3g}"
    return ", ".join(candidates)


# -----------------------------------------------------------------------------
# Check 3: Packaged Zarr — what's actually in each channel
# -----------------------------------------------------------------------------
def check_packaged_zarr(packaged_zarr: Path, base_centers: np.ndarray | None):
    header(f"[3] Packaged GraphCast Zarr — {packaged_zarr}")
    pkg = xr.open_zarr(packaged_zarr, consolidated=False)
    print(f"  dims:      {dict(pkg.sizes)}")
    print(f"  data_vars: {list(pkg.data_vars)}")

    if "era5" not in pkg.data_vars:
        print("  ⚠ 'era5' var missing.")
        pkg.close()
        return

    era5 = pkg["era5"]
    n_channels = era5.sizes.get("era5_channel", 0)
    print(f"  n channels: {n_channels}")

    # Build expected channel labels per the pipeline.py config
    expected_labels = []
    for v in EXPECTED_PRESSURE_VARS:
        for lvl in EXPECTED_LEVEL_INDICES:
            expected_labels.append(f"{v}_{lvl}")
    expected_labels.extend(EXPECTED_SURFACE_VARS)
    print(f"  expected channel layout (from pipeline.py + default.yml):")
    print(f"    {expected_labels}")
    print(f"    -> total expected: {len(expected_labels)}")

    if len(expected_labels) != n_channels:
        print(f"  ⚠ MISMATCH: packaged has {n_channels} channels, expected {len(expected_labels)}.")

    # Sample first timestep, compute channel-wise mean / std on the spatial grid
    print(f"\n  Per-channel mean/std at first timestep (vs base era5_center):")
    print(f"  {'idx':>3}  {'expected':>10}  {'data_mean':>14}  {'data_std':>14}  "
          f"{'center':>14}  match?")
    t0 = era5.isel(time=0).values  # (era5_channel, south_north, west_east)
    for i in range(min(n_channels, len(expected_labels))):
        ch = t0[i]
        m, s = float(np.nanmean(ch)), float(np.nanstd(ch))
        c    = float(base_centers[i]) if base_centers is not None and i < len(base_centers) else float("nan")
        label = expected_labels[i]

        # Sanity: does the data magnitude match what we'd expect from `label`'s var?
        var_kind = label.split("_")[0] if "_" in label else label
        ok = "?"
        if var_kind in PHYSICAL_HINT:
            lo, hi, _ = PHYSICAL_HINT[var_kind]
            ok = "✓" if lo <= m <= hi else "✗ data outside physical range for this label"

        # Also: does center match expected var kind?
        center_ok = "?"
        if not np.isnan(c) and var_kind in PHYSICAL_HINT:
            lo, hi, _ = PHYSICAL_HINT[var_kind]
            center_ok = "✓" if lo <= c <= hi else "✗ center suggests a different variable"

        print(f"  {i:>3}  {label:>10}  {m:>14.4g}  {s:>14.4g}  {c:>14.4g}  "
              f"data:{ok}  center:{center_ok}")
    pkg.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _run_checks(args):
    levels = check_graphcast_level_order(args.gc_nc)
    base_stats = check_base_zarr_layout(args.base_zarr)
    base_centers = base_stats[0] if base_stats is not None else None

    if args.packaged_zarr is not None:
        check_packaged_zarr(args.packaged_zarr, base_centers)

    header("Interpretation guide")
    print("""
  - If GraphCast 'level' is DESCENDING (1000 -> 50), then pipeline.py's
    `isel(level=0)` is 1000 hPa, but the base Zarr may have been built
    with level index 0 = 50 hPa (or vice versa). Pressure-level channels
    will be paired with the wrong era5_center / era5_scale.

  - In Check [2], look at the 'guess-from-magnitude' column. Channels whose
    center suggests temperature should align (in your expected layout) with
    't' channels, etc. If e.g. channel 0 has center ~273 K but pipeline.py
    puts 't_0' there with 50-hPa-temperature data (~200 K), the *normalization
    is wrong* even though the variable identity is technically correct —
    because the level pairing is flipped.

  - In Check [3], rows marked 'data:✗' or 'center:✗' show concrete mismatches
    between what's in each channel and what's expected.
""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gc-nc", required=True, type=Path,
                   help="One GraphCast prediction .nc file (e.g. graphcast_2024_01_01.nc)")
    p.add_argument("--base-zarr", required=True, type=Path,
                   help="Base CorrDiff Zarr (training Zarr that provides era5_center/era5_scale)")
    p.add_argument("--packaged-zarr", required=False, type=Path,
                   help="Optional: the packaged-for-CorrDiff GraphCast Zarr to inspect")
    p.add_argument("--output", required=False, type=Path, default=None,
                   help="Path to save the report. Default: "
                        "channel_alignment_report_<timestamp>.txt next to this script. "
                        "Pass an empty string '' to skip saving.")
    args = p.parse_args()

    # Resolve output path
    script_dir = Path(__file__).resolve().parent
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = script_dir / f"channel_alignment_report_{ts}.txt"
    elif str(args.output) == "":
        out_path = None
    else:
        out_path = args.output

    if out_path is None:
        _run_checks(args)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[saving report to: {out_path}]")
    f = open(out_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, f)
    try:
        _run_checks(args)
    finally:
        sys.stdout = original_stdout
        f.close()
        print(f"[report saved: {out_path}]")


if __name__ == "__main__":
    main()


# =============================================================================
# HPC run commands (Wahab / wrf-python env)
# =============================================================================
# By default the report is auto-saved next to this script:
#     aces/graphcast_to_corrdiff_zarr/debug/channel_alignment_report_<YYYYMMDD_HHMMSS>.txt
# so reports from different runs live side-by-side and never overwrite each
# other. Override with --output PATH, or pass --output '' to print only.
#
# The forecast-mode pipeline deletes the full-resolution prediction after
# cropping, so RAW_OUTPUTS/2014-2015/ contains ONLY *_cropped.nc files. That
# is also what graphcast_to_corrdiff_zarr/pipeline.py consumes, so it's the
# right input for this diagnostic.
#
# Period dirs are 15-day chunks: 2014_01_01_2014_01_15, 2014_02_01_2014_02_15,
# etc. Files inside are named graphcast_YYYY_MM_DD_cropped.nc.
#
#
# Case 1 — Minimal check (no packaged Zarr; quickest diagnostic):
#   inspects GraphCast level order + base Zarr channel layout only.
#
#   crun.python3 -p ~/envs/wrf-python python \
#       /home/apaudel/PROJECTS/graphcast/graphcast_fork_ap/aces/graphcast_to_corrdiff_zarr/debug/check_channel_alignment.py \
#       --gc-nc     /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/GRAPHCAST/RAW_OUTPUTS/2014-2015/2014_01_01_2014_01_15/graphcast_2014_01_01_cropped.nc \
#       --base-zarr /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/CORRDIFF/TRAINING/hampton_2007_2015_wpd_avg.zarr
#
#
# Case 2 — Full check against the packaged Zarr (adds per-channel data vs
#          era5_center cross-check; this is the one that pinpoints mismatches):
#
#   crun.python3 -p ~/envs/wrf-python python \
#       /home/apaudel/PROJECTS/graphcast/graphcast_fork_ap/aces/graphcast_to_corrdiff_zarr/debug/check_channel_alignment.py \
#       --gc-nc         /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/GRAPHCAST/RAW_OUTPUTS/2014-2015/2014_01_01_2014_01_15/graphcast_2014_01_01_cropped.nc \
#       --base-zarr     /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/CORRDIFF/TRAINING/hampton_2007_2015_wpd_avg.zarr \
#       --packaged-zarr /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/GRAPHCAST/ZARR_FOR_CORRDIFF/graphcast_2014_2015.zarr
#
#
# Case 3 — Use the full-resolution single-step verify-run NetCDF (from the
#          test_graphcast.yml run). Verify mode KEEPS the full-resolution
#          file, so this is the only way to inspect uncropped predictions.
#
#   crun.python3 -p ~/envs/wrf-python python \
#       /home/apaudel/PROJECTS/graphcast/graphcast_fork_ap/aces/graphcast_to_corrdiff_zarr/debug/check_channel_alignment.py \
#       --gc-nc         /home/apaudel/PROJECTS/graphcast/graphcast_fork_ap/aces/graphcast_pipeline/GRAPHCAST_VERIFY_ROLLOUT1/verify_20240101/graphcast_2024_01_01.nc \
#       --base-zarr     /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/CORRDIFF/TRAINING/hampton_2007_2015_wpd_avg.zarr \
#       --packaged-zarr /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/GRAPHCAST/ZARR_FOR_CORRDIFF/graphcast_2014_2015.zarr
#
#
# Custom output filename:
#   crun.python3 -p ~/envs/wrf-python python \
#       /home/apaudel/PROJECTS/graphcast/graphcast_fork_ap/aces/graphcast_to_corrdiff_zarr/debug/check_channel_alignment.py \
#       --gc-nc ... --base-zarr ... --packaged-zarr ... \
#       --output ~/reports/channel_alignment_2014_01_01.txt
# =============================================================================
