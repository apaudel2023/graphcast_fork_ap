"""
Compare per-channel mean/std between the base CorrDiff Zarr (real ERA5)
and the packaged GraphCast Zarr (GraphCast output stored in the same `era5`
slot for CorrDiff to consume).

Both Zarrs use the same `era5_channel` dimension of length 68 with the layout
(per the base Zarr's era5_variable coord):
    idx  0..12  -> t       (13 pressure levels)
    idx 13..25  -> u       (13)
    idx 26..38  -> v       (13)
    idx 39..51  -> q       (13)
    idx 52..64  -> z_pl    (13)
    idx 65      -> t2m
    idx 66      -> u10
    idx 67      -> v10

For each channel this script computes spatial+temporal mean and std on both
Zarrs (lazily, over the time intersection if available) and reports:

  - base mean/std         (climatological reference)
  - gc   mean/std         (what's in the packaged Zarr)
  - direct match          gc[i]   close to base[i]   ?
  - flipped match         gc[i]   close to base[mirror(i)] within the same
                          variable block (i.e. 50<->1000 level swap)

If pipeline.py has the level-ordering bug, the *flipped match* column will
light up across every pressure-level channel while *direct match* will fail.

Run:
    crun.python3 -p ~/envs/wrf-python python compare_channel_stats.py \
        --base-zarr     /raid/apaudel/.../hampton_2007_2015_wpd_avg.zarr \
        --packaged-zarr /raid/apaudel/.../graphcast_2014_2015.zarr
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


# Block layout (must mirror the base-Zarr era5_variable coord we already saw)
BLOCKS = [
    ("t",    0, 12),
    ("u",    13, 25),
    ("v",    26, 38),
    ("q",    39, 51),
    ("z_pl", 52, 64),
]
SURFACE_IDX = {65: "t2m", 66: "u10", 67: "v10"}


# Tolerance for declaring a "match": within this relative diff of base's std
MATCH_REL_TOL = 0.10   # 10%


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def header(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# ---------------------------------------------------------------------------
def compute_channel_stats(zpath: Path, time_subset=None) -> dict:
    """Return dict with keys: shape, time_range, mean[ch], std[ch]."""
    print(f"\nOpening: {zpath}")
    ds = xr.open_zarr(zpath, consolidated=False)
    if "era5" not in ds.data_vars:
        ds.close()
        raise ValueError(f"{zpath} has no 'era5' var")

    era5 = ds["era5"]
    print(f"  era5 shape: {era5.shape}, dims: {era5.dims}")

    if time_subset is not None:
        before = era5.sizes.get("time", 0)
        era5 = era5.sel(time=time_subset)
        print(f"  time subset: {before} -> {era5.sizes.get('time', 0)}")

    print(f"  time range: {era5['time'].values[0]} ... {era5['time'].values[-1]}")

    spatial_dims = [d for d in era5.dims if d not in ("era5_channel",)]
    print(f"  reducing over: {spatial_dims}")

    means = era5.mean(dim=spatial_dims, skipna=True).compute().values
    stds  = era5.std (dim=spatial_dims, skipna=True).compute().values

    out = {
        "shape": era5.shape,
        "t_start": str(era5["time"].values[0]),
        "t_end":   str(era5["time"].values[-1]),
        "mean": np.asarray(means, dtype=np.float64),
        "std":  np.asarray(stds,  dtype=np.float64),
    }
    ds.close()
    return out


def time_intersection_subset(base_path: Path, gc_path: Path):
    """Return an xarray time-slice covering the intersection of the two stores."""
    b = xr.open_zarr(base_path, consolidated=False)
    g = xr.open_zarr(gc_path,   consolidated=False)
    bt = b["time"].values
    gt = g["time"].values
    b.close(); g.close()

    lo = max(bt.min(), gt.min())
    hi = min(bt.max(), gt.max())
    print(f"\nTime intersection: {lo} ... {hi}")
    return slice(lo, hi)


# ---------------------------------------------------------------------------
def relclose(a: float, b: float, tol: float) -> bool:
    """Relative closeness vs base scale (|a-b| / (|b| + eps) <= tol)."""
    return abs(a - b) <= tol * (abs(b) + 1e-12)


def print_comparison(base: dict, gc: dict):
    bm, bs = base["mean"], base["std"]
    gm, gs = gc["mean"],   gc["std"]

    header("Pressure-level blocks: direct vs flipped match")
    print("Direct match = gc[i] close to base[i]")
    print("Flipped match = gc[i] close to base[mirror(i)] within same var block")
    print(f"(Match tolerance: |gc - base| <= {MATCH_REL_TOL*100:.0f}% of |base|)\n")

    n_direct = n_flipped = n_neither = 0

    for vname, lo, hi in BLOCKS:
        print(f"\n--- Block: {vname}  (idx {lo}..{hi}) ---")
        print(f"  {'idx':>3}  {'base_mean':>12}  {'base_std':>10}  "
              f"{'gc_mean':>12}  {'gc_std':>10}  {'direct?':>7}  {'flipped?':>8}  hint")

        for i in range(lo, hi + 1):
            mirror_i = lo + (hi - i)
            direct  = relclose(gm[i], bm[i], MATCH_REL_TOL)
            flipped = relclose(gm[i], bm[mirror_i], MATCH_REL_TOL)
            if direct:
                tag = "OK"
            elif flipped:
                tag = "** flipped — gc[%d] looks like base[%d] **" % (i, mirror_i)
            else:
                tag = "?? neither — investigate"

            if direct:
                n_direct += 1
            elif flipped:
                n_flipped += 1
            else:
                n_neither += 1

            d_mark = "Y" if direct else "."
            f_mark = "Y" if flipped else "."

            print(f"  {i:>3}  {bm[i]:>12.4g}  {bs[i]:>10.4g}  "
                  f"{gm[i]:>12.4g}  {gs[i]:>10.4g}  {d_mark:>7}  {f_mark:>8}  {tag}")

    header("Surface channels")
    print(f"  {'idx':>3}  {'name':>5}  {'base_mean':>12}  {'base_std':>10}  "
          f"{'gc_mean':>12}  {'gc_std':>10}  match?")
    for i, name in SURFACE_IDX.items():
        ok = relclose(gm[i], bm[i], MATCH_REL_TOL)
        print(f"  {i:>3}  {name:>5}  {bm[i]:>12.4g}  {bs[i]:>10.4g}  "
              f"{gm[i]:>12.4g}  {gs[i]:>10.4g}  {'Y' if ok else 'N'}")

    header("Summary across pressure-level channels (65 total)")
    print(f"  direct-matching channels : {n_direct}")
    print(f"  FLIPPED-matching channels: {n_flipped}")
    print(f"  neither                  : {n_neither}")
    if n_flipped > 0.6 * (n_direct + n_flipped + n_neither):
        print("\n  -> Diagnosis: pressure-level CHANNELS ARE FLIPPED.")
        print("     pipeline.py is using ASCENDING-pressure GraphCast levels with")
        print("     DESCENDING-pressure base Zarr stats. Reverse before packing.")
    elif n_direct > 0.6 * (n_direct + n_flipped + n_neither):
        print("\n  -> Diagnosis: pressure-level ordering is consistent.")
        print("     If you're still seeing wrong values downstream, the bug is")
        print("     elsewhere (regridding, time alignment, normalization application).")
    else:
        print("\n  -> Diagnosis: mixed / unclear. Read the per-channel rows above.")


# ---------------------------------------------------------------------------
def _run_checks(args):
    if args.no_time_subset:
        time_subset = None
    else:
        time_subset = time_intersection_subset(args.base_zarr, args.packaged_zarr)

    header("Computing base-Zarr per-channel stats (this may take a minute)")
    base = compute_channel_stats(args.base_zarr, time_subset)
    header("Computing packaged-Zarr per-channel stats")
    gc   = compute_channel_stats(args.packaged_zarr, time_subset)

    print_comparison(base, gc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-zarr",     required=True, type=Path)
    p.add_argument("--packaged-zarr", required=True, type=Path)
    p.add_argument("--no-time-subset", action="store_true",
                   help="Skip time-intersection; compute over each store's full time range.")
    p.add_argument("--output", required=False, type=Path, default=None,
                   help="Path to save the report. Default: "
                        "compare_channel_stats_<timestamp>.txt next to this script. "
                        "Pass an empty string '' to skip saving.")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = script_dir / f"compare_channel_stats_{ts}.txt"
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
# HPC run command
# =============================================================================
#
#   crun.python3 -p ~/envs/wrf-python python \
#       /home/apaudel/PROJECTS/graphcast/graphcast_fork_ap/aces/graphcast_to_corrdiff_zarr/debug/compare_channel_stats.py \
#       --base-zarr     /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/CORRDIFF/TRAINING/hampton_2007_2015_wpd_avg.zarr \
#       --packaged-zarr /raid/apaudel/AURORA-CORRDIFF-ARTIFACTS/DATASETS/GRAPHCAST/ZARR_FOR_CORRDIFF/graphcast_2014_2015.zarr
#
# This reads ALL timesteps in the time-intersection (2014-2015) from both
# stores. With 1472 timesteps x 68 channels x 112x112 grid that's a few GB —
# xarray streams it lazily so memory stays manageable.
#
# If you want to test a quick subset first, run on a smaller time slice in
# Python directly, or pass --no-time-subset and rely on default chunking.
# =============================================================================
