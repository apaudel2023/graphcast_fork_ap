"""
Side-by-side per-channel statistics across up to four Zarrs:

  base   — the CorrDiff training Zarr (real ERA5 + WRF; all samples)
  aurora — Aurora-output Zarr packaged for CorrDiff (optional)
  zarr1  — old packaged GraphCast Zarr (suspected buggy)
  zarr2  — new packaged GraphCast Zarr (after the level-pairing fix)

Computes mean and std of every channel in both the `era5` and `wrf` data
variables on each store and prints four tables:

  - ERA5 CENTER (mean)
  - ERA5 SCALE  (std)
  - WRF  CENTER (mean)
  - WRF  SCALE  (std)

Rows are channels, columns are {base, zarr1, zarr2}. Channel labels are read
from each store's *_variable / *_pressure coords. The base Zarr stores
era5_pressure as a level INDEX (0..N-1) into a descending-pressure ordering
(1000, 925, ..., 50 hPa); we decode that for the row labels only — the
per-channel reductions themselves are computed positionally on era5_channel,
which is consistent across all three Zarrs.

Per the user's request: NO time filtering. Each Zarr is reduced over its
own full time range (e.g. 26296 timesteps for the base; 1472 for the
packaged ones).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


# Variables we treat as intrinsically single-level (no pressure dim).
SURFACE_VARS = {"t2m", "u10", "v10", "msl", "tp", "sp", "tcwv", "skt"}


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
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# ---------------------------------------------------------------------------
# Coord decoding (build row labels for the tables)
# ---------------------------------------------------------------------------
def _decode_labels(zpath: Path, group: str) -> list[str]:
    """Build a human-readable label string per channel for the given group.

    group: "era5" or "wrf".

    Falls back gracefully if any coord is missing.
    """
    ds = xr.open_zarr(zpath, consolidated=False)
    try:
        var_coord  = f"{group}_variable"
        press_coord = f"{group}_pressure"
        chan_dim   = f"{group}_channel"

        if chan_dim not in ds.sizes:
            return []
        n = ds.sizes[chan_dim]

        vars_per_ch = (
            [str(v) for v in ds[var_coord].values]
            if var_coord in ds.coords else
            [f"ch_{i}" for i in range(n)]
        )

        if press_coord in ds.coords:
            press_raw = np.asarray(ds[press_coord].values, dtype=float)
        else:
            press_raw = np.full(n, np.nan)
    finally:
        ds.close()

    # Detect indices vs hPa (same heuristic as pipeline.py).
    nonsurf_finite = [
        float(p) for v, p in zip(vars_per_ch, press_raw)
        if v not in SURFACE_VARS and np.isfinite(p)
    ]
    looks_like_indices = bool(nonsurf_finite) and max(nonsurf_finite) < 20.0

    # Common GraphCast/ERA5 13-level list for descending decoding. Fallback
    # only — used for cosmetic labels, not for any computation.
    desc_13 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    labels = []
    for v, p in zip(vars_per_ch, press_raw):
        if v in SURFACE_VARS:
            labels.append(f"{v:<8}  surface ")
            continue
        if not np.isfinite(p):
            labels.append(f"{v:<8}  ?       ")
            continue
        if looks_like_indices:
            idx = int(round(float(p)))
            hpa = desc_13[idx] if 0 <= idx < len(desc_13) else int(p)
            labels.append(f"{v:<8} {hpa:>6.1f} hPa")
        else:
            labels.append(f"{v:<8} {float(p):>6.1f} hPa")
    return labels


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------
def compute_stats(zpath: Path, group: str) -> dict:
    """Return per-channel mean and std for `group` ('era5' or 'wrf')."""
    print(f"\n[{group}]  opening {zpath}")
    ds = xr.open_zarr(zpath, consolidated=False)
    try:
        if group not in ds.data_vars:
            print(f"  no '{group}' var — skipping")
            return {"present": False}

        da = ds[group]
        chan_dim = f"{group}_channel"
        if chan_dim not in da.dims:
            print(f"  '{group}' has no '{chan_dim}' dim — skipping")
            return {"present": False}

        print(f"  shape={da.shape}  dims={da.dims}")
        print(f"  time range: {ds['time'].values[0]} ... {ds['time'].values[-1]}  (no filtering)")

        reduce_dims = [d for d in da.dims if d != chan_dim]
        print(f"  reducing over: {reduce_dims}")

        means = da.mean(dim=reduce_dims, skipna=True).compute().values
        stds  = da.std (dim=reduce_dims, skipna=True).compute().values
    finally:
        ds.close()

    return {
        "present": True,
        "mean": np.asarray(means, dtype=np.float64),
        "std":  np.asarray(stds,  dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "        n/a"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:>11.4e}"
    return f"{x:>11.4f}"


def print_table(title: str, labels: list[str], cols: list[tuple[str, np.ndarray | None]]):
    """cols: list of (col_name, array_or_None). All arrays must be same length."""
    header(title)

    # Header line
    hdr = f"  {'idx':>3}  {'variable / level':<20}"
    for name, _ in cols:
        hdr += f"  {name:>11}"
    hdr += f"  {'|base-v2|':>11}  match?"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # Determine row count from the first non-None column
    n_rows = next((len(arr) for _, arr in cols if arr is not None), 0)
    if n_rows == 0 or len(labels) == 0:
        print("  (no rows to print)")
        return

    # Pad labels to match
    if len(labels) < n_rows:
        labels = labels + [f"ch_{i}" for i in range(len(labels), n_rows)]

    # Compare base vs v2 (last col) where both available
    base_arr = cols[0][1]
    v2_arr   = cols[-1][1] if len(cols) >= 3 else None

    for i in range(n_rows):
        row = f"  {i:>3}  {labels[i]:<20}"
        for _, arr in cols:
            row += f"  {_fmt(None if arr is None else float(arr[i]))}"

        if base_arr is not None and v2_arr is not None and i < len(base_arr) and i < len(v2_arr):
            diff = abs(float(base_arr[i]) - float(v2_arr[i]))
            denom = abs(float(base_arr[i])) + 1e-12
            ok = (diff / denom) <= 0.10
            row += f"  {diff:>11.4g}  {'Y' if ok else 'N'}"
        else:
            row += f"  {'n/a':>11}  ?"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run(args):
    # Column order: base, aurora (optional), zarr1, zarr2.
    paths = [("base", args.base_zarr)]
    if args.aurora_zarr is not None:
        paths.append(("aurora", args.aurora_zarr))
    paths.extend([("zarr1", args.zarr1), ("zarr2", args.zarr2)])

    # Labels: take from the base Zarr (it has the canonical *_variable / *_pressure coords).
    era5_labels = _decode_labels(args.base_zarr, "era5")
    wrf_labels  = _decode_labels(args.base_zarr, "wrf")

    era5_stats = {name: compute_stats(p, "era5") for name, p in paths}
    wrf_stats  = {name: compute_stats(p, "wrf")  for name, p in paths}

    def cols_for(stats_dict, key):
        return [
            (name, stats_dict[name][key] if stats_dict[name]["present"] else None)
            for name, _ in paths
        ]

    print_table(
        "ERA5 CENTER  (per-channel mean of `era5` over all samples)",
        era5_labels,
        cols=cols_for(era5_stats, "mean"),
    )
    print_table(
        "ERA5 SCALE  (per-channel std of `era5` over all samples)",
        era5_labels,
        cols=cols_for(era5_stats, "std"),
    )
    print_table(
        "WRF CENTER  (per-channel mean of `wrf` over all samples)",
        wrf_labels,
        cols=cols_for(wrf_stats, "mean"),
    )
    print_table(
        "WRF SCALE  (per-channel std of `wrf` over all samples)",
        wrf_labels,
        cols=cols_for(wrf_stats, "std"),
    )

    header("Notes on interpretation")
    print("""
  - 'match?' compares base vs zarr2 (10% relative tolerance). If the v2
    packaging fix is correct, the ERA5 center/scale columns for base and
    zarr2 should match channel-by-channel.
  - WRF data in the packaged Zarrs is grafted from the base Zarr via
    `add_wrf_fields(use_real=True)`, so the WRF tables should already match
    closely (only the time subset differs).
  - zarr1 was packaged before the level-pairing fix; for pressure-level
    channels its mean/std should LOOK FLIPPED relative to base (an obvious
    mirror inside each variable block).
""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-zarr",   required=True, type=Path, help="Base CorrDiff Zarr (real ERA5+WRF)")
    p.add_argument("--aurora-zarr", required=False, type=Path, default=None,
                   help="Optional: Aurora-output Zarr (inserted as the second column, after base)")
    p.add_argument("--zarr1",       required=True, type=Path, help="First packaged Zarr to compare (e.g. buggy v1)")
    p.add_argument("--zarr2",       required=True, type=Path, help="Second packaged Zarr to compare (e.g. fixed v2)")
    p.add_argument("--output", required=False, type=Path, default=None,
                   help="Path to save the report. Default: "
                        "compare_zarr_stats_<timestamp>.txt next to this script. "
                        "Pass an empty string '' to skip saving.")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = script_dir / f"compare_zarr_stats_{ts}.txt"
    elif str(args.output) == "":
        out_path = None
    else:
        out_path = args.output

    if out_path is None:
        _run(args); return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[saving report to: {out_path}]")
    f = open(out_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, f)
    try:
        _run(args)
    finally:
        sys.stdout = original_stdout
        f.close()
        print(f"[report saved: {out_path}]")


if __name__ == "__main__":
    main()
