"""GraphCast post-inference analysis: RMSE/MAE metrics, comparison plots, error evolution, GIF animations."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


class GraphCastAnalysis:
    """Analysis for a single forecast window (one init_time).

    Produces:
      - metrics CSV (RMSE, MAE per variable per level per timestep)
      - 3-panel comparison plots (truth | prediction | residual)
      - Error evolution plots (metric vs lead time)
      - GIF animations across rollout steps
    """

    _SURFACE_GROUPS = {
        "temperature": ["2m_temperature"],
        "wind": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "mslp": ["mean_sea_level_pressure"],
    }

    def __init__(
        self,
        predictions,
        ground_truth,
        analysis_dir,
        cmap="viridis",
        residual_cmap="RdBu_r",
        logger=None,
    ):
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.analysis_dir = Path(analysis_dir)
        self.cmap = cmap
        self.residual_cmap = residual_cmap
        self.logger = logger

        self._all_vars = sorted(set(predictions.data_vars) & set(ground_truth.data_vars))
        self._surface_vars = [v for v in self._all_vars if "level" not in predictions[v].dims]
        self._pressure_vars = [v for v in self._all_vars if "level" in predictions[v].dims]

        self._levels = []
        if self._pressure_vars and "level" in predictions.coords:
            self._levels = sorted(predictions.coords["level"].values.tolist())

        self._shared_times = np.intersect1d(
            predictions.coords["time"].values,
            ground_truth.coords["time"].values,
        )
        self._n_steps = len(self._shared_times)

        self._global_ranges = {}
        self._global_residual_ranges = {}
        self._precompute_ranges()

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _precompute_ranges(self):
        for var in self._all_vars:
            pred = self.predictions[var]
            truth = self.ground_truth[var]
            if "batch" in pred.dims:
                pred = pred.isel(batch=0)

            pred = pred.sel(time=self._shared_times)
            truth = truth.sel(time=self._shared_times)

            has_level = "level" in pred.dims
            levels = pred.coords["level"].values if has_level else [None]

            for lev in levels:
                p = pred.sel(level=lev) if lev is not None else pred
                t = truth.sel(level=lev) if lev is not None else truth

                key = (var, int(lev) if lev is not None else None)

                vmin = min(float(np.nanmin(t.values)), float(np.nanmin(p.values)))
                vmax = max(float(np.nanmax(t.values)), float(np.nanmax(p.values)))
                self._global_ranges[key] = (vmin, vmax)

                residual = p.values - t.values
                abs_max = float(np.nanmax(np.abs(residual)))
                self._global_residual_ranges[key] = (-abs_max, abs_max)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def compute_metrics(self):
        rows = []

        for var in self._all_vars:
            pred = self.predictions[var]
            truth = self.ground_truth[var]
            if "batch" in pred.dims:
                pred = pred.isel(batch=0)

            pred = pred.sel(time=self._shared_times)
            truth = truth.sel(time=self._shared_times)

            has_level = "level" in pred.dims
            levels = pred.coords["level"].values if has_level else [None]

            for lev in levels:
                p = pred.sel(level=lev) if lev is not None else pred
                t = truth.sel(level=lev) if lev is not None else truth

                error = p - t
                rmse_per_time = np.sqrt((error ** 2).mean(dim=["lat", "lon"]))
                mae_per_time = np.abs(error).mean(dim=["lat", "lon"])

                for i, time_val in enumerate(self._shared_times):
                    if np.issubdtype(type(time_val), np.timedelta64):
                        lead_h = int(time_val / np.timedelta64(1, "h"))
                        valid_str = f"+{lead_h}h"
                    else:
                        lead_h = (i + 1) * 6
                        valid_str = str(np.datetime_as_string(time_val, unit="h"))

                    rows.append({
                        "variable": var,
                        "level": int(lev) if lev is not None else "",
                        "step": i + 1,
                        "lead_hours": lead_h,
                        "valid_time": valid_str,
                        "rmse": float(rmse_per_time.sel(time=time_val)),
                        "mae": float(mae_per_time.sel(time=time_val)),
                    })

        self.metrics_df = pd.DataFrame(rows)
        return self.metrics_df

    def save_metrics_csv(self, path=None):
        if not hasattr(self, "metrics_df"):
            self.compute_metrics()

        metrics_dir = self.analysis_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(path) if path else metrics_dir / "metrics_summary.csv"
        self.metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
        self._log(f"  Saved metrics CSV -> {csv_path}")
        return csv_path

    # -------------------------------------------------------------------------
    # Error evolution plots
    # -------------------------------------------------------------------------

    def plot_error_evolution(self, metric="rmse"):
        if not hasattr(self, "metrics_df"):
            self.compute_metrics()

        evo_dir = self.analysis_dir / "figures" / "error_evolution"
        evo_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for var in self._pressure_vars:
            vdf = self.metrics_df[self.metrics_df["variable"] == var]
            if vdf.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            for lev in self._levels:
                ldf = vdf[vdf["level"] == lev]
                if ldf.empty:
                    continue
                ax.plot(ldf["lead_hours"], ldf[metric], marker="o", markersize=3, label=f"{int(lev)} hPa")

            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{var} — {metric.upper()} vs Lead Time (by pressure level)")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(evo_dir / f"pressure_{var}_{metric}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1

        for group_name, group_vars in self._SURFACE_GROUPS.items():
            plot_vars = [v for v in group_vars if v in self._surface_vars]
            if not plot_vars:
                continue

            fig, ax = plt.subplots(figsize=(10, 5))
            for var in plot_vars:
                vdf = self.metrics_df[(self.metrics_df["variable"] == var) & (self.metrics_df["level"] == "")]
                if vdf.empty:
                    continue
                ax.plot(vdf["lead_hours"], vdf[metric], marker="o", markersize=4, label=var)

            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Surface {group_name} — {metric.upper()} vs Lead Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(evo_dir / f"surface_{group_name}_{metric}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1

        self._log(f"  Generated {count} error evolution plots")

    # -------------------------------------------------------------------------
    # Comparison plots
    # -------------------------------------------------------------------------

    def _var_dir_name(self, variable, level=None):
        if level is not None:
            return f"{variable}_{int(level)}hPa"
        return variable

    def _get_step_label(self, time_val, step_idx):
        if np.issubdtype(type(time_val), np.timedelta64):
            lead_h = int(time_val / np.timedelta64(1, "h"))
            valid_str = f"+{lead_h}h"
        else:
            lead_h = (step_idx + 1) * 6
            valid_str = str(np.datetime_as_string(time_val, unit="h"))
        file_suffix = f"step_{step_idx+1:02d}_{lead_h:+04d}h"
        return lead_h, valid_str, file_suffix

    def plot_comparison(self, variable, level=None, step=0, cmap=None, residual_cmap=None):
        cmap = cmap or self.cmap
        residual_cmap = residual_cmap or self.residual_cmap

        pred = self.predictions[variable]
        truth = self.ground_truth[variable]
        if "batch" in pred.dims:
            pred = pred.isel(batch=0)

        pred = pred.sel(time=self._shared_times)
        truth = truth.sel(time=self._shared_times)

        if level is not None and "level" in pred.dims:
            pred = pred.sel(level=level)
            truth = truth.sel(level=level)

        time_val = self._shared_times[step]
        pred_step = pred.sel(time=time_val).values
        truth_step = truth.sel(time=time_val).values
        residual_step = pred_step - truth_step

        lead_h, valid_str, file_suffix = self._get_step_label(time_val, step)
        title_base = self._var_dir_name(variable, level)

        key = (variable, int(level) if level is not None else None)
        vmin, vmax = self._global_ranges[key]
        res_vmin, res_vmax = self._global_residual_ranges[key]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        im0 = axes[0].imshow(
            truth_step, origin="lower", extent=[0, 360, -90, 90],
            aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap,
        )
        axes[0].set_title("ERA5 Truth", fontsize=11)
        plt.colorbar(im0, ax=axes[0], shrink=0.75)

        im1 = axes[1].imshow(
            pred_step, origin="lower", extent=[0, 360, -90, 90],
            aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap,
        )
        axes[1].set_title("GraphCast Prediction", fontsize=11)
        plt.colorbar(im1, ax=axes[1], shrink=0.75)

        im2 = axes[2].imshow(
            residual_step, origin="lower", extent=[0, 360, -90, 90],
            aspect="auto", vmin=res_vmin, vmax=res_vmax, cmap=residual_cmap,
        )
        axes[2].set_title("Prediction − Truth", fontsize=11)
        plt.colorbar(im2, ax=axes[2], shrink=0.75)

        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        fig.suptitle(f"{title_base}  |  Step {step+1}  |  {valid_str}", fontsize=13, y=1.02)
        fig.tight_layout()

        fig_dir = self.analysis_dir / "figures" / "comparisons" / self._var_dir_name(variable, level)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"{file_suffix}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    def plot_all(self, variables=None, levels=None):
        if variables is None:
            variables = self._all_vars
        if levels is None:
            levels = self._levels

        total = 0
        for var in variables:
            if var in self._pressure_vars:
                for lev in levels:
                    for step in range(self._n_steps):
                        self.plot_comparison(var, level=lev, step=step)
                        total += 1
            else:
                for step in range(self._n_steps):
                    self.plot_comparison(var, level=None, step=step)
                    total += 1

        self._log(f"  Generated {total} comparison plots")

    # -------------------------------------------------------------------------
    # Animations
    # -------------------------------------------------------------------------

    def create_animations(self, variables=None, levels=None, fps=2):
        if variables is None:
            variables = self._all_vars
        if levels is None:
            levels = self._levels

        anim_dir = self.analysis_dir / "animations"
        anim_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for var in variables:
            level_list = levels if var in self._pressure_vars else [None]

            for lev in level_list:
                dir_name = self._var_dir_name(var, lev)
                fig_dir = self.analysis_dir / "figures" / "comparisons" / dir_name

                if not fig_dir.exists():
                    continue

                pngs = sorted(fig_dir.glob("step_*.png"))
                if not pngs:
                    continue

                frames = [Image.open(p) for p in pngs]
                gif_path = anim_dir / f"{dir_name}.gif"
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )
                count += 1

        self._log(f"  Created {count} GIF animations")

    # -------------------------------------------------------------------------
    # Crop preview
    # -------------------------------------------------------------------------

    # Default variables for crop preview: 3 surface + 1 per pressure level
    _CROP_PREVIEW_SURFACE = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind"]
    _CROP_PREVIEW_PRESSURE_LEVELS = [500]  # one representative level per pressure var

    def plot_crop_preview(self, crop_cfg):
        """Plot global map with red crop box + zoomed crop side-by-side.

        Generates one image per selected variable at the first timestep.
        Saves to analysis_dir/figures/crop_preview/

        Parameters
        ----------
        crop_cfg : dict
            Must contain center_lat, center_lon, window_size.
        """
        center_lat = crop_cfg["center_lat"]
        center_lon = crop_cfg["center_lon"] % 360
        window_size = crop_cfg["window_size"]

        preview_dir = self.analysis_dir / "figures" / "crop_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Use prediction data at the first shared timestep
        time_val = self._shared_times[0]

        # Determine the crop box in lat/lon coords
        pred_sample = self.predictions[self._all_vars[0]]
        if "batch" in pred_sample.dims:
            pred_sample = pred_sample.isel(batch=0)

        lats = pred_sample.coords["lat"].values
        lons = pred_sample.coords["lon"].values

        ilat = int(np.abs(lats - center_lat).argmin())
        ilon = int(np.abs(lons - center_lon).argmin())
        half = window_size // 2

        lat_lo = lats[max(ilat - half, 0)]
        lat_hi = lats[min(ilat + half - 1, len(lats) - 1)]
        lon_lo = lons[max(ilon - half, 0)]
        lon_hi = lons[min(ilon + half - 1, len(lons) - 1)]

        # Build list of (variable, level) to preview
        preview_list = []
        for v in self._CROP_PREVIEW_SURFACE:
            if v in self._surface_vars:
                preview_list.append((v, None))
        for lev in self._CROP_PREVIEW_PRESSURE_LEVELS:
            if lev in self._levels:
                for v in self._pressure_vars:
                    preview_list.append((v, lev))

        count = 0
        for var, level in preview_list:
            data = self.predictions[var]
            if "batch" in data.dims:
                data = data.isel(batch=0)
            data = data.sel(time=time_val)
            if level is not None and "level" in data.dims:
                data = data.sel(level=level)

            full = data.values
            label = self._var_dir_name(var, level)

            # Crop region
            lat_slice = slice(max(ilat - half, 0), min(ilat + half, len(lats)))
            lon_slice = slice(max(ilon - half, 0), min(ilon + half, len(lons)))
            cropped = full[lat_slice, lon_slice]

            fig, axes = plt.subplots(1, 2, figsize=(16, 4),
                                     gridspec_kw={"width_ratios": [2, 1]})

            # Left: global map with red box
            vmin, vmax = float(np.nanmin(full)), float(np.nanmax(full))
            im0 = axes[0].imshow(
                full, origin="lower", extent=[0, 360, -90, 90],
                aspect="auto", vmin=vmin, vmax=vmax, cmap=self.cmap,
            )
            # Draw red rectangle for crop region
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (float(lon_lo), float(lat_lo)),
                float(lon_hi - lon_lo),
                float(lat_hi - lat_lo),
                linewidth=2, edgecolor="red", facecolor="none",
            )
            axes[0].add_patch(rect)
            axes[0].set_title(f"{label} — Global (prediction)", fontsize=11)
            axes[0].set_xlabel("Longitude")
            axes[0].set_ylabel("Latitude")
            plt.colorbar(im0, ax=axes[0], shrink=0.75)

            # Right: zoomed crop region
            im1 = axes[1].imshow(
                cropped, origin="lower",
                extent=[float(lon_lo), float(lon_hi), float(lat_lo), float(lat_hi)],
                aspect="auto", vmin=vmin, vmax=vmax, cmap=self.cmap,
            )
            axes[1].set_title(f"Cropped region", fontsize=11)
            axes[1].set_xlabel("Longitude")
            axes[1].set_ylabel("Latitude")
            plt.colorbar(im1, ax=axes[1], shrink=0.75)

            fig.suptitle(
                f"Crop preview: {label}  |  "
                f"Center ({center_lat}, {crop_cfg['center_lon']}), window={window_size}",
                fontsize=12, y=1.02,
            )
            fig.tight_layout()

            fig_path = preview_dir / f"{label}_crop_preview.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            count += 1

        self._log(f"  Generated {count} crop preview plots in {preview_dir}")

    # -------------------------------------------------------------------------
    # Run all
    # -------------------------------------------------------------------------

    def run(self, levels=None, fps=2, metrics=True, plots=True, animations=True):
        """Run selected analysis steps.

        Parameters
        ----------
        levels : list, optional
            Pressure levels for plots/GIFs. None = all.
        fps : int
            GIF frame rate.
        metrics : bool
            Compute and save metrics CSV.
        plots : bool
            Generate comparison plots + error evolution.
        animations : bool
            Create GIF animations (requires plots to exist).
        """
        if metrics:
            self._log("  Computing metrics (RMSE, MAE) ...")
            self.compute_metrics()
            self.save_metrics_csv()

        if plots:
            self._log("  Generating comparison plots ...")
            self.plot_all(levels=levels)
            self._log("  Generating error evolution plots ...")
            for m in ("rmse", "mae"):
                self.plot_error_evolution(metric=m)

        if animations:
            self._log("  Creating GIF animations ...")
            self.create_animations(levels=levels, fps=fps)

        self._log(f"  Analysis complete -> {self.analysis_dir}")
