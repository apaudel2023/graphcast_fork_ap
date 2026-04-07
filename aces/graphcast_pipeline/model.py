"""GraphCast model loading and inference."""

import dataclasses
import functools
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import xarray

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout


class GraphCastModel:
    """Loads the GraphCast checkpoint once and runs inference repeatedly."""

    def __init__(self, cfg, logger):
        self.logger = logger
        self._load(cfg)

    def _load(self, cfg):
        ckpt_path = Path(cfg["model"]["ckpt_path"])
        stats_dir = Path(cfg["model"]["stats_dir"])

        self.logger.info(f"Loading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)

        self._params = ckpt.params
        self._state = {}
        self._model_config = ckpt.model_config
        self._task_config = ckpt.task_config
        self.logger.info(f"  Model config: {self._model_config}")
        self.logger.info(f"  Task config:  {self._task_config}")

        self.logger.info(f"Loading normalization stats: {stats_dir}")
        self._diffs_stddev = xarray.load_dataset(stats_dir / "diffs_stddev_by_level.nc").compute()
        self._mean = xarray.load_dataset(stats_dir / "mean_by_level.nc").compute()
        self._stddev = xarray.load_dataset(stats_dir / "stddev_by_level.nc").compute()

        self.logger.info("Building JIT-compiled forward function ...")
        self._run_forward_jitted = self._build_jitted_forward()
        self.logger.info("Model ready.")

    def _build_jitted_forward(self):
        diffs_stddev = self._diffs_stddev
        mean = self._mean
        stddev = self._stddev
        model_config = self._model_config
        task_config = self._task_config
        params = self._params
        state = self._state

        def construct_wrapped_graphcast(model_config, task_config):
            predictor = graphcast.GraphCast(model_config, task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=diffs_stddev,
                mean_by_level=mean,
                stddev_by_level=stddev,
            )
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
            return predictor

        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        forward_apply = functools.partial(
            run_forward.apply, model_config=model_config, task_config=task_config
        )
        forward_apply = functools.partial(forward_apply, params=params, state=state)
        forward_apply_jitted = jax.jit(forward_apply)

        def _call(rng, inputs, targets_template, forcings):
            preds, _ = forward_apply_jitted(
                rng=rng, inputs=inputs, targets_template=targets_template, forcings=forcings
            )
            return preds

        return _call

    def predict(self, batch, rollout_steps):
        """Run GraphCast inference on a prepared batch.

        Parameters
        ----------
        batch : xarray.Dataset
            Output from batch_builder.build_batch().
        rollout_steps : int

        Returns
        -------
        predictions : xarray.Dataset
            With time coord as lead-time timedeltas [6h, 12h, ...].
        """
        # Split into inputs / targets_template / forcings
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=slice("6h", f"{rollout_steps * 6}h"),
            **dataclasses.asdict(self._task_config),
        )
        targets_template = targets * np.nan

        self.logger.info(f"  inputs:           {dict(inputs.sizes)}")
        self.logger.info(f"  targets_template: {dict(targets_template.sizes)}")
        self.logger.info(f"  forcings:         {dict(forcings.sizes)}")

        # Validate no NaN in inputs
        for var in inputs.data_vars:
            n = int(np.isnan(inputs[var].values).sum())
            if n > 0:
                raise ValueError(f"NaN in input '{var}': {n} values")
        self.logger.info("  Inputs clean (no NaN).")

        # Autoregressive rollout
        self.logger.info("  Running rollout (first chunk includes JIT compile time) ...")
        predictions = rollout.chunked_prediction(
            self._run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )
        self.logger.info(f"  Rollout complete: {dict(predictions.sizes)}")
        return predictions
