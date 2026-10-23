"""Inference wrapper of the GRU soft sensor (feedback variant F3).

The soft sensor is trained by ``scripts/train_sensor_value_soft_sensor.py`` with the existing
pipeline of Befund 94 (arm N, ``DataSpec.new_study()``, process config of Befund 94) in the
operating mode ``SOFT_SENSOR`` on the moisture head alone: the composite network has one
independent network per target, and the moisture network sees the exogenous channels (actions and
disturbances) and its own history, which in ``SOFT_SENSOR`` is masked to the standardised training
mean (zero). It never sees a moisture value, and -- by the design of the composite network
(``models/composite.py``, "Was das kostet") -- no other measured target either.

Estimate at the end of 25 s block ``j``: the h = 1 prediction of a window whose history holds the
blocks ``j - L .. j - 1`` and whose single future step is block ``j`` (the GRU decoder is causal,
so a one-step future gives the same first output as the full horizon; tested). The exogenous values
of the block are already realised when the estimate is formed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.data.windows import WindowBatch

__all__ = ["SOFT_SENSOR_EXOGENOUS", "SoftSensor"]

SOFT_SENSOR_EXOGENOUS: tuple[str, ...] = (
    "inlet_temperature_celsius",
    "gas_velocity_m_per_s",
    "spray_rate_kg_per_h",
    "nozzle_pressure_bar",
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
    "suspension_solid_fraction",
)
"""Exogenous channels of the Befund 94 data in their order (actions, then disturbances)."""

_TARGET = "product_moisture_kg_per_kg"


class SoftSensor:
    """A trained moisture head with its scalers."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        if tuple(metadata["exogenous_names"]) != SOFT_SENSOR_EXOGENOUS:
            raise ValueError(f"unexpected exogenous channels {metadata['exogenous_names']}")
        if tuple(metadata["target_names"]) != (_TARGET,):
            raise ValueError("the soft sensor must be the moisture head alone")
        observed = metadata["data_spec"].get("data/observed_target_names")
        if observed is None or _TARGET in observed:
            raise ValueError("the checkpoint was not trained in the SOFT_SENSOR mode")
        self._model = model
        self.metadata = metadata
        self._input_scaler = metadata["input_scaler"]
        self._target_scaler = metadata["target_scaler"]
        self._n_exogenous = len(SOFT_SENSOR_EXOGENOUS)
        self.history_length = int(metadata["data_spec"]["data/history_length"])

    @classmethod
    def load(cls, path: Path) -> SoftSensor:
        """From a checkpoint written by the training script."""
        from twinlink_ml.utils.checkpoints import load_checkpoint

        model, metadata = load_checkpoint(path)
        return cls(model, metadata)

    def estimate(self, histories: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Moisture estimates ``(m,)`` from exogenous histories ``(m, L, 7)`` and blocks ``(m, 7)``.

        The moisture channel of the history is the standardised training mean (0), as in the
        windows of the ``SOFT_SENSOR`` mode.
        """
        histories = np.asarray(histories, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        m = histories.shape[0]
        if histories.shape[1:] != (self.history_length, self._n_exogenous):
            raise ValueError(f"history shape {histories.shape}")
        inputs = np.concatenate([histories, np.zeros((m, self.history_length, 1))], axis=-1)
        standardised = self._input_scaler.transform(inputs)
        standardised[..., self._n_exogenous] = 0.0  # unobserved target (SOFT_SENSOR masking)
        exogenous_scaler = self._input_scaler.select(range(self._n_exogenous))
        batch = WindowBatch(
            history_inputs=standardised,
            future_exogenous=exogenous_scaler.transform(current)[:, None, :],
            future_targets=np.zeros((m, 1, 1)),
            run_index=np.zeros(m),
            window_start=np.zeros(m),
        )
        prediction = np.asarray(self._model.predict(batch), dtype=np.float64)
        return self._target_scaler.inverse(prediction[:, 0, :])[:, 0]
