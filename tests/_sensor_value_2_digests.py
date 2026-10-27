"""Digests of the behaviour of existing configurations, for the bit-identity tests of study 2.

The reference values in ``tests/test_sensor_value_2.py`` were computed with these functions on
``origin/master`` 8f5db3e, BEFORE the window pipeline and the closed loop were extended for the
follow-up study (docs/29). The tests recompute them on the extended code; equal digests mean the
existing configurations behave bit for bit as before.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from typing import Any

import numpy as np


def _array_digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:20]


def window_digests() -> dict[str, str]:
    """Train/val/test windows and scalers of three existing data configurations."""
    from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data
    from twinlink_ml.evaluation.protocol import ObservationMode, spec_for_mode
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config("conf/process/greybox_gab_placeholder.yaml")
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    small = {
        "sample_time_s": 25.0,
        "run_duration_s": 3000.0,
        "n_train_max": 4,
        "n_val": 2,
        "n_test": 2,
        "history_length": 8,
        "horizon": 4,
        "stride": 4,
    }
    specs = {
        "legacy_two_targets": DataSpec(**small),
        "new_study_soft_moisture": spec_for_mode(
            DataSpec.new_study(**small, target_names=("product_moisture_kg_per_kg",)),
            ObservationMode.SOFT_SENSOR,
        ),
        "new_study_three_targets": DataSpec.new_study(
            **small,
            target_names=("energy_target", "bed_temperature_celsius", "product_moisture_kg_per_kg"),
        ),
    }
    out = {}
    for label, spec in specs.items():
        data = generate_data(config, spec, master_seed=5)
        train, validation, test = build_datasets(data, 4)
        parts = []
        for ds in (train, validation, test):
            assert ds is not None
            b = ds.full_batch()
            parts += [b.history_inputs, b.future_exogenous, b.future_targets]
            parts += [b.future_targets_truth, b.run_index, b.window_start]
        parts += [train.input_scaler.mean, train.input_scaler.scale]
        parts += [train.target_scaler.mean, train.target_scaler.scale]
        out[label] = _array_digest(*parts) + "|" + ",".join(train.exogenous_names)
    return out


def loop_digest() -> dict[str, str]:
    """Records of F0, F1 and F2 on two short scenarios of draw 0 (model mismatch, pre-roll)."""
    from twinlink_ml.headroom.config import load_headroom_config, study_process_config
    from twinlink_ml.headroom.scenarios import build_scenarios
    from twinlink_ml.outliers import load_outlier_config
    from twinlink_ml.sensitivity.config import load_sensitivity_config
    from twinlink_ml.sensor_value import (
        IdealMeasurement,
        InlineSensor,
        NoFeedback,
        load_sensor_value_config,
        parameter_draws,
        run_pairs,
        scenario_subset,
        warmed_up_states,
    )
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    config = load_sensor_value_config()
    study = load_headroom_config(config.headroom_config)
    nominal = study_process_config(study)
    draws, _ = parameter_draws(
        config.draws, load_sensitivity_config(config.sensitivity_config), nominal
    )
    subset = scenario_subset(build_scenarios(study, nominal, "observed"), 4)
    picks = [subset[0], subset[9]]
    n = round(1.0 * 3600.0 / picks[0].sample_time_s)
    scenarios = [
        dataclasses.replace(
            s, disturbances=s.disturbances[:, :n], moisture_limit=s.moisture_limit[:n]
        )
        for s in picks
    ]
    truth = draws[3].process
    states = warmed_up_states(study, nominal, truth, scenarios, 1800.0)
    measurement = load_process_config(study.process_config).measurement
    oc = load_outlier_config(config.outlier_config).outliers

    def feedbacks(variant: str) -> list[Any]:
        if variant == "F0":
            return [NoFeedback() for _ in scenarios]
        if variant == "F1":
            return [IdealMeasurement() for _ in scenarios]
        return [
            InlineSensor(
                nominal,
                measurement,
                25.0,
                outlier_rate=0.25,
                degrees_of_freedom=oc.t_degrees_of_freedom,
                scale_factor=oc.scale_factor,
                noise_seed=11 + i,
                outlier_seed=23 + i,
            )
            for i in range(len(scenarios))
        ]

    out = {}
    for variant in ("F0", "F1", "F2"):
        records = run_pairs(
            study,
            nominal,
            scenarios,
            [0.004, 0.01],
            feedbacks(variant),
            truth=truth,
            truth_initial_states=states,
            pre_roll_s=1800.0,
            measurement_interval_s=25.0,
            bias_time_constant_s=1800.0,
            variant=variant,
            draw=3,
            keep_arrays=True,
        )
        payload = json.dumps([r.as_dict() for r in records], sort_keys=True)
        out[variant] = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return out


if __name__ == "__main__":
    print(json.dumps({"windows": window_digests(), "loop": loop_digest()}, indent=1))
