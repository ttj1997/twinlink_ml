"""PR1c -- integration accuracy: re-simulate with halved (and quartered) internal step.

The stored data use 5 RK4 sub-steps of 5 s per 25 s sample. Here the same requested
actions/disturbances/initial states are re-simulated with 10 and 20 sub-steps (2.5 s, 1.25 s) and
the noise-free targets are compared. Acceptance guide (from the task): the deviation must be
far below the model error, rule of thumb < 1/10 of the LSTM moisture error.

    python scripts/verify/pr1c_integration.py
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
from _common import provenance, read_jsonl, require_clean_tree, write_artifact
from _data import make_data

CHANNELS = {"energy_target": 0, "bed_temperature_celsius": 1, "product_moisture_kg_per_kg": 2}
UNITS = {
    "energy_target": "kW",
    "bed_temperature_celsius": "K",
    "product_moisture_kg_per_kg": "kg/kg",
}


def lstm_reference_errors() -> dict[str, dict[str, float]]:
    """Median-over-seeds LSTM rmse_run_median_physical, n_train=64, h=36, from the records."""
    records = read_jsonl("arch_learning_curve_projected")
    out: dict[str, dict[str, float]] = {}
    for test_set in ("excited", "constant_holdout"):
        out[test_set] = {}
        for name in CHANNELS:
            values = [
                r["metrics"][f"rmse_run_median_physical/{name}/h36"]
                for r in records
                if r["tags"]["model"] == "lstm"
                and r["tags"]["test_set"] == test_set
                and r["params"]["n_train"] == 64
            ]
            assert len(values) == 5
            out[test_set][name] = float(np.median(values))
    return out


def rollout_with(data: Any, substeps: int) -> np.ndarray:
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator

    cfg = dataclasses.replace(
        data.process_config,
        numerics=dataclasses.replace(data.process_config.numerics, substeps=substeps),
    )
    sim = GreyBoxFluidBedGranulator(cfg)
    s = data.schedule
    initial = sim.state_from_physical(*(s.initial_states[:, i] for i in range(4)))
    traj = sim.rollout(
        s.actions,
        s.disturbances,
        run_id=s.run_id,
        seed=data.seeds.noise_seed,
        initial_state=initial,
    )
    return traj.outputs


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, col in CHANNELS.items():
        d = np.abs(a[..., col] - b[..., col])
        per_run_max = d.max(axis=1)
        result[name] = {
            "unit": UNITS[name],
            "max_abs": float(d.max()),
            "rms": float(np.sqrt(np.mean(d**2))),
            "median_of_per_run_max": float(np.median(per_run_max)),
            "worst_run_index": int(per_run_max.argmax()),
        }
    return result


def main() -> None:
    require_clean_tree()
    reference = lstm_reference_errors()
    report: dict[str, Any] = {
        "_provenance": provenance(),
        "lstm_reference_errors_n64_h36": reference,
    }
    pools = []
    for kind, seed, test_set in (
        ("excited", 1, "excited"),
        ("holdout", 999, "constant_holdout"),
        ("constant_train", 1, "constant_holdout"),
    ):
        data = make_data(kind, seed)
        base = rollout_with(data, 5)
        identical_to_stored = bool(np.array_equal(base, data.trajectory.outputs))
        half = rollout_with(data, 10)
        quarter = rollout_with(data, 20)
        d5_10, d10_20 = compare(base, half), compare(half, quarter)
        limits = {n: reference[test_set][n] / 10.0 for n in CHANNELS}
        entry: dict[str, Any] = {
            "kind": kind,
            "seed": seed,
            "n_runs": int(base.shape[0]),
            "substeps_5_reproduces_stored_outputs_bitwise": identical_to_stored,
            "delta_5_vs_10": d5_10,
            "delta_10_vs_20": d10_20,
            "convergence_ratio_max_abs_(RK4 smooth: ~16)": {
                n: (
                    d5_10[n]["max_abs"] / d10_20[n]["max_abs"] if d10_20[n]["max_abs"] > 0 else None
                )
                for n in CHANNELS
            },
            "limit_one_tenth_of_lstm_error": limits,
            "lstm_error_reference_test_set": test_set,
            "max_abs_delta_5_vs_10_over_limit": {
                n: d5_10[n]["max_abs"] / limits[n] for n in CHANNELS
            },
            "rms_delta_5_vs_10_over_limit": {n: d5_10[n]["rms"] / limits[n] for n in CHANNELS},
            "within_limit_max_abs": {n: bool(d5_10[n]["max_abs"] < limits[n]) for n in CHANNELS},
        }
        pools.append(entry)
        print(kind, seed, json.dumps(entry["max_abs_delta_5_vs_10_over_limit"]), flush=True)
    report["pools"] = pools
    write_artifact("verify_pr1c_integration", report)


if __name__ == "__main__":
    main()
