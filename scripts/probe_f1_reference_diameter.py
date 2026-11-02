r"""F1: clarify the reference diameter of fluidization_margin_upper/lower_m_per_s.

NO network training, NO change to src/. Calls terminal_velocity_m_per_s /
minimum_fluidization_velocity_m_per_s (fluidization.py, unchanged) with the
seed diameter instead of d50, in order to recompute u_t(d_Keim) -- in the model itself
(greybox.py:466-471) only d50 is used, see the quotation below.

Dataset: identical to D2 (mamba_reduced_comparison.jsonl reconstruction, 128 runs,
master_seed=1, greybox_gab_placeholder.yaml, dt=25s).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.fluidization import terminal_velocity_m_per_s

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
MASTER_SEED = 1


def _spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
    )


def main() -> None:
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, _spec(), master_seed=MASTER_SEED)

    trajectory = data.trajectory
    material = config.material
    gas = config.gas
    n_runs, n_steps, _ = trajectory.actions.shape
    gas_velocity = trajectory.actions[..., 1]

    # (a) against d50 -- as actually computed in the model (diagnostics unchanged)
    u_terminal_d50 = trajectory.diagnostics["terminal_velocity_m_per_s"]
    margin_upper_d50 = trajectory.diagnostics["fluidization_margin_upper_m_per_s"]
    violation_d50 = margin_upper_d50 < 0.0

    # (b) against d_Keim -- NOT implemented in the model, recomputed here with the same
    # correlation (Haider & Levenspiel, fluidization.py:68-94), imported unchanged.
    backend = NumpyBackend()
    bed_temperature = trajectory.states[..., 3]
    inlet_humidity = trajectory.diagnostics["inlet_humidity_kg_per_kg"]
    nucleus_diameter = np.full_like(bed_temperature, material.nucleus_diameter_m)
    u_terminal_nucleus = np.asarray(
        terminal_velocity_m_per_s(
            backend, gas, material, nucleus_diameter, bed_temperature, inlet_humidity
        )
    )
    margin_upper_nucleus = u_terminal_nucleus - gas_velocity
    violation_nucleus = margin_upper_nucleus < 0.0

    def _summary(mask: np.ndarray) -> dict:
        run_hits = mask.any(axis=1)
        return {
            "runs_affected": int(run_hits.sum()),
            "n_runs": n_runs,
            "run_fraction": float(run_hits.mean()),
            "points_affected": int(mask.sum()),
            "n_points": n_runs * n_steps,
            "point_fraction": float(mask.mean()),
        }

    result = {
        "nucleus_diameter_m": material.nucleus_diameter_m,
        "d50_mean_m": float(trajectory.diagnostics["d50_m"].mean()),
        "d50_min_m": float(trajectory.diagnostics["d50_m"].min()),
        "d50_max_m": float(trajectory.diagnostics["d50_m"].max()),
        "u_terminal_d50_mean_m_per_s": float(u_terminal_d50.mean()),
        "u_terminal_nucleus_mean_m_per_s": float(u_terminal_nucleus.mean()),
        "a_against_d50": _summary(violation_d50),
        "b_against_nucleus": _summary(violation_nucleus),
        "_elapsed_s": time.monotonic() - t0,
    }
    print(json.dumps(result, indent=2))
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_f1_reference_diameter.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ngeschrieben: {out_path}")


if __name__ == "__main__":
    main()
