r"""Probe D2+D4: Tragfaehigkeit von arch_composite.jsonl / mamba_reduced_comparison.jsonl.

Prueft die u>u_terminal-Verletzung in V0.

KEIN Netztraining (PersistenceModel/LinearARXModel sind geschlossene Baselines, kein
Gradientenverfahren), KEINE Aenderung an ``src/``.

Vorbedingung (siehe D1-Befund im Chat): arch_composite.jsonl hat einen ANDEREN
process/fingerprint (b24c5b728681bca9) als V0 (82f65f576a2e5253) und laesst sich mit den
aktuellen conf/process/*.yaml NICHT rekonstruieren -- dieses Skript arbeitet deshalb nur
mit mamba_reduced_comparison.jsonl (fp 82f65f576a2e5253, split/digest 5b9628a151344f37,
master_seed=1, siehe scripts/run_mamba_reduced_comparison.py::_spec/EXPECTED).
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.data.splits import SplitName
from twinlink_ml.data.windows import TrajectoryWindowDataset
from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models import LinearARXModel, PersistenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.config import MaterialConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
N_TRAIN = 16
MASTER_SEED = 1
BUDGET_POINTS = (8, 16, 32, 64)

# Fingerprint aktualisiert von 82f65f576a2e5253 (Wert oben in der Vorbedingung, historisch
# korrekt fuer mamba_reduced_comparison.jsonl) auf 40aae2b4526a5ed7: NumericsConfig.
# enforce_gas_velocity_projection hinzugefuegt (Default AUS). ProcessConfig.fingerprint
# hasht die gesamte Config, daher aendert sich der Wert auch bei unveraendertem Verhalten.
# Aequivalenz belegt durch test_gas_velocity_projection_disabled_matches_legacy_baseline
# (tests/test_gas_velocity_projection.py) -- siehe dort fuer den Bitgleichheits-Beleg.
EXPECTED = {
    "process/fingerprint": "40aae2b4526a5ed7",
    "split/digest": "5b9628a151344f37",
    "seed/data": 57731891,
    "seed/noise": 1868603354,
    "seed/model": 2509151119,
}


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


def _violation_masks(trajectory: Trajectory, material: MaterialConfig) -> dict[str, np.ndarray]:
    diagnostics = trajectory.diagnostics
    output_names = trajectory.metadata["output_names"]
    bed_temp_idx = output_names.index("bed_temperature_celsius")
    return {
        "u_above_u_terminal": diagnostics["fluidization_margin_upper_m_per_s"] < 0.0,
        "u_below_u_mf": diagnostics["fluidization_margin_lower_m_per_s"] < 0.0,
        "T_bed_above_80C": trajectory.outputs[:, :, bed_temp_idx] > 80.0,
        "moisture_above_critical": diagnostics["particle_moisture_kg_per_kg"]
        > material.critical_moisture_kg_per_kg,
    }


def main() -> None:
    t0 = time.monotonic()
    result: dict = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=MASTER_SEED)

    actual = {
        "process/fingerprint": data.process_config.fingerprint,
        "split/digest": data.split.digest,
        "seed/data": data.seeds.data_seed,
        "seed/noise": data.seeds.noise_seed,
        "seed/model": data.seeds.model_seed,
    }
    mismatches = {k: (v, actual[k]) for k, v in EXPECTED.items() if v != actual[k]}
    if mismatches:
        raise RuntimeError(f"Reproduktion weicht ab: {mismatches}")
    print("Reproduktion bitgleich zu mamba_reduced_comparison.jsonl bestaetigt:", actual)

    trajectory = data.trajectory
    run_id = trajectory.run_id
    n_runs, n_steps, _ = trajectory.actions.shape
    train_pool = set(data.split.train)  # 64 Laeufe
    holdout_pool = set(data.split.test)  # 32 Laeufe, "test" im Code, "Holdout" im Auftrag

    masks = _violation_masks(trajectory, config.material)

    # ---------------------------------------------------------------- D2a
    d2a: dict = {}
    run_any_violation: dict[str, bool] = dict.fromkeys(run_id, False)
    for crit, mask in masks.items():
        run_has = {run_id[i]: bool(mask[i].any()) for i in range(n_runs)}
        for rid, flag in run_has.items():
            if flag:
                run_any_violation[rid] = True
        train_hits = sum(run_has[r] for r in train_pool)
        holdout_hits = sum(run_has[r] for r in holdout_pool)
        d2a[crit] = {
            "train_pool_runs_affected": train_hits,
            "train_pool_n": len(train_pool),
            "train_pool_fraction": train_hits / len(train_pool),
            "holdout_runs_affected": holdout_hits,
            "holdout_n": len(holdout_pool),
            "holdout_fraction": holdout_hits / len(holdout_pool),
        }
    result["d2a"] = d2a

    # ---------------------------------------------------------------- D2b
    quarter = n_steps // 4
    edges = [0, quarter, 2 * quarter, 3 * quarter, n_steps]
    d2b: dict = {}
    for crit, mask in masks.items():
        counts = [int(mask[:, edges[q] : edges[q + 1]].sum()) for q in range(4)]
        total = sum(counts)
        d2b[crit] = {
            "points_q1_q2_q3_q4": counts,
            "fraction_q1_q2_q3_q4": [c / total if total else float("nan") for c in counts],
            "total_points": total,
        }
    result["d2b"] = d2b

    # ---------------------------------------------------------------- D2c
    d2c: dict = {}
    for crit, mask in masks.items():
        first_times = []
        for i in range(n_runs):
            hits = np.nonzero(mask[i])[0]
            if hits.size:
                first_times.append(float(trajectory.time_s[hits[0]]))
        if first_times:
            arr = np.asarray(first_times)
            d2c[crit] = {
                "n_affected_runs": len(first_times),
                "median_s": float(np.median(arr)),
                "min_s": float(np.min(arr)),
                "max_s": float(np.max(arr)),
            }
        else:
            d2c[crit] = {"n_affected_runs": 0}
    result["d2c"] = d2c

    # ---------------------------------------------------------------- D2d
    train_pool_ordered = data.split.train  # 64, Design-Reihenfolge
    d2d: dict = {}
    for n in BUDGET_POINTS:
        subset = train_pool_ordered[:n]
        violating = sum(1 for r in subset if run_any_violation[r])
        d2d[str(n)] = {"n_budget": n, "violating": violating, "surviving": n - violating}
    result["d2d"] = d2d

    print("=== D2 ===")
    print(json.dumps({"d2a": d2a, "d2b": d2b, "d2c": d2c, "d2d": d2d}, indent=2))

    # ---------------------------------------------------------------- D4
    print("=== D4: Persistenz + ARX, unmaskiert vs. laufweise ausgeschlossen ===")
    any_violation_holdout = {r for r in holdout_pool if run_any_violation[r]}
    masked_test_ids = tuple(r for r in data.split.test if r not in any_violation_holdout)
    print(
        f"Holdout: {len(holdout_pool)} Laeufe gesamt, "
        f"{len(any_violation_holdout)} mit >=1 Verletzung (irgendein Kriterium), "
        f"{len(masked_test_ids)} verbleiben nach Ausschluss."
    )

    train, validation, test_unmasked = build_datasets(
        data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL
    )
    split16 = data.split.subset(N_TRAIN)
    masked_split = dataclasses.replace(split16, test=masked_test_ids)
    test_masked = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, masked_split, SplitName.TEST, spec.window, scalers=train.scalers
    )

    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))
    d4: dict = {}
    for label, factory in (("persistence", PersistenceModel), ("arx_p4_q2", LinearARXModel)):
        model = factory()
        model.fit(train, seed=data.seeds.model_seed, validation=validation)
        unmasked_metrics = evaluator.evaluate(model, test_unmasked).as_flat_metrics()
        masked_metrics = evaluator.evaluate(model, test_masked).as_flat_metrics()
        entry: dict = {}
        for target in spec.target_names:
            for h in (1, spec.horizon):
                key = f"nrmse/{target}/h{h}"
                unmasked_v = unmasked_metrics.get(key)
                masked_v = masked_metrics.get(key)
                diff = None if unmasked_v is None or masked_v is None else masked_v - unmasked_v
                entry[key] = {"unmasked": unmasked_v, "masked": masked_v, "diff": diff}
        entry["n_test_runs_unmasked"] = unmasked_metrics.get("n_test_runs")
        entry["n_test_runs_masked"] = masked_metrics.get("n_test_runs")
        d4[label] = entry
        print(f"-- {label} --")
        print(json.dumps(entry, indent=2))
    result["d4"] = d4

    result["_elapsed_s"] = time.monotonic() - t0
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_sweep_contamination.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ngeschrieben: {out_path} ({result['_elapsed_s']:.1f} s)")


if __name__ == "__main__":
    main()
