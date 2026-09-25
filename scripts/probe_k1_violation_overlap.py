r"""CA2 (Nachtrag zu Befund 51/K1): quantifiziert die Überlappung der vier Verletzungs-
kriterien aus ``_violation_mask`` (u > u_terminal, u < u_mf, T_bed > 80 °C, Feuchte >
kritisch), auf derselben Punktpopulation wie K1s Zusatzmessung (``probe_k1_l1_clip_
sensitivity_projected.py``, 3 Seeds, Trainingspool-Block ``[:64]`` je Seed,
``_excited_spec`` aus ``probe_l1_clip_sensitivity.py``), einmal unprojiziert und einmal
projiziert (``enforce_gas_velocity_projection``).

Keine neuen Simulationsexperimente -- dieselben Seeds/dieselbe Config wie K1 (bzw. deren
unprojizierte Referenz), deterministisch reproduziert, nur mit einer feineren Auswertung
(komponentenweise statt nur Einzelkriterium/Union) als im ursprünglichen Befund 51
festgehalten wurde.

Ergebnis dient Befund 48/CA2: Befund 48 nahm an, die verletzende Region zerfalle etwa
hälftig in ``u > u_terminal`` und ``bed_temperature > 80 °C``, zwei UNABHÄNGIGE Hälften.
K1 (Befund 51) widerlegte das bereits an den Aggregatzahlen (Union sinkt nur 34 % statt
50 %, wenn ``u > u_terminal`` fast vollständig verschwindet) -- dieses Skript legt offen,
WELCHER Anteil der verbleibenden Verletzung auf welches Kriterium fällt und wie stark sie
sich überlappen.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.config import MaterialConfig, ProcessConfig
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_k1_violation_overlap.json"

TRAIN_SEEDS = (1, 2, 3)
N_TRAIN_POOL = 64
"""Trainingspool-Block je Seed, identisch zu ``probe_l1_clip_sensitivity.py``s
``excited.trajectory.actions[:64]`` -- dieselbe Punktpopulation wie K1s Zusatzmessung."""

EXPECTED_FINGERPRINTS = {
    False: "6ecedc6bd993fb3d",
    True: "bddca37f716f5cfc",
}


def _excited_spec() -> DataSpec:
    """Bitgleich zu ``probe_l1_clip_sensitivity.py::_excited_spec``."""
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


def _load_config(*, projected: bool) -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        base, numerics=dataclasses.replace(base.numerics, enforce_gas_velocity_projection=projected)
    )


def _violation_masks(trajectory: Trajectory, material: MaterialConfig) -> dict[str, np.ndarray]:
    """Bitgleich zu den vier ODER-verknüpften Kriterien in ``_violation_mask`` (Befund
    46/D2/F1), hier einzeln gehalten statt sofort vereinigt."""
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


def _pooled_masks(*, projected: bool) -> dict[str, np.ndarray]:
    """Vereinigt die vier Einzelmasken über die 3-Seed-Trainingspool-Population
    (192 Läufe x n_steps Punkte) zu EINEM flachen Array je Kriterium."""
    config = _load_config(projected=projected)
    spec = _excited_spec()
    pooled: dict[str, list[np.ndarray]] = {
        "u_above_u_terminal": [],
        "u_below_u_mf": [],
        "T_bed_above_80C": [],
        "moisture_above_critical": [],
    }
    fingerprint = None
    for seed in TRAIN_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            data = generate_data(config, spec, master_seed=seed)
        fingerprint = data.process_config.fingerprint
        masks = _violation_masks(data.trajectory, config.material)
        for name, mask in masks.items():
            pooled[name].append(mask[:N_TRAIN_POOL].reshape(-1))

    expected = EXPECTED_FINGERPRINTS[projected]
    if not fingerprints_equivalent(fingerprint, expected):
        raise RuntimeError(
            f"process/fingerprint={fingerprint!r} != erwartet {expected!r} (projected={projected}) "
            "-- Abbruch, Ergebnis nicht uebernommen."
        )
    return {name: np.concatenate(arrays) for name, arrays in pooled.items()}


def _summarise(masks: dict[str, np.ndarray]) -> dict[str, float]:
    stacked = np.stack(list(masks.values()), axis=0)  # (4, n_points)
    n_points = stacked.shape[1]
    n_criteria_hit = stacked.sum(axis=0)  # wie viele der 4 Kriterien je Punkt verletzt sind
    union = n_criteria_hit > 0
    only_bed_temp = masks["T_bed_above_80C"] & (n_criteria_hit == 1)

    summary = {f"{name}_fraction": float(mask.mean()) for name, mask in masks.items()}
    summary["union_fraction"] = float(union.mean())
    summary["multiple_criteria_fraction"] = float((n_criteria_hit >= 2).mean())
    summary["only_T_bed_above_80C_fraction"] = float(only_bed_temp.mean())
    summary["n_points"] = n_points
    return summary


def main() -> None:
    result: dict = {}
    for projected in (False, True):
        masks = _pooled_masks(projected=projected)
        result["projected" if projected else "unprojected"] = _summarise(masks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("=== CA2: Überlappung der Verletzungskriterien (3 Seeds, Trainingspool [:64]) ===")
    for label in ("unprojected", "projected"):
        s = result[label]
        print(f"\n-- {label} (n={s['n_points']}) --")
        print(f"  u > u_terminal        {100 * s['u_above_u_terminal_fraction']:6.2f} %")
        print(f"  u < u_mf              {100 * s['u_below_u_mf_fraction']:6.2f} %")
        print(f"  T_bed > 80 C          {100 * s['T_bed_above_80C_fraction']:6.2f} %")
        print(f"  Feuchte > kritisch    {100 * s['moisture_above_critical_fraction']:6.2f} %")
        print(f"  Union (irgendeines)   {100 * s['union_fraction']:6.2f} %")
        print(f"  >= 2 Kriterien gleichzeitig {100 * s['multiple_criteria_fraction']:6.2f} %")
        print(f"  NUR T_bed > 80 C      {100 * s['only_T_bed_above_80C_fraction']:6.2f} %")
    print(f"\nGeschrieben: {OUT_PATH}")


if __name__ == "__main__":
    main()
