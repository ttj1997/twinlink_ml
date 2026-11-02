r"""HA1: How large is GA1's safety margin relative to the prediction error of the
moisture soft sensor? Pure evaluation, no new run, no training.

Preparation for the M2 synthesis (question F6 from ``docs/08_synthese_m2_bestand.md``): the
comparison there, "safety margin of the agent against the margin justified by the
prediction error", did not exist anywhere in the lab book as a calculation of its own so
far -- this finding provides it before it may be cited in the synthesis.

Links three already committed artifacts without recomputing any of them:

  - GA1's safety margin (``rl_bandit_ga1_safety_margin.json``,
    ``median_across_seeds/moisture_margin_agent_binding_subset_kg_per_kg``): median over
    the 17 oracle-binding contexts per seed, of that the median over 3 seeds.
  - prediction error of the EXCITED-trained LSTM soft sensor (Befund 51/58, the training
    case recommended for the moisture soft sensor), ``arch_learning_curve_projected.jsonl``,
    ``model=lstm``, ``n_train=64``, ``test_set=constant_holdout``,
    ``rmse_run_median_physical/product_moisture_kg_per_kg/h36``, median over 5 seeds.
  - prediction error of the CONSTANT-trained LSTM soft sensor (Befund 55, pure
    operating data), ``arch_learning_curve_constant_trained.jsonl``, the same cell.

Convention (newly introduced in this evaluation, not derived from a plant or standard
specification): safety factor = margin / (3 x RMSE). Three standard deviations is a
convention common in statistics, but deliberately chosen here -- not a plant or
process limit.

Usage::

    python scripts/probe_bandit_env_ha1_margin_vs_prediction_error.py \
        --output artifacts/runs/rl_bandit_ha1_margin_vs_prediction_error.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twinlink_ml.utils.tracking import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GA1_MARGIN_JSON = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_ga1_safety_margin.json"
DEFAULT_EXCITED_TRAINED_JSONL = (
    REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl"
)
DEFAULT_CONSTANT_TRAINED_JSONL = (
    REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_constant_trained.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_ha1_margin_vs_prediction_error.json"
SAFETY_FACTOR_MULTIPLE = 3.0
TARGET_METRIC = "rmse_run_median_physical/product_moisture_kg_per_kg/h36"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _lstm_moisture_error(jsonl_path: Path) -> dict[str, object]:
    """Median RMSE (physical) of LSTM on product_moisture_kg_per_kg,
    n_train=64, test_set=constant_holdout, h=36, over all available seeds."""
    values: list[float] = []
    for record in read_jsonl(jsonl_path):
        if record.tags.get("model") != "lstm":
            continue
        if record.tags.get("n_train") != "64":
            continue
        if record.tags.get("test_set") != "constant_holdout":
            continue
        value = record.metrics.get(TARGET_METRIC)
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError(f"Keine passenden LSTM-Records in {jsonl_path}")
    return {
        "n_seeds": len(values),
        "values_kg_per_kg": sorted(values),
        "median_kg_per_kg": _median(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ga1-margin-json", type=Path, default=DEFAULT_GA1_MARGIN_JSON)
    parser.add_argument("--excited-trained-jsonl", type=Path, default=DEFAULT_EXCITED_TRAINED_JSONL)
    parser.add_argument(
        "--constant-trained-jsonl", type=Path, default=DEFAULT_CONSTANT_TRAINED_JSONL
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    ga1 = json.loads(arguments.ga1_margin_json.read_text(encoding="utf-8"))
    margin_kg_per_kg = ga1["median_across_seeds"]["moisture_margin_agent_binding_subset_kg_per_kg"]
    n_binding = ga1["n_oracle_binding_moisture"]
    n_points = ga1["n_points"]

    excited = _lstm_moisture_error(arguments.excited_trained_jsonl)
    constant = _lstm_moisture_error(arguments.constant_trained_jsonl)

    cases = {}
    for label, error_info, jsonl_path in (
        ("excited_trained", excited, arguments.excited_trained_jsonl),
        ("constant_trained", constant, arguments.constant_trained_jsonl),
    ):
        rmse = error_info["median_kg_per_kg"]
        threshold = SAFETY_FACTOR_MULTIPLE * rmse
        cases[label] = {
            "source_jsonl": str(jsonl_path.relative_to(REPO_ROOT)),
            "rmse_run_median_physical_kg_per_kg": rmse,
            "n_seeds": error_info["n_seeds"],
            "values_over_seeds_kg_per_kg": error_info["values_kg_per_kg"],
            f"{SAFETY_FACTOR_MULTIPLE:g}x_rmse_kg_per_kg": threshold,
            "margin_over_threshold_ratio": margin_kg_per_kg / threshold,
        }

    print("=" * 88)
    print("HA1: Sicherheitsmarge des Agenten gegen Vorhersagefehler des Feuchte-Softsensors")
    print("=" * 88)
    print(
        f"GA1-Marge (Median ueber {n_binding}/{n_points} orakel-bindende Kontexte, "
        f"Median ueber 3 Seeds): {margin_kg_per_kg:.6f} kg/kg"
    )
    for label, case in cases.items():
        print(f"\n--- {label} " + "-" * (60 - len(label)))
        print(
            f"  RMSE (Median ueber {case['n_seeds']} Seeds): {case['rmse_run_median_physical_kg_per_kg']:.6f} kg/kg"
        )
        print(
            f"  {SAFETY_FACTOR_MULTIPLE:g}x RMSE: {case[f'{SAFETY_FACTOR_MULTIPLE:g}x_rmse_kg_per_kg']:.6f} kg/kg"
        )
        print(
            f"  Marge / ({SAFETY_FACTOR_MULTIPLE:g}x RMSE) = {case['margin_over_threshold_ratio']:.2f}"
        )

    output = {
        "safety_factor_multiple": SAFETY_FACTOR_MULTIPLE,
        "target_metric": TARGET_METRIC,
        "ga1_margin_kg_per_kg": margin_kg_per_kg,
        "ga1_n_oracle_binding_moisture": n_binding,
        "ga1_n_points": n_points,
        "ga1_source_json": str(arguments.ga1_margin_json.relative_to(REPO_ROOT)),
        "cases": cases,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
