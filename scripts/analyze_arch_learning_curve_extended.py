r"""OA5: Acht-Architekturen-Vergleich, GRU/LSTM/Baselines aus dem Ur-Datensatz (Befund 51)
plus Transformer/xLSTM/Mamba aus der Erweiterung -- beide protokollgleich (identischer
Fingerprint, Split, Seed-Buendel je master_seed, siehe OA3-Pilot-Pruefung), deshalb hier
gemeinsam gelesen und ausgewertet, OHNE erneut zu trainieren.

Liest ausschliesslich:
  - artifacts/runs/arch_learning_curve_projected.jsonl (persistence/static/arx/gru/lstm)
  - artifacts/runs/arch_learning_curve_projected_extended.jsonl (transformer/xlstm/mamba)

Fuer jede (Architektur, Testmenge, Zielkanal, Horizont, Budgetpunkt)-Kombination: Median
und Spannweite (Min..Max) von ``rmse_run_median_physical`` und ``nrmse_run_median`` ueber
die 5 Seeds. Zusaetzlich:
  - Monotonie der Lernkurve (Median ueber n_train) je Architektur/Kanal/Testmenge/Horizont
    -- Verletzungen benannt, nicht nur gezaehlt (Persistenz konstruktionsbedingt
    ausgenommen, kein Training auf n_train).
  - Epochenobergrenze (OA5b): Anteil der Fits, die ``model/net_max_epochs`` erreichen,
    je Architektur/Budgetpunkt/Kopf (nur die fuenf Torch-Architekturen, Baselines haben
    keine Epochen).

Aufruf::

    python scripts/analyze_arch_learning_curve_extended.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_LOG = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl"
EXT_LOG = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected_extended.jsonl"

ARCHITECTURE_ORDER = (
    "persistence",
    "static",
    "arx",
    "gru",
    "lstm",
    "transformer",
    "xlstm",
    "mamba",
)
TORCH_ARCHITECTURES = ("gru", "lstm", "transformer", "xlstm", "mamba")
TEST_SETS = ("excited", "constant_holdout")
TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
HORIZONS = (1, 36)
BUDGET_POINTS = (8, 16, 32, 64)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric_values(
    records: list[dict],
    *,
    label: str,
    test_set: str,
    n_train: int,
    metric_prefix: str,
    target: str,
    horizon: int,
) -> list[float]:
    key = f"{metric_prefix}/{target}/h{horizon}"
    out = []
    for record in records:
        params = record["params"]
        if (
            params["model/label"] == label
            and params["test_set"] == test_set
            and params["n_train"] == n_train
        ):
            out.append(record["metrics"][key])
    return out


def _median_range(values: list[float]) -> tuple[float, float, float]:
    return (statistics.median(values), min(values), max(values))


def _print_tables(records: list[dict]) -> None:
    for test_set in TEST_SETS:
        print(f"\n{'=' * 100}\nTestmenge: {test_set}\n{'=' * 100}")
        for target in TARGETS:
            for horizon in HORIZONS:
                print(f"\n--- {target}, h={horizon} -- rmse_run_median_physical  |  nrmse_run_median ---")
                header = "".join(f"{label:>15}" for label in ARCHITECTURE_ORDER)
                print(f"{'n_train':>8} {header}")
                for n_train in BUDGET_POINTS:
                    rmse_cells = ""
                    nrmse_cells = ""
                    for label in ARCHITECTURE_ORDER:
                        rmse_vals = _metric_values(
                            records,
                            label=label,
                            test_set=test_set,
                            n_train=n_train,
                            metric_prefix="rmse_run_median_physical",
                            target=target,
                            horizon=horizon,
                        )
                        nrmse_vals = _metric_values(
                            records,
                            label=label,
                            test_set=test_set,
                            n_train=n_train,
                            metric_prefix="nrmse_run_median",
                            target=target,
                            horizon=horizon,
                        )
                        if rmse_vals:
                            med, lo, hi = _median_range(rmse_vals)
                            rmse_cells += f"{med:>15.4g}"
                        else:
                            rmse_cells += f"{'-':>15}"
                        if nrmse_vals:
                            med, lo, hi = _median_range(nrmse_vals)
                            nrmse_cells += f"{med:>15.4g}"
                        else:
                            nrmse_cells += f"{'-':>15}"
                    print(f"{n_train:>8} {rmse_cells}  (rmse)")
                    print(f"{'':>8} {nrmse_cells}  (nrmse)")


def _print_ranges(records: list[dict]) -> None:
    print(f"\n{'=' * 100}\nSpannweiten (Min..Max ueber 5 Seeds), rmse_run_median_physical\n{'=' * 100}")
    for test_set in TEST_SETS:
        for target in TARGETS:
            for horizon in HORIZONS:
                for label in ARCHITECTURE_ORDER:
                    for n_train in BUDGET_POINTS:
                        vals = _metric_values(
                            records,
                            label=label,
                            test_set=test_set,
                            n_train=n_train,
                            metric_prefix="rmse_run_median_physical",
                            target=target,
                            horizon=horizon,
                        )
                        if not vals:
                            continue
                        med, lo, hi = _median_range(vals)
                        print(
                            f"{test_set:>17} {target:>28} h{horizon:<3} {label:>12} "
                            f"n={n_train:<3} median={med:.4g} [{lo:.4g}..{hi:.4g}] (n_seeds={len(vals)})"
                        )


def _check_monotonicity(records: list[dict]) -> None:
    print(f"\n{'=' * 100}\nMonotonie (Median-rmse_run_median_physical ueber n_train, absteigend erwartet)\n{'=' * 100}")
    for test_set in TEST_SETS:
        for target in TARGETS:
            for horizon in HORIZONS:
                for label in ARCHITECTURE_ORDER:
                    medians = []
                    for n_train in BUDGET_POINTS:
                        vals = _metric_values(
                            records,
                            label=label,
                            test_set=test_set,
                            n_train=n_train,
                            metric_prefix="rmse_run_median_physical",
                            target=target,
                            horizon=horizon,
                        )
                        medians.append(statistics.median(vals) if vals else None)
                    if any(m is None for m in medians):
                        continue
                    violations = []
                    for i in range(1, len(medians)):
                        if medians[i] > medians[i - 1]:
                            violations.append(
                                f"n={BUDGET_POINTS[i - 1]}->n={BUDGET_POINTS[i]}: "
                                f"{medians[i - 1]:.4g}->{medians[i]:.4g}"
                            )
                    tag = "MONOTON" if not violations else "NICHT MONOTON: " + "; ".join(violations)
                    if label == "persistence":
                        tag += " (Artefakt: kein Training auf n_train, siehe K4-Konvention)"
                    print(f"{test_set:>17} {target:>28} h{horizon:<3} {label:>12}: {tag}")


def _epoch_ceiling(records: list[dict]) -> None:
    print(f"\n{'=' * 100}\nOA5b: Epochenobergrenze (training/epochs_run vs. model/net_max_epochs)\n{'=' * 100}")
    binding_by_arch: dict[str, list[bool]] = {label: [] for label in TORCH_ARCHITECTURES}
    for label in TORCH_ARCHITECTURES:
        for n_train in BUDGET_POINTS:
            per_head_binding: dict[str, list[bool]] = {target: [] for target in TARGETS}
            for record in records:
                params = record["params"]
                if (
                    params["model/label"] != label
                    or params["n_train"] != n_train
                    or params["test_set"] != "excited"
                ):
                    continue
                max_epochs = params.get("model/net_max_epochs")
                for target in TARGETS:
                    key = f"training/epochs_run/{target}"
                    if key not in params:
                        continue
                    bound = params[key] == max_epochs
                    per_head_binding[target].append(bound)
                    binding_by_arch[label].append(bound)
            cells = []
            for target in TARGETS:
                flags = per_head_binding[target]
                if not flags:
                    cells.append(f"{target}=n/a")
                    continue
                n_bound = sum(flags)
                cells.append(f"{target}={n_bound}/{len(flags)}")
            print(f"{label:>12} n_train={n_train:<3} " + "  ".join(cells))
    print("\nAnteil ueber alle Budgetpunkte/Koepfe/Seeds, je Architektur:")
    for label in TORCH_ARCHITECTURES:
        flags = binding_by_arch[label]
        if not flags:
            print(f"  {label:>12}: keine Trainingsdiagnostik gefunden")
            continue
        frac = sum(flags) / len(flags)
        print(f"  {label:>12}: {sum(flags)}/{len(flags)} ({frac:.1%}) an der Obergrenze")


def main() -> None:
    base_records = _load(BASE_LOG)
    ext_records = _load(EXT_LOG)
    base_labels = {r["params"]["model/label"] for r in base_records}
    ext_labels = {r["params"]["model/label"] for r in ext_records}
    print(f"Basis ({BASE_LOG.name}): {len(base_records)} Records, Labels {sorted(base_labels)}")
    print(f"Erweiterung ({EXT_LOG.name}): {len(ext_records)} Records, Labels {sorted(ext_labels)}")
    overlap = base_labels & ext_labels
    if overlap:
        raise RuntimeError(
            f"Ueberlappende Labels zwischen Basis und Erweiterung: {overlap} -- "
            "Kombination wuerde Records verlieren oder doppelt zaehlen. Abbruch."
        )

    records = base_records + ext_records
    _print_tables(records)
    _print_ranges(records)
    _check_monotonicity(records)
    _epoch_ceiling(records)


if __name__ == "__main__":
    main()
