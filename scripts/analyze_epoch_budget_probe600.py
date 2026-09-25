r"""TA4: Transformer/LSTM bei ``net_max_epochs=600`` gegen den 200-Epochen-Stand (Befund 58/59c).

Laedt drei bestehende Logs, OHNE erneut zu trainieren:

- ``arch_epoch_budget_probe600.jsonl`` (TA3, dieser Auftrag): Transformer/LSTM,
  ``n_train=64``, 5 Seeds, ``net_max_epochs=600``. AUSDRUECKLICH NICHT protokollgleich zu
  Befund 51/58 -- einzige Abweichung ist das Epochenbudget, siehe Commit-Nachricht.
  Baselines nicht enthalten (``--skip-baselines``).
- ``arch_learning_curve_projected.jsonl`` (Befund 51/28-Protokoll): LSTM bei 200 Epochen.
- ``arch_learning_curve_projected_extended.jsonl`` (Befund 58/OA4): Transformer bei
  200 Epochen.

Kernfrage (Befund 58/59c): ueberholt der Transformer LSTM auf ``bed_temperature_celsius``
(Konstantlauf-Holdout, h=36, ``n_train=64``), wenn beide auskonvergieren duerfen, statt
durch ``net_max_epochs=200`` gemeinsam abgeschnitten zu werden (Befund 59c: Transformer an
allen 5 Seeds, LSTM an 2/5 gecappt -- die urspruengliche Aufhol-Aussage war dadurch fuer
BEIDE Architekturen budget-konfundiert, nicht nur fuer den Transformer).

Aufruf::

    python scripts/analyze_epoch_budget_probe600.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE600_LOG = REPO_ROOT / "artifacts" / "runs" / "arch_epoch_budget_probe600.jsonl"
LSTM200_LOG = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl"
TRANSFORMER200_LOG = (
    REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected_extended.jsonl"
)

HEADS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TEST_SETS = ("excited", "constant_holdout")
HORIZON = 36
N_TRAIN = 64


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _filter(
    records: list[dict], *, label: str, test_set: str, n_train: int = N_TRAIN
) -> list[dict]:
    return [
        r
        for r in records
        if r["params"]["model/label"] == label
        and r["params"]["test_set"] == test_set
        and r["params"]["n_train"] == n_train
    ]


def _rmse(records: list[dict], head: str, horizon: int = HORIZON) -> list[tuple[int, float]]:
    key = f"rmse_run_median_physical/{head}/h{horizon}"
    return sorted((r["params"]["master_seed"], r["metrics"][key]) for r in records)


def _epochs_run(records: list[dict]) -> dict[int, dict[str, int]]:
    """Je Seed die erreichte Epochenzahl pro Kopf (ein Record je Testmenge traegt dieselben
    Trainingsdiagnosen -- hier ueber ``excited`` dedupliziert)."""
    out: dict[int, dict[str, int]] = {}
    for r in records:
        p = r["params"]
        if p["test_set"] != "excited":
            continue
        seed = p["master_seed"]
        out[seed] = {h: p[f"training/epochs_run/{h}"] for h in HEADS}
    return out


def main() -> None:
    probe600 = _load(PROBE600_LOG)
    lstm200 = _load(LSTM200_LOG)
    transformer200 = _load(TRANSFORMER200_LOG)

    print("=== TA4a: erreichte Epochenzahl je Seed (net_max_epochs=600) ===")
    for label in ("transformer", "lstm"):
        records = _filter(probe600, label=label, test_set="excited")
        epochs = _epochs_run(records)
        print(f"-- {label} --")
        for seed in sorted(epochs):
            capped = [h for h, v in epochs[seed].items() if v >= 600]
            print(f"  seed={seed}: {epochs[seed]}  capped_at_600={capped}")

    print()
    print("=== TA4b: rmse_run_median_physical, Transformer(600) gegen LSTM(600), h=36 ===")
    print(
        f"{'Kopf':28} {'Testmenge':18} {'Transformer(600)':>18} {'LSTM(600)':>12} {'Faktor T/L':>11}"
    )
    for head in HEADS:
        for test_set in TEST_SETS:
            t_vals = [
                v for _, v in _rmse(_filter(probe600, label="transformer", test_set=test_set), head)
            ]
            l_vals = [v for _, v in _rmse(_filter(probe600, label="lstm", test_set=test_set), head)]
            t_med = statistics.median(t_vals)
            l_med = statistics.median(l_vals)
            print(f"{head:28} {test_set:18} {t_med:18.5f} {l_med:12.5f} {t_med / l_med:11.3f}")

    print()
    print("=== TA4d: 600 gegen 200 Epochen, je Architektur, Konstantlauf-Holdout, h=36 ===")
    print(
        f"{'Kopf':28} {'Architektur':12} {'200ep (Befund 58)':>18} {'600ep (hier)':>13} {'Delta relativ':>14}"
    )
    for head in HEADS:
        t600 = statistics.median(
            v
            for _, v in _rmse(
                _filter(probe600, label="transformer", test_set="constant_holdout"), head
            )
        )
        t200 = statistics.median(
            v
            for _, v in _rmse(
                _filter(transformer200, label="transformer", test_set="constant_holdout"), head
            )
        )
        l600 = statistics.median(
            v for _, v in _rmse(_filter(probe600, label="lstm", test_set="constant_holdout"), head)
        )
        l200 = statistics.median(
            v for _, v in _rmse(_filter(lstm200, label="lstm", test_set="constant_holdout"), head)
        )
        print(
            f"{head:28} {'transformer':12} {t200:18.5f} {t600:13.5f} "
            f"{(t600 - t200) / t200 * 100:+13.1f}%"
        )
        print(
            f"{head:28} {'lstm':12} {l200:18.5f} {l600:13.5f} {(l600 - l200) / l200 * 100:+13.1f}%"
        )

    print()
    print("=== Kernfrage: Faktor Transformer/LSTM bei bed_temperature_celsius, ")
    print("    Konstantlauf-Holdout, h=36, n_train=64 -- 200ep (Befund 58) gegen 600ep (hier) ===")
    head = "bed_temperature_celsius"
    t200 = statistics.median(
        v
        for _, v in _rmse(
            _filter(transformer200, label="transformer", test_set="constant_holdout"), head
        )
    )
    l200 = statistics.median(
        v for _, v in _rmse(_filter(lstm200, label="lstm", test_set="constant_holdout"), head)
    )
    t600 = statistics.median(
        v
        for _, v in _rmse(_filter(probe600, label="transformer", test_set="constant_holdout"), head)
    )
    l600 = statistics.median(
        v for _, v in _rmse(_filter(probe600, label="lstm", test_set="constant_holdout"), head)
    )
    print(
        f"  200 Epochen (Befund 58): Transformer {t200:.5f} / LSTM {l200:.5f} = Faktor {t200 / l200:.3f}"
    )
    print(
        f"  600 Epochen (hier):      Transformer {t600:.5f} / LSTM {l600:.5f} = Faktor {t600 / l600:.3f}"
    )


if __name__ == "__main__":
    main()
