r"""Befund 63: resolving power applied to the eight-way comparison (Befund 51/58), plus
the GRU-versus-LSTM speed question (parameter count/hidden size).

Pure evaluation of already committed artifacts, no new run, no training.
Two independent questions, in one script, because both need the same data family
(``arch_learning_curve_projected.jsonl`` + ``..._extended.jsonl``, Befund 51/58,
same protocol, identical thread settings).

Part A -- speed question (task: does a different model size explain
why LSTM is faster than GRU in table C of the M2 inventory?).
Read ``model/n_parameters`` and ``model/net_hidden_size`` per architecture at ``n_train=64``,
do not estimate them.

Part B -- resolving power, TWO-SAMPLE procedure (correction of a first version that
used only the LSTM spread, see Befund 63 addendum). For the 5 seeds actually run
(not the 3 from the M2 project definition), the smallest detectable relative difference is
computed from the observed seed-to-seed spread of BOTH compared models (candidate AND
reference, :class:`~twinlink_ml.evaluation.resolution.SeedSpread`, range-based):

.. math::
   \Delta_{\min} = z_{1-\alpha/2}\,\sqrt{\frac{\hat\sigma_{\mathrm{Kandidat}}^2 +
   \hat\sigma_{\mathrm{Referenz}}^2}{n}}

instead of the one-sample approximation :math:`z\,\hat\sigma_{\mathrm{Referenz}}\,\sqrt{2/n}`,
which implicitly assumes equal spread on both sides. That does not hold for architectures with
very different seed-to-seed robustness (e.g. xLSTM's diverging optimisation on
`product_moisture_kg_per_kg`, Befund 58 OA5) -- the correction therefore does
change verdicts (the Befund 63 addendum documents which). LSTM remains the reference in
every claim checked here (not persistence: on the constant-run holdout, persistence is
bit-identical across all five seeds -- the holdout runs are fixed independently of `master_seed`,
persistence has no trainable parameters -- its spread there is zero by construction
and yields no threshold).

Primary metric ``rmse_run_median_physical`` (not ``nrmse_run_median``): all claims checked
here (Befund 58 OA5, Befund 59c, table A/B of the M2 inventory) are already formulated in
this metric -- a change of metric within the same check would introduce a new
inconsistency instead of resolving one.

Claims checked, with source:

  - GRU vs. LSTM, per channel, per test set, ``n_train=64`` (table A, M2 inventory).
  - Transformer vs. LSTM, ``bed_temperature_celsius``, constant run, 200 AND 600 epochs
    (Befund 58 OA5b / Befund 58 addendum, "factor 1.026" / "factor 1.000").
  - Transformer vs. LSTM, ``energy_target``, constant run, 200 AND 600 epochs
    (Befund 58 addendum, "factor 1.082" / "factor 1.158").
  - xLSTM vs. LSTM and Mamba vs. LSTM, per channel, per test set, ``n_train=64``
    ("lag clearly behind", Befund 58 conclusion).

Usage::

    python scripts/analyze_m2_resolution_verdicts.py \
        --output artifacts/runs/analyze_m2_resolution_verdicts.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from twinlink_ml.evaluation.resolution import SeedSpread
from twinlink_ml.utils.seeding import VarianceSource
from twinlink_ml.utils.tracking import RunRecord, read_jsonl

Z_95 = 1.960

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTED_JSONL = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl"
EXTENDED_JSONL = REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected_extended.jsonl"
EPOCH600_JSONL = REPO_ROOT / "artifacts" / "runs" / "arch_epoch_budget_probe600.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "runs" / "analyze_m2_resolution_verdicts.json"

METRIC = "rmse_run_median_physical"
HORIZON = "36"
N_TRAIN_HEADLINE = "64"
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TEST_SETS = ("excited", "constant_holdout")
CONFIDENCE = 0.95
PARAMETER_FIELDS = ("model/n_parameters", "model/net_hidden_size", "model/net_core")


def _metric_key(channel: str) -> str:
    return f"{METRIC}/{channel}/h{HORIZON}"


def _values(
    records: list[RunRecord], *, model: str, n_train: str, test_set: str, channel: str
) -> list[float]:
    key = _metric_key(channel)
    out = [
        record.metrics[key]
        for record in records
        if record.tags.get("model") == model
        and record.tags.get("n_train") == n_train
        and record.tags.get("test_set") == test_set
        and key in record.metrics
    ]
    return sorted(out)


def _spread(values: list[float]) -> SeedSpread:
    return SeedSpread(source=VarianceSource.JOINT, values=tuple(values))


def _verdict(
    *, candidate: str, candidate_values: list[float], reference: str, reference_values: list[float]
) -> dict[str, object]:
    """One comparison, together with its threshold -- never one number without the other.

    Two-sample threshold: uses the spread of BOTH models (not only of the reference),
    see the module docstring. `n` is the number of seeds on the reference side -- in this
    evaluation the same throughout for candidate and reference (5).
    """
    candidate_spread = _spread(candidate_values)
    reference_spread = _spread(reference_values)
    n = len(reference_values)
    combined_sigma = math.sqrt(
        candidate_spread.sigma_from_span**2 + reference_spread.sigma_from_span**2
    )
    threshold = Z_95 * combined_sigma / math.sqrt(n) / abs(reference_spread.median)
    improvement = (reference_spread.median - candidate_spread.median) / abs(reference_spread.median)
    resolved = abs(improvement) > threshold
    if not resolved:
        statement = (
            f"{candidate} vs. {reference}: {improvement:+.1%} -- unterhalb der "
            f"Auflösungsschwelle ({threshold:.1%}), nicht unterscheidbar"
        )
    else:
        direction = "besser" if improvement > 0.0 else "schlechter"
        statement = (
            f"{candidate} vs. {reference}: {improvement:+.1%} -- nachweisbar {direction} "
            f"(Schwelle {threshold:.1%})"
        )
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_median": candidate_spread.median,
        "reference_median": reference_spread.median,
        "candidate_values": candidate_values,
        "reference_values": reference_values,
        "candidate_sigma_relative": candidate_spread.sigma_from_span / abs(candidate_spread.median)
        if candidate_spread.median != 0.0
        else float("nan"),
        "reference_sigma_relative": reference_spread.sigma_from_span / abs(reference_spread.median)
        if reference_spread.median != 0.0
        else float("nan"),
        "threshold_relative": threshold,
        "improvement_relative": improvement,
        "over_resolution": resolved,
        "statement": statement,
    }


def _part_a_parameter_count(records: list[RunRecord]) -> dict[str, object]:
    info: dict[str, dict[str, object]] = {}
    for model in ("gru", "lstm"):
        for record in records:
            if record.tags.get("model") == model and record.tags.get("n_train") == N_TRAIN_HEADLINE:
                info[model] = {field: record.params.get(field) for field in PARAMETER_FIELDS}
                break
    gru_params = info["gru"]["model/n_parameters"]
    lstm_params = info["lstm"]["model/n_parameters"]
    same_hidden_size = info["gru"]["model/net_hidden_size"] == info["lstm"]["model/net_hidden_size"]
    parameter_ratio = lstm_params / gru_params
    # Inference times from the M2 inventory, table C (same data family).
    inference_us_gru = 323.09
    inference_us_lstm = 182.83
    size_explains_speed = same_hidden_size and lstm_params < gru_params
    return {
        "gru": info["gru"],
        "lstm": info["lstm"],
        "same_hidden_size": same_hidden_size,
        "lstm_over_gru_parameter_ratio": parameter_ratio,
        "inference_us_per_window_n_train_64": {"gru": inference_us_gru, "lstm": inference_us_lstm},
        "lstm_over_gru_speed_ratio": inference_us_gru / inference_us_lstm,
        "model_size_explains_speed_difference": size_explains_speed,
        "note": (
            "LSTM hat bei GLEICHER hidden_size 33% MEHR Parameter als GRU, ist aber "
            "schneller -- Modellgroesse erklaert den Unterschied NICHT, im Gegenteil, "
            "sie legt das Gegenteil nahe (mehr Gatter/Parameter -> eher langsamer "
            "erwartet). Unerklaert, so zu kennzeichnen."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    projected = read_jsonl(PROJECTED_JSONL)
    extended = read_jsonl(EXTENDED_JSONL)
    headline_records = projected + extended
    epoch600_records = read_jsonl(EPOCH600_JSONL)

    print("=" * 88)
    print("Befund 63, Teil A: erklaert Modellgroesse GRU langsamer/LSTM schneller?")
    print("=" * 88)
    part_a = _part_a_parameter_count(headline_records)
    print(json.dumps(part_a, indent=2, ensure_ascii=False))

    print("\n" + "=" * 88)
    print("Befund 63, Teil B: Auflösungsvermögen (LSTM-Seed-Streuung, n=5, 95% Konfidenz)")
    print("=" * 88)

    verdicts_by_channel_testset: dict[str, dict[str, object]] = {}
    for channel in CHANNELS:
        for test_set in TEST_SETS:
            lstm_values = _values(
                headline_records,
                model="lstm",
                n_train=N_TRAIN_HEADLINE,
                test_set=test_set,
                channel=channel,
            )
            entry_key = f"{channel}/{test_set}"
            candidates: dict[str, object] = {}
            for candidate in ("gru", "transformer", "xlstm", "mamba"):
                candidate_values = _values(
                    headline_records,
                    model=candidate,
                    n_train=N_TRAIN_HEADLINE,
                    test_set=test_set,
                    channel=channel,
                )
                if not candidate_values:
                    continue
                verdict = _verdict(
                    candidate=candidate,
                    candidate_values=candidate_values,
                    reference="lstm",
                    reference_values=lstm_values,
                )
                candidates[candidate] = verdict
                print(f"  [{channel} / {test_set}] {verdict['statement']}")
            verdicts_by_channel_testset[entry_key] = candidates

    print("\n" + "=" * 88)
    print("Befund 63, Teil B (Fortsetzung): Transformer vs. LSTM bei 200 vs. 600 Epochen")
    print("=" * 88)
    epoch_budget_verdicts: dict[str, object] = {}
    for channel in ("bed_temperature_celsius", "energy_target"):
        lstm_600 = [
            record.metrics[_metric_key(channel)]
            for record in epoch600_records
            if record.tags.get("model") == "lstm"
            and record.tags.get("test_set") == "constant_holdout"
            and _metric_key(channel) in record.metrics
        ]
        transformer_600 = [
            record.metrics[_metric_key(channel)]
            for record in epoch600_records
            if record.tags.get("model") == "transformer"
            and record.tags.get("test_set") == "constant_holdout"
            and _metric_key(channel) in record.metrics
        ]
        lstm_200 = _values(
            headline_records,
            model="lstm",
            n_train=N_TRAIN_HEADLINE,
            test_set="constant_holdout",
            channel=channel,
        )
        transformer_200 = _values(
            headline_records,
            model="transformer",
            n_train=N_TRAIN_HEADLINE,
            test_set="constant_holdout",
            channel=channel,
        )
        verdict_200 = _verdict(
            candidate="transformer_200ep",
            candidate_values=transformer_200,
            reference="lstm_200ep",
            reference_values=lstm_200,
        )
        verdict_600 = _verdict(
            candidate="transformer_600ep",
            candidate_values=transformer_600,
            reference="lstm_600ep",
            reference_values=lstm_600,
        )
        epoch_budget_verdicts[channel] = {"epochs_200": verdict_200, "epochs_600": verdict_600}
        print(f"  [{channel}] 200 Epochen: {verdict_200['statement']}")
        print(f"  [{channel}] 600 Epochen: {verdict_600['statement']}")

    output = {
        "metric": METRIC,
        "horizon": HORIZON,
        "n_train_headline": N_TRAIN_HEADLINE,
        "confidence": CONFIDENCE,
        "resolution_source": (
            "zwei-stichproben (Streuung von Kandidat UND lstm-Referenz kombiniert, "
            "z95*sqrt(sigma_c^2+sigma_r^2)/sqrt(n); Korrektur einer Erstfassung mit "
            "reiner LSTM-Streuung, siehe Befund 63 Nachtrag)"
        ),
        "part_a_parameter_count": part_a,
        "part_b_verdicts_n_train_64": verdicts_by_channel_testset,
        "part_b_verdicts_epoch_budget": epoch_budget_verdicts,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
