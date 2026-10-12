r"""Re-evaluation of the rank statements of Befunde 58, 63 and 67 with the paired resolution.

Pure evaluation of committed records, NO training, no new data. Written before the run (this file
is committed before it is executed; clean tree, ``git/dirty`` is logged).

Why: review items C1/C2 (docs/12). The verdicts of Befunde 63 and 67 use an unpaired two-sample
threshold from range-based spreads (z in 63 g, Welch-t with nu_eff 3.76 in 67/SX1), relative to the
median of the reference. All methods of Befunde 51/58/67 were run with the SAME ``master_seed`` set
(1-5; identical ``split/digest`` per seed, checked below), so the design is paired. The paired test
(``twinlink_ml.evaluation.paired_resolution``, commit ``cd7b562``) is applied to the same cells.

Rule (fixed here, before any computation)
-----------------------------------------
* PRIMARY verdict per comparison: paired t test on the differences ``reference - candidate`` of the
  physical metric per seed (``rmse_run_median_physical``, h = 36, as in Befunde 63/67), ``n - 1``
  degrees of freedom, two-sided 95 %. Verdict "candidate better" / "candidate worse" if
  ``|mean difference| > t * s_d / sqrt(n)``, else "unresolved"; no variance estimate (n < 2 or zero
  spread) = "unresolved (degenerate)".
* SENSITIVITY (reported, never used for the verdict): the same test on ``log(candidate)`` versus
  ``log(reference)`` (multiplicative effect; the absolute differences of heavy-tailed errors, e.g. a
  diverging xLSTM seed, are dominated by one seed). Cells where the two disagree are listed.
* DESCRIPTIVE next to every verdict: medians over the seeds, relative difference of the medians
  (the quantity of the old verdicts), and in how many of the seeds the candidate is worse. With five
  seeds a sign test can never reach p < 0.05 two-sided (5 of 5: p = 0.0625); the count is context,
  not a test.
* OLD verdict = the final one of the lab book: Welch-t verdict of Befund 67 / SX1
  (``sx1_resolution_t.json``) for Befund 63 cells, ``sx3_capacity_matched.json`` for Befund 67 SX3;
  the z verdict of Befund 63 g is carried along. Befund 58 states no threshold; its statements
  are mapped to explicit cells and tested here for the first time.
* Cells: ``n_train = 64`` for 63/67 as there; all four budget points 8/16/32/64 for 58. 3 channels x 2
  test sets (excited, constant-run holdout), h = 36, seeds 1-5. NO multiplicity correction
  (docs/12, C-L1 open): the number of tests is reported.
* Known confound, not resolvable here (docs/12 F1.4): GRU/LSTM/baselines were trained on Linux
  (torch 2.13+cu130, numpy 2.4.6), Transformer/xLSTM/Mamba and all SX3 variants on Windows. A paired
  difference between the two groups contains this platform difference.

    tools/run_here.sh scripts/analyze_paired_reevaluation.py
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation.paired_resolution import PairedComparison, paired_comparison
from twinlink_ml.utils.tracking import RunRecord, read_jsonl

METRIC = "rmse_run_median_physical"
HORIZON = 36
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TEST_SETS = ("excited", "constant_holdout")
SEEDS = (1, 2, 3, 4, 5)
BUDGETS = (8, 16, 32, 64)
SX3_VARIANTS = ("mlstm_cm", "slstm_cm", "slstm_l1_cm", "mixed_cm", "mamba_cm")
OUTPUT_NAME = "paired_reevaluation_58_63_67"

Sources = dict[str, list[RunRecord]]


def load_sources() -> Sources:
    main = read_jsonl(RUNS_DIR / "arch_learning_curve_projected.jsonl") + read_jsonl(
        RUNS_DIR / "arch_learning_curve_projected_extended.jsonl"
    )
    return {
        "main": main,
        "ep600": read_jsonl(RUNS_DIR / "arch_epoch_budget_probe600.jsonl"),
        "sx3": read_jsonl(RUNS_DIR / "sx3_capacity_matched_a.jsonl")
        + read_jsonl(RUNS_DIR / "sx3_capacity_matched_b.jsonl"),
    }


def series(
    sources: Sources, source: str, label: str, n_train: int, test_set: str, channel: str
) -> dict[int, float]:
    """Metric per master seed; a seed with two values is an error (never averaged)."""
    key = f"{METRIC}/{channel}/h{HORIZON}"
    out: dict[int, float] = {}
    for record in sources[source]:
        params = record.params
        if (
            params.get("model/label") != label
            or int(params["n_train"]) != n_train
            or params.get("test_set") != test_set
            or key not in record.metrics
        ):
            continue
        seed = int(params["master_seed"])
        if seed in out:
            raise SystemExit(
                f"duplicate record: {source}/{label} n={n_train} {test_set} seed {seed}"
            )
        out[seed] = float(record.metrics[key])
    if tuple(sorted(out)) != SEEDS:
        raise SystemExit(
            f"{source}/{label} n={n_train} {test_set} {channel}: seeds {sorted(out)}, "
            f"expected {SEEDS}"
        )
    return out


def check_pairing(sources: Sources) -> dict[str, Any]:
    """Every method of one master seed must come from the same data pool (split/digest)."""
    digests: dict[tuple[int, int], set[str]] = defaultdict(set)
    for records in sources.values():
        for record in records:
            digest = record.params.get("split/digest")
            if digest is None:
                raise SystemExit("record without split/digest")
            digests[(int(record.params["master_seed"]), int(record.params["n_train"]))].add(
                str(digest)
            )
    bad = {key: sorted(value) for key, value in digests.items() if len(value) != 1}
    if bad:
        raise SystemExit(f"split/digest differs between methods of one seed and n_train: {bad}")
    return {"points_checked": len(digests), "all_methods_share_the_pool": True}


def _verdict(result: PairedComparison) -> str:
    if result.degenerate:
        return "unresolved (degenerate)"
    if not result.is_resolved:
        return "unresolved"
    return "candidate better" if result.mean_difference > 0.0 else "candidate worse"


def compare(
    candidate: dict[int, float], reference: dict[int, float], cand: str, ref: str
) -> dict[str, Any]:
    primary = paired_comparison(candidate, reference, candidate_label=cand, reference_label=ref)
    logged = paired_comparison(
        {s: math.log(v) for s, v in candidate.items()},
        {s: math.log(v) for s, v in reference.items()},
        candidate_label=cand,
        reference_label=ref,
    )
    cand_values = sorted(candidate.values())
    ref_values = sorted(reference.values())
    median_c = cand_values[len(cand_values) // 2]
    median_r = ref_values[len(ref_values) // 2]
    return {
        "candidate": cand,
        "reference": ref,
        "verdict": _verdict(primary),
        "verdict_log_ratio": _verdict(logged),
        "mean_difference": primary.mean_difference,
        "threshold": primary.threshold,
        "p_value": primary.p_value,
        "t_critical": primary.critical_value,
        "degrees_of_freedom": primary.degrees_of_freedom,
        "mean_candidate": primary.mean_candidate,
        "mean_reference": primary.mean_reference,
        "median_candidate": median_c,
        "median_reference": median_r,
        "relative_difference_of_medians": (median_r - median_c) / abs(median_r),
        "seeds_candidate_worse": sum(candidate[s] > reference[s] for s in SEEDS),
        "per_seed_candidate": [candidate[s] for s in SEEDS],
        "per_seed_reference": [reference[s] for s in SEEDS],
        "log_ratio_p_value": logged.p_value,
    }


def _old_verdict(cell: dict[str, Any]) -> dict[str, str]:
    def word(resolved: bool) -> str:
        if not resolved:
            return "unresolved"
        return "candidate better" if cell["improvement_relative"] > 0.0 else "candidate worse"

    return {"t_welch": word(cell["t_range_dof"]["resolved"]), "z": word(cell["z"]["resolved"])}


def main() -> int:
    require_clean_tree()
    import json

    sources = load_sources()
    pairing = check_pairing(sources)
    sx1 = json.loads((RUNS_DIR / "sx1_resolution_t.json").read_text(encoding="utf-8"))["cells"]
    sx3 = json.loads((RUNS_DIR / "sx3_capacity_matched.json").read_text(encoding="utf-8"))

    n_tests = 0

    def run(
        source_c: str,
        cand: str,
        source_r: str,
        ref: str,
        n_train: int,
        test_set: str,
        channel: str,
    ) -> dict[str, Any]:
        nonlocal n_tests
        n_tests += 1
        c = series(sources, source_c, cand, n_train, test_set, channel)
        r = series(sources, source_r, ref, n_train, test_set, channel)
        out = compare(c, r, cand, ref)
        out.update({"channel": channel, "test_set": test_set, "n_train": n_train})
        return out

    # ---------------------------------------------------------------- Befund 63 / SX1
    part_63: list[dict[str, Any]] = []
    for channel, test_set, cand in itertools.product(
        CHANNELS, TEST_SETS, ("gru", "transformer", "xlstm", "mamba")
    ):
        cell = run("main", cand, "main", "lstm", 64, test_set, channel)
        cell["old"] = _old_verdict(sx1[f"{channel}/{test_set}/{cand}"])
        cell["source"] = "Befund 63 g / 67 SX1, n_train 64, 200 epochs"
        part_63.append(cell)
    for channel, test_set in (
        ("bed_temperature_celsius", "constant_holdout"),
        ("energy_target", "constant_holdout"),
    ):
        # old verdicts exist for these four cells (200 and 600 epochs); the 200-epoch cell is already
        # in part_63 (transformer, main) -- repeated here under the key of the old artifact
        cell200 = run("main", "transformer", "main", "lstm", 64, test_set, channel)
        cell200["old"] = _old_verdict(sx1[f"{channel}/{test_set}/transformer_200ep"])
        cell200["source"] = "Befund 63 d / 67 SX1, epoch budget 200"
        cell600 = run("ep600", "transformer", "ep600", "lstm", 64, test_set, channel)
        cell600["old"] = _old_verdict(sx1[f"{channel}/{test_set}/transformer_600ep"])
        cell600["source"] = "Befund 63 d / 67 SX1, epoch budget 600"
        part_63 += [cell200, cell600]
    for channel, test_set in itertools.product(CHANNELS, TEST_SETS):
        if (channel, test_set) in (
            ("bed_temperature_celsius", "constant_holdout"),
            ("energy_target", "constant_holdout"),
        ):
            continue
        cell600 = run("ep600", "transformer", "ep600", "lstm", 64, test_set, channel)
        cell600["old"] = None
        cell600["source"] = (
            "epoch budget 600, no old verdict (not one of the four cells of Befund 63)"
        )
        part_63.append(cell600)

    # ---------------------------------------------------------------- Befund 67 / SX3
    part_67: list[dict[str, Any]] = []
    for variant in SX3_VARIANTS:
        for channel, test_set in itertools.product(CHANNELS, TEST_SETS):
            cell = run("sx3", variant, "main", "lstm", 64, test_set, channel)
            cell["old"] = _old_verdict(
                sx3["versus_lstm"][variant]["cells"][f"{channel}/{test_set}"]
            )
            cell["source"] = "Befund 67 SX3, against the LSTM of Befund 51/58"
            part_67.append(cell)
    for channel, test_set in itertools.product(CHANNELS, TEST_SETS):
        for ref_label, key in (
            ("lstm", "lstm_small_vs_lstm"),
            ("xlstm", "lstm_small_vs_xlstm"),
            ("mamba", "lstm_small_vs_mamba"),
        ):
            cell = run("sx3", "lstm_small", "main", ref_label, 64, test_set, channel)
            old_cells = sx3["control"].get(key, {}).get("cells", {})
            cell["old"] = _old_verdict(old_cells[f"{channel}/{test_set}"]) if old_cells else None
            cell["source"] = f"Befund 67 SX3 control: small LSTM against {ref_label}"
            part_67.append(cell)
    for variant in ("slstm_cm", "slstm_l1_cm", "mixed_cm"):
        for channel, test_set in itertools.product(CHANNELS, TEST_SETS):
            cell = run("sx3", variant, "sx3", "mlstm_cm", 64, test_set, channel)
            old_key = f"{variant}_vs_mlstm_cm"
            old_cells = sx3["reading_b_slstm_vs_mlstm"].get(old_key, {}).get("cells", {})
            cell["old"] = _old_verdict(old_cells[f"{channel}/{test_set}"]) if old_cells else None
            cell["source"] = f"Befund 67 reading B: {variant} against mlstm_cm"
            part_67.append(cell)

    def count(cells: list[dict[str, Any]], variant: str, ref: str, verdict: str) -> int:
        return sum(
            1
            for c in cells
            if c["candidate"] == variant and c["reference"] == ref and c["verdict"] == verdict
        )

    readings: dict[str, Any] = {}
    for variant in SX3_VARIANTS:
        worse = count(part_67, variant, "lstm", "candidate worse")
        old_worse = sum(
            1
            for c in part_67
            if c["candidate"] == variant
            and c["reference"] == "lstm"
            and c["old"]["t_welch"] == "candidate worse"
        )
        readings[variant] = {
            "cells_resolvably_worse_new": worse,
            "cells_resolvably_worse_old": old_worse,
        }
    readings["reading_b_slstm_2L_vs_mlstm"] = {
        "better_new": count(part_67, "slstm_cm", "mlstm_cm", "candidate better"),
        "worse_new": count(part_67, "slstm_cm", "mlstm_cm", "candidate worse"),
        "better_old": sum(
            1
            for c in part_67
            if c["candidate"] == "slstm_cm"
            and c["reference"] == "mlstm_cm"
            and c["old"]
            and c["old"]["t_welch"] == "candidate better"
        ),
    }
    readings["control_small_lstm"] = {
        "worse_than_large_lstm_new": count(part_67, "lstm_small", "lstm", "candidate worse"),
        "better_than_small_xlstm_new": count(part_67, "lstm_small", "xlstm", "candidate better"),
        "better_than_small_mamba_new": count(part_67, "lstm_small", "mamba", "candidate better"),
        "better_than_large_lstm_new": count(part_67, "lstm_small", "lstm", "candidate better"),
    }
    control = readings["control_small_lstm"]
    control["size_effect_criterion_met_new"] = bool(
        control["worse_than_large_lstm_new"] >= 4
        and control["better_than_small_xlstm_new"] + control["better_than_small_mamba_new"] == 0
    )
    control["size_effect_criterion_met_old"] = sx3["control"]["size_effect_criterion_met"]

    # ---------------------------------------------------------------- Befund 58 statements
    part_58: list[dict[str, Any]] = []
    candidates_vs_lstm = ("gru", "transformer", "xlstm", "mamba", "static", "arx", "persistence")
    for n_train, channel, test_set, cand in itertools.product(
        BUDGETS, CHANNELS, TEST_SETS, candidates_vs_lstm
    ):
        part_58.append(run("main", cand, "main", "lstm", n_train, test_set, channel))
    for n_train, channel, test_set, cand in itertools.product(
        BUDGETS, CHANNELS, TEST_SETS, ("gru", "lstm", "transformer", "xlstm", "mamba")
    ):
        part_58.append(run("main", cand, "main", "static", n_train, test_set, channel))
    for n_train, channel, test_set, cand in itertools.product(
        BUDGETS, CHANNELS, TEST_SETS, ("gru", "lstm", "transformer", "xlstm", "mamba", "static")
    ):
        part_58.append(run("main", cand, "main", "persistence", n_train, test_set, channel))
    pairwise: list[dict[str, Any]] = []
    five = ("lstm", "gru", "transformer", "xlstm", "mamba")
    for n_train, channel, test_set in (
        (8, "product_moisture_kg_per_kg", "excited"),
        (64, "product_moisture_kg_per_kg", "excited"),
        (64, "energy_target", "excited"),
    ):
        for a, b in itertools.combinations(five, 2):
            pairwise.append(run("main", a, "main", b, n_train, test_set, channel))

    by_key = {
        (c["n_train"], c["channel"], c["test_set"], c["candidate"], c["reference"]): c
        for c in part_58
    }
    best_gru_lstm_beaten: list[dict[str, Any]] = []
    for n_train, channel, test_set in itertools.product(BUDGETS, CHANNELS, TEST_SETS):
        for cand in ("transformer", "xlstm", "mamba"):
            for ref in ("lstm", "gru"):
                if ref == "lstm":
                    match = by_key[(n_train, channel, test_set, cand, "lstm")]
                else:
                    match = run("main", cand, "main", "gru", n_train, test_set, channel)
                if match["verdict"] == "candidate better":
                    best_gru_lstm_beaten.append(match)

    changes = [
        {
            k: c[k]
            for k in (
                "channel",
                "test_set",
                "candidate",
                "reference",
                "source",
                "old",
                "verdict",
                "verdict_log_ratio",
                "relative_difference_of_medians",
                "mean_difference",
                "threshold",
                "p_value",
                "seeds_candidate_worse",
            )
        }
        for c in part_63 + part_67
        if c["old"] is not None and c["old"]["t_welch"] != c["verdict"]
    ]
    log_disagree = [
        {
            k: c[k]
            for k in (
                "channel",
                "test_set",
                "n_train",
                "candidate",
                "reference",
                "verdict",
                "verdict_log_ratio",
            )
        }
        for c in part_63 + part_67 + part_58
        if c["verdict"] != c["verdict_log_ratio"]
    ]

    payload = {
        "_provenance": provenance(),
        "rule": "see the module docstring of scripts/analyze_paired_reevaluation.py",
        "metric": METRIC,
        "horizon": HORIZON,
        "seeds": list(SEEDS),
        "n_tests_total": n_tests,
        "multiplicity_correction": "none (docs/12 C-L1 open)",
        "pairing_check": pairing,
        "paired_resolution_commit": "cd7b562",
        "part_63_67_sx1_cells": part_63,
        "part_67_sx3_cells": part_67,
        "part_67_readings": readings,
        "part_58_cells": part_58,
        "part_58_pairwise_cells": pairwise,
        "part_58_candidates_resolvably_better_than_lstm_or_gru": best_gru_lstm_beaten,
        "changed_verdicts_old_t_vs_new_paired": changes,
        "verdicts_where_log_ratio_sensitivity_disagrees": log_disagree,
    }
    write_artifact(OUTPUT_NAME, payload)

    print(f"tests: {n_tests}; changed verdicts (Befund 63/67 cells): {len(changes)}")
    for c in changes:
        print(
            f"  {c['channel']}/{c['test_set']} {c['candidate']} vs {c['reference']}: old {c['old']['t_welch']} -> new {c['verdict']} (log: {c['verdict_log_ratio']})"
        )
    print("readings:", readings)
    print("58: candidates resolvably better than LSTM or GRU:", len(best_gru_lstm_beaten))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
