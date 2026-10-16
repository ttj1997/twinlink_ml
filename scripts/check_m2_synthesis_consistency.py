r"""Consistency check of ``docs/08_synthese_m2.md`` (version 5): every number of the result
sections against its committed artifact.

No new run, no simulation: pure re-reading of committed artifacts. Three parts:

1. CLAIMS. Each claim names a literal fragment of the synthesis, the number(s) written in it
   and the value(s) computed from the artifact. The fragment must occur in the text; the number
   must agree with the computed value within the rounding of the written number.
2. COVERAGE. All numeric tokens of sections 1-8 (the changelog and the appendix table are
   excluded: they restate claims, and the appendix lists the artifact per claim) must lie inside
   a claim fragment, inside a declared exemption (design constants and plan facts, each with the
   reason and, where possible, a verification) or match a reference pattern (Befund numbers,
   section and figure numbers, labels such as A0, P1, HB1). A token outside all three fails the
   run: a number without a check cannot enter the text unnoticed.
3. NOT CHECKABLE. Claims whose source is not in the repository (raw plant data) are listed with
   the reason, not silently skipped.

Call::

    tools/run_here.sh scripts/check_m2_synthesis_consistency.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import REPO_ROOT, RUNS_DIR, provenance, require_clean_tree, write_artifact

SYNTHESIS = REPO_ROOT / "docs" / "08_synthese_m2.md"
ARTIFACT_NAME = "check_m2_synthesis_consistency_v5"
MICRO = "µ"


def load(name: str) -> Any:
    return json.loads((RUNS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def load_jsonl(name: str) -> list[dict[str, Any]]:
    with (RUNS_DIR / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def decimals(written: str) -> int:
    clean = written.replace(",", "")
    return len(clean.split(".")[1]) if "." in clean else 0


class Report:
    """Collects claims, their fragments and the verdicts."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.results: list[dict[str, Any]] = []
        self.fragments: list[str] = []

    def claim(self, fragment: str, pairs: list[tuple[str, float]], source: str) -> None:
        """``pairs``: (number as written in the text, value computed from the artifact)."""
        frag = normalise(fragment)
        present = frag in self.text
        self.fragments.append(frag)
        for written, actual in pairs:
            claimed = float(written.replace(",", ""))
            tolerance = 0.5 * 10 ** (-decimals(written)) + 1e-9
            ok = present and abs(claimed - actual) <= tolerance
            self.results.append(
                {
                    "fragment": frag,
                    "written": written,
                    "computed": actual,
                    "source": source,
                    "status": "confirmed" if ok else "DEVIATING",
                    "fragment_in_text": present,
                }
            )

    def truth(self, fragment: str, holds: bool, what: str, source: str) -> None:
        frag = normalise(fragment)
        present = frag in self.text
        self.fragments.append(frag)
        self.results.append(
            {
                "fragment": frag,
                "assertion": what,
                "source": source,
                "status": "confirmed" if (present and holds) else "DEVIATING",
                "fragment_in_text": present,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write the artifact (clean tree required)"
    )
    arguments = parser.parse_args()
    if arguments.write:
        require_clean_tree()
    raw = SYNTHESIS.read_text(encoding="utf-8")
    text = normalise(raw)
    report = Report(text)

    rerun = load("rerun_evaluation_rerun_full")
    holm = load("holm_rank_cells_rerun_full")
    timing = load("joint_timing_idle")
    hybrid = load("a1_inference_time_v2_idle")
    paired = load("paired_reevaluation_58_63_67")
    seeds = load("paired_reevaluation_83_addendum")
    a1 = load("a1_main_analysis")
    b1 = load("b1_main_analysis")
    m1 = load("m1_k1_comparison")["rerun_k1"]
    dt20 = load("m4r_test_dt20_umin126")
    dt10 = load("m4r_test_dt10_umin126")
    margin = load("bres_margin")
    heat = load("heat_recovery_comparison_r4_consistent")
    gate = load("gate_storage_shift")
    posthoc = load("gate_storage_shift_posthoc")
    n_records = load_jsonl("rerun_full_N")

    # ---- section 2 -------------------------------------------------------------------
    trees = load_jsonl("extra_trees_split_comparison")
    reductions = []
    for channel in ("energy_target", "product_moisture_kg_per_kg", "d50_um"):
        key = f"nrmse_run_median/{channel}/h1"
        run = statistics.median(
            r["metrics"][key] for r in trees if r["tags"]["split_kind"] == "run_level"
        )
        point = statistics.median(
            r["metrics"][key] for r in trees if r["tags"]["split_kind"] == "point_level"
        )
        reductions.append(100 * (1 - point / run))
    for fragment in (
        "lower by 29 to 68 % depending on the target",
        "an optimistic bias of 29 to 68 %",
    ):
        report.claim(
            fragment,
            [("29", min(reductions)), ("68", max(reductions))],
            "extra_trees_split_comparison.jsonl (median of nrmse_run_median at h1, point-level against run-level)",
        )
    report.claim(
        "only if it survives Holm (5 %)",
        [("5", 100 * seeds["alpha"])],
        "paired_reevaluation_83_addendum.json",
    )
    mult = rerun["multiplicity"]
    report.claim(
        "(336 cells in A0 and N, 168 in A1",
        [
            ("336", mult["rank_A0"]["tests"]),
            ("336", mult["rank_N"]["tests"]),
            ("168", mult["rank_A1"]["tests"]),
        ],
        "rerun_evaluation_rerun_full.json",
    )
    report.claim(
        "applied 632 uncorrected paired tests",
        [("632", paired["n_tests_total"])],
        "paired_reevaluation_58_63_67.json",
    )
    rows64 = [r for r in seeds["seeds_needed"] if r["n_train"] == 64]

    def need(channel: str, test_set: str, candidate: str) -> float:
        match = [
            r
            for r in rows64
            if r["channel"] == channel and r["test_set"] == test_set and r["candidate"] == candidate
        ]
        return match[0]["seeds_0.125"]

    easy = [
        r[k]
        for r in rows64
        if r["channel"] in ("energy_target", "bed_temperature_celsius")
        for k in ("seeds_0.125", "seeds_0.100")
    ]
    report.claim(
        "On energy and bed temperature, 3 to 11 seeds resolve a 10 to 12.5 % difference of GRU or Transformer against LSTM at 80 % power",
        [("3", min(easy)), ("11", max(easy)), ("80", 100 * seeds["power"])],
        "paired_reevaluation_83_addendum.json (n_train 64, delta 0.125 and 0.100)",
    )
    report.truth(
        "10 to 12.5 %",
        sorted(seeds["deltas"]) == [0.1, 0.125],
        "differences tested were 10 % and 12.5 %",
        "paired_reevaluation_83_addendum.json",
    )
    spread = [
        r["sd_relative_to_reference_mean"]
        for r in rows64
        if r["channel"] == "product_moisture_kg_per_kg"
    ]
    report.truth(
        "paired standard deviation is 25 to 60 % of the mean",
        min(spread) >= 0.23 and max(spread) <= 0.62,
        f"moisture sd/mean at n_train 64 ranges {min(spread):.3f}-{max(spread):.3f}",
        "paired_reevaluation_83_addendum.json",
    )
    report.claim(
        "so 37 and 45 seeds are needed for GRU (constant / excited) and 33 and 193 for the Transformer at 12.5 %",
        [
            ("37", need("product_moisture_kg_per_kg", "constant_holdout", "gru")),
            ("45", need("product_moisture_kg_per_kg", "excited", "gru")),
            ("33", need("product_moisture_kg_per_kg", "constant_holdout", "transformer")),
            ("193", need("product_moisture_kg_per_kg", "excited", "transformer")),
        ],
        "paired_reevaluation_83_addendum.json",
    )

    # ---- section 3 -------------------------------------------------------------------
    gru = holm["N"]["gru_vs_lstm"]
    report.claim(
        "(15 of 16 cells unresolved, one cell with GRU worse, uncorrected)",
        [
            ("15", gru["product_moisture_kg_per_kg"]["main_family"]["unresolved_uncorrected"]),
            ("16", gru["product_moisture_kg_per_kg"]["main_family"]["cells"]),
            ("1", gru["product_moisture_kg_per_kg"]["main_family"]["worse_uncorrected"]),
        ],
        "holm_rank_cells_rerun_full.json (arm N)",
    )
    report.claim(
        "in 5 of 16 and 9 of 16 cells without correction",
        [
            ("5", gru["energy_target"]["main_family"]["better_uncorrected"]),
            ("9", gru["bed_temperature_celsius"]["main_family"]["better_uncorrected"]),
            ("16", gru["energy_target"]["main_family"]["cells"]),
        ],
        "holm_rank_cells_rerun_full.json (arm N)",
    )
    report.truth(
        "and never worse",
        gru["energy_target"]["main_family"]["worse_uncorrected"] == 0
        and gru["bed_temperature_celsius"]["main_family"]["worse_uncorrected"] == 0,
        "GRU worse than LSTM in 0 cells on energy and bed temperature (arm N)",
        "holm_rank_cells_rerun_full.json",
    )
    no_holm = all(
        holm[arm]["gru_vs_lstm"][ch][fam][k] == 0
        for arm in ("A0", "A1", "N")
        for ch in holm[arm]["gru_vs_lstm"]
        for fam in ("main_family", "sensitivity_family")
        for k in ("better_holm", "worse_holm")
    )
    report.truth(
        "after Holm correction, no GRU-versus-LSTM cell is resolved in any channel",
        no_holm,
        "0 resolved GRU-against-LSTM cells after Holm, all arms, channels, both families",
        "holm_rank_cells_rerun_full.json",
    )
    lstm_row = next(r for r in timing["results"] if r["method"] == "lstm")
    gru_row = next(r for r in timing["results"] if r["method"] == "gru")
    report.claim(
        "about 25 % fewer parameters (42,819 against 57,027, Befund 28)",
        [
            ("25", 100 * (1 - gru_row["n_parameters"] / lstm_row["n_parameters"])),
            ("42,819", gru_row["n_parameters"]),
            ("57,027", lstm_row["n_parameters"]),
        ],
        "joint_timing_idle.json (n_parameters)",
    )
    by_cand = holm["N"]["by_candidate"]
    report.claim(
        "they are worse in 39 (Transformer), 38 (xLSTM) and 44 (Mamba) of 48 cells without correction",
        [
            ("39", by_cand["transformer"]["all"]["worse_uncorrected"]),
            ("38", by_cand["xlstm"]["all"]["worse_uncorrected"]),
            ("44", by_cand["mamba"]["all"]["worse_uncorrected"]),
            ("48", by_cand["mamba"]["all"]["cells"]),
        ],
        "holm_rank_cells_rerun_full.json (arm N)",
    )
    report.truth(
        "never better",
        all(
            by_cand[c]["all"]["better_uncorrected"] == 0 for c in ("transformer", "xlstm", "mamba")
        ),
        "Transformer, xLSTM, Mamba better than LSTM in 0 cells uncorrected (arm N)",
        "holm_rank_cells_rerun_full.json",
    )
    report.claim(
        "resolved in 14 of 48 cells for Mamba, in 2 of 48 for the Transformer and in 2 of 48 for xLSTM",
        [
            ("14", by_cand["mamba"]["all"]["worse_holm"]),
            ("2", by_cand["transformer"]["all"]["worse_holm"]),
            ("2", by_cand["xlstm"]["all"]["worse_holm"]),
        ],
        "holm_rank_cells_rerun_full.json (arm N)",
    )
    xl_moist = [
        c
        for c in rerun["rank_cells"]["N"]
        if c["candidate"] == "xlstm" and c["channel"] == "product_moisture_kg_per_kg"
    ]
    report.claim(
        "xLSTM stays unresolved in 7 of 16 cells (uncorrected)",
        [
            ("7", sum(c["verdict"] == "unresolved" for c in xl_moist)),
            ("16", len(xl_moist)),
        ],
        "rerun_evaluation_rerun_full.json (rank cells, arm N)",
    )
    report.claim(
        "the static regression is resolved worse in 21 of 48 cells after Holm, persistence in 19, ARX in 12",
        [
            ("21", by_cand["static"]["all"]["worse_holm"]),
            ("19", by_cand["persistence"]["all"]["worse_holm"]),
            ("12", by_cand["arx"]["all"]["worse_holm"]),
        ],
        "holm_rank_cells_rerun_full.json (arm N)",
    )
    report.claim(
        "after Holm, 70 of 336 rank cells remain in arm N (82 of 336 in A0)",
        [("70", mult["rank_N"]["resolved_holm"]), ("82", mult["rank_A0"]["resolved_holm"])],
        "rerun_evaluation_rerun_full.json (multiplicity)",
    )
    changed = rerun["rank_verdicts_changed_against_a0"]
    report.claim(
        "85 rank verdicts differ between A0 and an arm of the re-run, 45 in A1 and 40 in N",
        [
            ("85", len(changed)),
            ("45", sum(c["arm"] == "A1" for c in changed)),
            ("40", sum(c["arm"] == "N" for c in changed)),
        ],
        "rerun_evaluation_rerun_full.json (rank_verdicts_changed_against_a0)",
    )
    report.claim(
        "(3 of 192 tests in A1 against A0; 24 of 384 in N against A0",
        [
            ("3", mult["a1_vs_a0"]["resolved_holm"]),
            ("192", mult["a1_vs_a0"]["tests"]),
            ("24", mult["n_vs_a0"]["resolved_holm"]),
            ("384", mult["n_vs_a0"]["tests"]),
        ],
        "rerun_evaluation_rerun_full.json (multiplicity)",
    )
    ratios = []
    all_better = True
    for cell in paired["part_58_cells"]:
        if (
            cell["reference"] == "static"
            and cell["candidate"] in ("gru", "lstm", "transformer")
            and cell["channel"] == "energy_target"
            and cell["n_train"] == 64
            and cell.get("horizon", 36) in (36, "36")
        ):
            ratios.append(cell["median_reference"] / cell["median_candidate"])
            all_better &= (
                cell["verdict"] == "candidate better" and cell["seeds_candidate_worse"] == 0
            )
    report.claim(
        "by factors of 5.2 to 7.8 (GRU, LSTM, Transformer,",
        [("5.2", min(ratios)), ("7.8", max(ratios))],
        "paired_reevaluation_58_63_67.json (part_58_cells, ratio of medians against static)",
    )
    report.truth(
        "all five seeds better",
        all_better and len(ratios) == 6,
        "every one of the six cells resolved better, no seed worse",
        "paired_reevaluation_58_63_67.json",
    )

    # ---- section 4 -------------------------------------------------------------------
    names = {
        "persistence": "persistence",
        "static": "static regression",
        "arx": "ARX",
        "gru": "GRU",
        "lstm": "LSTM",
        "mamba": "Mamba",
        "xlstm": "xLSTM",
        "transformer": "Transformer",
    }
    for method, label in names.items():
        row = next(r for r in timing["results"] if r["method"] == method)
        latency = f"{round(row['latency_batch1_us_median']):,}"
        batch = f"{row['throughput_us_per_window_median']:,.1f}"
        params = f"{row['n_parameters']:,}"
        expected = f"| {label} | {params} | {latency} | {batch} |"
        report.truth(
            expected,
            True,
            f"{label}: parameters, latency of one window and batch cost per window equal the artifact",
            "joint_timing_idle.json",
        )
    windows = {r["n_test_windows"] for r in timing["results"]}
    report.claim(
        "= 64, 4352 excited-test windows", [("4352", next(iter(windows)))], "joint_timing_idle.json"
    )
    sample_time = n_records[0]["params"]["data/sample_time_s"]
    report.claim(
        "below the sample time of 25 s",
        [("25", sample_time)],
        "rerun_full_N.jsonl (data/sample_time_s)",
    )
    fit = {m: next(r for r in timing["results"] if r["method"] == m) for m in names}
    report.claim(
        "(2-epoch fit, setup included, indicative only): GRU 4.3 s, LSTM 4.7 s, Mamba 14.8 s, xLSTM 27.8 s, Transformer 67.6 s",
        [
            ("2", fit["gru"]["fit_epochs"]),
            ("4.3", fit["gru"]["fit_s_per_epoch_indicative"]),
            ("4.7", fit["lstm"]["fit_s_per_epoch_indicative"]),
            ("14.8", fit["mamba"]["fit_s_per_epoch_indicative"]),
            ("27.8", fit["xlstm"]["fit_s_per_epoch_indicative"]),
            ("67.6", fit["transformer"]["fit_s_per_epoch_indicative"]),
        ],
        "joint_timing_idle.json (fit_s_per_epoch_indicative)",
    )
    ht = hybrid["hybrid_total"]
    report.claim(
        f"one horizon of 36 steps costs 253,136 {MICRO}s in the simulator, of 256,123 {MICRO}s in total, i.e. 98.8 % simulator share",
        [
            ("36", hybrid["simulator_channel_costs"]["horizon_batch1"]["steps"]),
            ("253,136", ht["simulation_part_us"]),
            ("256,123", ht["realtime_batch1_us_per_prediction"]),
            ("98.8", 100 * ht["simulation_share"]),
        ],
        "a1_inference_time_v2_idle.json (hybrid_total)",
    )
    report.claim(
        f"the network alone (2,987 {MICRO}s)",
        [("2,987", ht["network_part_us"])],
        "a1_inference_time_v2_idle.json",
    )
    order_batch = sorted(
        ("lstm", "gru", "mamba", "xlstm", "transformer"),
        key=lambda m: fit[m]["throughput_us_per_window_median"],
    )
    order_one = sorted(
        ("gru", "lstm", "mamba", "xlstm", "transformer"),
        key=lambda m: fit[m]["latency_batch1_us_median"],
    )
    report.truth(
        "The batch order is LSTM < GRU < Mamba < xLSTM < Transformer, the single-window order GRU < LSTM < Mamba < xLSTM < Transformer",
        order_batch == ["lstm", "gru", "mamba", "xlstm", "transformer"]
        and order_one == ["gru", "lstm", "mamba", "xlstm", "transformer"],
        "orders recomputed from the medians",
        "joint_timing_idle.json",
    )

    # ---- section 5 -------------------------------------------------------------------
    report.claim(
        "In arm N 260 of 336 rank cells are resolved without correction and 70 after Holm",
        [
            ("260", mult["rank_N"]["resolved_uncorrected"]),
            ("336", mult["rank_N"]["tests"]),
            ("70", mult["rank_N"]["resolved_holm"]),
        ],
        "rerun_evaluation_rerun_full.json (multiplicity)",
    )

    # ---- section 6 -------------------------------------------------------------------
    ratio = a1["ha1"]["part_b_ratio_scratch_over_finetune"]
    report.claim(
        "falls from 12.69 (identical plant) and 10.27 (parameter deviation) at $N$ = 2 to 1.32 and 1.27 at $N$ = 64",
        [
            ("12.69", ratio["P0"]["2"]),
            ("10.27", ratio["P1"]["2"]),
            ("1.32", ratio["P0"]["64"]),
            ("1.27", ratio["P1"]["64"]),
        ],
        "a1_main_analysis.json (ha1)",
    )
    crossover = a1["ha1"]["crossover_finetune_vs_scratch"]
    reached = [p for p, v in crossover.items() if v["n_star"] is None]
    report.truth(
        "the crossover is not reached on four of the five plants",
        len(reached) == 4 and len(crossover) == 5,
        f"n_star is None on {sorted(reached)} of {len(crossover)} plants",
        "a1_main_analysis.json",
    )
    verdicts = {k: v["verdict"] for k, v in b1["verdicts"].items()}
    t_sets = {k: v["verdict"] for k, v in b1["verdicts"]["HB2"]["per_test_set"].items()}
    report.truth(
        "(HB1: not established)",
        verdicts["HB1"] == "not established",
        f"HB1 verdict: {verdicts['HB1']}",
        "b1_main_analysis.json",
    )
    report.truth(
        "(HB2: not established on the deployment-like test set, refuted on the neutral mix)",
        t_sets == {"T3": "not established", "T1": "refuted"},
        f"HB2 per test set: {t_sets}",
        "b1_main_analysis.json",
    )
    report.truth(
        "not established (HB3)",
        verdicts["HB3"] == "not established",
        f"HB3 verdict: {verdicts['HB3']}",
        "b1_main_analysis.json",
    )
    constant = load_jsonl("arch_learning_curve_constant_trained")
    excited = load_jsonl("arch_learning_curve_projected")

    def med(records: list[dict[str, Any]], model: str, n: str, test_set: str) -> float:
        key = "nrmse_run_median/product_moisture_kg_per_kg/h36"
        return statistics.median(
            r["metrics"][key]
            for r in records
            if r["tags"].get("model") == model
            and r["tags"].get("n_train") == n
            and r["tags"].get("test_set") == test_set
            and key in r["metrics"]
        )

    factors = [
        med(constant, m, n, "constant_holdout") / med(excited, m, n, "constant_holdout")
        for m in ("gru", "lstm")
        for n in ("8", "16", "32", "64")
    ]
    report.truth(
        "by a factor of 2 to 3 over the whole budget range (NRMSE, no catching up)",
        min(factors) >= 1.95 and max(factors) <= 3.2,
        f"constant-trained over excited-trained NRMSE ratio, GRU and LSTM, four budgets: {min(factors):.2f}-{max(factors):.2f}",
        "arch_learning_curve_constant_trained.jsonl, arch_learning_curve_projected.jsonl",
    )

    # ---- section 7 -------------------------------------------------------------------
    yaml_text = (REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml").read_text(
        encoding="utf-8"
    )
    report.truth(
        "the critical moisture of 0.20 kg/kg",
        bool(re.search(r"critical_moisture[^\n]*:\s*0\.200?\b", yaml_text)),
        "conf/process/greybox_gab_placeholder.yaml sets the critical moisture to 0.2",
        "conf/process/greybox_gab_placeholder.yaml",
    )
    report.truth(
        "the lower bound of the gas velocity (0.80 and 1.26 m/s)",
        abs(dt20["gas_velocity_lower_bound_m_per_s"] - 1.26) < 1e-12
        and "0.80 m/s" in margin["placeholders"],
        "1.26 m/s in the coupling-test artifact, 0.80 m/s in the placeholder text of bres_margin",
        "m4r_test_dt20_umin126.json, bres_margin.json",
    )

    # ---- section 8 -------------------------------------------------------------------
    s1 = m1["sensitivity_figures"]
    regret = (s1["0.80"]["CO2"]["regret_rel_median"], s1["1.26"]["CO2"]["regret_rel_median"])
    cost = (s1["0.80"]["CO2"]["cost_rel_median"], s1["1.26"]["CO2"]["cost_rel_median"])
    remaining = (s1["0.80"]["CO2"]["remaining_median"], s1["1.26"]["CO2"]["remaining_median"])
    report.claim(
        "saves a median of 9.0 % (lower gas-velocity bound 0.80 m/s) to 16.5 % (1.26 m/s)",
        [("9.0", 100 * regret[0]), ("16.5", 100 * regret[1])],
        "m1_k1_comparison.json (rerun_k1, CO2, median regret)",
    )
    report.claim(
        "costs 1.30 to 1.46 % and leaves 94 % (1.26 m/s) to 85 % (0.80 m/s)",
        [
            ("1.30", 100 * cost[0]),
            ("1.46", 100 * cost[1]),
            ("94", 100 * remaining[1]),
            ("85", 100 * remaining[0]),
        ],
        "m1_k1_comparison.json (rerun_k1, CO2, median cost and remaining benefit)",
    )
    b126, b080 = m1["1.26"]["bounds"]["M0"], m1["0.80"]["bounds"]["M0"]
    report.claim(
        "on its lower bound at 22 (1.26 m/s) and 17 (0.80 m/s) of 25 test points",
        [
            ("22", b126["n_on_lower_gas_bound"]),
            ("17", b080["n_on_lower_gas_bound"]),
            ("25", b126["n"]),
        ],
        "m1_k1_comparison.json (rerun_k1, bounds M0)",
    )
    report.claim(
        "the moisture limit binds at 13 (1.26 m/s) and 25 (0.80 m/s) of 25 points",
        [("13", b126["n_binding_moisture"]), ("25", b080["n_binding_moisture"])],
        "m1_k1_comparison.json (rerun_k1, bounds M0)",
    )
    w_scenarios = (
        sum(s["group"] == "W" for s in dt20["scenarios"].values())
        if isinstance(dt20["scenarios"], dict)
        else sum(s["group"] == "W" for s in dt20["scenarios"])
    )
    report.claim(
        "(0 of 6 scenarios at 20 K minimum approach, 1 of 6 at 10 K as a sensitivity",
        [
            ("0", dt20["n_scenarios_of_W_holding_best_seed"]),
            ("6", w_scenarios),
            ("1", dt10["n_scenarios_of_W_holding_best_seed"]),
            ("20", dt20["case_dt_min_k"]),
            ("10", dt10["case_dt_min_k"]),
        ],
        "m4r_test_dt20_umin126.json, m4r_test_dt10_umin126.json",
    )
    cell5 = margin["cells"]["5"]
    report.truth(
        "needs a margin of five times the base margin to be feasible in every seed",
        margin["gate_k"] == 5 and cell5["feasible_in_every_seed"],
        "gate_k 5, feasibility 1.0 in every seed at k = 5",
        "bres_margin.json",
    )
    report.claim(
        "at 6.9 % extra energy over the optimiser with the base margin, of which 6.5 points are the margin itself",
        [
            ("6.9", cell5["cost_energy_total_pct"]["median"]),
            ("6.5", cell5["cost_energy_margin_part_pct"]["median"]),
        ],
        "bres_margin.json (cells 5, medians over five seeds)",
    )
    ratios_partner = []
    for scenario in ("1", "3", "5"):
        a_kw = heat["source_variants"]["i_partner_exhausts"]["wu1"]["a"][scenario]["recovered_kw"]
        for dt in ("dt20", "dt10", "dt5"):
            c_kw = heat["source_variants"]["i_partner_exhausts"]["wu1"][dt]["c"][scenario][
                "recovered_kw"
            ]
            ratios_partner.append(a_kw / c_kw)
    report.claim(
        "the model recovers 60 to 88 % of an independent counter-current calculation",
        [("60", 100 * min(ratios_partner)), ("88", 100 * max(ratios_partner))],
        "heat_recovery_comparison_r4_consistent.json (partner recovery over counter-current, scenarios 1, 3, 5, 20/10/5 K)",
    )
    report.truth(
        "The saving read by the pre-registered rule is L1 (L2 only post hoc with a plain rule)",
        gate["gate"]["all_products"]["reading"] == "L1"
        and posthoc["gate_with_R1b_post_hoc"]["all_products"]["reading_with_R1b"] == "L2",
        "pre-registered gate reads L1 (all products), the post-hoc rule R1b reads L2",
        "gate_storage_shift.json, gate_storage_shift_posthoc.json",
    )

    # ---- exemptions: design constants and plan facts, each with its reason ---------------
    def grep(path: str, pattern: str) -> bool:
        return bool(re.search(pattern, (REPO_ROOT / path).read_text(encoding="utf-8")))

    n_set = sorted({int(r["tags"]["n_train"]) for r in n_records})
    exemptions: list[tuple[str, bool, str]] = [
        (
            r"n_{\mathrm{train}} \in \{8, 16, 32, 64\}",
            n_set == [8, 16, 32, 64],
            "n_train set of the N records",
        ),
        (
            r"n_{\mathrm{train}} \in \{8,16,32,64\}",
            n_set == [8, 16, 32, 64],
            "n_train set of the N records",
        ),
        (r"n_{\mathrm{train}}$ = 64", 64 in n_set, "largest n_train of the N records"),
        (
            "arm N, the preset version 2",
            {r["params"]["data/new_study_version"] for r in n_records} == {2},
            "records of arm N carry data/new_study_version 2",
        ),
        ("whose curve still falls at 64", 64 in n_set, "largest n_train of the N records"),
        (
            "time constant 600 s",
            grep("docs/02_prozess_setup.md", r"600"),
            "docs/02, bed-level controller",
        ),
        (
            "planned for month 21",
            grep("docs/01_projektsteckbrief.md", r"21"),
            "docs/01, critical schedule",
        ),
        (
            "horizon 36 (900 s)",
            abs(36 * sample_time - 900) < 1e-9,
            "36 steps x sample time 25 s from the records",
        ),
        (
            "The M2 definition of 10 to 15 %",
            grep("docs/01_projektsteckbrief.md", r"10\s*(?:bis|-|–)\s*15"),
            "docs/01 note",
        ),
        (
            "two 24-hour exports",
            True,
            "NOT CHECKABLE: raw plant data are never in the repository (CLAUDE.md)",
        ),
        (
            "of only 0.12 to 2.01 %",
            True,
            "NOT CHECKABLE: coefficient of variation of Befund 56a is computed from raw plant data",
        ),
        (
            "nozzle-pressure kink at 1 bar",
            grep("docs/12_review_2026-10-09.md", r"E-Q1"),
            "register item E-Q1",
        ),
    ]
    exempt_results = []
    for literal, verified, reason in exemptions:
        present = normalise(literal) in text
        exempt_results.append(
            {"exemption": literal, "in_text": present, "verified": verified, "reason": reason}
        )

    # ---- coverage ----------------------------------------------------------------------
    start = text.index("## 1. Question and frame")
    end = text.index("## Appendix")
    body = text[start:end]
    body = re.sub(r"`[^`]*`", " ", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
    for fragment in sorted(report.fragments, key=len, reverse=True):
        body = body.replace(fragment, " ")
    for literal, _, _ in sorted(exemptions, key=lambda e: -len(e[0])):
        body = body.replace(normalise(literal), " ")
    patterns = [
        r"Befunde?\s+\d+[a-z]?(?:\s*(?:,|and|/|to)\s*\d+[a-z]?)*(?:\s+(?:addendum\s+[a-z]|[a-z]\b))?",
        r"\bsections?\s+\d+(?:\s+to\s+\d+)?",
        r"\bFigures?\s+\d+(?:\s+and\s+\d+)?",
        r"docs/\d+",
        r"\b[A-Za-z]{1,3}-?\d+[a-z]?\b",
        r"R\^2",
        r"##\s+\d+\.",
        r"\b(?:five|four) seeds\b",
        r"\bhorizons? \d+(?: and \d+)?",
        r"\bAP\d\b",
        r"\$N\$ = \d+",
    ]
    for pattern in patterns:
        body = re.sub(pattern, " ", body)
    leftover = sorted({m.group() for m in re.finditer(r"\d[\d,\.]*\d|\d", body)})
    context = {
        tok: [
            body[max(0, m.start() - 30) : m.end() + 20] for m in re.finditer(re.escape(tok), body)
        ][:2]
        for tok in leftover
    }

    n_conf = sum(r["status"] == "confirmed" for r in report.results)
    n_dev = sum(r["status"] == "DEVIATING" for r in report.results)
    unverified = [e for e in exempt_results if not (e["in_text"] and e["verified"])]
    print(f"claims: {len(report.results)} checked, {n_conf} confirmed, {n_dev} deviating")
    print(
        f"exemptions: {len(exempt_results)} declared, {len(unverified)} not in text or not verified"
    )
    print(f"coverage: {len(leftover)} numeric tokens without a check: {leftover}")
    for row in report.results:
        if row["status"] != "confirmed":
            print("DEVIATING:", row)
    for row in unverified:
        print("EXEMPTION PROBLEM:", row)
    for tok, ctx in context.items():
        print("UNCOVERED", tok, ctx)
    if arguments.write:
        write_artifact(
            ARTIFACT_NAME,
            {
                "_provenance": provenance(),
                "synthesis": SYNTHESIS.name,
                "claims": report.results,
                "exemptions": exempt_results,
                "uncovered_numeric_tokens": leftover,
                "summary": {
                    "claims": len(report.results),
                    "confirmed": n_conf,
                    "deviating": n_dev,
                    "uncovered": len(leftover),
                },
            },
        )
    return 0 if n_dev == 0 and not leftover and not unverified else 1


if __name__ == "__main__":
    raise SystemExit(main())
