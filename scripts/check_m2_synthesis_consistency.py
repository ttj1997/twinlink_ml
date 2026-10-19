r"""Consistency check of ``docs/08_synthese_m2.md`` (version 5c): every number of the summary, of
sections 1-8 and of appendices A-E against its committed artifact or configuration.

No new run, no simulation: pure re-reading of committed artifacts and configuration files. Three parts:

1. CLAIMS. Each claim names a literal fragment of the synthesis, the number(s) written in it
   (digits, or a number word such as "five") and the value(s) computed from the artifact. The
   fragment must occur in the text; the number must agree with the computed value within the
   rounding of the written number (number words: exactly). Truth claims assert a fact computed from
   an artifact or a configuration for a fragment.
2. COVERAGE. All numeric tokens -- digits AND the number words one to twelve, twenty, hundred -- of
   the summary, of sections 1-8 and of appendices A-E (the two changelog sections and the
   traceability table of appendix F are excluded: they restate claims, and the appendix lists the
   artifact per claim) must lie inside a claim fragment, inside a declared exemption (design
   constants and plan facts, each with the reason and a verification against a configuration file or
   an artifact) or match a reference pattern (Befund numbers, section and figure numbers, labels such
   as A0, P1, HB1, enumerations "(1)"). Horizons and budgets written as "horizon N" or "$N$ = n" are
   removed only if the value is one of those of the records. A token outside all of these fails the
   run: a number without a check cannot enter the text unnoticed.
3. NOT CHECKABLE. Claims whose source is not in the repository (raw plant data) are listed with
   the reason, not silently skipped; where possible the lab-book text of the Befund is compared.

Version 5b (review 4, finding 22): the exemptions "600 s", "month 21" and "1 bar" are verified against
the configuration and the plan row instead of a bare grep; "0.80 m/s" against the configuration;
"Holm (5 %)" against ``analyze_rerun.ALPHA`` (the analysis of Befund 94); number words are covered;
the loose tolerances are replaced by claims with the rounding of the written number.

Version 5c (review 4b): the claims for the epoch cap (addendum to Befund 94, artifact
``befund94_addendum_epoch_cap_rerun_full``), for the sample time of the YAML configuration (5 s against
the 25 s of ``DataSpec``) and for the statement that no capacity-matched variant is better than GRU or LSTM
are added; the summary claims follow the new wording.

Call::

    tools/run_here.sh scripts/check_m2_synthesis_consistency.py           # check only
    tools/run_here.sh scripts/check_m2_synthesis_consistency.py --write   # write the artifact
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, RUNS_DIR, provenance, require_clean_tree, write_artifact
from analyze_83_addendum import holm as holm_adjust
from analyze_rerun import ALPHA as RERUN_ALPHA

SYNTHESIS = REPO_ROOT / "docs" / "08_synthese_m2.md"
CONFIG = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
ARTIFACT_NAME = "check_m2_synthesis_consistency_v5c"
MICRO = "µ"
WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "hundred": 100,
}
WORD_PATTERN = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)


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


def as_number(written: str) -> tuple[float, float]:
    """Value and tolerance of a written number (digits with their rounding, or a number word)."""
    if written.lower() in WORDS:
        return float(WORDS[written.lower()]), 1e-9
    return float(written.replace(",", "")), 0.5 * 10 ** (-decimals(written)) + 1e-9


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
            claimed, tolerance = as_number(written)
            ok = present and abs(claimed - float(actual)) <= tolerance
            self.results.append(
                {
                    "fragment": frag,
                    "written": written,
                    "computed": float(actual),
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
                "status": "confirmed" if (present and bool(holds)) else "DEVIATING",
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
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    rerun = load("rerun_evaluation_rerun_full")
    holm = load("holm_rank_cells_rerun_full")
    effects = load("befund98_addendum_effect_sizes_rerun_full")
    need94 = load("befund94_addendum_seeds_data_need_rerun_full")
    ratio55 = load("befund55_addendum_ratio_range")
    timing = load("joint_timing_idle")
    hybrid = load("a1_inference_time_v2_idle")
    paired = load("paired_reevaluation_58_63_67")
    seeds83 = load("paired_reevaluation_83_addendum")
    sx3 = load("sx3_capacity_matched")
    a1 = load("a1_main_analysis")
    b1 = load("b1_main_analysis")
    b1_limits = load("b1_pool_limits")
    m1 = load("m1_k1_comparison")["rerun_k1"]
    stability = load("m1_gas_bound_sensitivity_k1")
    dt20 = load("m4r_test_dt20_umin126")
    dt10 = load("m4r_test_dt10_umin126")
    margin = load("bres_margin")
    fa2 = load("verify_fa2_spread_calibration")
    heat = load("heat_recovery_comparison_r4_consistent")
    gate = load("gate_storage_shift")
    posthoc = load("gate_storage_shift_posthoc")
    timeslice = load("gate_storage_timeslice")
    n_records = load_jsonl("rerun_full_N")
    primary = [r for r in n_records if r["params"].get("eval/primary")]

    def params(model: str) -> dict[str, Any]:
        values = {
            json.dumps(
                {k: v for k, v in r["params"].items() if k.startswith("model/")}, sort_keys=True
            )
            for r in primary
            if r["tags"]["model"] == model
        }
        if len(values) != 1:
            raise SystemExit(f"{model}: model parameters differ between records")
        return json.loads(next(iter(values)))

    def common(key: str) -> Any:
        values = {json.dumps(r["params"][key]) for r in n_records}
        if len(values) != 1:
            raise SystemExit(f"{key} differs between records")
        return json.loads(next(iter(values)))

    seeds_n = sorted({int(r["params"]["master_seed"]) for r in n_records})
    methods_n = sorted({r["tags"]["model"] for r in n_records})
    test_sets_n = sorted({r["tags"]["test_set"] for r in n_records})
    targets = common("data/target_names")
    sample_time = common("data/sample_time_s")
    n_set = sorted({int(r["tags"]["n_train"]) for r in n_records})
    horizons_n = sorted(
        {int(k.rsplit("/h", 1)[1]) for r in primary[:1] for k in r["metrics"] if "/h" in k}
    )
    dof = {c["degrees_of_freedom"] for arm in rerun["rank_cells"].values() for c in arm}
    mult = rerun["multiplicity"]
    n_params = {m: params(m)["model/n_parameters"] for m in methods_n}

    import warnings

    from twinlink_ml.evaluation.experiments import NEW_STUDY_OPTIONS
    from twinlink_ml.simulation import load_process_config
    from twinlink_ml.simulation.excitation import ExcitationConfig
    from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulator = GreyBoxFluidBedGranulator(load_process_config(CONFIG))

    def five(fragment: str, source: str = "rerun_full_N.jsonl (master_seed)") -> None:
        report.claim(fragment, [("five", len(seeds_n))], source)

    # ---- summary ---------------------------------------------------------------------
    five("paired comparisons over five seeds")
    report.claim(
        "lowest errors of the eight methods tested",
        [("eight", len(methods_n))],
        "rerun_full_N.jsonl",
    )
    report.claim(
        "by a factor of about seven",
        [("seven", round(n_params["lstm"] / n_params["mamba"]))],
        "rerun_full_N.jsonl (model/n_parameters, LSTM over Mamba)",
    )
    curves = need94["part_b_data_need"]["curves"]
    moist8 = [
        curves[f"{m}/product_moisture_kg_per_kg/{t}"]["8"]["median_over_median_at_64"]
        for m in ("gru", "lstm")
        for t in ("excited", "constant_holdout")
    ]
    for fragment in (
        "the moisture error of GRU and LSTM at 8 training runs is 3.7 to 5.1 times the error at 64 runs",
        "the error of GRU and LSTM on the product moisture at 8 training runs is 3.7 to 5.1 times the error at 64 runs",
    ):
        report.claim(
            fragment,
            [("8", min(n_set)), ("3.7", min(moist8)), ("5.1", max(moist8)), ("64", max(n_set))],
            "befund94_addendum_seeds_data_need_rerun_full.json (part_b curves)",
        )
    runs = need94["part_b_data_need"]["runs_behind_each_point"]
    report.claim(
        "every point also uses 32 validation runs", [("32", runs["n_val"])], "befund94 addendum"
    )
    report.claim(
        "below the sample time of 25 s",
        [("25", sample_time)],
        "rerun_full_N.jsonl (data/sample_time_s)",
    )

    report.claim(
        "splits by run, two named test sets", [("two", len(test_sets_n))], "rerun_full_N.jsonl"
    )
    report.claim(
        "large and consistent across all seeds but not resolved under Holm with five seeds",
        [("five", len(seeds_n))],
        "rerun_full_N.jsonl (master_seed)",
    )
    gl_falls_all = [
        c for c in need94["part_b_data_need"]["falls_32_to_64"] if c["method"] in ("gru", "lstm")
    ]
    report.claim(
        "and the curves still fall at 64 runs in the medians (an uncorrected statement, see section 6), so the need beyond 64 runs is not determined",
        [("64", max(n_set)), ("64", max(n_set))],
        "befund94 addendum (falls_32_to_64)",
    )
    report.truth(
        "and the curves still fall at 64 runs",
        all(c["falls_uncorrected"] for c in gl_falls_all),
        "GRU and LSTM lower at 64 than at 32 runs in every cell (uncorrected)",
        "befund94 addendum",
    )
    # ---- section 1 -------------------------------------------------------------------
    report.claim(
        "The three predicted targets",
        [("three", len(targets))],
        "rerun_full_N.jsonl (data/target_names)",
    )
    report.truth(
        "the instantaneous total power of heater, fan, pump and atomiser in kW (absolute, no reference point",
        config["energy"]["target"] == "power_kw" and config["energy"]["reference_value"] is None,
        "energy.target power_kw, reference_value null",
        CONFIG.name,
    )
    n_states = len(simulator.state_names)
    report.claim(
        "grey-box model with four states",
        [("four", n_states)],
        "GreyBoxFluidBedGranulator.state_names",
    )
    report.claim(
        "Four states, no agglomeration",
        [("four", n_states)],
        "GreyBoxFluidBedGranulator.state_names",
    )

    # ---- section 2 -------------------------------------------------------------------
    five("as median and range over five seeds (Figures 1 and 2)")
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
    report.truth(
        "run-median NRMSE of the energy target, the product moisture and the median particle diameter",
        all(
            f"nrmse_run_median/{c}/h1" in trees[0]["metrics"]
            for c in ("energy_target", "product_moisture_kg_per_kg", "d50_um")
        ),
        "the three channels of the comparison exist in the records",
        "extra_trees_split_comparison.jsonl",
    )
    report.claim(
        "Two test sets, always named",
        [("two", len(test_sets_n))],
        "rerun_full_N.jsonl (tags test_set)",
    )
    report.claim(
        "with the t distribution on four degrees of freedom",
        [("four", next(iter(dof)))],
        "rerun_evaluation_rerun_full.json (degrees_of_freedom)",
    )
    report.claim(
        "only if it survives Holm (5 %)",
        [("5", 100 * RERUN_ALPHA)],
        "scripts/analyze_rerun.py ALPHA",
    )
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
    seeds_part = need94["part_a_seeds"]
    n64 = {(r["channel"], r["test_set"], r["candidate"]): r for r in seeds_part["N_n64_h36"]}
    easy = ("energy_target", "bed_temperature_celsius")
    both = ("excited", "constant_holdout")

    def need(channel: str, test_set: str, candidate: str, key: str = "seeds_0.125") -> int:
        return n64[(channel, test_set, candidate)][key]

    gru_easy = [need(c, t, "gru") for c in easy for t in both]
    tr_easy = [need(c, t, "transformer") for c in easy for t in both]
    report.claim(
        "a paired difference of 12.5 % between GRU and LSTM is resolved at 80 % power with 3 to 6 seeds on energy and bed temperature, with 5 seeds on the product moisture on the excited test set and with 16 on the constant-run holdout",
        [
            ("12.5", 100 * max(seeds83["deltas"])),
            ("80", 100 * seeds83["power"]),
            ("3", min(gru_easy)),
            ("6", max(gru_easy)),
            ("5", need("product_moisture_kg_per_kg", "excited", "gru")),
            ("16", need("product_moisture_kg_per_kg", "constant_holdout", "gru")),
        ],
        "befund94_addendum_seeds_data_need_rerun_full.json (N, n_train 64, h36)",
    )
    report.claim(
        "the Transformer needs 4 to 7 seeds on energy and bed temperature but 210 (excited) and 69 (constant-run holdout) on the moisture",
        [
            ("4", min(tr_easy)),
            ("7", max(tr_easy)),
            ("210", need("product_moisture_kg_per_kg", "excited", "transformer")),
            ("69", need("product_moisture_kg_per_kg", "constant_holdout", "transformer")),
        ],
        "befund94_addendum_seeds_data_need_rerun_full.json (N, n_train 64, h36)",
    )
    five(
        "planning figures estimated from five seeds",
        "befund94 addendum (seeds_row over five seeds)",
    )
    rows64 = {
        (r["channel"], r["test_set"], r["candidate"]): r["seeds_0.125"]
        for r in seeds83["seeds_needed"]
        if r["n_train"] == 64
    }
    report.claim(
        "the old data A0 gave 45 and 37 seeds for GRU and 193 and 33 for the Transformer on the moisture",
        [
            ("45", rows64[("product_moisture_kg_per_kg", "excited", "gru")]),
            ("37", rows64[("product_moisture_kg_per_kg", "constant_holdout", "gru")]),
            ("193", rows64[("product_moisture_kg_per_kg", "excited", "transformer")]),
            ("33", rows64[("product_moisture_kg_per_kg", "constant_holdout", "transformer")]),
        ],
        "paired_reevaluation_83_addendum.json",
    )
    spread = seeds_part["ranges_seeds_0.125_all_budgets_and_horizons"][
        "N/product_moisture_kg_per_kg/excited/gru"
    ]
    report.claim(
        "(GRU on the excited set 3 to 753 seeds)",
        [("3", spread["min"]), ("753", spread["max"])],
        "befund94 addendum (ranges over all budgets and horizons)",
    )

    # ---- section 3 -------------------------------------------------------------------
    report.claim(
        "Eight methods (persistence, static regression, ARX, GRU, LSTM, Transformer, xLSTM, Mamba), three targets, two test sets",
        [("eight", len(methods_n)), ("three", len(targets)), ("two", len(test_sets_n))],
        "rerun_full_N.jsonl",
    )
    five("$n_{\\mathrm{train}} \\in \\{8,16,32,64\\}$, five seeds, horizons 1 and 36")
    five("median and range over five seeds, arm N, excited test set")
    report.claim(
        "LSTM 57,027 parameters, GRU 42,819, Transformer 23,523, xLSTM 8,649 and Mamba 8,139",
        [
            ("57,027", n_params["lstm"]),
            ("42,819", n_params["gru"]),
            ("23,523", n_params["transformer"]),
            ("8,649", n_params["xlstm"]),
            ("8,139", n_params["mamba"]),
        ],
        "rerun_full_N.jsonl (model/n_parameters)",
    )
    report.claim(
        "(at most 200 epochs, early stopping)",
        [("200", params("gru")["model/net_max_epochs"])],
        "rerun_full_N.jsonl",
    )
    sp = sx3["parameters"]
    mixing = ("slstm_cm", "slstm_l1_cm", "mixed_cm")
    report.claim(
        "(sLSTM only, and mixed mLSTM and sLSTM, 47,955 to 50,739 parameters)",
        [("47,955", min(sp[v] for v in mixing)), ("50,739", max(sp[v] for v in mixing))],
        "sx3_capacity_matched.json (parameters)",
    )
    med = sx3["medians"]

    def sx3_ratio(variant: str, cell: str) -> float:
        return med[variant][cell] / med["lstm"][cell]

    moist_cells = (
        "product_moisture_kg_per_kg/excited",
        "product_moisture_kg_per_kg/constant_holdout",
    )
    other_cells = [k for k in med["lstm"] if not k.startswith("product_moisture")]
    report.claim(
        "stay 1.6 to 3.3 times behind it on the moisture",
        [
            ("1.6", min(sx3_ratio(v, c) for v in mixing for c in moist_cells)),
            ("3.3", max(sx3_ratio(v, c) for v in mixing for c in moist_cells)),
        ],
        "sx3_capacity_matched.json (medians)",
    )
    report.truth(
        "come close to the LSTM on energy and bed temperature",
        max(sx3_ratio(v, c) for v in mixing for c in other_cells) < 1.2,
        "sLSTM and mixed variants within a factor 1.2 of the LSTM median on energy and bed temperature",
        "sx3_capacity_matched.json (medians)",
    )
    matched = ("mlstm_cm", "slstm_cm", "slstm_l1_cm", "mixed_cm", "mamba_cm")
    report.truth(
        "no capacity-matched variant is better than GRU or LSTM",
        all(med[v][c] >= min(med["lstm"][c], med["gru"][c]) for v in matched for c in med["lstm"]),
        "median of every capacity-matched variant in every cell is not below the better of GRU and LSTM",
        "sx3_capacity_matched.json (medians)",
    )
    report.truth(
        "at equal capacity the sLSTM and the mixed variant largely close the gap to the LSTM on energy and bed temperature but not on the moisture",
        max(sx3_ratio(v, c) for v in mixing for c in other_cells) < 1.2
        and min(sx3_ratio(v, c) for v in mixing for c in moist_cells) > 1.5,
        "sLSTM and mixed variants within 1.2 of the LSTM on energy and bed temperature, above 1.5 on moisture",
        "sx3_capacity_matched.json (medians)",
    )
    cap = load("befund94_addendum_epoch_cap_rerun_full")
    cap_networks = ("gru", "lstm", "transformer", "xlstm", "mamba")
    cap64 = {n: cap[n]["by_n_train"]["64"] for n in cap_networks}
    cap_all = {n: cap[n]["total"] for n in cap_networks}
    cap_source = "befund94_addendum_epoch_cap_rerun_full.json"
    report.claim(
        "the epoch cap binds for part of the heads of all five networks",
        [("five", len(cap_networks))],
        cap_source,
    )
    report.claim(
        "All networks stop at the cap of 200 epochs at the latest",
        [("200", cap["gru"]["epoch_cap"])],
        cap_source,
    )
    report.claim(
        "Heads that reach it at 64 runs: Transformer 10 of 15, Mamba 9 of 15, LSTM 5 of 15, GRU 4 of 15, xLSTM 2 of 15",
        [
            ("10", cap64["transformer"]["at_cap"]),
            ("15", cap64["transformer"]["trainings"]),
            ("9", cap64["mamba"]["at_cap"]),
            ("15", cap64["mamba"]["trainings"]),
            ("5", cap64["lstm"]["at_cap"]),
            ("15", cap64["lstm"]["trainings"]),
            ("4", cap64["gru"]["at_cap"]),
            ("15", cap64["gru"]["trainings"]),
            ("2", cap64["xlstm"]["at_cap"]),
            ("15", cap64["xlstm"]["trainings"]),
        ],
        cap_source,
    )
    report.claim(
        "their best epoch lies in the last 20 epochs",
        [("20", cap["last"])],
        cap_source,
    )
    report.truth(
        "their best epoch lies in the last 20 epochs",
        all(cap64[n]["at_cap_and_best_in_last"] == cap64[n]["at_cap"] for n in cap_networks),
        "at 64 runs every head that reaches the cap has its best epoch in the last 20 epochs",
        cap_source,
    )
    report.claim(
        "Over all four budgets the cap of 200 epochs is reached by 44 of 60 head-trainings of the Transformer, 33 of the GRU, 32 of the LSTM, 22 of Mamba and 4 of the xLSTM",
        [
            ("four", len(n_set)),
            ("200", cap["gru"]["epoch_cap"]),
            ("44", cap_all["transformer"]["at_cap"]),
            ("60", cap_all["transformer"]["trainings"]),
            ("33", cap_all["gru"]["at_cap"]),
            ("32", cap_all["lstm"]["at_cap"]),
            ("22", cap_all["mamba"]["at_cap"]),
            ("4", cap_all["xlstm"]["at_cap"]),
        ],
        cap_source,
    )
    fam = next(f for f in seeds83["families"] if f["family"] == "b63_67")
    mamba_cm = [
        c for c in fam["cells"] if c["candidate"] == "mamba_cm" and c["reference"] == "lstm"
    ]
    small = [c for c in fam["cells"] if c["candidate"] == "lstm_small" and c["reference"] == "lstm"]
    report.claim(
        "the capacity-matched mLSTM-only variant (50,301) diverges on the energy target",
        [("50,301", sp["mlstm_cm"])],
        "sx3_capacity_matched.json (parameters)",
    )
    report.claim(
        "a Mamba of 53,211 parameters is not better than the small one (resolved worse than LSTM in 6 of 6 cells, 3 of them after Holm in the family of 96 tests",
        [
            ("53,211", sp["mamba_cm"]),
            ("6", sum(c["verdict"] == "candidate worse" for c in mamba_cm)),
            ("6", len(mamba_cm)),
            ("3", sum(c["survives_holm"] for c in mamba_cm)),
            ("96", fam["tests"]),
        ],
        "sx3_capacity_matched.json, paired_reevaluation_83_addendum.json (family b63_67)",
    )
    report.claim(
        "a small LSTM of 8,517 parameters is not resolved worse than the large one in any cell",
        [("8,517", sp["lstm_small"])],
        "sx3_capacity_matched.json",
    )
    report.truth(
        "is not resolved worse than the large one in any cell",
        len(small) > 0 and not any(c["verdict"] == "candidate worse" for c in small),
        "small LSTM against LSTM: no cell worse (paired re-evaluation)",
        "paired_reevaluation_83_addendum.json (family b63_67)",
    )
    threshold = effects["large_ratio_threshold"]
    report.claim(
        "but whose median is at least 1.5 times that of LSTM (or at most 1/1.5 of it) with all five seeds on that side",
        [("1.5", threshold), ("1", 1), ("1.5", threshold), ("five", len(seeds_n))],
        "befund98_addendum_effect_sizes_rerun_full.json (large_ratio_threshold)",
    )
    report.claim(
        "not resolved under Holm with five seeds and a family of 336 tests",
        [("five", len(seeds_n)), ("336", mult["rank_N"]["tests"])],
        "rerun_evaluation_rerun_full.json",
    )
    gru = holm["N"]["gru_vs_lstm"]
    report.claim(
        "(15 of 16 cells unresolved, one cell with GRU worse, uncorrected)",
        [
            ("15", gru["product_moisture_kg_per_kg"]["main_family"]["unresolved_uncorrected"]),
            ("16", gru["product_moisture_kg_per_kg"]["main_family"]["cells"]),
            ("one", gru["product_moisture_kg_per_kg"]["main_family"]["worse_uncorrected"]),
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
    by_h = holm["N"]["gru_by_channel_and_horizon"]
    report.claim(
        "(on bed temperature mostly at horizon 1: 7 of the 9 cells)",
        [
            ("1", 1),
            ("7", by_h.get("bed_temperature_celsius/h1/candidate better", 0)),
            ("9", gru["bed_temperature_celsius"]["main_family"]["better_uncorrected"]),
        ],
        "holm_rank_cells_rerun_full.json (gru_by_channel_and_horizon, N)",
    )
    eff = effects["N"]["summary"]
    report.claim(
        "(median ratio of the medians 0.94 on energy and 0.98 on bed temperature)",
        [
            ("0.94", eff["gru"]["energy_target"]["ratio_median"]),
            ("0.98", eff["gru"]["bed_temperature_celsius"]["ratio_median"]),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    no_holm = all(
        holm[arm]["gru_vs_lstm"][ch][fam_][k] == 0
        for arm in ("A0", "A1", "N")
        for ch in holm[arm]["gru_vs_lstm"]
        for fam_ in ("main_family", "sensitivity_family")
        for k in ("better_holm", "worse_holm")
    )
    report.truth(
        "After Holm correction, no GRU-versus-LSTM cell is resolved in any channel",
        no_holm,
        "0 resolved GRU-against-LSTM cells after Holm, all arms, channels, both families",
        "holm_rank_cells_rerun_full.json",
    )
    report.truth(
        "after Holm no GRU-against-LSTM cell is resolved",
        no_holm,
        "as above (summary)",
        "holm_rank_cells_rerun_full.json",
    )
    report.claim(
        "about 25 % fewer parameters (42,819 against 57,027, Befund 28)",
        [
            ("25", 100 * (1 - n_params["gru"] / n_params["lstm"])),
            ("42,819", n_params["gru"]),
            ("57,027", n_params["lstm"]),
        ],
        "rerun_full_N.jsonl (model/n_parameters)",
    )
    report.claim(
        "(mLSTM-only variant, 8.6k parameters)",
        [("8.6", n_params["xlstm"] / 1000)],
        "rerun_full_N.jsonl",
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
    report.claim(
        "every seed is worse in all 48 cells, the median ratio is 3.3",
        [
            ("48", eff["xlstm"]["all"]["all_seeds_worse"]),
            ("48", eff["xlstm"]["all"]["cells"]),
            ("3.3", eff["xlstm"]["all"]["ratio_median"]),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    xe = effects["N"]["xlstm_energy_h36"]
    report.claim(
        "on the energy target at horizon 36 the xLSTM median is 10.4 to 24.5 times the LSTM's in all 8 cells",
        [
            ("36", 36),
            ("10.4", min(r["ratio_xlstm_over_lstm"] for r in xe)),
            ("24.5", max(r["ratio_xlstm_over_lstm"] for r in xe)),
            ("8", len(xe)),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json (xlstm_energy_h36)",
    )
    report.truth(
        "with every seed worse, and none of these cells survives Holm",
        all(
            r["seeds_xlstm_worse_than_lstm"] == len(seeds_n) and r["p_holm"] >= RERUN_ALPHA
            for r in xe
        ),
        "all seeds worse and Holm-adjusted p >= 0.05 in all 8 cells",
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    x64 = {r["test_set"]: r["vs_static"] for r in xe if r["n_train"] == 64}
    report.claim(
        "times that of the static regression, but with 4 and 3 of 5 seeds worse this is not resolved",
        [
            ("4", x64["excited"]["seeds_xlstm_worse"]),
            ("3", x64["constant_holdout"]["seeds_xlstm_worse"]),
            ("5", len(seeds_n)),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json (vs_static, n_train 64)",
    )
    report.claim(
        "the xLSTM median on the energy target is also 3.4 (excited) and 2.3 (constant-run holdout)",
        [
            ("3.4", x64["excited"]["ratio_xlstm_over_static"]),
            ("2.3", x64["constant_holdout"]["ratio_xlstm_over_static"]),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json (vs_static, n_train 64)",
    )
    report.truth(
        "this is not resolved",
        all(v["verdict_uncorrected"] == "unresolved" for v in x64.values()),
        "xLSTM against static at n_train 64: unresolved on both test sets",
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    xl_moist = [
        c
        for c in rerun["rank_cells"]["N"]
        if c["candidate"] == "xlstm" and c["channel"] == "product_moisture_kg_per_kg"
    ]
    report.claim(
        "it stays unresolved in 7 of 16 cells (uncorrected)",
        [("7", sum(c["verdict"] == "unresolved" for c in xl_moist)), ("16", len(xl_moist))],
        "rerun_evaluation_rerun_full.json (rank cells, arm N)",
    )
    tm = eff["transformer"]
    report.claim(
        "large and consistent on the moisture (11 of 16 cells, median ratio 1.67)",
        [
            ("11", tm["product_moisture_kg_per_kg"]["large_consistent_unresolved"]),
            ("16", tm["product_moisture_kg_per_kg"]["cells"]),
            ("1.67", tm["product_moisture_kg_per_kg"]["ratio_median"]),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    report.claim(
        "small on energy and bed temperature (median ratio 1.26 and 1.16)",
        [
            ("1.26", tm["energy_target"]["ratio_median"]),
            ("1.16", tm["bed_temperature_celsius"]["ratio_median"]),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    report.claim(
        "Mamba is resolved behind LSTM in 14 of 48 cells after Holm (median ratio 3.2)",
        [
            ("14", eff["mamba"]["all"]["worse_holm"]),
            ("48", eff["mamba"]["all"]["cells"]),
            ("3.2", eff["mamba"]["all"]["ratio_median"]),
        ],
        "befund98_addendum_effect_sizes_rerun_full.json",
    )
    report.claim(
        "static regression in 21 of 48 cells, persistence in 19, ARX in 12",
        [
            ("21", by_cand["static"]["all"]["worse_holm"]),
            ("48", by_cand["static"]["all"]["cells"]),
            ("19", by_cand["persistence"]["all"]["worse_holm"]),
            ("12", by_cand["arx"]["all"]["worse_holm"]),
        ],
        "holm_rank_cells_rerun_full.json (arm N)",
    )
    report.claim(
        "persistence is better than LSTM in one cell (energy, uncorrected)",
        [("one", by_cand["persistence"]["energy_target"]["better_uncorrected"])],
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

    def holm_survivors(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
        adjusted = holm_adjust(
            np.array([1.0 if c["p_value"] != c["p_value"] else c["p_value"] for c in cells])
        )
        return [
            c
            for c, p in zip(cells, adjusted, strict=True)
            if c["verdict"] in ("candidate better", "candidate worse") and p < RERUN_ALPHA
        ]

    a1_surv = holm_survivors(rerun["a1_vs_a0"])
    n_surv = holm_survivors(rerun["n_vs_a0"])
    report.claim(
        "after Holm 3 of 192 tests remain (GRU and LSTM, all worse in A1",
        [("3", len(a1_surv)), ("192", len(rerun["a1_vs_a0"]))],
        "rerun_evaluation_rerun_full.json (a1_vs_a0, Holm recomputed)",
    )
    report.truth(
        "(GRU and LSTM, all worse in A1",
        {c["candidate"].split("[")[0] for c in a1_surv} == {"gru", "lstm"}
        and all(c["verdict"] == "candidate worse" for c in a1_surv)
        and len(a1_surv) == mult["a1_vs_a0"]["resolved_holm"],
        f"A1 against A0 after Holm: {Counter((c['candidate'], c['verdict']) for c in a1_surv)}",
        "rerun_evaluation_rerun_full.json",
    )
    report.claim(
        "Of 384 tests 24 survive Holm, and all of them are persistence",
        [("384", len(rerun["n_vs_a0"])), ("24", len(n_surv))],
        "rerun_evaluation_rerun_full.json (n_vs_a0, Holm recomputed)",
    )
    report.truth(
        "and all of them are persistence",
        all(c["candidate"].startswith("persistence") for c in n_surv)
        and len(n_surv) == mult["n_vs_a0"]["resolved_holm"],
        f"N against A0 after Holm: {Counter((c['candidate'], c['verdict']) for c in n_surv)}",
        "rerun_evaluation_rerun_full.json",
    )
    report.claim(
        "N against A0 mixes three changes (planned inputs, solid-fraction process, per-run cap",
        [("three", len([k for k in NEW_STUDY_OPTIONS if k != "new_study_version"]))],
        "twinlink_ml.evaluation.experiments.NEW_STUDY_OPTIONS",
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
        "all five seeds better; Befund 83 e",
        all_better and len(ratios) == 6 and len(seeds_n) == 5,
        "every one of the six cells resolved better, no seed worse",
        "paired_reevaluation_58_63_67.json",
    )

    five("is not supported by five seeds")
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
        expected = (
            f"| {label} | {row['n_parameters']:,} | {round(row['latency_batch1_us_median']):,} "
            f"| {row['throughput_us_per_window_median']:,.1f} |"
        )
        report.truth(
            expected,
            True,
            f"{label}: parameters, latency, batch cost equal the artifact",
            "joint_timing_idle.json",
        )
    env = timing["environment"]
    report.claim(
        "on an idle machine (Befund 97): one thread, warm-up excluded, one process",
        [("one", env["torch_num_threads"]), ("one", timing["design"]["threads"])],
        "joint_timing_idle.json (environment, design)",
    )
    windows = {r["n_test_windows"] for r in timing["results"]}
    report.claim(
        "= 64, 4352 excited-test windows", [("4352", next(iter(windows)))], "joint_timing_idle.json"
    )
    report.claim(
        "inference cost of the eight methods",
        [("eight", len(timing["results"]))],
        "joint_timing_idle.json",
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
            ("one", 1),
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
    report.truth(
        "Latency of one window and cost per window in a batch",
        "latency_batch1_us_median" in fit["gru"],
        "latency is measured at a batch of 1 window",
        "joint_timing_idle.json",
    )
    report.truth(
        "| method | parameters | latency, one window |",
        True,
        "table header (batch of 1)",
        "joint_timing_idle.json",
    )
    report.truth(
        "latency of one window and cost per window in a batch, microseconds",
        True,
        "caption of Figure 3",
        "joint_timing_idle.json",
    )

    # ---- section 5 -------------------------------------------------------------------
    report.claim(
        "With five seeds and a family of 336 tests",
        [("five", len(seeds_n)), ("336", mult["rank_N"]["tests"])],
        "rerun_evaluation_rerun_full.json",
    )
    five("Five seeds resolve large differences everywhere")
    gru_all = [
        need(c, t, "gru", k) for c in easy for t in both for k in ("seeds_0.125", "seeds_0.100")
    ]
    tr_all = [
        need(c, t, "transformer", k)
        for c in easy
        for t in both
        for k in ("seeds_0.125", "seeds_0.100")
    ]
    report.claim(
        "for a 10 to 12.5 % difference of GRU against LSTM on energy and bed temperature (3 to 8 seeds, arm N, $n_{\\mathrm{train}}$ = 64), not for the Transformer (4 to 10 seeds)",
        [
            ("10", 100 * min(seeds83["deltas"])),
            ("12.5", 100 * max(seeds83["deltas"])),
            ("3", min(gru_all)),
            ("8", max(gru_all)),
            ("64", 64),
            ("4", min(tr_all)),
            ("10", max(tr_all)),
        ],
        "befund94_addendum_seeds_data_need_rerun_full.json (N, n_train 64, h36)",
    )
    report.claim(
        "In arm N 260 of 336 rank cells are resolved without correction, 251 after Benjamini-Hochberg and 70 after Holm",
        [
            ("260", mult["rank_N"]["resolved_uncorrected"]),
            ("336", mult["rank_N"]["tests"]),
            ("251", mult["rank_N"]["resolved_bh"]),
            ("70", mult["rank_N"]["resolved_holm"]),
        ],
        "rerun_evaluation_rerun_full.json (multiplicity)",
    )

    # ---- section 6 and appendix D: data need -----------------------------------------
    falls = need94["part_b_data_need"]["falls_32_to_64"]
    gl_falls = [c for c in falls if c["method"] in ("gru", "lstm")]
    report.claim(
        "the curves still fall between 32 and 64 runs",
        [("32", 32), ("64", max(n_set))],
        "befund94 addendum (falls_32_to_64)",
    )
    report.truth(
        "the curves still fall between 32 and 64 runs",
        all(c["falls_uncorrected"] for c in gl_falls),
        "GRU and LSTM lower at 64 than at 32 runs in every cell (uncorrected)",
        "befund94 addendum",
    )
    beats = need94["part_b_data_need"]["beats_baseline"]

    def first(net: str, base: str, channel: str, test_set: str) -> int | None:
        return beats[f"{net}/{base}/{channel}/{test_set}"]["first_n_train_all_seeds_better"]

    chans = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
    static_first = [first(n, "static", c, t) for n in ("gru", "lstm") for c in chans for t in both]
    pers_exc = [first(n, "persistence", c, "excited") for n in ("gru", "lstm") for c in chans]
    pers_moist = [
        first(n, "persistence", "product_moisture_kg_per_kg", "constant_holdout")
        for n in ("gru", "lstm")
    ]
    report.claim(
        "GRU and LSTM beat the static regression in every seed from 8 runs; on the constant-run holdout they beat persistence on the moisture in every seed only from 32 runs",
        [("8", max(static_first)), ("32", min(pers_moist)), ("32", max(pers_moist))],
        "befund94 addendum (beats_baseline)",
    )
    report.claim(
        "the static regression from 8 runs; persistence from 8 runs on the excited set, and on the constant-run holdout from 32 runs on the moisture",
        [("8", max(static_first)), ("8", max(pers_exc)), ("32", max(pers_moist))],
        "befund94 addendum (beats_baseline)",
    )
    report.claim(
        "Paired tests over five seeds, 2640 tests without multiplicity correction",
        [("five", len(a1["design"]["seeds"])), ("2640", a1["design"]["n_paired_tests_total"])],
        "a1_main_analysis.json (design)",
    )
    report.truth(
        "Paired tests over five seeds",
        len(a1["design"]["seeds"]) == 5 and a1["design"]["multiplicity_correction"] == "none",
        "five seeds, no correction",
        "a1_main_analysis.json (design)",
    )
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
    report.claim(
        "reduces the error at 2 real runs by a factor of 12.69 (identical plant) and 10.27 (parameter deviation)",
        [("2", 2), ("12.69", ratio["P0"]["2"]), ("10.27", ratio["P1"]["2"])],
        "a1_main_analysis.json (ha1)",
    )
    crossover = a1["ha1"]["crossover_finetune_vs_scratch"]
    reached = [p for p, v in crossover.items() if v["n_star"] is None]
    report.claim(
        "the crossover is not reached on four of the five plants",
        [("four", len(reached)), ("five", len(crossover))],
        "a1_main_analysis.json (crossover)",
    )
    hyb = [
        c
        for c in a1["comparisons"]
        if c["candidate"] == "hybrid"
        and c["reference"] == "pretrain_finetune"
        and c["test_set"] == "excited"
    ]
    per_plant_n = Counter((c["plant"], c["n"]) for c in hyb if c["verdict"] == "better")
    report.claim(
        "The hybrid is not better than pre-training plus fine-tuning in two of three channels at any $N$ (resolved better in 1 of 90 cells, worse in 34)",
        [
            ("two", 2),
            ("three", len(a1["design"]["channels"])),
            ("1", sum(c["verdict"] == "better" for c in hyb)),
            ("90", len(hyb)),
            ("34", sum(c["verdict"] == "worse" for c in hyb)),
        ],
        "a1_main_analysis.json (comparisons, hybrid against pretrain_finetune, excited)",
    )
    report.truth(
        "not better than pre-training plus fine-tuning in two of three channels at any $N$",
        max(per_plant_n.values(), default=0) < 2,
        "no (plant, N) with the hybrid better in two or more channels",
        "a1_main_analysis.json",
    )
    report.claim(
        "With the simulator making up 98.8 % of its inference cost",
        [("98.8", 100 * ht["simulation_share"])],
        "a1_inference_time_v2_idle.json",
    )
    five("Five seeds, an LSTM, plants that are variants", "a1_main_analysis.json (design seeds)")
    verdicts = {k: v["verdict"] for k, v in b1["verdicts"].items()}
    t_sets = {k: v["verdict"] for k, v in b1["verdicts"]["HB2"]["per_test_set"].items()}
    report.claim(
        "(ten seeds, paired t;", [("ten", len(b1["seeds"]))], "b1_main_analysis.json (seeds)"
    )
    report.claim(
        "LSTM, product moisture, horizon 24)",
        [("24", int(re.search(r"\d+", str(b1["primary"].get("horizon", b1["primary"]))).group()))],
        "b1_main_analysis.json (primary)",
    )
    report.truth(
        "(HB1: not established)",
        verdicts["HB1"] == "not established",
        f"HB1: {verdicts['HB1']}",
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
        f"HB3: {verdicts['HB3']}",
        "b1_main_analysis.json",
    )
    shares = b1_limits["summary"]
    at_cap = [
        shares[p]["shares_mean_over_seeds"]["at_cap"][1] for p in ("b0", "b1", "b2", "b3", "b4")
    ]
    report.claim(
        "(the gas velocity sits on it in 20 to 24 % of the steps)",
        [("20", 100 * min(at_cap)), ("24", 100 * max(at_cap))],
        "b1_pool_limits.json (at_cap, gas velocity)",
    )
    b2 = shares["b2"]["shares_mean_over_seeds"]["ratelim_changed"]
    channels_b1 = b1_limits["channels"]
    report.claim(
        "(plan B2: 18 % of the inlet-temperature and 35 % of the spray-rate steps)",
        [
            ("18", 100 * b2[channels_b1.index("inlet_temperature_celsius")]),
            ("35", 100 * b2[channels_b1.index("spray_rate_kg_per_h")]),
        ],
        "b1_pool_limits.json (ratelim_changed, plan b2)",
    )
    ranges55 = ratio55["ratio_ranges"]
    moist55 = [
        ranges55[f"product_moisture_kg_per_kg/{m}"][k]
        for m in ("gru", "lstm")
        for k in ("min", "max")
    ]
    report.claim(
        "by a factor of 2 to 3 over the whole budget range (GRU and LSTM, run-median NRMSE at horizon 36 on the constant-run holdout",
        [("2", min(moist55)), ("3", max(moist55)), ("36", 36)],
        "befund55_addendum_ratio_range.json",
    )
    report.truth(
        "on the constant-run holdout, legacy data configuration of arm A0",
        ratio55["test_set"] == "constant_holdout"
        and all(
            v["process/fingerprint"] == ['"2d2679eeed716577"']
            and v["data/new_study_version"] == ["null"]
            for v in ratio55["data_configuration"].values()
        ),
        "test set constant_holdout, legacy fingerprint, no preset version",
        "befund55_addendum_ratio_range.json",
    )
    report.claim(
        "for the LSTM at the largest budget it is not separable from the seed spread with five seeds, and the two training arms",
        [("five", len(seeds_n)), ("two", len(ratio55["sources"]))],
        "befund55_addendum_ratio_range.json (two record files, five seeds per arm)",
    )
    lstm64 = ratio55["table"]["product_moisture_kg_per_kg/lstm"]["64"]
    report.truth(
        "for the LSTM at the largest budget it is not separable from the seed spread",
        not lstm64["difference_exceeds_larger_seed_range"],
        "LSTM, n_train 64: difference of medians below the larger seed range",
        "befund55_addendum_ratio_range.json",
    )

    # ---- section 7 -------------------------------------------------------------------
    report.truth(
        "the critical moisture of 0.20 kg/kg",
        abs(config["material"]["critical_moisture_kg_per_kg"] - 0.20) < 1e-12,
        "material.critical_moisture_kg_per_kg of the process configuration",
        CONFIG.name,
    )
    report.truth(
        "the lower bound of the gas velocity (0.80 and 1.26 m/s)",
        abs(config["actions"]["gas_velocity_m_per_s"]["minimum"] - 0.80) < 1e-12
        and abs(dt20["gas_velocity_lower_bound_m_per_s"] - 1.26) < 1e-12,
        "0.80 m/s = actions.gas_velocity_m_per_s.minimum of the configuration; 1.26 m/s in the coupling-test artifact",
        f"{CONFIG.name}, m4r_test_dt20_umin126.json",
    )
    report.claim(
        "parameter counts that differ by a factor of about seven",
        [("seven", round(n_params["lstm"] / n_params["mamba"]))],
        "rerun_full_N.jsonl",
    )
    five("Five seeds. Large differences are resolved")
    report.claim(
        "A single timing run on a single machine (Befund 97); the networks used for the timing were fitted for two epochs only",
        [("two", timing["design"]["part_b"]["fit_epochs_networks"])],
        "joint_timing_idle.json (design)",
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
    st = stability["statements"]
    s1_changes = [
        f["relative_change"]
        for f in st["S1_adapting_the_operating_point_saves_energy"]["figures"].values()
    ]
    report.claim(
        "(its figures change by 74 to 87 %)",
        [("74", 100 * min(s1_changes)), ("87", 100 * max(s1_changes))],
        "m1_gas_bound_sensitivity_k1.json (S1 relative changes)",
    )
    report.truth(
        "saving itself is not stable between the two bounds",
        not st["S1_adapting_the_operating_point_saves_energy"]["stable"]
        and st["S2_the_margin_costs_energy"]["stable"]
        and st["S3_most_of_the_control_benefit_survives_the_margin"]["stable"],
        "S1 not stable, S2 and S3 stable",
        "m1_gas_bound_sensitivity_k1.json",
    )
    b126, b080 = m1["1.26"]["bounds"]["M0"], m1["0.80"]["bounds"]["M0"]
    report.claim(
        "lower bound at 22 (1.26 m/s) and 17 (0.80 m/s) of 25 test points",
        [
            ("22", b126["n_on_lower_gas_bound"]),
            ("17", b080["n_on_lower_gas_bound"]),
            ("25", b126["n"]),
        ],
        "m1_k1_comparison.json (rerun_k1, bounds M0)",
    )
    report.claim(
        "the moisture limit binds at 13 (1.26 m/s) and 25 (0.80 m/s) of 25 points",
        [("13", b126["n_binding_moisture"]), ("25", b080["n_binding_moisture"]), ("25", b126["n"])],
        "m1_k1_comparison.json (rerun_k1, bounds M0)",
    )
    report.claim(
        "The margin m (0.006 kg/kg) is the median half-width of the calibrated band",
        [("0.006", margin["m_kg_per_kg"])],
        "bres_margin.json (m_kg_per_kg)",
    )
    report.claim(
        "calibrated to 95 % coverage on validation runs",
        [("95", 100 * fa2["target_coverage"])],
        "verify_fa2_spread_calibration.json (target_coverage)",
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
    tested = sorted(int(k) for k in margin["cells"])
    report.claim(
        "the policy is feasible in every seed at five times the base margin, the smallest tested factor that meets the gate (factors 1, 2, 3 and 5 tested, 4 not)",
        [
            ("five", margin["gate_k"]),
            ("1", tested[0]),
            ("2", tested[1]),
            ("3", tested[2]),
            ("5", tested[3]),
            ("4", 4),
        ],
        "bres_margin.json (gate_k, cells)",
    )
    report.truth(
        "the smallest tested factor that meets the gate (factors 1, 2, 3 and 5 tested, 4 not)",
        tested == [1, 2, 3, 5]
        and margin["cells"]["5"]["feasible_in_every_seed"]
        and not any(margin["cells"][str(k)]["feasible_in_every_seed"] for k in (1, 2, 3)),
        "tested factors 1, 2, 3, 5; feasible in every seed only at 5",
        "bres_margin.json",
    )
    cell5 = margin["cells"]["5"]
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
        "heat_recovery_comparison_r4_consistent.json (variant (i), scenarios 1, 3, 5, 20/10/5 K)",
    )
    report.truth(
        "The saving read by the pre-registered rule is L1 (L2 only post hoc with a plain rule)",
        gate["gate"]["all_products"]["reading"] == "L1"
        and posthoc["gate_with_R1b_post_hoc"]["all_products"]["reading_with_R1b"] == "L2",
        "pre-registered gate L1 (all products), post-hoc rule R1b L2",
        "gate_storage_shift.json, gate_storage_shift_posthoc.json",
    )
    report.truth(
        "storage reads L2 by its pre-registered gate",
        timeslice["gate"]["verdict"]["reading"] == "L2",
        "time-slice gate reading L2",
        "gate_storage_timeslice.json",
    )

    # ---- appendix A ------------------------------------------------------------------
    report.truth(
        "Python 3.14.2, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.0 (CPU build), Windows 11; one thread per process",
        common("python") == "3.14.2"
        and common("version/numpy") == "2.5.3"
        and common("version/scipy") == "1.18.1"
        and common("version/torch") == "2.14.0+cpu"
        and common("platform").startswith("Windows-11")
        and common("torch/num_threads") == 1
        and env["python"] == "3.14.2"
        and env["torch"] == "2.14.0+cpu",
        "versions, platform and threads of every arm-N record and of the timing environment",
        "rerun_full_N.jsonl, joint_timing_idle.json (environment)",
    )
    report.truth(
        "AMD Ryzen 7 9700X with 16 logical CPUs",
        "Ryzen 7 9700X" in env["processor"] and env["logical_cpus"] == 16,
        f"processor {env['processor']!r}, logical CPUs {env['logical_cpus']}",
        "joint_timing_idle.json (environment)",
    )

    # ---- appendix B ------------------------------------------------------------------
    p = {m: params(m) for m in methods_n}
    rows_b = [
        f"| persistence | `{p['persistence']['model/model']}` | {n_params['persistence']} | last measured value held over the horizon |",
        f"| static regression | `{p['static']['model/model']}` | {n_params['static']} | quadratic in the current inputs, ridge {p['static']['model/ridge']!r} |",
        f"| ARX | `{p['arx']['model/model']}` | {n_params['arx']} | {p['arx']['model/n_lags_target']} target lags, {p['arx']['model/n_lags_exogenous']} input lags, ridge {p['arx']['model/ridge']!r} |",
    ]
    for method, label in (("gru", "GRU"), ("lstm", "LSTM")):
        q = p[method]
        rows_b.append(
            f"| {label} | `{q['model/model']}` | {n_params[method]:,} | hidden size {q['model/net_hidden_size']}, {q['model/net_num_layers']} layer |"
        )
    q = p["transformer"]
    rows_b.append(
        f"| Transformer | `{q['model/model']}` | {n_params['transformer']:,} | width {q['model/net_hidden_size']}, {q['model/net_num_layers']} layer, {q['model/net_num_heads']} heads |"
    )
    q = p["xlstm"]
    rows_b.append(
        f"| xLSTM | `{q['model/model']}` | {n_params['xlstm']:,} | mLSTM cells only, hidden size {q['model/net_hidden_size']}, {q['model/net_num_layers']} layer |"
    )
    q = p["mamba"]
    rows_b.append(
        f"| Mamba | `{q['model/model']}` | {n_params['mamba']:,} | hidden size {q['model/net_hidden_size']}, state size {q['model/net_state_size']}, {q['model/net_num_layers']} layer |"
    )
    for row in rows_b:
        report.truth(
            row,
            True,
            "row assembled from the model parameters of the arm-N records",
            "rerun_full_N.jsonl",
        )
    nets = [p[m] for m in ("gru", "lstm", "transformer", "xlstm", "mamba")]

    def same(key: str) -> Any:
        values = {json.dumps(n[key]) for n in nets}
        if len(values) != 1:
            raise SystemExit(f"{key} differs between the networks")
        return json.loads(next(iter(values)))

    torch_source = (REPO_ROOT / "src" / "twinlink_ml" / "models" / "torch_sequence.py").read_text(
        encoding="utf-8"
    )
    report.claim(
        "a composite of 3 independent heads, one per target; optimiser AdamW, learning rate 0.001, weight decay 0.0001, batch size 256, at most 200 epochs, early stopping with patience 20 on the validation runs, gradient clipping at 1.0",
        [
            ("3", same("model/composite/n_submodels")),
            ("one", 1),
            ("0.001", same("model/net_learning_rate")),
            ("0.0001", same("model/net_weight_decay")),
            ("256", same("model/net_batch_size")),
            ("200", same("model/net_max_epochs")),
            ("20", same("model/net_patience")),
            ("1.0", same("model/net_gradient_clip")),
        ],
        "rerun_full_N.jsonl (model parameters of the five networks)",
    )
    report.claim(
        "Common to the five networks",
        [("five", sum(1 for m in methods_n if "model/net_core" in p[m]))],
        "rerun_full_N.jsonl (model/net_core)",
    )
    report.truth(
        "optimiser AdamW",
        "optim.AdamW(" in torch_source,
        "TorchSequenceModel uses torch.optim.AdamW",
        "src/twinlink_ml/models/torch_sequence.py",
    )
    report.truth(
        "no dropout, no teacher forcing, equal weight of all horizon steps",
        same("model/net_dropout") == 0.0
        and same("model/net_teacher_forcing") == 0.0
        and same("model/net_horizon_weight_decay") == 1.0,
        "dropout 0, teacher forcing 0, horizon weight decay 1.0",
        "rerun_full_N.jsonl",
    )
    report.claim(
        "input windows of 32 past steps, prediction of 36 steps (900 s) ahead, window stride 8",
        [
            ("32", common("data/history_length")),
            ("36", common("data/horizon")),
            ("900", common("data/horizon") * sample_time),
            ("8", common("data/stride")),
        ],
        "rerun_full_N.jsonl (data/*)",
    )

    # ---- appendix C ------------------------------------------------------------------
    act = config["actions"]
    dist = config["disturbances"]
    report.claim(
        "Manipulated variables (4), box and rate limit: inlet temperature 60 to 120 degC (0.5 K/s), gas velocity 0.8 to 2.5 m/s (0.05 m/s per s), spray rate 0.4 to 3.0 kg/h (0.02 kg/h per s), nozzle pressure 1.0 to 4.0 bar (0.05 bar/s)",
        [
            ("4", len(simulator.action_names)),
            ("60", act["inlet_temperature_celsius"]["minimum"]),
            ("120", act["inlet_temperature_celsius"]["maximum"]),
            ("0.5", act["inlet_temperature_celsius"]["max_rate_per_s"]),
            ("0.8", act["gas_velocity_m_per_s"]["minimum"]),
            ("2.5", act["gas_velocity_m_per_s"]["maximum"]),
            ("0.05", act["gas_velocity_m_per_s"]["max_rate_per_s"]),
            ("0.4", act["spray_rate_kg_per_h"]["minimum"]),
            ("3.0", act["spray_rate_kg_per_h"]["maximum"]),
            ("0.02", act["spray_rate_kg_per_h"]["max_rate_per_s"]),
            ("1.0", act["nozzle_pressure_bar"]["minimum"]),
            ("4.0", act["nozzle_pressure_bar"]["maximum"]),
            ("0.05", act["nozzle_pressure_bar"]["max_rate_per_s"]),
        ],
        f"{CONFIG.name} (actions)",
    )
    report.claim(
        "Disturbances (3): ambient temperature 5 to 35 degC, ambient relative humidity 0.25 to 0.85, solid fraction of the suspension 0.20 to 0.40 (preset version 2: own disturbance process)",
        [
            ("3", len(simulator.disturbance_names)),
            ("5", dist["ambient_temperature_celsius"]["minimum"]),
            ("35", dist["ambient_temperature_celsius"]["maximum"]),
            ("0.25", dist["ambient_relative_humidity"]["minimum"]),
            ("0.85", dist["ambient_relative_humidity"]["maximum"]),
            ("0.20", dist["suspension_solid_fraction"]["minimum"]),
            ("0.40", dist["suspension_solid_fraction"]["maximum"]),
            ("2", common("data/new_study_version")),
        ],
        f"{CONFIG.name} (disturbances), rerun_full_N.jsonl",
    )
    report.claim(
        "States (4): solids mass, water mass, particle number, bed temperature. Outputs (4): energy target dE (kW), bed temperature, product moisture (kg/kg) and median particle diameter; the first three are the targets of the comparison",
        [("4", n_states), ("4", len(simulator.output_names)), ("three", len(targets))],
        "GreyBoxFluidBedGranulator, rerun_full_N.jsonl",
    )
    report.claim(
        "sample time 25 s (set by the data specification `DataSpec`, overriding the 5 s of the YAML configuration), integration with a fourth-order Runge-Kutta scheme and 5 substeps per sample; a run lasts 28,800 s (8 h); the prediction horizon is 36 steps (900 s), the history 32 steps",
        [
            ("25", sample_time),
            ("5", config["numerics"]["sample_time_s"]),
            ("5", config["numerics"]["substeps"]),
            ("28,800", common("data/run_duration_s")),
            ("8", common("data/run_duration_s") / 3600),
            ("36", common("data/horizon")),
            ("900", common("data/horizon") * sample_time),
            ("32", common("data/history_length")),
        ],
        f"{CONFIG.name} (numerics), rerun_full_N.jsonl",
    )
    report.truth(
        "fourth-order Runge-Kutta scheme",
        config["numerics"]["integrator"] == "rk4",
        "integrator rk4",
        CONFIG.name,
    )
    exc = ExcitationConfig()
    weights = exc.signal_weights
    report.claim(
        "a step, PRBS or ramp signal (weights 0.40, 0.30, 0.30), amplitudes 15, 35, 70 and 100 % of the span",
        [
            ("0.40", weights["step"]),
            ("0.30", weights["prbs"]),
            ("0.30", weights["ramp"]),
            ("15", 100 * exc.amplitude_levels[0]),
            ("35", 100 * exc.amplitude_levels[1]),
            ("70", 100 * exc.amplitude_levels[2]),
            ("100", 100 * exc.amplitude_levels[3]),
        ],
        "twinlink_ml.simulation.excitation.ExcitationConfig defaults",
    )
    report.truth(
        "Sobol sampling of the levels, randomised initial conditions, rate limits applied",
        exc.sampler == "sobol" and exc.randomise_initial_conditions and exc.apply_rate_limits,
        "ExcitationConfig defaults",
        "twinlink_ml.simulation.excitation",
    )
    report.claim(
        "10 % of the runs are constant. Constant-run holdout: 32 runs with constant set points",
        [("10", 100 * common("data/constant_run_fraction")), ("32", common("holdout/n_runs"))],
        "rerun_full_N.jsonl (data/constant_run_fraction, holdout/n_runs)",
    )
    report.claim(
        "up to 64 training, 32 validation and 32 test runs per seed",
        [
            ("64", common("data/n_train_max")),
            ("32", common("data/n_val")),
            ("32", common("data/n_test")),
        ],
        "rerun_full_N.jsonl",
    )
    report.truth(
        "every parameter of the process configuration is a placeholder",
        config["provenance"]["status"] == "placeholder",
        "provenance.status placeholder",
        CONFIG.name,
    )

    # ---- appendix D ------------------------------------------------------------------
    report.claim(
        "data need (arm N, horizon 36, median over five seeds)",
        [("36", 36), ("five", len(seeds_n))],
        "rerun_full_N.jsonl",
    )
    report.claim(
        "the data need for a given error beyond 64 runs is not determined by these data",
        [("64", max(n_set))],
        "befund94 addendum (falls_32_to_64)",
    )
    report.claim(
        "Still falling at 64 runs:", [("64", max(n_set))], "befund94 addendum (falls_32_to_64)"
    )
    report.claim(
        "the data need for a given error beyond 64 runs is therefore not determined",
        [("64", max(n_set))],
        "befund94 addendum (falls_32_to_64)",
    )
    report.claim(
        "Error at 8, 16 and 32 training runs relative to the error at 64 runs",
        [("8", n_set[0]), ("16", n_set[1]), ("32", n_set[2]), ("64", n_set[3])],
        "rerun_full_N.jsonl (n_train)",
    )
    labels_d = {
        "energy_target": "energy target",
        "product_moisture_kg_per_kg": "product moisture",
        "bed_temperature_celsius": "bed temperature",
    }
    for channel, label in labels_d.items():
        cells = []
        for m in ("gru", "lstm"):
            for t in both:
                c = curves[f"{m}/{channel}/{t}"]
                cells.append(
                    " / ".join(f"{c[n]['median_over_median_at_64']:.2f}" for n in ("8", "16", "32"))
                )
        report.truth(
            f"| {label} | " + " | ".join(cells) + " |",
            True,
            f"{label}: ratios to the error at 64 runs equal the artifact",
            "befund94_addendum_seeds_data_need_rerun_full.json",
        )
    report.claim(
        "product moisture 7.08e-4 / 6.17e-4 kg/kg on the excited set and 6.71e-4 / 5.43e-4 kg/kg on the constant-run holdout (persistence 1.18e-2 and 1.62e-3 kg/kg)",
        [
            ("7.08e-4", curves["gru/product_moisture_kg_per_kg/excited"]["64"]["median"]),
            ("6.17e-4", curves["lstm/product_moisture_kg_per_kg/excited"]["64"]["median"]),
            ("6.71e-4", curves["gru/product_moisture_kg_per_kg/constant_holdout"]["64"]["median"]),
            ("5.43e-4", curves["lstm/product_moisture_kg_per_kg/constant_holdout"]["64"]["median"]),
            ("1.18e-2", curves["persistence/product_moisture_kg_per_kg/excited"]["64"]["median"]),
            (
                "1.62e-3",
                curves["persistence/product_moisture_kg_per_kg/constant_holdout"]["64"]["median"],
            ),
        ],
        "befund94_addendum_seeds_data_need_rerun_full.json (curves, n_train 64)",
    )
    report.claim(
        "Absolute errors at 64 runs (GRU / LSTM)", [("64", max(n_set))], "rerun_full_N.jsonl"
    )
    report.claim(
        "in a paired test of the error at 64 against 32 runs, GRU and LSTM are lower at 64 runs in all 12 cells uncorrected, in 2 of them after Holm (family of 48 tests)",
        [
            ("64", 64),
            ("32", 32),
            ("12", sum(c["falls_uncorrected"] for c in gl_falls)),
            ("12", len(gl_falls)),
            ("2", sum(c["falls_holm"] for c in gl_falls)),
            ("48", need94["part_b_data_need"]["falls_32_to_64_summary"]["tests"]),
        ],
        "befund94_addendum_seeds_data_need_rerun_full.json (falls_32_to_64)",
    )
    report.claim(
        "training runs plus 32 validation runs for early stopping (appendix C); each run is 8 h of simulated operation",
        [("32", runs["n_val"]), ("8", runs["run_duration_s"] / 3600)],
        "befund94 addendum (runs_behind_each_point)",
    )
    five("Caveats: simulator with placeholder parameters; fixed, untuned protocol; five seeds")

    # ---- appendix E ------------------------------------------------------------------
    report.claim(
        "Median ratio at least 1.5 (or at most 1/1.5) with all seeds on that side",
        [("1.5", threshold), ("1", 1), ("1.5", threshold)],
        "befund98_addendum_effect_sizes_rerun_full.json (large_ratio_threshold)",
    )

    # ---- exemptions: design constants and plan facts, each with its reason ---------------
    docs01 = (REPO_ROOT / "docs" / "01_projektsteckbrief.md").read_text(encoding="utf-8")
    register = (REPO_ROOT / "docs" / "12_review_2026-10-09.md").read_text(encoding="utf-8")
    labbook = (REPO_ROOT / "docs" / "04_stand_ap1.md").read_text(encoding="utf-8")
    eq1 = next((line for line in register.splitlines() if "| E-Q1*" in line), "")
    b56 = labbook[labbook.index("### 56.") : labbook.index("### 57.")]
    review_file = REPO_ROOT / "docs" / "reviews" / "2026-10-19" / "review4_synthesis_v5.md"
    versions = {r["params"]["data/new_study_version"] for r in n_records}
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
            versions == {2},
            "records of arm N carry data/new_study_version 2",
        ),
        ("Preset version 2", versions == {2}, "records of arm N carry data/new_study_version 2"),
        (
            "N the preset version 2",
            versions == {2},
            "records of arm N carry data/new_study_version 2",
        ),
        ("whose curve still falls at 64", 64 in n_set, "largest n_train of the N records"),
        (
            "time constant 600 s",
            config["operation"]["mass_control_time_constant_s"] == 600.0,
            f"operation.mass_control_time_constant_s of {CONFIG.name}",
        ),
        (
            "planned for month 21",
            bool(re.search(r"Dyssol-Granulationsmodelle[^\n]*~?M21", docs01)),
            "docs/01: the Dyssol granulation models are planned for month M21",
        ),
        (
            "horizon 36 (900 s)",
            abs(36 * sample_time - 900) < 1e-9,
            "36 steps x sample time 25 s from the records",
        ),
        (
            "The M2 definition of 10 to 15 %",
            bool(re.search(r"10\s*(?:bis|-|–)\s*15", docs01)),
            "docs/01 note",
        ),
        (
            "two 24-hour exports, two products, one-minute sampling",
            True,
            "NOT CHECKABLE: raw plant data are never in the repository (CLAUDE.md)",
        ),
        (
            "of only 0.12 to 2.01 %",
            "0,12" in b56 and "2,01" in b56,
            "NOT CHECKABLE against an artifact (raw plant data); compared with the text of Befund 56",
        ),
        (
            "nozzle-pressure kink at 1 bar",
            "1.0 bar" in eq1 and config["actions"]["nozzle_pressure_bar"]["minimum"] == 1.0,
            "register row E-Q1 (kink at 1.0 bar) and nozzle_pressure_bar.minimum of the configuration",
        ),
        (
            "revised after review 4",
            review_file.exists(),
            "review 4 report in docs/reviews/2026-10-19",
        ),
    ]
    exempt_results = []
    for literal, verified, reason in exemptions:
        present = normalise(literal) in text
        exempt_results.append(
            {"exemption": literal, "in_text": present, "verified": bool(verified), "reason": reason}
        )

    # ---- coverage ----------------------------------------------------------------------
    pieces = []
    for start_head, end_head in (
        ("## Summary", "## Changes from version 5"),
        ("## 1. Question and frame", "## Appendix F"),
    ):
        pieces.append(text[text.index(start_head) : text.index(end_head)])
    body = " ".join(pieces)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
    for fragment in sorted(report.fragments, key=len, reverse=True):
        body = body.replace(fragment, " ")
    for literal, _, _ in sorted(exemptions, key=lambda e: -len(e[0])):
        body = body.replace(normalise(literal), " ")
    body = re.sub(r"`[^`]*`", " ", body)
    allowed_horizons = set(horizons_n) | {
        int(re.search(r"\d+", str(b1["primary"].get("horizon", b1["primary"]))).group())
    }
    allowed_n_study_a = set(a1["design"]["sizes"])
    bad_references: list[str] = []

    def verified(pattern: str, allowed: set[int]) -> None:
        nonlocal body

        def check(match: re.Match[str]) -> str:
            numbers = [int(x) for x in re.findall(r"\d+", match.group())]
            if all(n in allowed for n in numbers):
                return " "
            bad_references.append(match.group())
            return match.group()

        body = re.sub(pattern, check, body)

    verified(r"\bhorizons? \d+(?: and \d+)?", allowed_horizons)
    verified(r"\$N\$ = \d+", allowed_n_study_a)
    patterns = [
        r"Befunde?\s+\d+[a-z]?(?:\s*(?:,|and|/|to|-)\s*\d+[a-z]?)*(?:\s+(?:addendum\s+(?:[a-z]|\d)\b|[a-z]\b))?",
        r"\bsections?\s+\d+(?:\s+(?:to|and)\s+\d+)?",
        r"\bFigures?\s+\d+(?:\s+(?:and|to)\s+\d+)?",
        r"docs/\d+",
        r"\b[A-Za-z]{1,3}-?\d+[a-z]?\b",
        r"R\^2",
        r"##\s+\d+\.",
        r"\(\d\)",
        r"\bAP\d\b",
    ]
    for pattern in patterns:
        body = re.sub(pattern, " ", body)
    digits = {m.group() for m in re.finditer(r"\d[\d,\.]*\d|\d", body)}
    words = {m.group().lower() for m in WORD_PATTERN.finditer(body)}
    leftover = sorted(digits | words)
    context = {}
    for tok in leftover:
        rx = (
            re.compile(r"\b" + re.escape(tok) + r"\b", re.IGNORECASE)
            if tok.isalpha()
            else re.compile(re.escape(tok))
        )
        context[tok] = [body[max(0, m.start() - 40) : m.end() + 25] for m in rx.finditer(body)]

    n_conf = sum(r["status"] == "confirmed" for r in report.results)
    n_dev = sum(r["status"] == "DEVIATING" for r in report.results)
    unverified = [e for e in exempt_results if not (e["in_text"] and e["verified"])]
    print(f"claims: {len(report.results)} checked, {n_conf} confirmed, {n_dev} deviating")
    print(
        f"exemptions: {len(exempt_results)} declared, {len(unverified)} not in text or not verified"
    )
    print(f"verified references outside their allowed values: {bad_references}")
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
                "bad_references": bad_references,
                "uncovered_numeric_tokens": leftover,
                "summary": {
                    "claims": len(report.results),
                    "confirmed": n_conf,
                    "deviating": n_dev,
                    "exemptions": len(exempt_results),
                    "uncovered": len(leftover),
                },
            },
        )
    return 0 if n_dev == 0 and not leftover and not unverified and not bad_references else 1


if __name__ == "__main__":
    raise SystemExit(main())
