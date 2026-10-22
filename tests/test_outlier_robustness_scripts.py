"""Outlier robustness study (docs/27): driver plan, budget cuts, pre-registered reading rules."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_outlier_robustness as ana  # noqa: E402
import run_outlier_robustness as driver  # noqa: E402

B, W, U = ana.BETTER, ana.WORSE, ana.UNRESOLVED


def test_full_plan_and_the_two_budget_cuts():
    full = driver.build_jobs(
        seeds=[1, 2, 3, 4, 5], bursts_only_at_025=False, skip_window_check=False
    )
    conditions = {j.condition.id for j in full}
    assert len(conditions) == 10 + 4
    assert "p0_none_none" in conditions and "p0_none_hampel8" in conditions
    assert "p0.05_burst_hampel8" in conditions
    assert {"p0.25_independent_hampel4", "p0.25_burst_hampel16"} <= conditions
    assert len(full) == 10 * 5 * 3 + 4 * 5
    assert full[0].tier == 1 and full[0].models == ("transformer",) and full[0].sizes == (64,)
    assert all(j.models == ("gru",) for j in full if j.tier == 3)
    cut1 = driver.build_jobs(
        seeds=[1, 2, 3, 4, 5], bursts_only_at_025=True, skip_window_check=False
    )
    assert not any(j.condition.id.startswith("p0.05_burst") for j in cut1)
    assert len(cut1) == 8 * 5 * 3 + 4 * 5
    cut2 = driver.build_jobs(seeds=[1, 2, 3, 4, 5], bursts_only_at_025=True, skip_window_check=True)
    assert len(cut2) == 8 * 5 * 3
    tiers = [j.tier for j in cut2]
    assert tiers == sorted(tiers)


def test_jobs_cover_every_model_budget_and_test_set():
    jobs = driver.build_jobs(seeds=[1], bursts_only_at_025=False, skip_window_check=True)
    per_condition: dict[str, set[tuple[str, int]]] = {}
    for job in jobs:
        per_condition.setdefault(job.condition.id, set()).update(
            (m, n) for m in job.models for n in job.sizes
        )
    for covered in per_condition.values():
        assert covered == {(m, n) for m in ana.MODELS for n in ana.SIZES}
    job = next(j for j in jobs if j.models != ("transformer",))
    assert job.expected_records == 4 * 2 * 2
    command = job.command(Path("x.jsonl"))
    assert command[command.index("--condition") + 1] == job.condition.id


def test_seventy_two_rank_cells_per_condition():
    assert len(ana.PAIRS) == 9 and len(ana.cells()) == 8


def _rows(verdicts: dict[tuple[str, str], str]) -> list[dict]:
    rows = []
    for cand, ref in ana.PAIRS:
        for n, t, h in ana.cells():
            v = verdicts.get((cand, ref), U)
            rows.append(
                {
                    "candidate": cand,
                    "reference": ref,
                    "n_train": n,
                    "test_set": t,
                    "horizon": h,
                    "block": "net_order" if (cand, ref) in ana.NET_ORDER else "nets_vs_baselines",
                    "verdict_holm": v,
                    "verdict_uncorrected": v,
                }
            )
    return rows


REFERENCE = {
    ("transformer", "lstm"): W,
    ("transformer", "gru"): W,
    **dict.fromkeys(ana.NETS_VS_BASELINES, W),
}


def test_identical_verdicts_are_unchanged():
    out = ana.compare_to_reference(_rows(REFERENCE), _rows(REFERENCE), "verdict_holm")
    assert out["unchanged"] and out["lead"] == 48 and not out["lead_vanished"]


def test_reversal_and_new_net_order_cell_change_the_ranking():
    reversed_ = {**REFERENCE, ("transformer", "lstm"): B}
    assert not ana.compare_to_reference(_rows(reversed_), _rows(REFERENCE), "verdict_holm")[
        "criteria"
    ]["a_no_reversal"]
    new_order = {**REFERENCE, ("gru", "lstm"): B}
    out = ana.compare_to_reference(_rows(new_order), _rows(REFERENCE), "verdict_holm")
    assert not out["unchanged"] and not out["criteria"]["c_net_order_no_new_and_at_most_half_lost"]


def test_lead_halving_rule_and_vanishing():
    half = {k: v for k, v in REFERENCE.items() if k[0] != "persistence"}  # 24 of 48 remain
    assert ana.compare_to_reference(_rows(half), _rows(REFERENCE), "verdict_holm")["unchanged"]
    less = {k: v for k, v in half.items() if k != ("arx", "gru")}  # 16 of 48
    out = ana.compare_to_reference(_rows(less), _rows(REFERENCE), "verdict_holm")
    assert not out["criteria"]["d_lead_not_halved"]
    none = {k: v for k, v in REFERENCE.items() if k in ana.NET_ORDER}
    assert ana.compare_to_reference(_rows(none), _rows(REFERENCE), "verdict_holm")["lead_vanished"]


def _cmp(unchanged: bool, vanished: bool = False) -> dict:
    return {
        "unchanged": unchanged,
        "criteria": {"x": unchanged},
        "lead": 0 if vanished else 10,
        "lead_vanished": vanished,
    }


def test_readings_r1_r2_r3_none():
    def run(n, f):
        comparisons = {
            "p0.25_independent_none": n,
            "p0.25_independent_hampel8": f,
            "p0.25_burst_none": n,
            "p0.25_burst_hampel8": f,
        }
        return ana.readings(comparisons, lead_reference=40)["independent"]["reading"]

    assert run(_cmp(True), _cmp(True)) == "R1"
    assert run(_cmp(False), _cmp(True)) == "R2"
    assert run(_cmp(False, True), _cmp(False, True)) == "R3"
    assert run(_cmp(True), _cmp(False, True)) == "R3"
    assert run(_cmp(False), _cmp(False)) == "none"
    assert run(_cmp(True), _cmp(False)) == "none"
    comparisons = {
        "p0.25_independent_none": _cmp(True),
        "p0.25_independent_hampel8": _cmp(True),
        "p0.25_burst_none": _cmp(True),
        "p0.25_burst_hampel8": _cmp(True),
    }
    assert ana.readings(comparisons, lead_reference=0)["burst"]["reading"] is None


def _synthetic(conditions: list[str], side: list[str]) -> tuple[list[dict], list[dict]]:
    """Records of a complete run and matching Befund 94 records; networks better than baselines."""
    import random

    rng = random.Random(0)
    level = {"persistence": 3.0, "arx": 2.0, "gru": 1.0, "lstm": 1.05, "transformer": 1.5}
    records, old = [], []
    for condition in conditions + side:
        models = ("gru",) if condition in side else ana.MODELS
        for model in models:
            for n in ana.SIZES:
                for s in ana.SEEDS:
                    for t in ana.TEST_SETS:
                        metrics, b94 = {}, {}
                        for h in ana.HORIZONS:
                            value = level[model] * (1 + 0.02 * rng.random()) * 1e-3
                            for metric in (ana.MAIN_METRIC, *ana.SECONDARY_METRICS):
                                metrics[ana.metric_key(metric, h)] = value
                            b94[ana.metric_key(ana.B94_METRIC, h)] = value
                        params = {
                            "condition/id": condition,
                            "model/label": model,
                            "n_train": n,
                            "master_seed": s,
                            "test_set": t,
                            "git/dirty": False,
                            "git/commit": "abc",
                            "outlier_config/digest": "d",
                            "preset/new_study_version": 2,
                            "process/fingerprint": ana.PROCESS_FINGERPRINT,
                            "pool/projected_steps": 0,
                            "holdout_pool/projected_steps": 0,
                            "pool/outlier/realised_rate": 0.0,
                            "holdout_pool/outlier/realised_rate": 0.0,
                            "pool/preprocess/replaced_fraction": 0.0,
                            "holdout_pool/preprocess/replaced_fraction": 0.0,
                            f"training/epochs_run/{ana.CHANNEL}": 10,
                        }
                        records.append({"params": params, "metrics": metrics})
                        if condition == ana.REFERENCE:
                            old.append(
                                {
                                    "params": {**params, "eval/primary": True},
                                    "metrics": b94,
                                }
                            )
    return records, old


def test_evaluate_end_to_end_on_synthetic_records():
    jobs = driver.build_jobs(
        seeds=list(ana.SEEDS), bursts_only_at_025=True, skip_window_check=False
    )
    conditions = sorted({j.condition.id for j in jobs})
    side = [c for c in conditions if c.endswith(("hampel4", "hampel16"))]
    main = [c for c in conditions if c not in side]
    records, old = _synthetic(main, side)
    result = ana.evaluate(records, old, main, side)
    assert result["gate"]["passed"], result["gate"]["failures"][:5]
    assert result["readings"]["independent"]["reading"] == "R1"
    assert result["readings"]["burst"]["reading"] == "R1"
    assert len(result["rank_cells"][ana.REFERENCE]) == 72
    assert set(result["side_check"]) == set(side)
    broken = [
        dict(r, params={**r["params"], "git/dirty": True}) if i == 0 else r
        for i, r in enumerate(records)
    ]
    assert not ana.evaluate(broken, old, main, side)["gate"]["passed"]
