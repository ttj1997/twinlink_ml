r"""Storage test 1 (Befund 99), RE-CHECK: exhaust humidity of plant A with an admissible solid fraction.

Why. Befund 99 reconstructs the exhaust humidity of plant A from the spray rate with the solid
fraction 0.3 (placeholder, band 0.2-0.5). The operating-point study of plant A (branch
``plant-a-operating-point``, ``docs/23_plant_a_operating_point.md`` section 5, artifact
``study_plant_a_operating_point``) shows that the dryer energy balance of the SAME two measured
days closes with a non-negative wall loss only for a solid fraction of at least 0.575 (product 1)
and 0.625 (product 2); one value consistent with both days is 0.634. The value 0.3 is therefore
inadmissible. This script checks whether any Befund 99 number moves when the solid fraction is
admissible. It does NOT change Befund 99, its scripts or its artifacts (they are read only).

What is re-run. Exactly the pre-registered computation of ``scripts/gate_storage_timeslice.py``
(same code, grids, capacities, levels, gate definitions, decomposition rule, sensitivity
variants) and the post-hoc references R3/R4 of ``scripts/analyze_storage_timeslice_sensitivity.py``,
each with ONE change: the solid fraction of every scenario is set to ``x_s`` in
{0.60, 0.634, 0.70} (0.60 is admissible on product 1 only; reported, not dropped). The solid
fraction enters only the reconstructed exhaust humidity of plant A (and through it the exhaust
state used for plant C). The sensitivity variants ``solid_fraction=0.2/0.5`` are dropped (the
comparison across the three values replaces them).

Gate realism. Befund 99 counts a scenario only when all placeholders lie in their bands; the
solid-fraction band 0.2-0.5 is what this re-check replaces, so here a scenario counts when all
OTHER placeholders lie in their Befund 99 bands (``gate_cells(..., realistic=...)``). Nothing else
of the gate changes.

Control. A subset of Befund 99's runs (every 20th grid scenario and the bases R1-R4) is re-run at
the old value 0.3 and compared with the Befund 99 artifacts (largest absolute difference of
``E_direct`` and ``E_ideal``); the headline numbers of docs/19 are recomputed from the Befund 99
grid with the same summary function that is applied to the new runs, so the comparison is
like for like.

Call::

    TWINLINK_PLANT_DATA_PROD1=... TWINLINK_PLANT_DATA_PROD2=... \
        tools/run_here.sh scripts/recheck_storage_timeslice_solid_fraction.py \
        --solid-fraction 0.634 --name recheck_storage_timeslice_solid_fraction_0634

and the same with ``--solid-fraction 0.6 --skip-control`` / ``0.7 --skip-control`` (names
``..._060`` / ``..._070``); one artifact per value, so a long run keeps what it has.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_storage_timeslice_sensitivity as posthoc
import gate_storage_timeslice as gate
from _common import NO_HEAT_NETWORK, RUNS_DIR, provenance, require_clean_tree, write_artifact

from twinlink_ml.storage_timeslice import (
    CAPACITIES_KWH,
    MeasuredDay,
    Placeholders,
    Scenario,
    ScenarioResult,
    StreamFactory,
)
from twinlink_ml.storage_timeslice.config import BAND_CAPACITY_MAX_KWH, is_realistic
from twinlink_ml.storage_timeslice.gate import (
    GATE_SAVING,
    gate_cells,
    gate_verdict,
    rank_sensitivity,
    subset_verdicts,
)
from twinlink_ml.storage_timeslice.plant import exhaust_humidity

SOLID_FRACTIONS = (0.60, 0.634, 0.70)
BEFUND_99_SOLID_FRACTION = 0.3
BEFUND_99_GATE = "gate_storage_timeslice"
BEFUND_99_POSTHOC = "gate_storage_timeslice_posthoc_sensitivity"
CONTROL_STRIDE = 20
DROPPED_VARIANTS = ("solid_fraction=0.2", "solid_fraction=0.5")
ADMISSIBLE_FROM = {"product_1": 0.575, "product_2": 0.625}
"""Lower bound of the solid fraction by the dryer energy balance (docs/23, section 5)."""

# scenarios of the table in docs/19 section 6 (labels of the main grid)
TABLE_LABELS = {
    "max_fraction": "A|product_2|day|shift|shift-10K|compA1200|hall200|water0|loss0",
    "max_kwh": "A+C|product_2|week|shift|shift-10K|compA800|hall0|water0|loss0",
    "R1": "A+C|product_1|day|shift|shift+0K|compA800|hall200|water200|loss0",
    "R2": "A+C|product_1|week|weekly|shift+0K|compA800|hall200|water200|loss0",
    **{
        f"closest_{day}_comp{level:g}": f"A|{day}|day|flat|shift+0K|compA{level:g}|hall0|water0|loss0"
        for day in ("product_1", "product_2")
        for level in gate.COMPRESSOR_LEVELS_KW
    },
}


def realistic_except_solid_fraction(scenario: Scenario) -> bool:
    """Befund 99 bands for every placeholder except the solid fraction (the re-checked one)."""
    return is_realistic(scenario.with_placeholders(solid_fraction=Placeholders().solid_fraction))


def with_solid_fraction(scenario: Scenario, x_s: float) -> Scenario:
    return scenario.with_placeholders(solid_fraction=x_s)


# ------------------------------------------------------------------------- summary of a grid


def _index(capacity_kwh: float) -> int:
    return CAPACITIES_KWH.index(capacity_kwh)


def headline(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The numbers of docs/19 sections 1, 5, 6 that are computed from the main grid.

    ``records`` are grid records as written by ``gate_storage_timeslice.record_of`` (field
    ``realistic`` decides membership); applied identically to Befund 99 and to the re-check.
    """
    cell_idx = [i for i, c in enumerate(CAPACITIES_KWH) if 0.0 < c <= BAND_CAPACITY_MAX_KWH]
    i5 = _index(5000.0)

    def best(r: dict[str, Any]) -> float:
        return max(r["saving_fraction"][i] for i in cell_idx)

    rows = [r for r in records if r["realistic"]]
    hit = [r for r in rows if best(r) >= GATE_SAVING - 1e-9]
    count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in hit:
        for key in ("day", "site", "mode", "horizon", "hall_kw", "water_kw", "loss_per_h"):
            count[key][str(r[key])] += 1
        count["ambient_shift_k"][f"{r['ambient_shift_k']:+g}"] += 1
        if r["site"] != "C":
            count["compressor_a_kw_sites_A_and_A+C"][f"{r['compressor_a_kw']:g}"] += 1

    def max_where(pred: Callable[[dict[str, Any]], bool]) -> float | None:
        values = [best(r) for r in rows if pred(r)]
        return max(values) if values else None

    by_label = {r["label"]: r for r in records}
    # loss effect: day horizon, 5 MWh, loss 0 against 0.5 %/h of the same scenario
    loss_delta = []
    for r in rows:
        if r["horizon"] == "day" and r["loss_per_h"] == 0.0:
            other = by_label.get(r["label"].replace("|loss0", "|loss0.005"))
            if other is not None:
                loss_delta.append(r["saving_kwh_per_day"][i5] - other["saving_kwh_per_day"][i5])
    # day against week horizon in mode shift, 5 MWh (kWh/d)
    dw = []
    for r in rows:
        if r["horizon"] == "day" and r["mode"] == "shift":
            other = by_label.get(r["label"].replace("|day|shift|", "|week|shift|"))
            if other is not None:
                dw.append(abs(r["saving_kwh_per_day"][i5] - other["saving_kwh_per_day"][i5]))
    table = {}
    for tag, label in TABLE_LABELS.items():
        r = by_label.get(label)
        if r is None:
            continue
        table[tag] = {
            "label": label,
            "direct_kwh_per_day": r["direct_kwh_per_day"],
            "saving_kwh_per_day": dict(
                zip([f"{c:g}" for c in CAPACITIES_KWH], r["saving_kwh_per_day"], strict=True)
            ),
            "saving_fraction": dict(
                zip([f"{c:g}" for c in CAPACITIES_KWH], r["saving_fraction"], strict=True)
            ),
            "capacity_at_90_percent_of_max_saving_kwh": r[
                "capacity_at_90_percent_of_max_saving_kwh"
            ],
            "max_rule_gap_fraction": max(r["gap_fraction"]),
            "ideal_level_celsius_5MWh": r["ideal_level_celsius"][i5],
        }
    return {
        "n_grid_scenarios": len(records),
        "n_grid_scenarios_counted": len(rows),
        "n_scenarios_with_cell_saving_at_least_5_percent": len(hit),
        "share_scenarios_with_cell_saving_at_least_5_percent": len(hit) / len(rows)
        if rows
        else None,
        "count_by": {k: dict(sorted(v.items())) for k, v in count.items()},
        "max_saving_fraction_product_1": max_where(lambda r: r["day"] == "product_1"),
        "max_saving_fraction_product_2": max_where(lambda r: r["day"] == "product_2"),
        "max_saving_fraction_mode_flat": max_where(lambda r: r["mode"] == "flat"),
        "max_saving_fraction_closest_A_flat_0K_no_hall_no_water_loss0": max_where(
            lambda r: r["site"] == "A"
            and r["mode"] == "flat"
            and r["hall_kw"] == 0.0
            and r["water_kw"] == 0.0
            and r["ambient_shift_k"] == 0.0
            and r["loss_per_h"] == 0.0
        ),
        "loss_effect_day_5MWh_kwh_per_day": {
            "mean": float(np.mean(loss_delta)) if loss_delta else None,
            "max": float(np.max(loss_delta)) if loss_delta else None,
            "n_pairs": len(loss_delta),
        },
        "day_vs_week_shift_5MWh_max_abs_difference_kwh_per_day": float(np.max(dw)) if dw else None,
        "table": table,
    }


# ------------------------------------------------------------------------- runs per solid fraction


def day_state(days: dict[str, MeasuredDay], x_s: float) -> dict[str, Any]:
    """Reconstructed exhaust state of plant A and the exhaust duties at ``x_s``."""
    ph = Placeholders(solid_fraction=x_s)
    factory = StreamFactory(days)
    out: dict[str, Any] = {}
    for label, day in days.items():
        y_out, _, capped = exhaust_humidity(
            day,
            solid_fraction=x_s,
            relative_humidity=ph.relative_humidity,
            normal_density_kg_per_nm3=ph.normal_density_kg_per_nm3,
            gas=ph.gas,
        )
        notes = factory.build(with_solid_fraction(Scenario("A+C", label, "day", "flat"), x_s)).notes
        out[label] = {
            "solid_fraction": x_s,
            "admissible_by_energy_balance": x_s >= ADMISSIBLE_FROM[label],
            "exhaust_humidity_reconstructed_mean_kg_per_kg": float(np.mean(y_out)),
            "exhaust_capped_at_saturation_fraction": float(np.mean(capped)),
            "exhaust_a_mean_kw_down_to_T_kalt": notes["exhaust_a_mean_kw"],
            "exhaust_c_flat_mean_kw_volume_basis_two_lines": notes[
                "exhaust_c_flat_mean_kw_volume_basis"
            ],
        }
    return out


def variants_of(base: Scenario, swap_day: bool = False) -> dict[str, Scenario]:
    v = gate.sensitivity_variants(base)
    for name in DROPPED_VARIANTS:
        v.pop(name)
    if swap_day:
        v.pop("day=product_2", None)
        v["day=product_1"] = replace(base, day="product_1")
    return v


def run_one(
    x_s: float, days: dict[str, MeasuredDay], workers: int, b99: dict[str, Any]
) -> dict[str, Any]:
    started = time.time()
    grid = [with_solid_fraction(s, x_s) for s in gate.main_grid()]
    refs = {k: with_solid_fraction(s, x_s) for k, s in gate.reference_scenarios().items()}
    refs_ph = {k: with_solid_fraction(s, x_s) for k, s in posthoc.references().items()}
    sens = {rid: variants_of(b) for rid, b in refs.items()}
    sens.update({rid: variants_of(b, swap_day=True) for rid, b in refs_ph.items()})
    tasks: list[dict[str, Any]] = [{"key": f"grid|{i}", "scenario": s} for i, s in enumerate(grid)]
    for rid, b in {**refs, **refs_ph}.items():
        tasks.append({"key": f"{rid}|base", "scenario": b})
        tasks.extend({"key": f"{rid}|var|{n}", "scenario": s} for n, s in sens[rid].items())
    done = gate.run_tasks(tasks, days, workers)
    print(f"x_s={x_s}: batch 1 done ({time.time() - started:.0f}s)", flush=True)
    scenario_of = {t["key"]: t["scenario"] for t in tasks}

    # gate exactly as Befund 99 (grid + pre-registered R1/R2 runs; R3/R4 were post-hoc, outside)
    in_gate = [k for k in done if not k.startswith(("R3|", "R4|"))]
    cells, counts = gate_cells(
        [(scenario_of[k], done[k][0]) for k in in_gate], realistic=realistic_except_solid_fraction
    )
    closest_keys = [
        k
        for k in in_gate
        if k.startswith("grid|")
        and scenario_of[k].site == "A"
        and scenario_of[k].mode == "flat"
        and scenario_of[k].hall_kw == 0.0
        and scenario_of[k].water_kw == 0.0
        and scenario_of[k].ambient_shift_k == 0.0
        and scenario_of[k].loss_per_h == 0.0
    ]
    closest_cells, closest_counts = gate_cells(
        [(scenario_of[k], done[k][0]) for k in closest_keys],
        realistic=realistic_except_solid_fraction,
    )
    groupings = {
        "site": lambda s: s.site,
        "horizon": lambda s: s.horizon,
        "day": lambda s: s.day,
        "mode": lambda s: s.mode,
    }
    gate_out = {
        "verdict": gate_verdict(cells, counts),
        "per_group": subset_verdicts(cells, groupings),
        "closest_to_measured_days_site_A_flat_no_hall_no_water": gate_verdict(
            closest_cells, closest_counts
        ),
    }
    print(f"x_s={x_s}: gate {gate_out['verdict']['reading']}", flush=True)

    # decomposition: Befund 99's four scenarios (same labels) and, if they differ, the ones the
    # pre-registered rule selects in this run
    grid_by_label = {scenario_of[k].label: scenario_of[k] for k in done if k.startswith("grid|")}
    chosen = {
        "R1": refs["R1"],
        "R2": refs["R2"],
        "max_fraction": grid_by_label[b99["decomposition"]["max_fraction"]["scenario"]],
        "max_kwh": grid_by_label[b99["decomposition"]["max_kwh"]["scenario"]],
    }
    best_frac = max(cells, key=lambda c: c.saving_fraction)
    best_kwh = max(cells, key=lambda c: c.saving_kwh_per_day)
    if best_frac.label != chosen["max_fraction"].label:
        chosen["max_fraction_rule_this_run"] = best_frac.scenario
    if best_kwh.label != chosen["max_kwh"].label:
        chosen["max_kwh_rule_this_run"] = best_kwh.scenario
    tasks2: list[dict[str, Any]] = []
    for tag, scen in chosen.items():
        tasks2.extend(gate.decomposition_tasks(tag, scen))
    done2 = gate.run_tasks(tasks2, days, workers)
    decomposition = {
        tag: gate.summarise_decomposition(tag, scen, done2) for tag, scen in chosen.items()
    }

    sensitivity: dict[str, Any] = {}
    for rid, b in {**refs, **refs_ph}.items():
        base_res = done[f"{rid}|base"][0]
        row = next(r for r in base_res.rows if r.capacity_kwh == gate.SENSITIVITY_CAPACITY_KWH)
        sensitivity[rid] = {
            "base": b.label,
            "post_hoc": rid in refs_ph,
            "base_direct_kwh_per_day": base_res.direct_kwh_per_day,
            "base_saving_kwh_per_day": base_res.direct_kwh_per_day - row.ideal_kwh_per_day,
            "ranking": rank_sensitivity(
                base_res,
                {n: done[f"{rid}|var|{n}"][0] for n in sens[rid]},
                gate.SENSITIVITY_CAPACITY_KWH,
            ),
            "notes": done[f"{rid}|base"][1],
        }

    records = []
    for k in done:
        if k.startswith("grid|"):
            rec = gate.record_of(scenario_of[k], done[k][0])
            rec["realistic"] = realistic_except_solid_fraction(scenario_of[k])
            records.append(rec)
    all_results: dict[str, ScenarioResult] = {k: v[0] for k, v in done.items()}
    all_results.update({f"decomp|{k}": v[0] for k, v in done2.items()})
    return {
        "solid_fraction": x_s,
        "days": day_state(days, x_s),
        "gate": gate_out,
        "headline": headline(records),
        "decomposition": decomposition,
        "sensitivity": sensitivity,
        "checks": gate.check_summary(all_results),
        "grid": records,
        "runtime_seconds": time.time() - started,
    }


def control(
    days: dict[str, MeasuredDay], workers: int, b99: dict[str, Any], b99_ph: dict[str, Any]
) -> dict[str, Any]:
    """Re-run part of Befund 99 at x_s = 0.3 and compare with its artifacts."""
    grid = gate.main_grid()
    tasks = [{"key": f"grid|{i}", "scenario": grid[i]} for i in range(0, len(grid), CONTROL_STRIDE)]
    for rid, b in {**gate.reference_scenarios(), **posthoc.references()}.items():
        tasks.append({"key": f"{rid}|base", "scenario": b})
    done = gate.run_tasks(tasks, days, workers)
    b99_grid = {r["label"]: r for r in b99["grid"]}
    worst_direct = worst_ideal = 0.0
    for key, (res, _) in done.items():
        if key.startswith("grid|"):
            ref = b99_grid[res.label]
            ref_direct, ref_ideal = ref["direct_kwh_per_day"], ref["ideal_kwh_per_day"]
        elif key.startswith(("R1", "R2")):
            ref = b99["reference_runs"][key]
            ref_direct, ref_ideal = ref["direct_kwh_per_day"], ref["ideal_kwh_per_day"]
        else:
            rid = key.split("|")[0]
            ref_direct = b99_ph["sensitivity"][rid]["base_direct_kwh_per_day"]
            saving = b99_ph["sensitivity"][rid]["base_saving_kwh_per_day"]
            i5 = _index(gate.SENSITIVITY_CAPACITY_KWH)
            ref_ideal = [float("nan")] * len(CAPACITIES_KWH)
            ref_ideal[i5] = ref_direct - saving
        worst_direct = max(worst_direct, abs(res.direct_kwh_per_day - ref_direct))
        for row, value in zip(res.rows, ref_ideal, strict=True):
            if not np.isnan(value):
                worst_ideal = max(worst_ideal, abs(row.ideal_kwh_per_day - value))
    return {
        "solid_fraction": BEFUND_99_SOLID_FRACTION,
        "n_runs": len(done),
        "max_abs_difference_direct_kwh_per_day": worst_direct,
        "max_abs_difference_ideal_kwh_per_day": worst_ideal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="artifact name (never overwritten)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--solid-fraction", type=float, required=True, choices=SOLID_FRACTIONS, help="one value"
    )
    parser.add_argument(
        "--skip-control", action="store_true", help="skip the control at 0.3 (run it once)"
    )
    args = parser.parse_args()
    require_clean_tree()
    started = time.time()
    b99 = json.loads((RUNS_DIR / f"{BEFUND_99_GATE}.json").read_text(encoding="utf-8"))
    b99_ph = json.loads((RUNS_DIR / f"{BEFUND_99_POSTHOC}.json").read_text(encoding="utf-8"))
    days, hashes = gate.load_days()
    if hashes != b99["raw_file_sha256"]:
        raise SystemExit("raw files differ from the ones of Befund 99")
    ctrl = None if args.skip_control else control(days, args.workers, b99, b99_ph)
    print("control:", ctrl, flush=True)
    b99_headline = headline(b99["grid"])
    run = run_one(args.solid_fraction, days, args.workers, b99)
    payload: dict[str, Any] = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "name": args.name,
        "befund_99_artifacts": [BEFUND_99_GATE, BEFUND_99_POSTHOC],
        "raw_file_sha256": hashes,
        "solid_fraction": args.solid_fraction,
        "solid_fractions_of_the_recheck": list(SOLID_FRACTIONS),
        "admissible_from_by_energy_balance": ADMISSIBLE_FROM,
        "gate_realism": "Befund 99 bands for every placeholder except the solid fraction",
        "dropped_sensitivity_variants": list(DROPPED_VARIANTS),
        "control_at_0.3": ctrl,
        "befund_99": {
            "solid_fraction": BEFUND_99_SOLID_FRACTION,
            "days": day_state(days, BEFUND_99_SOLID_FRACTION),
            "headline": b99_headline,
            "gate": b99["gate"],
        },
        "run": run,
        "runtime_seconds": time.time() - started,
    }
    write_artifact(args.name, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
