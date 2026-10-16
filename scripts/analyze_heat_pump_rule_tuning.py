r"""Heat-pump study, DECLARED POST-HOC: does a tuned constant-value rule close the gap.

Not part of the pre-registration of ``scripts/gate_heat_pump.py``; written AFTER seeing the gate
reading L1 of ``artifacts/runs/gate_heat_pump.json`` and committed before its own run. It does not
change the pre-registered reading; it asks what the L1 cells mean.

Observation that motivated it (artifact ``gate_heat_pump``): rule gaps of up to 4 points occur
already with a CONSTANT electricity signal (amplitude 0) in the shift mode; a trace of the largest
such cell (site A, product 1, day, shift, 500 kW, CO2 2030, 2 MWh) shows the mechanism: at night
the compressor heat is at 30 %, the evaporator of the pump runs short of free source heat and its
marginal effect falls to 1/COP, but the rules still run it (``gas_value``) or never pre-charge the
storage with pump heat in the evening (``myopic``, ``trailing``); the programme pre-charges.

Question: how much of each L1 gap closes with the simplest tunable rule without foresight -- the
step optimum with a CONSTANT value of stored heat ``v = kappa * c_U`` -- when ``kappa`` is chosen
per cell IN HINDSIGHT from {0.05, 0.1, 0.2, ..., 0.9, 1 - 1e-6} (and the level as before)? The
hindsight choice makes this an OPTIMISTIC bound for a once-tuned constant; a remaining gap is
what a constant cannot do (time-of-day or state-dependent value, i.e. a schedule or a planner).

Cells: every realistic cell of the artifact with ``D >= 2`` points (and their runs' other
capacities). Each run is recomputed (the programme and the three named rules are checked against
the artifact). Also reported: the breakdown of the L1 cells of the artifact by mode, amplitude,
criterion, site, day and horizon (read from the artifact, nothing recomputed).

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... \
        tools/run_here.sh scripts/analyze_heat_pump_rule_tuning.py \
        --name gate_heat_pump_rule_tuning_posthoc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_heat_pump as ghp
from _common import (  # type: ignore[import-untyped]
    NO_HEAT_NETWORK,
    REPO_ROOT,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.heatpump import HeatPumpSpec, Signal, design_pairs, site_cascade
from twinlink_ml.heatpump.storage import RULES, coupled_pieces, simulate_rule, solve_foresight
from twinlink_ml.storage_timeslice.profiles import Clock
from twinlink_ml.storage_timeslice.table import storage_stream_temperatures

SOURCE = REPO_ROOT / "artifacts" / "runs" / "gate_heat_pump.json"
KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 - 1e-6)
GAP = 0.02


def _run(task: dict[str, Any]) -> dict[str, Any]:
    from twinlink_ml.heatpump.signal import signal_series

    meta = task["meta"]
    scen = ghp.scenario(meta["site"], meta["day"], meta["mode"], horizon=meta["horizon"])
    crit = ghp.criterion_of(meta["criterion"])
    signal = Signal(amplitude=meta["amplitude"], weekend_reduction=meta["weekend_reduction"])
    pairs = {p.label: p for p in design_pairs(HeatPumpSpec(), minimum_pinch_k=20.0)}
    pair = pairs[task["pair"]]
    bank = ghp._factory().build(scen)
    ph = scen.placeholders
    levels = np.asarray(ghp.LEVELS_B)
    tc, td = storage_stream_temperatures(
        levels, minimum_pinch_k=ph.minimum_pinch_k, approach_k=ph.approach_k
    )
    cas = site_cascade(
        bank, scen, np.concatenate([[pair.evaporator_shifted, pair.condenser_shifted], tc, td])
    )
    elec = signal_series(crit.electricity_value_per_kwh, signal, Clock.of(scen.n_days))
    caps = np.asarray([c for c in ghp.CAPACITIES_B if c > 0.0])
    per_day = 1.0 / scen.n_days
    size = float(meta["size_kw"])
    lp = np.full((len(levels), len(caps)), np.inf)
    named = {r: np.full((len(levels), len(caps)), np.inf) for r in RULES}
    tuned = np.full((len(KAPPAS), len(levels), len(caps)), np.inf)
    tuned_ok = np.zeros((len(KAPPAS), len(levels), len(caps)), dtype=bool)
    for li in range(len(levels)):
        pc = coupled_pieces(
            cas,
            evaporator_shifted=pair.evaporator_shifted,
            condenser_shifted=pair.condenser_shifted,
            cop=pair.cop,
            charge_shifted=float(tc[li]),
            discharge_shifted=float(td[li]),
        )
        common: dict[str, Any] = {
            "q_max_kw": size,
            "loss_per_h": scen.loss_per_h,
            "dt_h": bank.dt_h,
            "heat_value": crit.heat_value_per_kwh,
            "electricity": elec,
        }
        for ci, cap in enumerate(caps):
            lp[li, ci] = solve_foresight(pc, capacity_kwh=float(cap), **common).objective
        for r in RULES:
            named[r][li] = simulate_rule(pc, r, capacities_kwh=caps, **common).objective
        for ki, kappa in enumerate(KAPPAS):
            run = simulate_rule(
                pc,
                "myopic",
                capacities_kwh=caps,
                stored_value=np.full(cas.n_steps, kappa * crit.heat_value_per_kwh),
                **common,
            )
            tuned[ki, li] = run.objective
            tuned_ok[ki, li] = run.converged
    j_none = task["objective_none_per_day"]
    best_lp = np.min(lp, axis=0) * per_day
    best_named = np.min(np.stack([np.min(v, axis=0) for v in named.values()]), axis=0) * per_day
    flat = tuned.reshape(len(KAPPAS) * len(levels), len(caps))
    k_best = np.argmin(flat, axis=0)
    best_tuned = flat[k_best, np.arange(len(caps))] * per_day
    ok = tuned_ok.reshape(len(KAPPAS) * len(levels), len(caps))[k_best, np.arange(len(caps))]
    return {
        "key": task["key"],
        "meta": meta,
        "capacities_kwh": caps.tolist(),
        "foresight_recomputed_per_day": best_lp.tolist(),
        "foresight_artifact_per_day": task["objective_foresight_per_day"],
        "named_rules_recomputed_per_day": best_named.tolist(),
        "gap_artifact": task["gap_fraction"],
        "gap_named_recomputed": ((best_named - best_lp) / j_none).tolist(),
        "gap_tuned_constant": ((best_tuned - best_lp) / j_none).tolist(),
        "kappa_best": [float(KAPPAS[k // len(levels)]) for k in k_best],
        "level_best": [float(levels[k % len(levels)]) for k in k_best],
        "tuned_converged": ok.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    args = parser.parse_args()
    require_clean_tree()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    grid: dict[str, Any] = source["stage_b"]["grid"]
    l1_cells = [
        (k, r["meta"], r["capacities_kwh"][i])
        for k, r in grid.items()
        if r["realistic"]
        for i in range(len(r["capacities_kwh"]))
        if r["gap_fraction"][i] >= GAP - 1e-9
    ]
    breakdown = {
        axis: dict(Counter(str(m[axis]) for _, m, _ in l1_cells))
        for axis in ("mode", "amplitude", "criterion", "kind", "site", "day", "horizon", "size_kw")
    }
    breakdown["capacity_kwh"] = dict(Counter(str(c) for _, _, c in l1_cells))
    keys = sorted({k for k, _, _ in l1_cells})
    tasks = [
        {
            "key": k,
            "meta": grid[k]["meta"],
            "pair": grid[k]["pair"],
            "objective_none_per_day": grid[k]["objective_none_per_day"],
            "objective_foresight_per_day": grid[k]["objective_foresight_per_day"],
            "gap_fraction": grid[k]["gap_fraction"],
        }
        for k in keys
    ]
    days, hashes = ghp.load_days()
    print(f"{len(l1_cells)} L1 cells in {len(tasks)} runs", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=ghp._init_worker, initargs=(days,)
    ) as pool:
        results = list(pool.map(_run, tasks))
    cells_after: list[dict[str, Any]] = []
    for res in results:
        for i, cap in enumerate(res["capacities_kwh"]):
            if res["gap_artifact"][i] >= GAP - 1e-9:
                cells_after.append(
                    {
                        "key": res["key"],
                        "capacity_kwh": cap,
                        "gap_artifact": res["gap_artifact"][i],
                        "gap_tuned_constant": res["gap_tuned_constant"][i],
                        "kappa_best": res["kappa_best"][i],
                        "mode": res["meta"]["mode"],
                        "amplitude": res["meta"]["amplitude"],
                        "criterion": res["meta"]["criterion"],
                    }
                )
    still = [c for c in cells_after if c["gap_tuned_constant"] >= GAP - 1e-9]
    max_dev_lp = max(
        abs(a - b) / max(1.0, abs(b))
        for r in results
        for a, b in zip(
            r["foresight_recomputed_per_day"], r["foresight_artifact_per_day"], strict=True
        )
    )
    max_dev_gap = max(
        abs(a - b)
        for r in results
        for a, b in zip(r["gap_named_recomputed"], r["gap_artifact"], strict=True)
    )
    summary = {
        "n_l1_cells": len(l1_cells),
        "n_l1_runs": len(tasks),
        "n_cells_still_gap_at_least_2_points_with_tuned_constant": len(still),
        "max_gap_artifact": max(c["gap_artifact"] for c in cells_after),
        "max_gap_tuned_constant": max(c["gap_tuned_constant"] for c in cells_after),
        "median_gap_artifact": float(np.median([c["gap_artifact"] for c in cells_after])),
        "median_gap_tuned_constant": float(
            np.median([c["gap_tuned_constant"] for c in cells_after])
        ),
        "still_by_mode": dict(Counter(c["mode"] for c in still)),
        "still_by_amplitude": dict(Counter(str(c["amplitude"]) for c in still)),
        "still_by_criterion": dict(Counter(c["criterion"] for c in still)),
        "kappa_best_counts": dict(Counter(str(c["kappa_best"]) for c in cells_after)),
        "check_max_relative_deviation_foresight_vs_artifact": max_dev_lp,
        "check_max_deviation_named_rule_gap_vs_artifact": max_dev_gap,
    }
    print(json.dumps(summary, indent=1), flush=True)
    write_artifact(
        args.name,
        {
            "_provenance": provenance(NO_HEAT_NETWORK),
            "name": args.name,
            "declared_post_hoc": True,
            "source_artifact": "gate_heat_pump",
            "source_provenance": source["provenance"],
            "raw_file_sha256": hashes,
            "kappas": list(KAPPAS),
            "l1_breakdown_from_source": breakdown,
            "summary": summary,
            "cells": cells_after,
            "runs": results,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
