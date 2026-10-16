r"""Storage test 1, POST-HOC: sensitivity around two scenarios with a non-zero saving.

DECLARED POST-HOC. The pre-registered reference scenarios R1 and R2 of
``scripts/gate_storage_timeslice.py`` (site A+C, product 1) have a saving of 0 at 5 MWh, so a
change of one placeholder can only open a potential there, never close one; the ranking of
those runs is therefore of limited use. After seeing this, two further reference scenarios
are fixed here (before this script was run), both with a non-zero saving in the gate run:

    R3  site A+C, product day 2, week horizon, mode shift, weather 0 K, compressor 800 kW,
        hall 200 kW, process water 200 kW, loss 0.
    R4  site A, product day 2, day horizon, mode shift, weather 0 K, compressor 1200 kW,
        hall 200 kW, process water 0 kW, loss 0.

Same variants (one placeholder at a time over its band; the variant ``day=product_2`` becomes
``day=product_1``), same metric (change of the saving at C = 5 MWh in kWh/d and in percentage
points of E_direct), same code as the pre-registered study. Nothing here changes the gate.

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... \
        tools/run_here.sh scripts/analyze_storage_timeslice_sensitivity.py \
        --name gate_storage_timeslice_posthoc_sensitivity
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_storage_timeslice as gate
from _common import NO_HEAT_NETWORK, provenance, require_clean_tree, write_artifact

from twinlink_ml.storage_timeslice import Scenario
from twinlink_ml.storage_timeslice.gate import rank_sensitivity


def references() -> dict[str, Scenario]:
    return {
        "R3": Scenario("A+C", "product_2", "week", "shift", 0.0, 800.0, 200.0, 200.0, 0.0),
        "R4": Scenario("A", "product_2", "day", "shift", 0.0, 1200.0, 200.0, 0.0, 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    require_clean_tree()
    started = time.time()
    days, hashes = gate.load_days()
    variants: dict[str, dict[str, Scenario]] = {}
    tasks: list[dict[str, Any]] = []
    for rid, base in references().items():
        v = gate.sensitivity_variants(base)
        v.pop("day=product_2", None)
        v["day=product_1"] = replace(base, day="product_1")
        variants[rid] = v
        tasks.append({"key": f"{rid}|base", "scenario": base})
        tasks.extend({"key": f"{rid}|var|{n}", "scenario": s} for n, s in v.items())
    done = gate.run_tasks(tasks, days, args.workers)
    out: dict[str, Any] = {}
    for rid, base in references().items():
        base_res = done[f"{rid}|base"][0]
        ranking = rank_sensitivity(
            base_res,
            {n: done[f"{rid}|var|{n}"][0] for n in variants[rid]},
            gate.SENSITIVITY_CAPACITY_KWH,
        )
        out[rid] = {
            "base": base.label,
            "capacity_kwh": gate.SENSITIVITY_CAPACITY_KWH,
            "base_direct_kwh_per_day": base_res.direct_kwh_per_day,
            "base_saving_kwh_per_day": base_res.direct_kwh_per_day
            - next(
                r.ideal_kwh_per_day
                for r in base_res.rows
                if r.capacity_kwh == gate.SENSITIVITY_CAPACITY_KWH
            ),
            "ranking": ranking,
        }
    payload = {
        "_provenance": provenance(NO_HEAT_NETWORK),
        "name": args.name,
        "post_hoc": True,
        "raw_file_sha256": hashes,
        "sensitivity": out,
        "runtime_seconds": time.time() - started,
    }
    write_artifact(args.name, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
