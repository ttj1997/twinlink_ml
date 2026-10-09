r"""Run-time of the corrected cascade (Befund 79, step 4): ``approach_check`` "along_exchanger" against "warm_end".

Measured single-threaded on the states of real solutions, so that the estimate of the re-optimisations
(the coupling test drafted after M1) rests on numbers of this machine:

* one cascade (``site_network.evaluate_site`` with ONE interconnection, as in every objective call of the
  table search and of the polishing: profiles built inside the call);
* one site evaluation over all 441 interconnections (profiles built once, as in S2);
* one objective call of the table search (``joint_search.table_objective``: two table interpolations and
  one cascade) and of the polishing (``joint_search.evaluate_true``: two steady-state solves and one cascade);
* one steady-state solve (reference).

States: the best S2 and the best M3' of seed 0 of the scenarios (30.0; 0.82) and (32.5; 0.60) of
``m4c_confirm_lab_umin126``; tables: ``m3_line_tables_umin126`` at the grid points 22 and 21. Every figure is
the median of 5 rounds of the median of repeated calls; the CPU load of the machine is read before and after
(``Win32_Processor.LoadPercentage``) and stored, because a loaded machine biases the absolute times.

    PYTHONPATH=src python scripts/probe_cascade_runtime.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_m3_joint as m3  # noqa: E402
import run_m3x_confirm as c  # noqa: E402
from _common import (  # noqa: E402
    REPO_ROOT,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

import twinlink_ml  # noqa: E402
from twinlink_ml.evaluation import joint_search as js  # noqa: E402
from twinlink_ml.evaluation import site_network as sn  # noqa: E402
from twinlink_ml.evaluation.line_tables import LineTable  # noqa: E402
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning  # noqa: E402
from twinlink_ml.simulation.oracle_search import evaluate_point  # noqa: E402

ARTIFACT = "probe_cascade_runtime"
ROUNDS = 5
MODES = ("warm_end", "along_exchanger")
SCENARIOS = {"2": 22, "3": 21}
"""M4-C scenario id -> grid index of the table artifact used for the table objective (nearest grid point)."""


def _cpu_load_percent() -> float | None:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _timed(function: Any, repeats: int) -> float:
    """Median over ``ROUNDS`` rounds of the median time of one call [s]."""
    rounds = []
    for _ in range(ROUNDS):
        calls = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            calls.append(time.perf_counter() - start)
        rounds.append(statistics.median(calls))
    return statistics.median(rounds)


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT)
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    load_before = _cpu_load_percent()
    configs, k, _, _ = m3._setup()
    art = json.loads((RUNS_DIR / "m4c_confirm_lab_umin126.json").read_text("utf-8"))
    tables = json.loads((RUNS_DIR / "m3_line_tables_umin126.json").read_text("utf-8"))
    table_rows = {row["grid_index"]: row for row in tables["scenarios"]}
    allocations = sn.all_allocations(2, order_variants=True)
    results: dict[str, Any] = {}
    for sid, grid_index in SCENARIOS.items():
        scenario = art["scenarios"][sid]
        t_amb, phi = scenario["t_amb"], scenario["phi"]
        table_a = LineTable.from_dict(table_rows[grid_index]["A"])
        table_b = LineTable.from_dict(table_rows[grid_index]["B"])
        for label, record in (("S2", scenario["best_s2"]), ("M3'", scenario["per_seed_m3"]["0"])):
            x = record["set_points"]
            allocation = tuple(tuple(p) for p in record["allocation"])
            lines = []
            for name, (t_in, u) in (("A", x[:2]), ("B", x[2:])):
                solution = evaluate_point(
                    t_in,
                    u,
                    1.0,
                    t_amb,
                    phi,
                    max_iterations=30,
                    tolerance=1e-7,
                    base_config=configs[name],
                )
                lines.append(
                    sn.line_state(
                        name, solution, configs[name], t_amb=t_amb, phi_amb=phi, t_in=t_in
                    )
                )
            lines_t = (lines[0], lines[1])
            entry: dict[str, Any] = {
                "steady_state_solve_s": _timed(
                    lambda x=x, t_amb=t_amb, phi=phi: evaluate_point(
                        x[0],
                        x[1],
                        1.0,
                        t_amb,
                        phi,
                        max_iterations=30,
                        tolerance=1e-7,
                        base_config=configs["A"],
                    ),
                    3,
                )
            }
            for mode in MODES:
                site = sn.SiteSpec(k, approach_check=mode)
                spec = SiteOracleSpec(c.THRESHOLD, site)
                entry[mode] = {
                    "one_cascade_s": _timed(
                        lambda site=site, allocation=allocation, lines_t=lines_t: sn.evaluate_site(
                            lines_t, site, [allocation]
                        ),
                        200,
                    ),
                    "site_441_s": _timed(
                        lambda site=site, lines_t=lines_t: sn.evaluate_site(
                            lines_t, site, allocations
                        ),
                        3,
                    ),
                    "table_objective_s": _timed(
                        lambda spec=spec,
                        allocation=allocation,
                        x=x,
                        ta=table_a,
                        tb=table_b: js.table_objective(x, allocation, ta, tb, spec),
                        200,
                    ),
                    "polish_objective_s": _timed(
                        lambda spec=spec,
                        allocation=allocation,
                        x=x,
                        t_amb=t_amb,
                        phi=phi: js.evaluate_true(x, allocation, configs, t_amb, phi, spec),
                        3,
                    ),
                }
            results[f"{scenario['group']} ({t_amb}; {phi}) {label}"] = entry
            print(
                label,
                t_amb,
                phi,
                {m: {k_: round(v * 1e3, 3) for k_, v in entry[m].items()} for m in MODES},
            )
    ratios = {
        key: statistics.median(
            [entry["along_exchanger"][key] / entry["warm_end"][key] for entry in results.values()]
        )
        for key in ("one_cascade_s", "site_441_s", "table_objective_s", "polish_objective_s")
    }
    medians = {
        mode: {
            key: statistics.median([entry[mode][key] for entry in results.values()])
            for key in ("one_cascade_s", "site_441_s", "table_objective_s", "polish_objective_s")
        }
        for mode in MODES
    }
    medians["steady_state_solve_s"] = statistics.median(
        [e["steady_state_solve_s"] for e in results.values()]
    )
    load_after = _cpu_load_percent()
    write_artifact(
        ARTIFACT,
        {
            "_provenance": provenance(sn.SiteSpec(k)),
            "purpose": "run-time of the cascade with the check along the exchanger (Befund 79, step 4)",
            "single_threaded": True,
            "rounds": ROUNDS,
            "cpu_load_percent_before_after": [load_before, load_after],
            "median_over_states_s": medians,
            "ratio_along_over_warm_end_median": ratios,
            "per_state": results,
        },
    )
    print("ratios", {k_: round(v, 2) for k_, v in ratios.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
