r"""M0: cost per call of the building blocks of the model-based optimisation (read-only probe).

Replaces the estimates of ``docs/09_entwurf_modellbasierte_optimierung.md`` 1.6 by
measurements, single process, nothing else started by this script:

1. ``evaluate_point`` (one steady-state solve) at the search precision of Befund 52
   (30 iterations, tolerance 1e-7) and at the solver defaults (100, 1e-13), on a fixed set of
   operating points (numpy generator, seed 0, uniform in the action box and the disturbance
   ranges; spray rate and solid fraction at their nominal values). Failures (solver raises) are
   timed separately. Three rounds in the same order to show the variability.
2. ``evaluate_cascade`` on the partner data case of Befund 60 (sources Q1/Q2, sinks S1/S2),
   all six connection cases, both recovery modes.
3. ``evaluate_pinch_target`` on the same data case (``T_kalt`` 29.1, ``dT_min`` 20,
   50 segments below the dew point, the setting of Befund 60).

The processor load (Windows, ``Win32_Processor.LoadPercentage``) is recorded before and after
each phase: the open question of Befund 52 (the original run was 3.5 times slower than its
reproduction, without any load record) is answered here for THIS run. Wall clock via
``time.perf_counter``. No decision rule: the numbers replace estimates.

    PYTHONPATH=<worktree>/src python scripts/probe_m0_cost_per_call.py
"""

from __future__ import annotations

import contextlib
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import run_pinch_benchmark as pinch_data  # noqa: E402
from _common import REPO_ROOT, provenance, require_clean_tree, write_artifact  # noqa: E402

import twinlink_ml  # noqa: E402
from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade  # noqa: E402
from twinlink_ml.heatnetwork.config import HeatNetworkConfig  # noqa: E402
from twinlink_ml.heatnetwork.pinch import evaluate_pinch_target  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402
from twinlink_ml.simulation.oracle_search import action_bounds, evaluate_point  # noqa: E402

ARTIFACT_NAME = "m0_cost_per_call"
N_POINTS = 24
N_ROUNDS = 3
CASCADE_REPEATS = 200
PINCH_REPEATS = 20


def _cpu_load_percent() -> float | None:
    """Processor load in percent (Windows), None if it cannot be read."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage "
                "-Average).Average",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "n": len(values),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _time_solves(
    base_config: Any, points: list[tuple[float, ...]], *, max_iterations: int, tolerance: float
) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    for _ in range(N_ROUNDS):
        ok: list[float] = []
        failed: list[float] = []
        for t_in, u_gas, p_nozzle, t_amb, phi in points:
            start = time.perf_counter()
            try:
                evaluate_point(
                    t_in,
                    u_gas,
                    p_nozzle,
                    t_amb,
                    phi,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    base_config=base_config,
                )
                ok.append(time.perf_counter() - start)
            except (RuntimeError, ValueError):
                failed.append(time.perf_counter() - start)
        rounds.append(
            {
                "solved": _summary(ok) if ok else None,
                "failed": _summary(failed) if failed else None,
                "n_solved": len(ok),
                "n_failed": len(failed),
            }
        )
    return {
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "rounds": rounds,
        "median_of_round_medians_s": statistics.median(
            r["solved"]["median"] for r in rounds if r["solved"]
        ),
    }


def main() -> int:
    require_clean_tree()
    assert Path(twinlink_ml.__file__).resolve().is_relative_to(REPO_ROOT), (
        "twinlink_ml is imported from outside this checkout; set PYTHONPATH=<checkout>/src"
    )
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)
    base_config = load_process_config(
        REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
    )

    rng = np.random.default_rng(0)
    bounds = action_bounds(base_config)
    d = base_config.disturbances
    points = [
        (
            *(float(rng.uniform(lo, hi)) for lo, hi in bounds),
            float(
                rng.uniform(
                    d.ambient_temperature_celsius.minimum, d.ambient_temperature_celsius.maximum
                )
            ),
            float(
                rng.uniform(
                    d.ambient_relative_humidity.minimum, d.ambient_relative_humidity.maximum
                )
            ),
        )
        for _ in range(N_POINTS)
    ]
    # warm-up (imports, first-call overhead), not counted
    for point in points[:3]:
        with contextlib.suppress(RuntimeError, ValueError):
            evaluate_point(*point, max_iterations=30, tolerance=1e-7, base_config=base_config)

    load: dict[str, float | None] = {"start": _cpu_load_percent()}
    solve_search = _time_solves(base_config, points, max_iterations=30, tolerance=1e-7)
    load["after_search_precision"] = _cpu_load_percent()
    solve_default = _time_solves(base_config, points[:12], max_iterations=100, tolerance=1e-13)
    load["after_solver_defaults"] = _cpu_load_percent()

    network = HeatNetworkConfig(
        sources=(pinch_data.SOURCE_1, pinch_data.SOURCE_2),
        sinks=(pinch_data.SINK_1, pinch_data.SINK_2),
        minimum_pinch_k=20.0,
        recovery_mode="condensing",
        cold_temperature_celsius=29.1,
        fuel_lower_heating_value_j_per_kg=pinch_data.FUEL_LOWER_HEATING_VALUE_J_PER_KG,
        heater_efficiency=pinch_data.HEATER_EFFICIENCY,
    )
    sources = tuple(SourceState.from_config(c) for c in network.sources)
    sinks = tuple(SinkState.from_config(c) for c in network.sinks)
    cascade: dict[str, Any] = {}
    for mode in ("sensible", "condensing"):
        cfg = HeatNetworkConfig(
            sources=network.sources,
            sinks=network.sinks,
            minimum_pinch_k=20.0,
            recovery_mode=mode,
            cold_temperature_celsius=29.1,
            fuel_lower_heating_value_j_per_kg=pinch_data.FUEL_LOWER_HEATING_VALUE_J_PER_KG,
            heater_efficiency=pinch_data.HEATER_EFFICIENCY,
        )
        for case, stages in pinch_data.CASES.items():
            times = []
            for _ in range(CASCADE_REPEATS):
                start = time.perf_counter()
                evaluate_cascade(cfg, stages, sources, sinks)
                times.append(time.perf_counter() - start)
            cascade[f"{mode}/{case}"] = _summary(times)
    load["after_cascade"] = _cpu_load_percent()

    pinch_times = []
    for _ in range(PINCH_REPEATS):
        start = time.perf_counter()
        evaluate_pinch_target(
            (pinch_data.SOURCE_1, pinch_data.SOURCE_2),
            (pinch_data.SINK_1, pinch_data.SINK_2),
            pinch_data.GAS,
            cold_temperature_celsius=29.1,
            minimum_pinch_k=20.0,
            n_segments_below_dew_point=50,
        )
        pinch_times.append(time.perf_counter() - start)
    load["after_pinch"] = _cpu_load_percent()

    write_artifact(
        ARTIFACT_NAME,
        {
            "_provenance": provenance(),
            "environment": {
                "os.cpu_count": os.cpu_count(),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "cpu_load_percent": load,
                "numpy": np.__version__,
            },
            "n_points": N_POINTS,
            "n_rounds": N_ROUNDS,
            "solve_search_precision": solve_search,
            "solve_solver_defaults_first_12_points": solve_default,
            "cascade_seconds_per_call": cascade,
            "pinch_target_seconds_per_call": _summary(pinch_times),
            "config_fingerprint": base_config.fingerprint,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
