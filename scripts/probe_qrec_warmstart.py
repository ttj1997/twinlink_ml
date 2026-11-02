r"""DB1: is the nominal initial state the cause of the Q_rec batch-mean deviation.

Addendum to Befund 54 (DA2c) -- NO new sweep, only a second batch-rollout variant at the
current parameter value that has already been computed anyway (``tau_4 = 16000`` s, config
unchanged, 3x3 grid from ``artifacts/runs/probe_tau4_sensitivity.json``, set points per grid
point already computed there are reused -- no new BB1-BB3 optimisation).

Befund 54 (DA2c) found: the batch-averaged recoverable power ``Q_rec`` at ``tau_4 = 16000`` s
deviates by a median of 35.2 % from the stationary value, although ``bed_temperature_
celsius``/moisture at the END OF THE BATCH lie practically at the fixed point (Befund 53b).
Explanation there: the NOMINAL initial state (``bed_temperature_celsius = 55`` C) lies above
the stationary fixed point (30-45 C) at most grid points -- the start-up phase delivers a
warmer, initially drier exhaust air and pulls the batch mean upwards.

This script checks the explanation directly: the same batch rollout (28800 s,
``Delta_t = 25`` s, projection on, the same set points optimal per grid point), but with the
STEADY STATE from ``solve_steady_state`` as initial state instead of the nominal one
(``operation.initial_bed_temperature_celsius = 55`` C etc.) -- everything else unchanged.
``study_operating_point_regret.py``/``study_operating_point_transient.py`` loaded via
``importlib``, no line there changed; ``rollout_constant_setpoint`` itself does not accept an
initial-state parameter, hence a separate, thin rollout function here with identical
action/disturbance tiling, only with a passed ``initial_state``.

Additionally reported: the point in time (in s) from which the RUNNING (cumulative) mean
Q_rec stays permanently within 5 % of the stationary value -- per grid point and start
variant -- as the duration of the start-up process.

Invocation::

    python scripts/probe_qrec_warmstart.py \
        --output artifacts/runs/probe_qrec_warmstart.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from twinlink_ml.evaluation.recoverable_power import compute_recoverable_power_kw
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
REGRET_SCRIPT_PATH = REPO_ROOT / "scripts" / "study_operating_point_regret.py"
TRANSIENT_SCRIPT_PATH = REPO_ROOT / "scripts" / "study_operating_point_transient.py"
TAU4_SENSITIVITY_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_tau4_sensitivity.json"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_qrec_warmstart.json"

EXPECTED_BASE_FINGERPRINT = "2acd06f38963c3af"
CONVERGENCE_BAND = 0.05
"""5 % -- Auftragsvorgabe fuer die Anfahr-Konvergenzzeit."""


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def rollout_with_initial_state(
    config: Any,
    *,
    t_in: float,
    u_gas: float,
    p_nozzle: float,
    t_amb: float,
    phi_amb: float,
    spray_rate: float,
    solid_fraction: float,
    n_steps: int,
    run_id: str,
    initial_state: np.ndarray | None,
) -> Any:
    """Like ``study_operating_point_transient.rollout_constant_setpoint``, only with an
    optionally passed ``initial_state`` instead of the nominal start hard-wired there
    (that module itself stays unchanged, see the module docstring). ``None``
    uses the nominal initial state -- identical to the original."""
    simulator = GreyBoxFluidBedGranulator(config)
    actions = np.tile(
        np.array([[t_in, u_gas, spray_rate, p_nozzle]], dtype=np.float64), (1, n_steps, 1)
    )
    disturbances = np.tile(
        np.array([[t_amb, phi_amb, solid_fraction]], dtype=np.float64), (1, n_steps, 1)
    )
    nominal_state = simulator.initial_state(1)  # also resets the sensor model
    state = nominal_state if initial_state is None else initial_state
    return simulator.rollout(actions, disturbances, run_id=(run_id,), seed=0, initial_state=state)


def q_rec_series_kw(config: Any, trajectory: Any) -> np.ndarray:
    bed_temperature = trajectory.states[0, :, 3]
    outlet_temperature = bed_temperature - config.exhaust.outlet_duct_heat_loss_k
    outlet_humidity = trajectory.diagnostics["outlet_humidity_kg_per_kg"][0, :]
    backend = NumpyBackend()
    return np.asarray(
        compute_recoverable_power_kw(backend, config.gas, outlet_temperature, outlet_humidity)
    )


def convergence_time_s(
    running_mean: np.ndarray, steady_value: float, time_s: np.ndarray, *, band: float
) -> float:
    """First time from which ``running_mean`` stays PERMANENTLY within ``band`` relative to
    ``steady_value`` (searched backwards from the last violation) -- 0.0 if never
    violated."""
    tolerance = band * abs(steady_value)
    violations = np.flatnonzero(np.abs(running_mean - steady_value) > tolerance)
    if violations.size == 0:
        return 0.0
    last_violation = violations[-1]
    if last_violation + 1 >= len(time_s):
        return float(time_s[-1])
    return float(time_s[last_violation + 1])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--tau4-sensitivity-json", type=Path, default=TAU4_SENSITIVITY_PATH)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()

    env = environment_fingerprint()
    if env["git/dirty"]:
        print("WARNUNG: Arbeitsbaum ist dirty.", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base_config = load_process_config(arguments.config)
    if not fingerprints_equivalent(base_config.fingerprint, EXPECTED_BASE_FINGERPRINT):
        raise RuntimeError(
            f"Basis-Config hat Fingerprint {base_config.fingerprint!r}, erwartet "
            f"{EXPECTED_BASE_FINGERPRINT!r} -- weicht von Befund 52/54 ab."
        )

    regret_module = _load_module("study_operating_point_regret", REGRET_SCRIPT_PATH)
    transient_module = _load_module("study_operating_point_transient", TRANSIENT_SCRIPT_PATH)

    tau4_sensitivity = json.loads(arguments.tau4_sensitivity_json.read_text())
    current_entry = tau4_sensitivity["results_by_value"]["16000"]["stationary"]
    if not fingerprints_equivalent(current_entry["config_fingerprint"], EXPECTED_BASE_FINGERPRINT):
        raise RuntimeError(
            f"{arguments.tau4_sensitivity_json}['16000'] hat config_fingerprint "
            f"{current_entry['config_fingerprint']!r}, erwartet {EXPECTED_BASE_FINGERPRINT!r}."
        )
    grid_results = current_entry["grid_results"]

    transient_config = transient_module.build_transient_config(base_config)
    spray_rate = base_config.actions.spray_rate_kg_per_h.nominal
    solid_fraction = base_config.disturbances.suspension_solid_fraction.nominal

    print("=" * 78)
    print("DB1: Q_rec-Chargenmittel -- nominaler vs. eingeschwungener Anfangszustand")
    print(f"Config       {arguments.config}  (Fingerprint {base_config.fingerprint}, tau_4=16000s)")
    print(f"Gitter       3x3 = {len(grid_results)} Punkte (aus {arguments.tau4_sensitivity_json})")
    print(f"git/commit={env['git/commit']}  git/dirty={env['git/dirty']}")
    print("=" * 78, flush=True)

    per_point: list[dict[str, Any]] = []
    for grid_result in grid_results:
        t_in = grid_result["inlet_temperature_celsius"]
        u_gas = grid_result["gas_velocity_m_per_s"]
        p_nozzle = grid_result["nozzle_pressure_bar"]
        t_amb = grid_result["ambient_temperature_celsius"]
        phi_amb = grid_result["ambient_relative_humidity"]
        index = grid_result["_grid_index"]

        solution = regret_module.evaluate_point(
            t_in,
            u_gas,
            p_nozzle,
            t_amb,
            phi_amb,
            max_iterations=regret_module.DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=regret_module.DEFAULT_SEARCH_TOLERANCE,
            base_config=base_config,
        )
        outlet_temperature = (
            solution.bed_temperature_celsius - base_config.exhaust.outlet_duct_heat_loss_k
        )
        q_rec_steady = float(
            compute_recoverable_power_kw(
                NumpyBackend(),
                base_config.gas,
                np.asarray(outlet_temperature),
                np.asarray(solution.outlet_humidity_kg_per_kg),
            )
        )
        warm_initial_state = np.asarray(solution.state, dtype=np.float64)[None, :]

        point_result: dict[str, Any] = {
            "grid_index": index,
            "ambient_temperature_celsius": t_amb,
            "ambient_relative_humidity": phi_amb,
            "q_rec_steady_kw": q_rec_steady,
        }
        for label, initial_state in (("nominal", None), ("warmstart", warm_initial_state)):
            trajectory = rollout_with_initial_state(
                transient_config,
                t_in=t_in,
                u_gas=u_gas,
                p_nozzle=p_nozzle,
                t_amb=t_amb,
                phi_amb=phi_amb,
                spray_rate=spray_rate,
                solid_fraction=solid_fraction,
                n_steps=transient_module.N_STEPS,
                run_id=f"db1_{label}_{index:02d}",
                initial_state=initial_state,
            )
            q_rec = q_rec_series_kw(transient_config, trajectory)
            running_mean = np.cumsum(q_rec) / np.arange(1, len(q_rec) + 1)
            batch_mean = float(q_rec.mean())
            deviation_pct = 100.0 * (batch_mean - q_rec_steady) / abs(q_rec_steady)
            t_conv = convergence_time_s(
                running_mean, q_rec_steady, trajectory.time_s, band=CONVERGENCE_BAND
            )
            point_result[label] = {
                "batch_mean_q_rec_kw": batch_mean,
                "batch_mean_deviation_pct": deviation_pct,
                "convergence_time_s": t_conv,
            }
        per_point.append(point_result)
        print(
            f"  Punkt {index + 1}: T_amb={t_amb:.1f} phi={phi_amb:.2f}  "
            f"nominal={point_result['nominal']['batch_mean_deviation_pct']:+.2f}% "
            f"(t_conv={point_result['nominal']['convergence_time_s']:.0f}s)  "
            f"warmstart={point_result['warmstart']['batch_mean_deviation_pct']:+.3f}% "
            f"(t_conv={point_result['warmstart']['convergence_time_s']:.0f}s)",
            flush=True,
        )

    def _agg(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values)
        return {"median": float(np.median(arr)), "max": float(np.max(np.abs(arr)))}

    summary = {
        label: {
            "batch_mean_deviation_pct": _agg(
                [p[label]["batch_mean_deviation_pct"] for p in per_point]
            ),
            "convergence_time_s": _agg([p[label]["convergence_time_s"] for p in per_point]),
        }
        for label in ("nominal", "warmstart")
    }

    print("\n" + "=" * 78)
    print("Zusammenfassung (Median / Max|.|)")
    for label in ("nominal", "warmstart"):
        s = summary[label]
        print(
            f"  {label:10s}  Abweichung Chargenmittel: {s['batch_mean_deviation_pct']['median']:+.2f}% / "
            f"{s['batch_mean_deviation_pct']['max']:.2f}%   "
            f"Anfahrzeit (5%-Band): {s['convergence_time_s']['median']:.0f}s / "
            f"{s['convergence_time_s']['max']:.0f}s"
        )
    print("=" * 78, flush=True)

    output = {
        "environment": env,
        "config_fingerprint": base_config.fingerprint,
        "transient_config_fingerprint": transient_config.fingerprint,
        "source_tau4_sensitivity": str(arguments.tau4_sensitivity_json),
        "convergence_band": CONVERGENCE_BAND,
        "per_grid_point": per_point,
        "summary": summary,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
