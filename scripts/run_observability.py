r"""Driver of the observability study (docs/NN_observability.md): steady states and step responses.

Model evaluation only, no training. Reads ``conf/observability/observability.yaml``, which names
the sensor-value config of Befund 111 (truth line, 20 parameter draws; read, not copied).

1. **Operating points** P1-P7 from the per-pair command means of Befund 111 (record named in the
   config, SHA-256 recorded): P1-P5 the 5/25/50/75/95 % percentiles of the inlet-temperature
   command among the pairs at the gas-velocity minimum (0.8 m/s), P6/P7 the pairs at the
   inlet-temperature minimum (below 61 degC) at the 50/95 % percentile of their gas velocity
   (rounded to 0.1 K and 0.01 m/s). Disturbances, nozzle pressure, spray rate and solid fraction at
   the nominal values of the lab line -- identical for every draw.
2. **Lab line, steady:** per operating point and draw the exact steady state
   (:func:`~twinlink_ml.simulation.steady_state.solve_steady_state`) of the truth with the draw's
   seven material parameters; the nominal model alongside (reference).
   Context (not part of the reading): the nominal model's steady shifts for the input amplitudes
   of the "observed" scenarios (inlet temperature + slow std, spray rate + throughput amplitude,
   solid fraction + its std), each alone.
3. **Plant-like, placeholder:** per operating point and placeholder product flow the commands of
   the lab point with the dry solid feed raised to ``ratio * dry-air flow`` at unchanged water feed,
   air, inlet temperature and every model parameter
   (:func:`~twinlink_ml.observability.core.scale_solid_to_air_ratio` on the nominal model); then the
   same steady states per draw at these commands.
4. **Dynamics:** at the configured operating point (P3) three draws chosen by the kinetics score
   ``s = u(critical moisture) - u(nu_max)`` (unit Sobol' coordinates; high = slow drying): the
   largest s ("slow"), the smallest ("fast") and the draw at sorted position ``(n - 1) // 2``
   ("median"). From each draw's own steady state the configured steps (spray rate and inlet
   temperature, factors 1.1 and 0.9, through the rate limit of the action box) over 24 h at the
   model sample time 5 s. Per step, measurement and pair of draws: the distance of the two
   trajectories against the sensor sigma at their mean value, (a) absolute and (b) of the
   deviations from each draw's unstepped reference; maximum ratio and total time above
   ``threshold_sigma``. Measurements: exhaust humidity (model diagnostic and water-balance value),
   exhaust temperature (sigma of one sensor), bed temperature (sigma of one sensor and sigma /
   sqrt(6)).

Output ``artifacts/runs/<name>.json.gz`` (never overwritten). The analysis, with the reading fixed
before the run, is ``scripts/analyze_observability.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import NO_HEAT_NETWORK, REPO_ROOT, RUNS_DIR, provenance  # noqa: E402

from twinlink_ml.headroom.config import load_headroom_config, study_process_config  # noqa: E402
from twinlink_ml.observability import (  # noqa: E402
    load_observability_config,
    operating_point_config,
    scale_solid_to_air_ratio,
    sensor_sigma,
    separation,
    steady_measurements,
    step_responses,
)
from twinlink_ml.sensitivity.config import load_sensitivity_config  # noqa: E402
from twinlink_ml.sensor_value import load_sensor_value_config, parameter_draws  # noqa: E402
from twinlink_ml.sensor_value.draws import truth_config  # noqa: E402
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402

MEASUREMENTS = {
    # name: (series key, temperature channel, redundancy)
    "Y_out": ("outlet_humidity_kg_per_kg", False, 1),
    "Y_out_balance": ("outlet_humidity_balance_kg_per_kg", False, 1),
    "T_out": ("exhaust_temperature_celsius", True, 1),
    "T_bed": ("bed_temperature_celsius", True, 1),
    "T_bed_x6": ("bed_temperature_celsius", True, None),  # None: redundant_bed_sensors
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operating_points(settings: Any, u_min: float) -> list[dict[str, Any]]:
    """P1-P7 from the per-pair command means of the Befund 111 record (rule in the docstring)."""
    with gzip.open(settings.source, "rt", encoding="utf-8") as f:
        record = json.load(f)
    rows = record["results"][settings.variant]["criteria"][settings.criterion]["records"]
    t = np.array([r["inlet_command_mean"] for r in rows])
    u = np.array([r["gas_velocity_command_mean"] for r in rows])
    at_min = u <= u_min + settings.minimum_velocity_tolerance_m_per_s
    low = t < settings.low_inlet_below_celsius
    points = []
    for p in settings.inlet_percentiles_at_minimum_velocity:
        points.append(
            {
                "rule": f"inlet P{p:g} at the velocity minimum",
                "inlet_temperature_celsius": round(
                    float(np.percentile(t[at_min], p)), settings.round_inlet_celsius
                ),
                "gas_velocity_m_per_s": u_min,
            }
        )
    for p in settings.velocity_percentiles_at_low_inlet:
        points.append(
            {
                "rule": f"velocity P{p:g} at the inlet minimum",
                "inlet_temperature_celsius": round(
                    float(np.percentile(t[low], 50)), settings.round_inlet_celsius
                ),
                "gas_velocity_m_per_s": round(
                    float(np.percentile(u[low], p)), settings.round_velocity_m_per_s
                ),
            }
        )
    for i, point in enumerate(points):
        point["name"] = f"P{i + 1}"
    counts = {
        "pairs": len(rows),
        "at_velocity_minimum": int(at_min.sum()),
        "low_inlet": int(low.sum()),
    }
    return [{**p, "source_counts": counts} for p in points]


def steady_block(base_config: Any, draws: list[Any]) -> dict[str, Any]:
    """Nominal model and every draw at the commands of ``base_config``."""
    out: dict[str, Any] = {"nominal": steady_measurements(base_config).as_dict(), "draws": []}
    for draw in draws:
        try:
            s = steady_measurements(truth_config(base_config, draw.values)).as_dict()
        except (RuntimeError, ValueError) as error:  # bracket failure: recorded, not dropped
            s = {"failure": f"{type(error).__name__}: {error}"}
        out["draws"].append({"draw": draw.index, **s})
    return out


def input_context(base_config: Any, observed: Any) -> dict[str, Any]:
    """Nominal model: steady shifts for the amplitudes of the "observed" scenarios (context only).

    Each input moved alone from the operating point: inlet temperature by the slow std, spray rate
    by the throughput amplitude, solid fraction by its std (fraction of the half band). Not part of
    the reading; it shows how much an unknown input offset moves the measurements.
    """
    reference = steady_measurements(base_config)
    t_in = base_config.actions.inlet_temperature_celsius.nominal
    u = base_config.actions.gas_velocity_m_per_s.nominal
    spray = base_config.actions.spray_rate_kg_per_h.nominal
    w = base_config.disturbances.suspension_solid_fraction.nominal
    sf_rel = (
        observed.solid_fraction_relative_half_band
        * observed.solid_fraction_std_fraction_of_half_band
    )
    cases = {
        "inlet_temperature_plus_slow_std": {
            "inlet_temperature_celsius": t_in + observed.inlet_temperature_slow[0]
        },
        "spray_rate_plus_throughput_amplitude": {
            "spray_rate_kg_per_h": spray * (1.0 + observed.throughput_relative_amplitude)
        },
        "solid_fraction_plus_std": {"solid_fraction": w * (1.0 + sf_rel)},
    }
    out: dict[str, Any] = {}
    for name, change in cases.items():
        kwargs = {"inlet_temperature_celsius": t_in, "gas_velocity_m_per_s": u, **change}
        moved = steady_measurements(operating_point_config(base_config, **kwargs))
        out[name] = {
            "change": change,
            "delta_moisture_kg_per_kg": moved.moisture_kg_per_kg - reference.moisture_kg_per_kg,
            "delta_Y_out": moved.outlet_humidity_kg_per_kg - reference.outlet_humidity_kg_per_kg,
            "delta_Y_out_balance": moved.outlet_humidity_balance_kg_per_kg
            - reference.outlet_humidity_balance_kg_per_kg,
            "delta_T_bed": moved.bed_temperature_celsius - reference.bed_temperature_celsius,
        }
    return out


def kinetics_selection(draws: list[Any], parameters: tuple[str, str]) -> dict[str, int]:
    """Slow, median and fast draw by s = u(critical moisture) - u(nu_max)."""
    score = np.array([d.unit[parameters[0]] - d.unit[parameters[1]] for d in draws])
    order = np.argsort(score, kind="stable")
    return {
        "slow": int(draws[order[-1]].index),
        "median": int(draws[order[(len(draws) - 1) // 2]].index),
        "fast": int(draws[order[0]].index),
    }


def dynamics_block(
    config: Any, base_config: Any, draws: list[Any], measurement: Any, selection: dict[str, int]
) -> dict[str, Any]:
    dyn = config.dynamics
    by_index = {d.index: d for d in draws}
    dt = base_config.numerics.sample_time_s
    thin = round(dyn.record_every_s / dt)
    series: dict[str, dict[str, np.ndarray]] = {}
    initial: dict[str, Any] = {}
    for label, index in selection.items():
        truth = truth_config(base_config, by_index[index].values)
        steady = steady_measurements(truth)
        initial[label] = steady.as_dict()
        series[label] = step_responses(truth, steady.state, dyn.steps, dyn.horizon_s)
    time_s = next(iter(series.values()))["time_s"]
    labels = list(selection)
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]
    stats: dict[str, Any] = {}
    for j, step in enumerate(dyn.steps, start=1):
        per_step: dict[str, Any] = {}
        for name, (key, temperature, redundancy) in MEASUREMENTS.items():
            n = config.snr.redundant_bed_sensors if redundancy is None else redundancy
            per_measurement: dict[str, Any] = {}
            for a, b in pairs:
                ya, yb = series[a][key][j], series[b][key][j]
                sigma = sensor_sigma(0.5 * (ya + yb), measurement, temperature=temperature)
                sigma = sigma / math.sqrt(n)
                absolute = separation(ya, yb, sigma, dyn.threshold_sigma, dt, time_s)
                da, db = ya - series[a][key][0], yb - series[b][key][0]
                deviation = separation(da, db, sigma, dyn.threshold_sigma, dt, time_s)
                per_measurement[f"{a}-{b}"] = {
                    "absolute": absolute.as_dict(),
                    "deviation": deviation.as_dict(),
                }
            per_step[name] = per_measurement
        stats[step.name] = per_step
    settling: dict[str, Any] = {}
    for label in labels:
        x = series[label]["moisture_kg_per_kg"]
        out = {}
        for j, step in enumerate(dyn.steps, start=1):
            change = x[j] - x[0]
            final = change[-1]
            reached = np.flatnonzero(np.abs(change) >= 0.95 * abs(final)) if final else []
            out[step.name] = {
                "final_moisture_change_kg_per_kg": float(final),
                "t95_s": float(time_s[reached[0]]) if len(reached) else None,
                "max_abs_moisture_change_kg_per_kg": float(np.abs(change).max()),
            }
        settling[label] = out
    thinned = {
        label: {
            k: np.round(v[..., thin - 1 :: thin], 10).tolist()
            for k, v in s.items()
            if k != "time_s"
        }
        for label, s in series.items()
    }
    return {
        "operating_point": dyn.operating_point,
        "selection": selection,
        "selection_scores": {
            label: by_index[i].unit[dyn.kinetics_parameters[0]]
            - by_index[i].unit[dyn.kinetics_parameters[1]]
            for label, i in selection.items()
        },
        "initial_steady_states": initial,
        "steps": [{"name": s.name, "channel": s.channel, "factor": s.factor} for s in dyn.steps],
        "sample_time_s": dt,
        "separation": stats,
        "settling": settling,
        "series_time_s": time_s[thin - 1 :: thin].tolist(),
        "series": thinned,
        "series_row_order": ["reference", *[s.name for s in dyn.steps]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "conf/observability/observability.yaml")
    )
    parser.add_argument("--name", default="observability_v1")
    parser.add_argument("--output", default=None, help="explicit output path (smoke runs)")
    parser.add_argument("--smoke", action="store_true", help="3 draws, 2 points, 1 h dynamics")
    args = parser.parse_args()
    if args.smoke and args.output is None:
        parser.error("--smoke needs --output outside artifacts/")
    output = Path(args.output) if args.output else RUNS_DIR / f"{args.name}.json.gz"
    if output.exists():
        raise SystemExit(f"{output} exists; artifacts are never overwritten")

    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    prov = provenance(NO_HEAT_NETWORK, ignore_paths=[output] if not args.smoke else ())
    started = time.perf_counter()
    config = load_observability_config(args.config)
    sv = load_sensor_value_config(config.sensor_value_config)
    study = load_headroom_config(sv.headroom_config)
    nominal = study_process_config(study)
    measurement = load_process_config(study.process_config).measurement
    draws, rejected = parameter_draws(
        sv.draws, load_sensitivity_config(sv.sensitivity_config), nominal
    )
    points = operating_points(config.operating_points, nominal.actions.gas_velocity_m_per_s.minimum)
    if args.smoke:
        import dataclasses

        draws = draws[:3]
        points = [points[2], points[5]]
        config = dataclasses.replace(
            config, dynamics=dataclasses.replace(config.dynamics, horizon_s=3600.0)
        )

    def point_config(p: dict[str, Any]) -> Any:
        return operating_point_config(
            nominal,
            inlet_temperature_celsius=p["inlet_temperature_celsius"],
            gas_velocity_m_per_s=p["gas_velocity_m_per_s"],
        )

    lab = {}
    plant: dict[str, Any] = {}
    for p in points:
        base = point_config(p)
        lab[p["name"]] = steady_block(base, draws)
        lab[p["name"]]["input_context"] = input_context(base, study.scenarios.observed)
        print(f"lab {p['name']} done ({time.perf_counter() - started:.0f} s)", flush=True)
        for product, ratio in zip(
            config.plant_like.product_dry_solid_t_per_h, config.plant_like.ratios(), strict=True
        ):
            scaled, info = scale_solid_to_air_ratio(base, ratio, config.plant_like.ratio_iterations)
            key = f"{product:g}t_per_h"
            plant.setdefault(key, {})[p["name"]] = {"scaling": info, **steady_block(scaled, draws)}
        print(f"plant-like {p['name']} done ({time.perf_counter() - started:.0f} s)", flush=True)

    selection = kinetics_selection(draws, config.dynamics.kinetics_parameters)
    dyn_point = next(p for p in points if p["name"] == config.dynamics.operating_point)
    dynamics = dynamics_block(config, point_config(dyn_point), draws, measurement, selection)
    print(f"dynamics done ({time.perf_counter() - started:.0f} s)", flush=True)

    record = {
        "provenance": prov,
        "study": "observability (docs/NN_observability.md)",
        "config": config.raw,
        "config_fingerprint": config.fingerprint,
        "sensor_value_config": str(config.sensor_value_config.relative_to(REPO_ROOT)),
        "sensor_value_fingerprint": sv.fingerprint,
        "sensitivity_fingerprint": load_sensitivity_config(sv.sensitivity_config).fingerprint,
        "operating_point_source": {
            "path": str(config.operating_points.source.relative_to(REPO_ROOT)),
            "sha256": sha256(config.operating_points.source),
        },
        "smoke": bool(args.smoke),
        "measurement_config": {
            "relative_noise_std": measurement.relative_noise_std,
            "absolute_noise_std_temperature_k": measurement.absolute_noise_std_temperature_k,
            "sensor_time_constant_s": measurement.sensor_time_constant_s,
        },
        "exhaust_outlet_duct_heat_loss_k": nominal.exhaust.outlet_duct_heat_loss_k,
        "nominal_disturbances": {
            "spray_rate_kg_per_h": nominal.actions.spray_rate_kg_per_h.nominal,
            "solid_fraction": nominal.disturbances.suspension_solid_fraction.nominal,
            "ambient_temperature_celsius": nominal.disturbances.ambient_temperature_celsius.nominal,
            "ambient_relative_humidity": nominal.disturbances.ambient_relative_humidity.nominal,
            "nozzle_pressure_bar": nominal.actions.nozzle_pressure_bar.nominal,
        },
        "draws": [d.as_dict() for d in draws],
        "rejected_sequence_positions": rejected,
        "operating_points": points,
        "lab": lab,
        "plant_like": {
            "label": "plant-like, placeholder",
            "ratios_kg_per_kg": dict(
                zip(
                    [f"{p:g}t_per_h" for p in config.plant_like.product_dry_solid_t_per_h],
                    config.plant_like.ratios(),
                    strict=True,
                )
            ),
            "points": plant,
        },
        "dynamics": dynamics,
        "runtime_s": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as f:
        json.dump(record, f)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB, {record['runtime_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
