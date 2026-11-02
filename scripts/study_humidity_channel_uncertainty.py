r"""MA2: uncertainty of the moisture channel in kW -- error propagation at ONE operating point.

Occasion: the consortium partner now calculates with a condensing heat exchanger (MA1,
``heatnetwork``). This raises the importance of the exhaust air loading ``Y_out`` -- in
condensing mode, moisture delivers additional LATENT recovery instead of only raising the
dew point (and thus the sensible cooling limit). ``Y_out`` is an algebraic
function of bed temperature and product moisture (``simulation.exhaust.
reconstruct_exhaust_state``), both quantities whose prediction errors are already measured in the
learning-curve datasets as ``rmse_run_median_physical`` -- not to be
determined anew, only to be propagated.

Procedure (no new model, no new physics -- a pure perturbation calculation at a fixed
reference point with already existing building blocks):

1. Reference point: steady state of ``greybox_gab_placeholder.yaml`` at NOMINAL
   actions/disturbances (``solve_steady_state`` without variant) -- the same nominal point
   that ``simulation.exhaust``'s module docstring also refers to (Befund 31).
2. Bed temperature and product moisture at this point are perturbed INDIVIDUALLY by their measured
   error (see ``BED_TEMPERATURE_RMSE_K``/``PRODUCT_MOISTURE_RMSE_KG_PER_KG``
   below, with source), then BOTH simultaneously in the unfavourable
   sign combination (the one that maximises \|ΔQ_rec\| -- by trying all four
   sign combinations, not assumed).
3. Per perturbation: ``reconstruct_exhaust_state`` returns the perturbed ``Y_out``/``T_out`` anew
   (mass flow and inlet air humidity unchanged -- these are actions/disturbances, not
   predictions). ``Q_rec`` is recomputed from it in BOTH recovery modes
   (sensible: ``evaluation.recoverable_power.compute_recoverable_power_kw``; condensing:
   the same MA1 formula as ``heatnetwork.balance.evaluate_cascade``, rebuilt here directly from
   ``thermo.py`` -- a single point needs no cascade).
4. Scale caveat (as in ``heatnetwork``/``evaluation.recoverable_power`` throughout):
   the same ΔT_out/ΔY_out is evaluated at BOTH mass flows --
   ``dry_air_mass_flow_kg_per_s`` from the steady state itself (lab simulator,
   approx. 0.03 kg/s) and ``DISTRICT_MASS_FLOW_KG_PER_S`` (consortium model, 8 kg/s). Every
   district number is a scaling calculation, not a statement about a plant.

Not part of this task: a new error model, a re-evaluation of the learning curves themselves
(Befund 51/55 remain the source), a cascade with several sources/sinks.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from twinlink_ml.evaluation.recoverable_power import (
    DEW_POINT_MARGIN_K,
    DISTRICT_MASS_FLOW_KG_PER_S,
    compute_recoverable_power_kw,
)
from twinlink_ml.heatnetwork.config import DEFAULT_COLD_TEMPERATURE_CELSIUS
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.exhaust import reconstruct_exhaust_state
from twinlink_ml.simulation.sorption import isotherm_from_material
from twinlink_ml.simulation.steady_state import solve_steady_state
from twinlink_ml.simulation.thermo import (
    absolute_humidity_kg_per_kg,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "study_humidity_channel_uncertainty.json"

LAB_MASS_FLOW_LABEL = "lab_simulator"
DISTRICT_MASS_FLOW_LABEL = "district_partner_model"

# --- MA2a: errors used per channel, with source ----------------------------------------
#
# Source: artifacts/runs/arch_learning_curve_constant_trained.jsonl (Befund 55, commit
# 1b85ce4), model LSTM (best model per channel at this budget point, see below),
# n_train=64 (largest tested budget -- the most favourable approximation of "enough
# operating data"), test_set="constant_holdout" (constant-run holdout, HOLDOUT_SEED=999 --
# the operationally relevant one, because it does NOT depend on excitation experiments, see
# Befund 55 occasion), horizon h=36, median over 5 seeds of
# "rmse_run_median_physical/<kanal>/h36". Deliberately NOT the "own_pool"/"excited" test set
# (which checked generalisation to excited runs, irrelevant here).
BED_TEMPERATURE_RMSE_K = 0.5393090806307859
"""Median (5 Seeds) von rmse_run_median_physical/bed_temperature_celsius/h36, LSTM,
n_train=64, test_set=constant_holdout, arch_learning_curve_constant_trained.jsonl."""

PRODUCT_MOISTURE_RMSE_KG_PER_KG = 0.0014883589441298013
"""Median (5 Seeds) von rmse_run_median_physical/product_moisture_kg_per_kg/h36, LSTM,
n_train=64, test_set=constant_holdout, arch_learning_curve_constant_trained.jsonl."""


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """Reference operating point (nominal point) including the reconstructed exhaust air state."""

    bed_temperature_celsius: float
    particle_moisture_kg_per_kg: float
    dry_air_mass_flow_kg_per_s: float
    inlet_absolute_humidity_kg_per_kg: float
    outlet_temperature_celsius: float
    outlet_humidity_kg_per_kg: float


def _build_reference_point() -> ReferencePoint:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator)
    if not solution.feasible:
        raise RuntimeError(
            f"Nominalpunkt nicht zulaessig: {solution.infeasibility_reasons} -- "
            "MA2 braucht einen gueltigen Referenzpunkt."
        )

    backend = NumpyBackend()
    inlet_absolute_humidity = float(
        backend.to_numpy(
            absolute_humidity_kg_per_kg(
                backend,
                config.gas,
                backend.asarray(config.disturbances.ambient_temperature_celsius.nominal),
                backend.asarray(config.disturbances.ambient_relative_humidity.nominal),
            )
        )
    )

    isotherm = isotherm_from_material(config.material)
    t_gas_out, y_gas_out, outlet_temperature = reconstruct_exhaust_state(
        backend,
        config.gas,
        config.material,
        isotherm,
        config.exhaust,
        bed_temperature_celsius=backend.asarray(solution.bed_temperature_celsius),
        particle_moisture_kg_per_kg=backend.asarray(solution.particle_moisture_kg_per_kg),
        dry_air_mass_flow_kg_per_s=backend.asarray(solution.dry_air_mass_flow_kg_per_s),
        inlet_absolute_humidity_kg_per_kg=backend.asarray(inlet_absolute_humidity),
    )
    del t_gas_out  # == bed_temperature_celsius by model assumption, not needed here

    # Sanity check: reconstruct_exhaust_state() must reproduce the solver bit for bit
    # (see the exhaust.py module docstring, "Gegen solve_steady_state() ... bitgleiche
    # Uebereinstimmung").
    reconstructed_y = float(backend.to_numpy(y_gas_out))
    if abs(reconstructed_y - solution.outlet_humidity_kg_per_kg) > 1e-9:
        raise RuntimeError(
            f"reconstruct_exhaust_state()={reconstructed_y!r} weicht vom Loeser "
            f"({solution.outlet_humidity_kg_per_kg!r}) ab -- Referenzpunkt inkonsistent."
        )

    return ReferencePoint(
        bed_temperature_celsius=solution.bed_temperature_celsius,
        particle_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        dry_air_mass_flow_kg_per_s=solution.dry_air_mass_flow_kg_per_s,
        inlet_absolute_humidity_kg_per_kg=inlet_absolute_humidity,
        outlet_temperature_celsius=float(backend.to_numpy(outlet_temperature)),
        outlet_humidity_kg_per_kg=reconstructed_y,
    )


def _reconstruct(
    reference: ReferencePoint,
    config_gas: object,
    config_material: object,
    config_exhaust: object,
    isotherm: object,
    *,
    bed_temperature_celsius: float,
    particle_moisture_kg_per_kg: float,
) -> tuple[float, float]:
    """Exhaust air state at perturbed bed temperature/product moisture, otherwise unchanged."""
    backend = NumpyBackend()
    _t_gas_out, y_out, t_out = reconstruct_exhaust_state(
        backend,
        config_gas,  # type: ignore[arg-type]
        config_material,  # type: ignore[arg-type]
        isotherm,  # type: ignore[arg-type]
        config_exhaust,  # type: ignore[arg-type]
        bed_temperature_celsius=backend.asarray(bed_temperature_celsius),
        particle_moisture_kg_per_kg=backend.asarray(particle_moisture_kg_per_kg),
        dry_air_mass_flow_kg_per_s=backend.asarray(reference.dry_air_mass_flow_kg_per_s),
        inlet_absolute_humidity_kg_per_kg=backend.asarray(
            reference.inlet_absolute_humidity_kg_per_kg
        ),
    )
    return float(backend.to_numpy(t_out)), float(backend.to_numpy(y_out))


def _q_rec_condensing_kw(
    gas: object,
    outlet_temperature_celsius: float,
    outlet_humidity_kg_per_kg: float,
    *,
    mass_flow_kg_per_s: float,
    cold_temperature_celsius: float,
) -> float:
    """MA1 formula (``heatnetwork.balance``) directly from ``thermo.py`` -- a single point
    needs no cascade, see module docstring."""
    backend = NumpyBackend()
    y_sat_cold = float(
        backend.to_numpy(
            saturation_humidity_kg_per_kg(
                backend,
                gas,  # type: ignore[arg-type]
                backend.asarray(cold_temperature_celsius),
            )
        )
    )
    condensate_kg_per_s = max(0.0, outlet_humidity_kg_per_kg - y_sat_cold) * mass_flow_kg_per_s
    dh_v = float(
        backend.to_numpy(latent_heat_j_per_kg(backend, backend.asarray(cold_temperature_celsius)))
    )
    q_latent_w = condensate_kg_per_s * dh_v
    q_sensible_w = (
        mass_flow_kg_per_s
        * (outlet_temperature_celsius - cold_temperature_celsius)
        * (gas.cp_dry_air_j_per_kg_k + outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k)  # type: ignore[attr-defined]
    )
    return (q_sensible_w + q_latent_w) / 1000.0


def main() -> None:
    reference = _build_reference_point()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    isotherm = isotherm_from_material(config.material)

    print(
        f"Referenzpunkt: T_bed={reference.bed_temperature_celsius:.3f} C, "
        f"X_p={reference.particle_moisture_kg_per_kg:.5f} kg/kg, "
        f"m_L={reference.dry_air_mass_flow_kg_per_s:.5f} kg/s (Labor), "
        f"Y_in={reference.inlet_absolute_humidity_kg_per_kg:.5f} kg/kg\n"
        f"  -> T_out={reference.outlet_temperature_celsius:.3f} C, "
        f"Y_out={reference.outlet_humidity_kg_per_kg:.5f} kg/kg"
    )

    def q_rec_both_modes(
        t_out: float, y_out: float, mass_flow_kg_per_s: float
    ) -> tuple[float, float]:
        q_sensible = compute_recoverable_power_kw(
            NumpyBackend(),
            config.gas,
            t_out,
            y_out,
            mass_flow_kg_per_s=mass_flow_kg_per_s,
            dew_point_margin_k=DEW_POINT_MARGIN_K,
        )
        q_condensing = _q_rec_condensing_kw(
            config.gas,
            t_out,
            y_out,
            mass_flow_kg_per_s=mass_flow_kg_per_s,
            cold_temperature_celsius=DEFAULT_COLD_TEMPERATURE_CELSIUS,
        )
        return float(q_sensible), float(q_condensing)

    mass_flow_scales = (
        (LAB_MASS_FLOW_LABEL, reference.dry_air_mass_flow_kg_per_s),
        (DISTRICT_MASS_FLOW_LABEL, DISTRICT_MASS_FLOW_KG_PER_S),
    )

    baseline_by_scale = {
        label: q_rec_both_modes(
            reference.outlet_temperature_celsius, reference.outlet_humidity_kg_per_kg, flow
        )
        for label, flow in mass_flow_scales
    }

    scenarios: dict[str, tuple[float, float]] = {
        "temperature_plus": (BED_TEMPERATURE_RMSE_K, 0.0),
        "temperature_minus": (-BED_TEMPERATURE_RMSE_K, 0.0),
        "moisture_plus": (0.0, PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "moisture_minus": (0.0, -PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_pp": (BED_TEMPERATURE_RMSE_K, PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_pm": (BED_TEMPERATURE_RMSE_K, -PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_mp": (-BED_TEMPERATURE_RMSE_K, PRODUCT_MOISTURE_RMSE_KG_PER_KG),
        "both_mm": (-BED_TEMPERATURE_RMSE_K, -PRODUCT_MOISTURE_RMSE_KG_PER_KG),
    }

    results: dict[str, object] = {
        "reference": {
            "bed_temperature_celsius": reference.bed_temperature_celsius,
            "particle_moisture_kg_per_kg": reference.particle_moisture_kg_per_kg,
            "dry_air_mass_flow_kg_per_s_lab": reference.dry_air_mass_flow_kg_per_s,
            "inlet_absolute_humidity_kg_per_kg": reference.inlet_absolute_humidity_kg_per_kg,
            "outlet_temperature_celsius": reference.outlet_temperature_celsius,
            "outlet_humidity_kg_per_kg": reference.outlet_humidity_kg_per_kg,
        },
        "errors": {
            "bed_temperature_rmse_k": BED_TEMPERATURE_RMSE_K,
            "product_moisture_rmse_kg_per_kg": PRODUCT_MOISTURE_RMSE_KG_PER_KG,
            "source": "artifacts/runs/arch_learning_curve_constant_trained.jsonl, LSTM, "
            "n_train=64, test_set=constant_holdout, h=36, Median 5 Seeds",
        },
        "baseline_kw": {
            label: {"sensible": s, "condensing": c} for label, (s, c) in baseline_by_scale.items()
        },
        "scenarios": {},
    }

    print(
        f"\n{'Szenario':<16}{'dT [K]':>9}{'dX_p [kg/kg]':>14}{'Y_out':>10}"
        f"{'dQ_sens[kW]':>13}{'dQ_sens[%]':>11}{'dQ_cond[kW]':>13}{'dQ_cond[%]':>11}  Skala"
    )
    for name, (dt, dx) in scenarios.items():
        t_out, y_out = _reconstruct(
            reference,
            config.gas,
            config.material,
            config.exhaust,
            isotherm,
            bed_temperature_celsius=reference.bed_temperature_celsius + dt,
            particle_moisture_kg_per_kg=reference.particle_moisture_kg_per_kg + dx,
        )
        results["scenarios"][name] = {
            "delta_bed_temperature_k": dt,
            "delta_particle_moisture_kg_per_kg": dx,
            "outlet_temperature_celsius": t_out,
            "outlet_humidity_kg_per_kg": y_out,
            "by_scale": {},
        }
        for label, flow in mass_flow_scales:
            q_sensible, q_condensing = q_rec_both_modes(t_out, y_out, flow)
            base_sensible, base_condensing = baseline_by_scale[label]
            d_sensible_kw = q_sensible - base_sensible
            d_condensing_kw = q_condensing - base_condensing
            d_sensible_pct = (
                100.0 * d_sensible_kw / base_sensible if base_sensible else float("nan")
            )
            d_condensing_pct = (
                100.0 * d_condensing_kw / base_condensing if base_condensing else float("nan")
            )
            results["scenarios"][name]["by_scale"][label] = {
                "q_sensible_kw": q_sensible,
                "q_condensing_kw": q_condensing,
                "delta_sensible_kw": d_sensible_kw,
                "delta_sensible_pct": d_sensible_pct,
                "delta_condensing_kw": d_condensing_kw,
                "delta_condensing_pct": d_condensing_pct,
            }
            print(
                f"{name:<16}{dt:>9.3f}{dx:>14.6f}{y_out:>10.5f}"
                f"{d_sensible_kw:>13.4f}{d_sensible_pct:>10.2f}%"
                f"{d_condensing_kw:>13.4f}{d_condensing_pct:>10.2f}%  {label}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGeschrieben: {OUT_PATH}")


if __name__ == "__main__":
    main()
