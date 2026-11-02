r"""Diagnostics of the process model -- prerequisite for an interpretable comparison.

The first three functions here answer questions that must be settled **before** the first
architecture comparison:

1. :func:`steady_state_report` -- does the operating point lie in the plausible range at all?
2. :func:`time_constant_report` -- which time constants does the system have, and which
   sequence length follows from them?
3. :func:`static_explained_variance` -- **is the target dynamic at all?**

A fourth, :func:`mass_balance_residual`, answers a different question -- not before the
architecture comparison but before every training run: does the solid balance of a
generated dataset close within the numerical tolerance? It is the basis of
``tests/test_greybox_balances.py::test_solid_balance_closes_within_tolerance_in_continuous_mode``
and meant as a dataset assertion (see its own docstring), not as a
dataset report like the first three.

On point 3, the most unpleasant one
-----------------------------------
If a target can already be explained by a static mapping of the current actions and
disturbances, there is nothing for a sequence model to learn. The comparison
GRU / xLSTM / Mamba / Transformer then measures optimisation noise, and the result is
arbitrary -- an artifact that looks exactly like a result.

For the instantaneous power of the grey-box model this static share is about
:math:`R^2 \approx 0.98`: the heater power is algebraic in
:math:`(T_{in}, u_{gas}, T_{amb})`, and the process state acts only through the gas density
and the bed pressure drop. The dynamic variables are the **quality variables**
(particle moisture, particle size), not the energy consumption.

This metric therefore belongs in every dataset report, and the baseline list of the
work plan needs a fourth entry: a **static regression** on the
current inputs. Persistence is the wrong reference for this target -- it
loses every action step and thus makes every architecture look good.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .backend import FloatArray
from .base import Trajectory
from .config import ProcessConfig
from .greybox import IDX_SOLID_MASS, GreyBoxFluidBedGranulator
from .steady_state import solve_steady_state

_SECONDS_PER_HOUR = 3600.0
"""Dieselbe Umrechnung wie ``greybox._SECONDS_PER_HOUR`` -- hier dupliziert statt
importiert, weil es sich um eine private Modulkonstante handelt."""

_MODE_MATCH_RTOL = 1e-6
"""Relative Toleranz für den Wertevergleich in :func:`_match_single_mode`. Die beiden
strukturellen Moden (Feststoff-Regelkreis, Feststoff-Verweilzeit) sind reine
Konfigurationskonstanten mit geschlossener Form (s. dort) -- der Jacobi-Pfad in
:meth:`GreyBoxFluidBedGranulator.time_constants` reproduziert sie numerisch praktisch
bitgleich (siehe ``tests/test_diagnostics.py``, ``tau_3_s``-Bitgleichheits-Test), 1e-6
relativ lässt trotzdem Luft für andere Konfigurationen/Backends."""


@dataclass(frozen=True, slots=True)
class SteadyStateReport:
    """Steady state under constant, nominal excitation."""

    bed_temperature_celsius: float
    particle_moisture_kg_per_kg: float
    diameter_um: float
    evaporation_kg_per_h: float
    dry_air_flow_kg_per_h: float
    product_flow_kg_per_h: float
    saturation_degree: float
    minimum_fluidization_velocity_m_per_s: float
    terminal_velocity_m_per_s: float
    power_total_kw: float
    power_shares: dict[str, float]
    residual_inf_norm: float
    """Skaliertes Residuum des Lösers am Berichtspunkt, siehe ``steady_state.py``."""
    feasible: bool
    """Zulässigkeitsflag des Lösers (Fluidisierungsfenster, Sättigung, Feuchtebereich)."""

    def format(self) -> str:
        """Human-readable report for run logs and coordination with TUHH-A."""
        shares = ", ".join(f"{k} {v:5.1%}" for k, v in self.power_shares.items())
        return (
            f"Betttemperatur          {self.bed_temperature_celsius:8.2f} °C\n"
            f"Partikelfeuchte         {self.particle_moisture_kg_per_kg:8.5f} kg/kg\n"
            f"d50                     {self.diameter_um:8.1f} µm\n"
            f"Verdampfungsrate        {self.evaporation_kg_per_h:8.3f} kg/h\n"
            f"Trockenluftstrom        {self.dry_air_flow_kg_per_h:8.1f} kg/h\n"
            f"Produktmassenstrom      {self.product_flow_kg_per_h:8.3f} kg/h\n"
            f"Sättigungsgrad Abluft   {self.saturation_degree:8.4f} -\n"
            f"Fluidisierungsfenster   {self.minimum_fluidization_velocity_m_per_s:8.3f}"
            f" .. {self.terminal_velocity_m_per_s:.3f} m/s\n"
            f"Leistung                {self.power_total_kw:8.3f} kW  ({shares})\n"
            f"Zulässig                {'ja' if self.feasible else 'nein':>8s}\n"
            f"Residuum (skaliert)     {self.residual_inf_norm:8.2e}"
        )


def steady_state_report(simulator: GreyBoxFluidBedGranulator) -> SteadyStateReport:
    """Exact steady state under nominal excitation.

    Since step 3 replaces the transient integration with the exact solver from
    :mod:`twinlink_ml.simulation.steady_state`. A simulation duration is thus no longer a
    parameter -- the earlier transient version reported, at
    ``duration_s = 30 000 s``, an intermediate state instead of the steady state
    (610.1 µm instead of 750.0 µm for d50, see ``docs/04_stand_ap1.md``, Befund A). The
    signature is thus narrower than before: a ``ProcessSimulator`` in general cannot
    supply a right-hand side whose root could be solved for (Dyssol cannot),
    so the exact report is restricted to :class:`GreyBoxFluidBedGranulator`.
    """
    solution = solve_steady_state(simulator)
    total = solution.diagnostics["power_total_kw"]
    shares = {
        key: solution.diagnostics[f"power_{key}_kw"] / total
        for key in ("heater", "fan", "pump", "atomiser")
    }
    return SteadyStateReport(
        bed_temperature_celsius=solution.bed_temperature_celsius,
        particle_moisture_kg_per_kg=solution.particle_moisture_kg_per_kg,
        diameter_um=solution.diameter_m * 1e6,
        evaporation_kg_per_h=solution.diagnostics["evaporation_rate_kg_per_s"] * 3600.0,
        dry_air_flow_kg_per_h=solution.dry_air_mass_flow_kg_per_s * 3600.0,
        product_flow_kg_per_h=solution.diagnostics["product_mass_flow_kg_per_s"] * 3600.0,
        saturation_degree=solution.diagnostics["saturation_degree"],
        minimum_fluidization_velocity_m_per_s=solution.diagnostics[
            "minimum_fluidization_velocity_m_per_s"
        ],
        terminal_velocity_m_per_s=solution.diagnostics["terminal_velocity_m_per_s"],
        power_total_kw=total,
        power_shares=shares,
        residual_inf_norm=solution.residual_inf_norm,
        feasible=solution.feasible,
    )


def _solid_residence_time_s(
    config: ProcessConfig, spray_rate_kg_per_h: float, solid_fraction: float
) -> float:
    """Closed form of the solid residence time (Befund 47/M1, ``docs/04_stand_ap1.md``).

    Bed mass set point (``geometry.bed_mass_solid_setpoint_kg``) divided by the
    steady solid feed at the linearisation point (spray suspension plus
    seed flow, the same balance as ``GreyBoxFluidBedGranulator.evaluate_derivative``,
    ``solid_inflow = spray_rate * solid_fraction + seed_flow``). A pure
    configuration constant at the nominal operating point -- independent of
    ``bed_temperature_celsius`` as long as spray rate/solid fraction/bed mass set point
    stay unchanged (numerically confirmed at three steady states, Befund 47/M1).
    """
    spray_solid_kg_per_s = spray_rate_kg_per_h / _SECONDS_PER_HOUR * solid_fraction
    seed_flow_kg_per_s = config.operation.seed_mass_flow_kg_per_h / _SECONDS_PER_HOUR
    solid_inflow_kg_per_s = spray_solid_kg_per_s + seed_flow_kg_per_s
    return config.geometry.bed_mass_solid_setpoint_kg / solid_inflow_kg_per_s


def _match_single_mode(taus: list[float], reference: float, *, rtol: float) -> int | None:
    """Index in ``taus`` whose value deviates by less than ``rtol`` relative from ``reference``.

    ``None`` for no match or for more than one match.

    The rank position (``tau_1_s`` .. ``tau_4_s``, sorted from fast to slow) is
    no stable identity: which rank the solid control loop mode
    (``mass_control_time_constant_s``, see :func:`time_constant_report`) takes depends
    on the operating point -- at 60 °C inlet temperature rank 2, at 90/120 °C rank 3
    (numerically checked). Hence identification via the numerical value of the respective
    closed form, not via the position in the sorted list. More than one
    match (two modes by chance within the tolerance) is not guessed but treated as
    not identifiable.
    """
    matches = [i for i, tau in enumerate(taus) if math.isclose(tau, reference, rel_tol=rtol)]
    return matches[0] if len(matches) == 1 else None


def time_constant_report(
    simulator: GreyBoxFluidBedGranulator, *, mode_match_rtol: float = _MODE_MATCH_RTOL
) -> dict[str, float | str | None]:
    """Time constants at the exact steady state plus derived metrics.

    Since step 3b the linearisation point comes from the exact solver
    (:mod:`twinlink_ml.simulation.steady_state`) instead of from a transient
    integration -- the same reasoning as for :func:`steady_state_report` (Befund A):
    two different notions of the steady state in the same module would be exactly the
    divergence that was avoided there. Numerically almost without consequence (see
    ``tests/test_diagnostics.py``): three of the four time constants change relatively by
    ~1e-10 (noise of two different floating-point paths on the same point), the
    fourth (solid control loop, time constant = ``mass_control_time_constant_s``) is
    path-independent and stays bit-identical.

    ``recommended_sequence_length_fast`` = ``ceil(5 * tau_schnellste / dt)``: lower bound
    for the sampling resolution -- this many steps are needed for the fastest mode
    to be visible at all. **NOT a recommendation for the window length** -- use
    ``recommended_sequence_length_coupled_slow`` for that. ``steps_for_slowest_mode``
    (``ceil(3 * tau_langsamste / dt)``) computes the same factor-3 window length against the
    slowest of ALL four modes, including the solid residence time (Befund 47/M1,
    ~16000 s) -- unusably large as a window length, see ``docs/04_stand_ap1.md``,
    Befund 49.

    Two of the four modes are pure configuration constants with a closed form,
    independent of the operating point (Befund 47/M1): the solid control loop
    (``tau = mass_control_time_constant_s``) and the solid residence time (see
    :func:`_solid_residence_time_s`). Only the remaining, coupled subsystem
    (water mass, bed temperature) depends on the operating point and thus determines the
    window length actually needed -- ``tau_coupled_fast_s``/``tau_coupled_slow_s`` are
    its two modes, identified by value comparison (see :func:`_match_single_mode`,
    tolerance ``mode_match_rtol``), not by rank position in ``tau_1_s``..``tau_4_s``
    (those are kept unchanged, backward compatibility of the logs). If one of the
    two structural modes cannot be assigned unambiguously (no or more than one
    match within the tolerance -- e.g. for a configuration with coincidentally
    coinciding time constants or a different number of states),
    ``tau_mass_control_s``/``tau_residence_s``/``tau_coupled_fast_s``/
    ``tau_coupled_slow_s``/``recommended_sequence_length_coupled_slow`` stay ``None``,
    with a note in ``mode_identification_note``.
    """
    solution = solve_steady_state(simulator)
    backend = simulator.backend
    state = backend.asarray(solution.state[None, :])
    action = backend.asarray(simulator.nominal_action(1))
    disturbance = backend.asarray(simulator.nominal_disturbance(1))
    try:
        time_constants = dict(simulator.time_constants(state, action, disturbance))
    except NotImplementedError as error:
        raise TypeError(f"{type(simulator).__name__} liefert keine Zeitkonstanten") from error
    taus = sorted(v for k, v in time_constants.items() if k.startswith("tau_"))
    report: dict[str, float | str | None] = dict(time_constants)
    if taus:
        dt = simulator.sample_time_s
        report["sample_time_s"] = dt
        report["fastest_mode_samples"] = taus[0] / dt
        report["recommended_sequence_length_fast"] = float(np.ceil(5.0 * taus[0] / dt))
        report["steps_for_slowest_mode"] = float(np.ceil(3.0 * taus[-1] / dt))

        action_np = backend.to_numpy(action)[0]
        disturbance_np = backend.to_numpy(disturbance)[0]
        mass_control_tau = simulator.config.operation.mass_control_time_constant_s
        residence_tau = _solid_residence_time_s(
            simulator.config, float(action_np[2]), float(disturbance_np[2])
        )
        mass_control_idx = _match_single_mode(taus, mass_control_tau, rtol=mode_match_rtol)
        residence_idx = _match_single_mode(taus, residence_tau, rtol=mode_match_rtol)

        report["tau_mass_control_s"] = (
            taus[mass_control_idx] if mass_control_idx is not None else None
        )
        report["tau_residence_s"] = taus[residence_idx] if residence_idx is not None else None
        report["tau_coupled_fast_s"] = None
        report["tau_coupled_slow_s"] = None
        report["recommended_sequence_length_coupled_slow"] = None
        report["mode_identification_note"] = None

        if mass_control_idx is None or residence_idx is None:
            report["mode_identification_note"] = (
                "mass_control- oder residence-Mode nicht eindeutig identifizierbar "
                f"(rtol={mode_match_rtol:g}) -- coupled_fast/slow und "
                "recommended_sequence_length_coupled_slow nicht berechnet."
            )
        else:
            structural_indices = {mass_control_idx, residence_idx}
            coupled_indices = [i for i in range(len(taus)) if i not in structural_indices]
            if len(coupled_indices) != 2:
                report["mode_identification_note"] = (
                    f"{len(coupled_indices)} statt 2 verbleibende Moden nach Abzug von "
                    "mass_control/residence -- Zustandszahl weicht vom erwarteten Fall "
                    "(N_STATE=4, ein gekoppeltes Paar) ab, coupled_fast/slow nicht "
                    "berechnet."
                )
            else:
                coupled_fast, coupled_slow = sorted(taus[i] for i in coupled_indices)
                report["tau_coupled_fast_s"] = coupled_fast
                report["tau_coupled_slow_s"] = coupled_slow
                report["recommended_sequence_length_coupled_slow"] = float(
                    np.ceil(3.0 * coupled_slow / dt)
                )
    return report


def static_explained_variance(
    trajectory: Trajectory, *, quadratic: bool = True
) -> dict[str, float]:
    r"""Share of the variance of each output variable that a static mapping explains.

    Regression of :math:`y[k]` on :math:`(\mathbf u[k], \mathbf z[k])` -- without any
    history. Optionally with quadratic terms, so as not to confuse nonlinearity with
    dynamics.

    Reading:

    * :math:`R^2 > 0.95` -- the variable is practically static. A sequence model can show
      nothing here; the comparison is not meaningful.
    * :math:`R^2 < 0.8` -- there is a real dynamic share. Only here is the question
      "which architecture" sensibly posed.

    This metric is deliberately to be computed **before** the first model training, for every
    dataset and every target. It costs seconds and prevents months of
    interpreting noise.
    """
    n_runs, n_steps, _ = trajectory.actions.shape
    inputs = np.concatenate([trajectory.actions, trajectory.disturbances], axis=-1)
    inputs = inputs.reshape(n_runs * n_steps, -1)
    if quadratic:
        inputs = np.concatenate([inputs, inputs**2], axis=-1)
    design = np.concatenate([inputs, np.ones((inputs.shape[0], 1))], axis=-1)

    names = trajectory.metadata.get("output_names", ())
    outputs = trajectory.outputs.reshape(n_runs * n_steps, -1)
    result: dict[str, float] = {}
    for index in range(outputs.shape[1]):
        target = outputs[:, index]
        variance = float(np.var(target))
        label = names[index] if index < len(names) else f"output_{index}"
        if variance <= 0.0:
            result[label] = float("nan")
            continue
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        result[label] = 1.0 - float(np.var(residual)) / variance
    return result


def mass_balance_residual(trajectory: Trajectory, *, solid_index: int = IDX_SOLID_MASS) -> float:
    """Relative residual of the integral solid balance of a rollout.

    Meant as an assertion for generated datasets: a dataset with a
    balance residual above :math:`10^{-6}` should not go into training.
    """
    dt = float(trajectory.time_s[1] - trajectory.time_s[0])
    product = trajectory.diagnostics["product_mass_flow_kg_per_s"]
    accumulated = trajectory.states[:, -1, solid_index] - trajectory.states[:, 0, solid_index]
    withdrawn = np.sum(product, axis=1) * dt
    scale = np.maximum(np.abs(accumulated) + np.abs(withdrawn), 1e-12)
    return float(np.max(np.abs(accumulated + withdrawn - _solid_inflow(trajectory, dt)) / scale))


def _solid_inflow(trajectory: Trajectory, dt: float) -> FloatArray:
    spray_rate = trajectory.actions[..., 2] / 3600.0
    solid_fraction = trajectory.disturbances[..., 2]
    return np.asarray(np.sum(spray_rate * solid_fraction, axis=1) * dt, dtype=np.float64)
