r"""Closed loop of the sensor-value study: truth plant, nominal model, feedback, bias-corrected R1.

One call of :func:`run_pairs` runs one truth config (one parameter draw, or the nominal model) on
a batch of scenarios, each with its own margin and its own feedback object, in lockstep on one
:class:`~twinlink_ml.headroom.truth.TruthPlant` batch.

Per pair
--------
* **Initial condition** (all variants alike): the nominal initial condition of Befund 105
  (:func:`~twinlink_ml.headroom.study.initial_conditions`: steady optimum at the first disturbance
  sample with bound ``limit - start_margin``); the truth has run at that command for ``warm_up_s``
  (:func:`~twinlink_ml.headroom.study.warm_up`, the robustness rule of Befund 105; passed in, as
  it depends on the draw and the scenario only).
* **Pre-roll**: ``pre_roll_s`` of the first disturbance sample held constant, prepended to the
  scenario, with the controller and the bias filter in the loop -- the plant is in continuous
  operation, and with a model mismatch of several 0.01 kg/kg the truth would otherwise start far
  above the limit whatever the margin. Energy and violations of the pre-roll are not counted.
* **Controller** :class:`BiasCorrectedController`: R1 (:class:`~twinlink_ml.headroom.controllers.
  QuasiStationaryController`, nominal model, its own steady optimiser per pair) with the moisture
  bound ``limit - margin - b``, i.e. :math:`X_{model} + b \le X_{lim} - m`. With :math:`b = 0` it
  is R1 exactly (the bound is the same floating-point number).
* **Model output** :math:`\hat x`: the nominal model simulated alongside (a second truth plant
  with the nominal config) with the same realised commands and disturbances from the nominal
  initial state -- the open-loop model prediction of the current moisture. Without a model
  mismatch it equals the truth bit for bit, so with an ideal measurement :math:`b \equiv 0`.
* **Feedback grid**: 25 s blocks (five truth samples); at the end of every block the feedback
  value :math:`y` and :math:`\hat x` give :math:`e = y - \hat x`, filtered by
  :class:`~.feedback.BiasFilter`. The controller uses the bias at each interval start.
  Exogenous block means for the soft sensor: realised commands (rate limited exactly as in the
  truth plant, checked at every interval end), the realised inlet temperature including the
  scenario's offset, spray rate and nozzle pressure as applied, ambient air and solid fraction.
* **Feedback interface** (follow-up study, docs/29): each feedback object updates its own bias
  estimator (:meth:`~.feedback.Feedback.bias_filter`) through :meth:`~.feedback.Feedback.feed`;
  for F0-F3 that is the update of the original study, arithmetic unchanged (digest test). F5 also
  receives the true process responses at the block ends (bed temperature, exhaust temperature =
  bed temperature minus the duct loss of the truth config, exhaust humidity of the truth model),
  which it measures itself with the sensor model.

A failed steady optimisation holds the previous command and is counted (as Befund 105).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from twinlink_ml.evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH
from twinlink_ml.headroom.config import HeadroomStudyConfig
from twinlink_ml.headroom.controllers import QuasiStationaryController
from twinlink_ml.headroom.scenarios import Scenario
from twinlink_ml.headroom.study import Solvers, initial_conditions, warm_up
from twinlink_ml.headroom.truth import TruthPlant
from twinlink_ml.simulation import ProcessConfig

from .feedback import Feedback

__all__ = [
    "BiasCorrectedController",
    "PairRecord",
    "extend_scenario",
    "run_pairs",
    "warmed_up_states",
]

_HOUR_S = 3600.0


class BiasCorrectedController:
    """R1 with a zeroth-order bias correction: steady optimum with bound ``limit - margin - b``."""

    def __init__(self, solvers: Solvers, margin: float) -> None:
        self._r1 = QuasiStationaryController(solvers.steady, margin)
        self.margin = margin

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        limit: float,
        bias: float,
    ) -> np.ndarray:
        """Command for interval ``k``; ``bias = 0`` gives the R1 command exactly."""
        return self._r1.command(k, state, last_command, disturbance, limit - bias)


@dataclass
class PairRecord:
    """Truth record of one pair over the evaluation window (JSON-serialisable via ``as_dict``)."""

    variant: str
    draw: int | None
    scenario: str
    kind: str
    index: int
    margin: float
    energy_kwh: float
    co2_kg: float
    electric_kwh: float
    gas_kwh: float
    max_violation: float
    violation_samples: int
    max_bed_temperature: float
    failures: int
    failures_pre_roll: int
    bias_end_pre_roll: float
    bias_one_hour_before_end_pre_roll: float
    bias_min: float
    bias_max: float
    bias_mean: float
    model_error_mean: float
    feedback_error_rmse: float
    feedback_error_max_abs: float
    inlet_command_mean: float
    gas_velocity_command_mean: float
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return dataclasses.asdict(self)


def extend_scenario(scenario: Scenario, pre_roll_s: float) -> Scenario:
    """The scenario with ``pre_roll_s`` of its first sample prepended (constant disturbance)."""
    n = round(pre_roll_s / scenario.sample_time_s)
    if n == 0:
        return scenario
    if scenario.change_time_s is not None:
        raise ValueError("the pre-roll is defined for scenarios without a changeover")
    head = np.repeat(scenario.disturbances[:, :1], n, axis=1)
    return dataclasses.replace(
        scenario,
        disturbances=np.concatenate([head, scenario.disturbances], axis=1),
        moisture_limit=np.concatenate(
            [np.full(n, scenario.moisture_limit[0]), scenario.moisture_limit]
        ),
    )


def pair_initial_conditions(
    study: HeadroomStudyConfig, nominal: ProcessConfig, scenarios: Sequence[Scenario]
) -> tuple[list[Solvers], np.ndarray, np.ndarray]:
    """Fresh solvers per pair and the nominal initial states ``(n, 4)`` and commands ``(n, 2)``."""
    pair_solvers = [Solvers(study, nominal) for _ in scenarios]
    states, commands = [], []
    for solvers, scenario in zip(pair_solvers, scenarios, strict=True):
        s, c = initial_conditions(solvers, [scenario])
        states.append(s[0])
        commands.append(c[0])
    return pair_solvers, np.array(states), np.array(commands)


def warmed_up_states(
    study: HeadroomStudyConfig,
    nominal: ProcessConfig,
    truth: ProcessConfig,
    scenarios: Sequence[Scenario],
    warm_up_s: float,
) -> np.ndarray:
    """Truth states ``(n, 4)`` after ``warm_up_s`` at the nominal initial commands."""
    _, states, commands = pair_initial_conditions(study, nominal, scenarios)
    return warm_up(truth, states, commands, scenarios, warm_up_s)


def run_pairs(
    study: HeadroomStudyConfig,
    nominal: ProcessConfig,
    scenarios: Sequence[Scenario],
    margins: Sequence[float],
    feedbacks: Sequence[Feedback],
    *,
    truth: ProcessConfig | None,
    truth_initial_states: np.ndarray | None,
    pre_roll_s: float,
    measurement_interval_s: float,
    bias_time_constant_s: float,
    variant: str,
    draw: int | None,
    keep_arrays: bool = False,
) -> list[PairRecord]:
    """Run the pairs of one truth config, see the module docstring.

    ``truth=None`` runs the nominal model as truth (no mismatch); ``truth_initial_states`` are
    the warmed-up truth states ``(n, 4)`` (``None``: the nominal initial states, no warm-up).
    """
    n = len(scenarios)
    if not (len(margins) == len(feedbacks) == n):
        raise ValueError("one margin and one feedback per scenario")
    dt = study.scenarios.sample_time_s
    spi = round(study.control.interval_s / dt)
    spb = round(measurement_interval_s / dt)
    if spb * dt != measurement_interval_s or spi % spb:
        raise ValueError("the feedback grid must divide the control interval into whole samples")
    blocks = spi // spb
    pair_solvers, nominal_states, initial_commands = pair_initial_conditions(
        study, nominal, scenarios
    )
    truth_config = nominal if truth is None else truth
    truth_states = nominal_states if truth_initial_states is None else truth_initial_states
    extended = [extend_scenario(s, pre_roll_s) for s in scenarios]
    n_pre = round(pre_roll_s / dt)
    if n_pre % spi:
        raise ValueError("the pre-roll must be a whole number of control intervals")
    k_pre = n_pre // spi
    plant_args = {
        "interval_s": study.control.interval_s,
        "gas_heater_efficiency": study.control.gas_heater_efficiency,
    }
    plant = TruthPlant(truth_config, extended, truth_states, initial_commands, **plant_args)
    need_model = any(f.uses_model_output for f in feedbacks)
    model = (
        TruthPlant(nominal, extended, nominal_states, initial_commands, **plant_args)
        if need_model
        else None
    )
    controllers = [
        BiasCorrectedController(s, float(m)) for s, m in zip(pair_solvers, margins, strict=True)
    ]
    filters = [f.bias_filter(bias_time_constant_s, measurement_interval_s) for f in feedbacks]
    duct_loss = truth_config.exhaust.outlet_duct_heat_loss_k
    actions = truth_config.actions
    rates = np.array(
        [
            actions.inlet_temperature_celsius.max_rate_per_s,
            actions.gas_velocity_m_per_s.max_rate_per_s,
        ]
    )
    step_limit = rates * dt
    nozzle = actions.nozzle_pressure_bar.nominal
    dist_all = np.stack([s.disturbances for s in extended])  # (n, 5, samples)
    realised = np.array(initial_commands, dtype=np.float64)
    failures = np.zeros(n, dtype=int)
    failures_pre = np.zeros(n, dtype=int)
    bias_trace = np.zeros((n, plant.n_intervals))
    errors_fb: list[list[float]] = [[] for _ in range(n)]
    model_err: list[list[float]] = [[] for _ in range(n)]
    last = np.array(initial_commands, dtype=np.float64)
    for k in range(plant.n_intervals):
        disturbance = plant.current_disturbance()
        limit = plant.current_limit()
        last_realised = plant.last_commands
        new = np.empty_like(last)
        for i, controller in enumerate(controllers):
            bias = filters[i].value
            bias_trace[i, k] = bias
            try:
                new[i] = controller.command(
                    k, plant.state[i], last_realised[i], disturbance[i], limit[i], bias
                )
            except RuntimeError:
                if k < k_pre:
                    failures_pre[i] += 1
                else:
                    failures[i] += 1
                new[i] = last[i]
        start = k * spi
        exo = np.empty((n, spi, 7))
        for j in range(spi):
            realised = realised + np.clip(new - realised, -step_limit, step_limit)
            d = dist_all[:, :, start + j]
            exo[:, j] = np.stack(
                [
                    realised[:, 0] + d[:, 4],
                    realised[:, 1],
                    d[:, 3],
                    np.full(n, nozzle),
                    d[:, 0],
                    d[:, 1],
                    d[:, 2],
                ],
                axis=-1,
            )
        plant.advance(new)
        if not np.array_equal(realised, plant.last_commands):
            raise RuntimeError("rate-limit tracker differs from the truth plant")
        if model is not None:
            model.advance(new)
            true_moisture = plant.moisture_so_far[:, start : start + spi]
            model_moisture = model.moisture_so_far[:, start : start + spi]
            ends = np.arange(spb - 1, spi, spb)
            block_exo = exo.reshape(n, blocks, spb, 7).mean(axis=2)
            for i, feedback in enumerate(feedbacks):
                if not feedback.uses_model_output:
                    continue
                x_true = true_moisture[i, ends]
                x_model = model_moisture[i, ends]
                responses = None
                if feedback.needs_responses:
                    bed = plant.bed_temperature_so_far[i, start : start + spi][ends]
                    outlet = plant.outlet_humidity_so_far[i, start : start + spi][ends]
                    responses = np.stack([bed, bed - duct_loss, outlet], axis=-1)
                errors = feedback.feed(filters[i], x_true, x_model, block_exo[i], responses)
                if k >= k_pre:
                    errors_fb[i].extend(errors.tolist())
                    model_err[i].extend((x_model - x_true).tolist())
        last = new
    result = plant.result()
    moisture = result.moisture[:, n_pre:]
    limit_eval = result.moisture_limit[:, n_pre:]
    excess = moisture - limit_eval
    tolerance = study.control.violation_tolerance_kg_per_kg
    energy = result.energy_kwh_per_interval[:, k_pre:]
    electric = result.electric_kwh_per_interval[:, k_pre:]
    gas = result.gas_kwh_per_interval[:, k_pre:]
    hour_intervals = round(_HOUR_S / study.control.interval_s)
    records = []
    for i, scenario in enumerate(scenarios):
        fb_err = np.asarray(errors_fb[i])
        m_err = np.asarray(model_err[i])
        bias_eval = bias_trace[i, k_pre:]
        record = PairRecord(
            variant=variant,
            draw=draw,
            scenario=scenario.name,
            kind=scenario.kind,
            index=scenario.index,
            margin=float(margins[i]),
            energy_kwh=float(energy[i].sum()),
            co2_kg=float(
                ELECTRICITY_CO2_KG_PER_KWH * electric[i].sum() + GAS_CO2_KG_PER_KWH * gas[i].sum()
            ),
            electric_kwh=float(electric[i].sum()),
            gas_kwh=float(gas[i].sum()),
            max_violation=float(max(excess[i].max(), 0.0)),
            violation_samples=int(np.sum(excess[i] > tolerance)),
            max_bed_temperature=float(result.bed_temperature[i, n_pre:].max()),
            failures=int(failures[i]),
            failures_pre_roll=int(failures_pre[i]),
            bias_end_pre_roll=float(bias_trace[i, k_pre]) if k_pre < plant.n_intervals else 0.0,
            bias_one_hour_before_end_pre_roll=float(bias_trace[i, max(k_pre - hour_intervals, 0)]),
            bias_min=float(bias_eval.min()),
            bias_max=float(bias_eval.max()),
            bias_mean=float(bias_eval.mean()),
            model_error_mean=float(m_err.mean()) if m_err.size else 0.0,
            feedback_error_rmse=float(np.sqrt(np.mean(fb_err**2))) if fb_err.size else 0.0,
            feedback_error_max_abs=float(np.abs(fb_err).max()) if fb_err.size else 0.0,
            inlet_command_mean=float(result.commands[i, k_pre:, 0].mean()),
            gas_velocity_command_mean=float(result.commands[i, k_pre:, 1].mean()),
        )
        outliers = getattr(feedbacks[i], "outliers_drawn", None)
        if outliers is not None:
            record.extra["outlier_share"] = outliers / max(feedbacks[i].samples, 1)  # type: ignore[attr-defined]
        for counter in ("samples_taken", "results_arrived"):  # F4
            value = getattr(feedbacks[i], counter, None)
            if value is not None:
                record.extra[counter] = int(value)
        if keep_arrays:
            record.extra["energy_kwh_per_interval"] = energy[i].tolist()
            record.extra["moisture_max_per_interval"] = (
                moisture[i].reshape(-1, spi).max(axis=1).tolist()
            )
            record.extra["bias_per_interval"] = bias_eval.tolist()
            record.extra["commands"] = result.commands[i, k_pre:].tolist()
        records.append(record)
    return records
