r"""Heat-distribution service and the recovery-case rule.

:class:`HeatDistributionService` wraps the campaign-network model
(``twinlink_ml.campaign_network``, Befunde 118 and 119): one source of waste heat with water as
the carrier, consumers as cold streams, optionally a hot-water tank on the loop.

Two modes
---------
``cascade_rule``
    The rule without foresight: the supply water goes to the stage with the highest required
    temperature first, its return cascades down (``run_k1``); with a tank the storage rule K2 or
    K2b decides hour by hour (``run_k2``). The answer is the heat per consumer and hour, the
    storage flows and the key figures of ``key_figures``.
``planning``
    One linear programme over the whole request with the on/off plan of the consumers known
    (``rolling_plan`` with the plan as the forecast, one window: the rolling planning P1 of Befund
    119 without re-planning). The reduced model behind it (``cut_data_for_streams``) knows the
    heat used per hour, not its split to the consumers, and needs the same efficiency of the
    replaced supply for every consumer.

What comes from where
---------------------
The consumers, the source power, the tank size and the content of the tank at the first hour come
from the request; supply temperature, return floor, minimum approach, storage loss, discharge
limit, pump share and the CO2 factors come from the injected configuration and are placeholders
(``conf/campaign_network``).

Stateless with a tank
---------------------
The content of the tank is state of the plant, not of the service: a request carries the content
at its first hour (``storage_content_kwh``) and the response returns the content after every
hour. Because the cascade rule has no foresight, calling the service hour by hour with the content
handed over gives the same operation as one call over all hours (tested in the research code).

The one computation of this module
----------------------------------
The key figures of the mode ``planning`` repeat the formulas of ``evaluate.key_figures`` for a
total instead of a sum over consumers (as ``plan_figures`` of the driver of Befund 119 does).

:class:`RecoveryCaseService` is the rule of Befund 121, see its docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import numpy as np

from ..campaign_network.cascade import CascadeResult, run_k1, run_k2
from ..campaign_network.config import CampaignNetworkConfig, load_campaign_network_config
from ..campaign_network.evaluate import key_figures
from ..campaign_network.planning import (
    PlanResult,
    TrueForecast,
    cut_data_for_streams,
    rolling_plan,
)
from ..campaign_network.profiles import ColdStream, Consumers
from ..simulation.backend import FloatArray
from .base import (
    API_VERSION,
    PARTNER_WORKBOOK_SCOPE,
    SCHEMA_VERSION,
    SITE_NETWORK_SCOPE,
    Notice,
    ParameterStatus,
    RequestRejectedError,
    ResponseMeta,
    Service,
    ValueRange,
)
from .schemas import (
    HeatDistributionDescription,
    HeatDistributionRequest,
    HeatDistributionResponse,
    HeatKeyFigures,
    RecoveryCaseDescription,
    RecoveryCaseRequest,
    RecoveryCaseResponse,
    SinkHeat,
)

__all__ = [
    "DIFFERENCE_THRESHOLD_K",
    "HIT_RATE_PHYSICAL_PRESSURISED",
    "HIT_RATE_PHYSICAL_UNPRESSURISED",
    "HOURLY_SAMPLE_TIME_S",
    "UNPRESSURISED_BRANCH_CELSIUS",
    "USABLE_EXCESS_OFFSET_K",
    "HeatDistributionService",
    "RecoveryCaseService",
]

HOURLY_SAMPLE_TIME_S = 3600.0
_HEAT_MODEL_LABEL = "campaign_network"
_RULE_MODEL_LABEL = "exhaust_order_rule"


def _cold_streams(request: HeatDistributionRequest, *, full_flow: bool) -> tuple[ColdStream, ...]:
    """The consumers of the request as cold streams of the research code.

    ``full_flow=False``: the heat-capacity flow is zero in the hours a consumer stands still (the
    cascade rule reads the demand from the flow). ``full_flow=True``: the flow while running (the
    planning multiplies with the on/off state itself).
    """
    streams: list[ColdStream] = []
    for sink in request.sinks:
        flow = np.asarray(sink.heat_capacity_flow_kw_per_k, dtype=np.float64)
        if not full_flow and sink.running is not None:
            flow = np.asarray(sink.running, dtype=np.float64) * flow
        streams.append(
            ColdStream(
                name=sink.name,
                t_in=np.asarray(sink.inlet_temperature_celsius, dtype=np.float64),
                t_out=np.asarray(sink.outlet_temperature_celsius, dtype=np.float64),
                cp=flow,
                gas_efficiency=sink.replaced_supply_efficiency,
                stage_celsius=sink.required_temperature_celsius,
                group=sink.name if sink.group is None else sink.group,
            )
        )
    return tuple(streams)


def _series(values: FloatArray) -> tuple[float, ...]:
    return tuple(np.asarray(values, dtype=np.float64).tolist())


class HeatDistributionService(Service[HeatDistributionRequest, HeatDistributionResponse]):
    """Distribution of the heat of one source to consumers, by rule or by planning.

    Parameters
    ----------
    config
        Network settings: source temperatures, minimum approach, storage loss and discharge limit,
        pump share, CO2 factors, solver settings. Its consumers are not used.
    parameter_status
        Status of these settings (the study configuration is a placeholder).
    model_scope
        Which site and model the answers rest on.
    """

    name: ClassVar[str] = "heat_distribution"
    request_model = HeatDistributionRequest
    response_model = HeatDistributionResponse
    description_model = HeatDistributionDescription

    def __init__(
        self,
        config: CampaignNetworkConfig,
        *,
        parameter_status: ParameterStatus = "placeholder",
        model_scope: str = SITE_NETWORK_SCOPE,
    ) -> None:
        self._config = config
        self._status: ParameterStatus = parameter_status
        self._scope = model_scope

    @classmethod
    def from_config(
        cls, path: Path | str | None = None, variant: str = "main"
    ) -> HeatDistributionService:
        """The network settings of the campaign-network study (``conf/campaign_network``)."""
        if path is None:
            return cls(load_campaign_network_config(variant=variant))
        return cls(load_campaign_network_config(path, variant=variant))

    # ------------------------------------------------------------------ interface

    @property
    def model_fingerprint(self) -> str:
        """Fingerprint of the network settings."""
        return f"campaign_network:{self._config.fingerprint}"

    def describe(self) -> HeatDistributionDescription:
        """Version, fingerprint, modes and the network settings the answers rest on."""
        config = self._config
        return HeatDistributionDescription(
            service=self.name,
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            model_fingerprint=self.model_fingerprint,
            model_label=_HEAT_MODEL_LABEL,
            model_scope=self._scope,
            parameter_status=self._status,
            required_extras=("api",),
            valid_input_ranges={},
            modes=("cascade_rule", "planning"),
            storage_rules=("k2", "k2b"),
            sample_time_s=HOURLY_SAMPLE_TIME_S,
            source_supply_temperature_celsius=config.source.supply_celsius,
            source_return_floor_celsius=config.source.floor_celsius,
            minimum_approach_k=config.approach_k,
            storage_loss_per_h=config.storage.loss_per_h,
            pump_electricity_fraction=config.pump_fraction,
        )

    def handle(self, request: HeatDistributionRequest) -> HeatDistributionResponse:
        """Assignment by the cascade rule, or the planned operation of the tank."""
        if abs(request.sample_time_s - HOURLY_SAMPLE_TIME_S) > 1e-9:
            raise RequestRejectedError(
                f"the network model works on an hourly grid ({HOURLY_SAMPLE_TIME_S} s), the "
                f"request is on {request.sample_time_s} s; resample before the call"
            )
        try:
            body = (
                self._planning(request)
                if request.mode == "planning"
                else self._cascade_rule(request)
            )
        except ValueError as error:
            raise RequestRejectedError(str(error)) from error
        return HeatDistributionResponse(
            meta=ResponseMeta(
                schema_version=SCHEMA_VERSION,
                service=self.name,
                api_version=API_VERSION,
                model_fingerprint=self.model_fingerprint,
                model_label=_HEAT_MODEL_LABEL,
                model_scope=self._scope,
                parameter_status=self._status,
            ),
            mode=request.mode,
            sample_time_s=HOURLY_SAMPLE_TIME_S,
            **body,
        )

    # ------------------------------------------------------------------- internal

    def _cascade_rule(self, request: HeatDistributionRequest) -> dict[str, Any]:
        config = self._config
        streams = _cold_streams(request, full_flow=False)
        n = streams[0].cp.size
        consumers = Consumers(streams=streams, ambient=np.zeros(n, dtype=np.float64))
        power = request.source_power_kw
        result: CascadeResult
        if request.storage_capacity_kwh > 0.0:
            result = run_k2(
                consumers,
                power,
                request.storage_capacity_kwh / power,
                config,
                rule=request.storage_rule,
                initial_content_kwh=request.storage_content_kwh,
            )
        else:
            result = run_k1(consumers, power, config)
        figures = key_figures(result, consumers, power, config)
        total = np.sum([result.duty_kw[s.name] for s in streams], axis=0)
        return {
            "sink_heat": tuple(
                SinkHeat(name=s.name, heat_delivered_kw=_series(result.duty_kw[s.name]))
                for s in streams
            ),
            "total_heat_used_kw": _series(total),
            "storage_charge_kw": _series(result.charge_kw),
            "storage_discharge_kw": _series(result.discharge_kw),
            "storage_content_kwh": _series(result.content_kwh),
            "key_figures": HeatKeyFigures(
                source_energy_kwh=figures["source_kwh"],
                heat_used_kwh=figures["use_kwh"],
                utilisation=figures["utilisation"],
                replaced_gas_kwh=figures["gas_kwh"],
                co2_replaced_gas_kg=figures["co2_gas_kg"],
                pump_electricity_kwh=figures["pump_electricity_kwh"],
                co2_pump_kg=figures["co2_pump_kg"],
                co2_net_saving_kg=figures["co2_net_kg"],
                storage_charged_kwh=figures["charge_kwh"],
                storage_discharged_kwh=figures["discharge_kwh"],
                storage_content_end_kwh=figures["content_end_kwh"],
            ),
        }

    def _planning(self, request: HeatDistributionRequest) -> dict[str, Any]:
        config = self._config
        efficiencies = {sink.replaced_supply_efficiency for sink in request.sinks}
        if len(efficiencies) != 1:
            raise RequestRejectedError(
                "the planning model needs the same replaced_supply_efficiency for every sink, got "
                f"{sorted(efficiencies)}; use the mode 'cascade_rule'"
            )
        streams = _cold_streams(request, full_flow=True)
        n = streams[0].cp.size
        on = {
            sink.name: np.asarray(sink.running, dtype=bool)
            for sink in request.sinks
            if sink.running is not None
        }
        cuts = cut_data_for_streams(
            streams,
            on.keys(),
            supply_celsius=config.source.supply_celsius,
            floor_celsius=config.source.floor_celsius,
            approach_k=config.approach_k,
        )
        power = request.source_power_kw
        result: PlanResult = rolling_plan(
            config,
            cuts,
            on,
            TrueForecast(on),
            power,
            request.storage_capacity_kwh / power,
            horizon_h=n,
            replan_every_h=n,
            initial_content_kwh=request.storage_content_kwh,
        )
        # The formulas of campaign_network.evaluate.key_figures for a total: every consumer has
        # the same efficiency here, so the gas replaced is the heat used divided by it.
        (efficiency,) = efficiencies
        source = power * n
        use = float(np.sum(result.use_kw))
        gas = use / efficiency
        pump = config.pump_fraction * use
        co2_gas = config.gas_co2_kg_per_kwh * gas
        co2_pump = config.electricity_co2_kg_per_kwh * pump
        return {
            "sink_heat": None,
            "total_heat_used_kw": _series(result.use_kw),
            "storage_charge_kw": _series(result.charge_kw),
            "storage_discharge_kw": _series(result.discharge_kw),
            "storage_content_kwh": _series(result.content_kwh),
            "key_figures": HeatKeyFigures(
                source_energy_kwh=source,
                heat_used_kwh=use,
                utilisation=use / source,
                replaced_gas_kwh=gas,
                co2_replaced_gas_kg=co2_gas,
                pump_electricity_kwh=pump,
                co2_pump_kg=co2_pump,
                co2_net_saving_kg=co2_gas - co2_pump,
                storage_charged_kwh=float(np.sum(result.charge_kw)),
                storage_discharged_kwh=float(np.sum(result.discharge_kw)),
                storage_content_end_kwh=float(result.content_kwh[-1]),
            ),
            "lp_failures": result.lp_failures,
        }


# ------------------------------------------------------------------ recovery case

DIFFERENCE_THRESHOLD_K: Final = -0.541
"""First split of the decision trees of both physical modes (Befund 121, artifact
``case_ranking_v1_analysis``, ``rule/fixed/physical_*/strict/text``): exhaust 1 minus exhaust 2."""

USABLE_EXCESS_OFFSET_K: Final = 20.0
"""Exhaust 2 has usable heat where it is warmer than ambient plus this offset (two exchangers of
10 K in series; ``case_ranking.rule.APPROACH_TOTAL_K``)."""

UNPRESSURISED_BRANCH_CELSIUS: Final = 118.491
"""Above this temperature of exhaust 2 the tree of the unpressurised mode decides by the supply
humidity, which this rule does not read."""

HIT_RATE_PHYSICAL_UNPRESSURISED: Final = 0.9658303464755078
HIT_RATE_PHYSICAL_PRESSURISED: Final = 0.9708482676224611
"""Test accuracy of the depth-3 trees of the two physical modes on the study's own test stream,
target "case 1 strictly best" (``rule/fixed/physical_*/strict/test_accuracy`` of the artifact)."""

REFERENCE: Final = "Befund 121 (docs/38_case_ranking.md, section 6.4)"
VALID_FOR: Final = (
    "the two physical modes (unpressurised and pressurised) of the rebuild of the partner "
    "workbook with its fixed circuit flows, inside the investigated box of docs/38 section 2; "
    "not the partner mode, not the optimised circuit flows, not a plant"
)
HIT_RATE_BASIS: Final = (
    "test accuracy of the depth-3 decision trees of the study for the statement 'case 1 is "
    "strictly best'; this rule is the effective leaves of those trees (exactly those of the "
    "pressurised mode; the unpressurised tree has one more branch above 118.5 degC of exhaust 2)"
)


class RecoveryCaseService(Service[RecoveryCaseRequest, RecoveryCaseResponse]):
    """Rule of Befund 121: which heat-recovery case of the two-dryer arrangement of the workbook.

    Case 1 (each exhaust feeds its own sink) is expected to be best where exhaust 1 is not colder
    than exhaust 2 by more than 0.541 K **and** exhaust 2 is warmer than ambient plus 20 K; where
    exhaust 2 is the warmer one the study names the crossed case 2; where exhaust 1 is warm enough
    but exhaust 2 has no usable heat the study names no case and the service recommends none.

    The research code holds the study that found the rule (decision trees fitted on its samples),
    not the rule as a function, so the logic is written here: the effective leaves of the trees of
    the physical modes, with the hit rates of those trees. The rule describes where the rebuild of
    the workbook ranks case 1 first; it is not physics and not a plant statement.

    Parameters
    ----------
    ambient_box_celsius
        Range of the ambient temperature the study covered; a request outside it is remarked.
    model_scope
        Which arrangement and model the answers rest on.
    """

    name: ClassVar[str] = "recovery_case"
    request_model = RecoveryCaseRequest
    response_model = RecoveryCaseResponse
    description_model = RecoveryCaseDescription

    def __init__(
        self,
        ambient_box_celsius: tuple[float, float],
        *,
        model_scope: str = PARTNER_WORKBOOK_SCOPE,
    ) -> None:
        low, high = ambient_box_celsius
        if not low < high:
            raise ValueError("ambient_box_celsius must be increasing")
        self._ambient_box = (float(low), float(high))
        self._scope = model_scope

    @classmethod
    def from_config(cls, path: Path | str | None = None) -> RecoveryCaseService:
        """The box of the case-ranking study (``conf/case_ranking``)."""
        from ..case_ranking.config import load_config

        config = load_config() if path is None else load_config(path)
        interval = config.box.interval("ambient_celsius")
        return cls((interval.low, interval.high))

    @property
    def model_fingerprint(self) -> str:
        """Name of the rule with its thresholds."""
        return (
            f"{_RULE_MODEL_LABEL}:difference>{DIFFERENCE_THRESHOLD_K!r}"
            f"/excess>{USABLE_EXCESS_OFFSET_K!r}"
        )

    def describe(self) -> RecoveryCaseDescription:
        """Version, rule, thresholds, where the rule is valid and how often it was right."""
        return RecoveryCaseDescription(
            service=self.name,
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            model_fingerprint=self.model_fingerprint,
            model_label=_RULE_MODEL_LABEL,
            model_scope=self._scope,
            parameter_status="placeholder",
            required_extras=("api",),
            valid_input_ranges={
                "ambient_temperature_celsius": ValueRange(
                    minimum=self._ambient_box[0], maximum=self._ambient_box[1]
                )
            },
            cases=(1, 2),
            difference_threshold_k=DIFFERENCE_THRESHOLD_K,
            usable_excess_offset_k=USABLE_EXCESS_OFFSET_K,
            reference=REFERENCE,
            valid_for=VALID_FOR,
            hit_rate_physical_unpressurised=HIT_RATE_PHYSICAL_UNPRESSURISED,
            hit_rate_physical_pressurised=HIT_RATE_PHYSICAL_PRESSURISED,
            hit_rate_basis=HIT_RATE_BASIS,
        )

    def handle(self, request: RecoveryCaseRequest) -> RecoveryCaseResponse:
        """Case 1, case 2 or no recommendation, with reference, validity and hit rates."""
        difference = request.exhaust_1_temperature_celsius - request.exhaust_2_temperature_celsius
        excess = (
            request.exhaust_2_temperature_celsius
            - request.ambient_temperature_celsius
            - USABLE_EXCESS_OFFSET_K
        )
        warm_enough = difference > DIFFERENCE_THRESHOLD_K
        case_1 = warm_enough and excess > 0.0
        notices: list[Notice] = []
        if warm_enough and not case_1:
            notices.append(
                Notice(
                    code="no_case_named",
                    message="exhaust 1 is warm enough, but exhaust 2 has no heat above ambient "
                    f"plus {USABLE_EXCESS_OFFSET_K:g} K: the study does not find case 1 best "
                    "here and names no other case",
                )
            )
        if warm_enough and request.exhaust_2_temperature_celsius > UNPRESSURISED_BRANCH_CELSIUS:
            notices.append(
                Notice(
                    code="unpressurised_branch_not_represented",
                    message=f"above {UNPRESSURISED_BRANCH_CELSIUS:g} degC of exhaust 2 the tree of "
                    "the unpressurised mode decides by the supply humidity, which this rule does "
                    "not read",
                )
            )
        low, high = self._ambient_box
        if not low <= request.ambient_temperature_celsius <= high:
            notices.append(
                Notice(
                    code="outside_investigated_box",
                    message=f"ambient temperature {request.ambient_temperature_celsius!r} degC "
                    f"outside the investigated [{low!r}, {high!r}] degC",
                )
            )
        recommended: Literal[1, 2] | None = 1 if case_1 else (None if warm_enough else 2)
        return RecoveryCaseResponse(
            meta=ResponseMeta(
                schema_version=SCHEMA_VERSION,
                service=self.name,
                api_version=API_VERSION,
                model_fingerprint=self.model_fingerprint,
                model_label=_RULE_MODEL_LABEL,
                model_scope=self._scope,
                parameter_status="placeholder",
                notices=tuple(notices),
            ),
            recommended_case=recommended,
            case_1_expected_best=case_1,
            exhaust_temperature_difference_k=difference,
            exhaust_2_usable_excess_k=excess,
            reference=REFERENCE,
            valid_for=VALID_FOR,
            hit_rate_physical_unpressurised=HIT_RATE_PHYSICAL_UNPRESSURISED,
            hit_rate_physical_pressurised=HIT_RATE_PHYSICAL_PRESSURISED,
            hit_rate_basis=HIT_RATE_BASIS,
        )
