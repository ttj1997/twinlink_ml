r"""Exakter Beharrungszustands-Löser für das Grey-Box-Modell.

Warum eine Modulfunktion und keine dritte ABC-Implementierung
---------------------------------------------------------------
``03_code_konventionen.md`` verlangt ABCs für die vier Kernabstraktionen, darunter
``ProcessSimulator``. Ein Beharrungszustands-Löser ist aber keine Datenquelle -- er
gehört nicht hinter dieselbe Schnittstelle wie ``rollout``. Eine eigene Abstraktion dafür
wäre trotzdem denkbar (``SteadyStateSolver``), aber mit genau einer Implementierung ist
sie Vorratsbau: Dyssol kann im stationären Fließschema-Modus (M7) einen zweiten Fall
liefern, ohne dass klar ist, ob dessen Schnittstelle dieselbe Form hat (Dyssol liefert
keine RHS-Funktion, gegen deren Nullstelle sich bisezieren ließe). Die Abstraktion wird
gezogen, sobald der zweite Fall ansteht -- nicht vorher.

Warum das nominell vierdimensionale Problem sich reduziert
-------------------------------------------------------------
Statt blind in 4D zu suchen, nutzt der Löser die Struktur der Bilanzen aus
``greybox.py``:

1. **Feststoffmasse** -- ``_product_mass_flow`` hat einen Vorsteueranteil
   (``solid_inflow``), der die Regelabweichung im Beharrungszustand exakt zum
   Verschwinden bringt: :math:`m_s = m_{s,\mathrm{soll}}` (``bed_mass_solid_setpoint_kg``).
2. **Durchmesser/Partikelanzahl** -- folgen geschlossen aus der Zahlenbilanz. Mit
   :math:`\dot m_{prod} = \dot m_{s,in}` (aus Punkt 1) reduziert sich :math:`\dot N = 0`
   auf

   .. math:: d_{50} = d_{\mathrm{nuc}}\,
       \left(\frac{\dot m_{spray}\,w_{solid} + \dot m_{seed}}
       {b_{\mathrm{nuc}}\,\dot m_{spray}\,w_{solid} + \dot m_{seed}}\right)^{1/3}

   Für :math:`\dot m_{seed} = 0` reduziert sich das exakt auf
   :meth:`~twinlink_ml.simulation.config.MaterialConfig.nucleation_limited_diameter_m`
   (:math:`d_{50} = d_{\mathrm{nuc}}\,b_{\mathrm{nuc}}^{-1/3}`). Am Nominalpunkt von
   ``greybox_placeholder.yaml`` ist die Keimbildung aus dem Sprühfeststoff
   :math:`b_{\mathrm{nuc}}\,\dot m_{spray}\,w_{solid} \approx 3.6\,\mathrm{g/h}`
   -- eine Keimzufuhr in dieser Größenordnung oder darüber verschiebt :math:`d_{50}`
   spürbar (bei 50 g/h, dem 14-fachen: 750 µm -> 316 µm; bei 2 kg/h: 160 µm, unterhalb
   des Fluidisierungsfensters). Das ist keine Fehlerquelle im Löser, sondern die
   Modellstruktur -- das Zulässigkeitsflag fängt den Fall ab, siehe unten.
3. **Wassermasse und Betttemperatur** -- beide verbleibenden Bilanzen sind bei festem
   jeweils anderem Zustand monoton (:math:`\dot m_w` fällt monoton in :math:`m_w`,
   :math:`\dot T_{bed}` fällt monoton in :math:`T_{bed}`). Zwei geschachtelte
   Bisektionen genügen: außen über :math:`T_{bed}`, innen über die Partikelfeuchte
   :math:`X_p = m_w/m_s` bei festem :math:`T_{bed}`. Kein Newton, kein Jacobi -- die
   rechte Seite enthält ``clip``/``max``-Operationen (Sättigungsgrad, Zustandsboden)
   und ist nicht überall differenzierbar.

Beide Bisektionen rufen ausschließlich
:meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.evaluate_derivative`
auf -- dieselbe rechte Seite wie Integrator und ``time_constants``, keine zweite,
abweichende Physik.

Zulässigkeit ist ein Flag, keine Exception
-------------------------------------------
Geprüft werden :math:`u_{mf}(d_{50}) < u_{gas} < u_t(d_{50})`, Sättigungsgrad
:math:`\le 1` und ob die Partikelfeuchte im plausiblen Bereich
:math:`[X_{eq}, X_{cr}]` liegt (oberhalb :math:`X_{cr}` signalisiert das Modell, dass
die Trocknungskapazität für den gegebenen Sprühstrom nicht ausreicht). Harte Fehler gibt
es nur an zwei Stellen: Batch-Betrieb (kein Beharrungszustand definiert) und eine
verletzte Bracket-Vorzeichenbedingung (die Monotonieannahme selbst ist verletzt -- eine
Bisektion würde sonst still den Bracket-Rand zurückgeben, und der landet unbemerkt als
falscher Wert im Ergebnis).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np

from .backend import ArrayBackend, FloatArray
from .greybox import (
    IDX_BED_TEMPERATURE,
    IDX_PARTICLE_COUNT,
    IDX_SOLID_MASS,
    IDX_WATER_MASS,
    GreyBoxFluidBedGranulator,
)
from .sorption import isotherm_from_material, sorption_safety_grid
from .thermo import (
    latent_heat_j_per_kg,
    relative_humidity_from_absolute_humidity,
    saturation_humidity_kg_per_kg,
    saturation_pressure_pa,
)

_DEFAULT_MAX_ITERATIONS = 100
_DEFAULT_TOLERANCE = 1e-13
_DRY_AIR_GAS_CONSTANT_RATIO = 0.622

BED_TEMPERATURE_SEARCH_MARGIN_K: Final = 10.0
"""Untere Grenze der äußeren Bisektion im ersten Versuch: ``T_amb + 10 K`` (Stand vor Befund 78)."""

BED_TEMPERATURE_FALLBACK_MARGIN_K: Final = 0.5
"""Untere Grenze des Rückfallintervalls ``[T_amb + 0.5, T_amb + 10]`` (Befund 78, Prüfbefund K1).

Das Rückfallintervall wird NUR durchsucht, wenn das Energiebilanz-Residuum an
``T_amb + 10`` und am oberen Rand negativ ist (die Nullstelle liegt unterhalb von
``T_amb + 10``; genau dann scheiterte der Löser vorher mit "kein Vorzeichenwechsel"). Überall,
wo der Löser vorher eine Lösung fand, ist der Rechenweg unverändert und das Ergebnis bitgleich.
Physikalische Begründung der neuen Grenze: die Zuluft trägt die absolute Feuchte der
Umgebung, ``Y_in <= Y_s(T_amb) < Y_s(T_bed)`` für jedes ``T_bed > T_amb`` -- oberhalb der
Umgebungstemperatur gibt es die Kondensationsbranche von ``evaporation_rate_kg_per_s``
(``Y_in > Y_s(T_bed)``) nicht. 0.5 K hält Abstand zu gesättigter Zuluft (``phi -> 1``)."""

BED_TEMPERATURE_BRACKET_RULE: Final = (
    "T_amb+10 K; fallback [T_amb+0.5 K, T_amb+10 K] iff the residual at T_amb+10 K is negative"
)
"""Kennung der Suchgrenze für Provenienzblöcke (Befund 78: in jedes neue Artefakt)."""

LOWER_SEARCH_BOUND_TOLERANCE_K: Final = 1.0e-6
"""Abstand, unterhalb dessen eine Lösung als "auf der unteren Suchgrenze" gemeldet wird."""

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SteadyStateSolution:
    """Exakter Beharrungszustand des Grey-Box-Modells bei konstanter, nomineller Anregung.

    ``diagnostics`` enthält alle Diagnosegrößen von
    :meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.evaluate_derivative`
    am Lösungspunkt, unverändert übernommen (Schlüssel wie in ``greybox._rhs``).
    ``dry_air_mass_flow_kg_per_s``, ``outlet_humidity_kg_per_kg`` und
    ``bed_temperature_celsius`` sind zusätzlich als explizite Felder geführt, ebenso die
    neu abgeleiteten ``dew_point_celsius``, ``exhaust_sensible_kw`` und
    ``exhaust_latent_kw`` -- gemeinsam bilden sie die Schnittstelle zum
    Standort-Fließschema (Massenstrom / Feuchtebeladung / Temperatur je Luftstrom).

    Bezugszustand von ``exhaust_sensible_kw`` und ``exhaust_latent_kw``: **Umgebung**
    (``T_amb``, Wasser flüssig) -- nicht Betttemperatur. Beide Größen beantworten "wie
    viel Enthalpie trägt die Abluft über den Umgebungszustand hinaus", die Frage des
    kondensierenden Wärmeübertragers zur Wärmerückgewinnung. Das ist ein anderer
    Bezugszustand als ``heat_of_evaporation`` in ``greybox._rhs`` (dort: Betttemperatur,
    die Frage "wie viel Wärme braucht das Bett für die Verdampfung") -- eine
    Enthalpiegröße ohne genannten Bezugszustand ist keine physikalische Größe, deshalb
    hier ausdrücklich benannt statt nur im Code sichtbar.
    """

    state: FloatArray
    """Zustandsvektor ``[m_s, m_w, N, T_bed]``, Reihenfolge wie ``IDX_*`` in ``greybox.py``."""

    diameter_m: float
    residual_inf_norm: float
    r"""Skaliertes Residuum :math:`\|\dot{\mathbf x}/\mathrm{scale}\|_\infty` am Lösungspunkt."""

    feasible: bool
    infeasibility_reasons: tuple[str, ...]
    diagnostics: dict[str, float]

    dry_air_mass_flow_kg_per_s: float
    outlet_humidity_kg_per_kg: float
    bed_temperature_celsius: float
    dew_point_celsius: float

    exhaust_sensible_kw: float
    """Enthalpiestrom der Abluft oberhalb des Umgebungszustands (T_amb), sensibler Anteil."""

    exhaust_latent_kw: float
    """Enthalpiestrom der Abluft oberhalb des Umgebungszustands (T_amb), latenter Anteil
    -- Verdampfungsenthalpie bei ``T_amb``, nicht bei ``T_bed`` (siehe Klassendocstring)."""

    lower_search_bound_celsius: float = math.nan
    """Unterste Betttemperatur, die die äußere Bisektion durchsuchen konnte (Voreinstellung:
    ``T_amb + 0.5``, siehe :data:`BED_TEMPERATURE_FALLBACK_MARGIN_K`; sonst die untere Grenze
    des übergebenen Brackets)."""

    bed_temperature_fallback_used: bool = False
    """``True``, wenn im Rückfallintervall ``[T_amb + 0.5, T_amb + 10]`` gesucht wurde."""

    drying_regime: bool = True
    """``Y_in < Y_out < Y_s(T_bed)`` am Lösungspunkt (die Luft nimmt Wasser auf und ist
    ungesättigt); ``False`` ist ein Unzulässigkeitsgrund."""

    @property
    def solid_mass_kg(self) -> float:
        return float(self.state[IDX_SOLID_MASS])

    @property
    def water_mass_kg(self) -> float:
        return float(self.state[IDX_WATER_MASS])

    @property
    def particle_count(self) -> float:
        return float(self.state[IDX_PARTICLE_COUNT])

    @property
    def particle_moisture_kg_per_kg(self) -> float:
        return self.water_mass_kg / self.solid_mass_kg

    @property
    def exhaust_dry_air_mass_flow_kg_per_s(self) -> float:
        """Dry-air flow of the exhaust that LEAVES the line.

        With exhaust recirculation the purge, ``(1 - r)`` of the bed flow; without, the bed flow
        ``dry_air_mass_flow_kg_per_s``. ``exhaust_sensible_kw``/``exhaust_latent_kw`` refer to
        this flow.
        """
        return self.diagnostics.get(
            "exhaust_dry_air_mass_flow_kg_per_s", self.dry_air_mass_flow_kg_per_s
        )


def solve_steady_state(
    simulator: GreyBoxFluidBedGranulator,
    *,
    moisture_bracket_kg_per_kg: tuple[float, float] | None = None,
    temperature_bracket_celsius: tuple[float, float] | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> SteadyStateSolution:
    r"""Exakter Beharrungszustand bei nomineller Anregung, siehe Moduldocstring.

    :param moisture_bracket_kg_per_kg: Bracket für die innere Bisektion (Partikelfeuchte).
        Voreinstellung ``[X_eq, max(10*X_cr, 1.0)]`` -- weit genug, weil
        :math:`\dot m_w \to -\infty` für :math:`X_p \to \infty` (der
        Produktabzugsterm dominiert), also existiert ein Vorzeichenwechsel im Bracket,
        sofern die Monotonieannahme selbst hält.
    :param temperature_bracket_celsius: Bracket für die äußere Bisektion (Betttemperatur).
        Voreinstellung aus den nominellen Stell-/Störgrößen abgeleitet (siehe Code).
    :param max_iterations: Bisektionsschritte je Ebene (äußere Temperatur-, innere
        Feuchte-Bisektion). WARNUNG an künftige Aufrufer: die Voreinstellung hier
        (:data:`_DEFAULT_MAX_ITERATIONS`, 100) reproduziert Befund 52
        (``scripts/study_operating_point_regret.py``,
        ``artifacts/runs/bb1_bb3_operating_point_regret.json``) NICHT bitgleich -- mit
        dieser Präzision scheitern 6 der 25 dort berichteten Gitterpunkte an einem
        Bracket-Vorzeichenfehler in der äußeren Temperatur-Bisektion (empirisch geprüft,
        siehe ``twinlink_ml.rl.env``-Moduldocstring). Ursache: an der unteren
        Bracket-Grenze (``T_amb + 10``) liegt die wahre Nullstelle knapp außerhalb, und ob
        der dort ausgewertete Residuenwert numerisches Rauschen mit positivem oder
        negativem Vorzeichen zeigt, hängt von der Bisektionstiefe der INNEREN
        Feuchte-Bisektion an genau diesem Punkt ab -- verschiedene Präzisionen können am
        selben Bracket-Rand verschiedene Vorzeichen liefern. Befund 52 selbst berichtet
        seine Gitterpunkte NICHT mit dieser Voreinstellung, sondern durchgehend mit der
        SUCH-Präzision (``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``, 30
        Schritte/1e-7 in ``study_operating_point_regret.py``) -- wer Befund 52 an einem
        beliebigen Betriebspunkt reproduzieren will, muss dieselbe Präzision explizit
        übergeben, nicht die Voreinstellung dieser Funktion.
    :param tolerance: Abbruchtoleranz je Bisektionsebene, siehe ``max_iterations`` für die
        Bitgleichheits-Warnung -- dieselbe Einschränkung gilt für :data:`_DEFAULT_TOLERANCE`.
    :raises ValueError: im Batch-Betrieb -- ohne Produktaustrag wächst die Feststoffmasse
        unbeschränkt, ein Beharrungszustand existiert nicht.
    :raises RuntimeError: wenn eine der beiden Bisektionen an ihren Bracket-Rändern
        keinen Vorzeichenwechsel vorfindet -- die Monotonieannahme ist für diese Config
        verletzt, kein numerisches Detail.
    """
    config = simulator.config
    if config.operation.mode == "batch":
        raise ValueError(
            "Kein Beharrungszustand im Batch-Betrieb definiert: ohne Produktaustrag "
            "wächst die Feststoffmasse unbeschränkt (siehe GreyBoxFluidBedGranulator."
            "_product_mass_flow)."
        )

    backend = simulator.backend
    material, geometry, operation = config.material, config.geometry, config.operation

    action = simulator.nominal_action(1)
    disturbance = simulator.nominal_disturbance(1)
    action_arr = backend.asarray(action)
    disturbance_arr = backend.asarray(disturbance)

    inlet_temperature = float(action[0, 0])
    gas_velocity = float(action[0, 1])
    spray_rate = float(action[0, 2]) / 3600.0
    solid_fraction = float(disturbance[0, 2])
    ambient_temperature = float(disturbance[0, 0])

    # -- Schritt 1: Feststoffmasse ------------------------------------------------
    solid_mass = geometry.bed_mass_solid_setpoint_kg

    # -- Schritt 2: Durchmesser/Partikelanzahl aus der Zahlenbilanz ---------------
    spray_solid = spray_rate * solid_fraction
    seed_flow = operation.seed_mass_flow_kg_per_h / 3600.0
    solid_inflow = spray_solid + seed_flow
    nucleation_denominator = material.nucleation_solid_fraction * spray_solid + seed_flow
    diameter_m = material.nucleus_diameter_m * (solid_inflow / nucleation_denominator) ** (
        1.0 / 3.0
    )
    bed_particle_mass = material.particle_density_kg_per_m3 * math.pi * diameter_m**3 / 6.0
    particle_count = solid_mass / bed_particle_mass

    # -- Schritt 3: Wassermasse und Betttemperatur, geschachtelte Bisektion -------
    def evaluate(moisture: float, temperature: float) -> tuple[FloatArray, dict[str, float]]:
        water_mass = moisture * solid_mass
        state_np = np.array(
            [[solid_mass, water_mass, particle_count, temperature]], dtype=np.float64
        )
        derivative, diagnostics = simulator.evaluate_derivative(
            backend.asarray(state_np), action_arr, disturbance_arr
        )
        derivative_np = np.asarray(backend.to_numpy(derivative)[0], dtype=np.float64)
        diagnostics_scalar = {
            key: float(backend.to_numpy(value)[0]) for key, value in diagnostics.items()
        }
        return derivative_np, diagnostics_scalar

    # Untere Bracket-Grenze: konservative, zustandsunabhängige Schranke (dieselbe
    # Logik wie greybox._floor's Sicherheitsraster-Fallback), nicht das exakte lokale
    # X_eq -- vor der Bisektion ist phi_Abluft am Lösungspunkt noch nicht bekannt. Für
    # ConstantIsotherm ist das exakt equilibrium_moisture_kg_per_kg (die Isotherme
    # ignoriert phi/T), keine Verhaltensänderung gegenüber vorher.
    isotherm = isotherm_from_material(material)
    phi_min, _phi_max, _t_min, t_max = sorption_safety_grid(
        backend, config.gas, config.actions, config.disturbances
    )
    moisture_safety_lower_bound = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(backend, backend.asarray(phi_min), backend.asarray(t_max))
        )
    )
    moisture_low, moisture_high = moisture_bracket_kg_per_kg or (
        moisture_safety_lower_bound,
        max(10.0 * material.critical_moisture_kg_per_kg, 1.0),
    )

    def solve_moisture_for_temperature(
        temperature: float,
    ) -> tuple[float, FloatArray, dict[str, float]]:
        def water_residual(moisture: float) -> tuple[float, tuple[FloatArray, dict[str, float]]]:
            derivative_np, diagnostics_scalar = evaluate(moisture, temperature)
            return float(derivative_np[IDX_WATER_MASS]), (derivative_np, diagnostics_scalar)

        moisture, (derivative_np, diagnostics_scalar) = _bisect_with_payload(
            water_residual,
            moisture_low,
            moisture_high,
            max_iterations=max_iterations,
            tolerance=tolerance,
            description=(
                f"Wasserbilanz bei T_bed={temperature:.4f} °C (dm_w/dt fällt monoton in m_w)"
            ),
        )
        return moisture, derivative_np, diagnostics_scalar

    # Untere Grenze (Befund 78, Prüfbefund K1). Erster Versuch wie bisher ab ``T_amb + 10``.
    # Liegt die Nullstelle darunter (Residuum an ``T_amb + 10`` und am oberen Rand negativ --
    # der Fall, in dem der Löser vorher mit "kein Vorzeichenwechsel" scheiterte), wird das
    # Rückfallintervall ``[T_amb + 0.5, T_amb + 10]`` durchsucht. Die frühere Begründung der
    # 10-K-Marge (Kondensationsbranche nahe ``T_amb``) gilt nur UNTERHALB der Umgebungs-
    # temperatur, siehe :data:`BED_TEMPERATURE_FALLBACK_MARGIN_K`. Ein übergebenes
    # ``temperature_bracket_celsius`` wird unverändert verwendet, ohne Rückfall.
    default_bracket = temperature_bracket_celsius is None
    temperature_low, temperature_high = temperature_bracket_celsius or (
        ambient_temperature + BED_TEMPERATURE_SEARCH_MARGIN_K,
        max(inlet_temperature, operation.suspension_temperature_celsius) + 5.0,
    )
    lower_search_bound = (
        ambient_temperature + BED_TEMPERATURE_FALLBACK_MARGIN_K
        if default_bracket
        else temperature_low
    )

    def temperature_residual(
        temperature: float,
    ) -> tuple[float, tuple[float, FloatArray, dict[str, float]]]:
        moisture, derivative_np, diagnostics_scalar = solve_moisture_for_temperature(temperature)
        return float(derivative_np[IDX_BED_TEMPERATURE]), (
            moisture,
            derivative_np,
            diagnostics_scalar,
        )

    low_value = high_value = None
    fallback_used = False
    if default_bracket:
        # Dieselben Auswertungen in derselben Reihenfolge wie in ``_bisect_with_payload`` --
        # sie werden dorthin durchgereicht, der Rechenweg ohne Rückfall bleibt bitgleich.
        low_value = temperature_residual(temperature_low)
        high_value = temperature_residual(temperature_high)
        if low_value[0] < 0.0 and high_value[0] < 0.0:
            fallback_used = True
            temperature_high, high_value = temperature_low, low_value
            temperature_low, low_value = lower_search_bound, None
    temperature, (moisture, derivative_np, diagnostics_scalar) = _bisect_with_payload(
        temperature_residual,
        temperature_low,
        temperature_high,
        max_iterations=max_iterations,
        tolerance=tolerance,
        description=(
            "Energiebilanz (dT_bed/dt fällt monoton in T_bed)"
            + (", Rückfallintervall ab T_amb + 0.5 K" if fallback_used else "")
        ),
        low_value=low_value,
        high_value=high_value,
    )

    water_mass = moisture * solid_mass
    state = np.array([solid_mass, water_mass, particle_count, temperature], dtype=np.float64)

    # -- Skaliertes Residuum -------------------------------------------------------
    scale = np.array(
        [
            max(abs(solid_mass), 1e-9),
            max(abs(water_mass), 1e-9),
            max(abs(particle_count), 1e-9),
            max(abs(temperature), 1.0),
        ]
    )
    residual_inf_norm = float(np.max(np.abs(derivative_np / scale)))

    # -- Abgeleitete Felder ----------------------------------------------------------
    # Bezugszustand: Umgebung (T_amb, Wasser flüssig) -- siehe Klassendocstring von
    # SteadyStateSolution. Jeder Term nutzt die Feuchte seines eigenen Stroms: die
    # Abluft trägt Y_out, deshalb dort gewichtet -- nicht Y_in wie bei der
    # Heizerleistung in energy.py, deren Luftstrom noch nichts verdampft hat.
    gas = config.gas
    dry_air_flow = diagnostics_scalar["dry_air_mass_flow_kg_per_s"]
    outlet_humidity = diagnostics_scalar["outlet_humidity_kg_per_kg"]
    inlet_humidity = diagnostics_scalar["inlet_humidity_kg_per_kg"]
    # Exhaust recirculation: only the purge (1 - r) of the bed flow leaves the line, and its
    # latent part is counted against the FRESH air humidity (the recirculated water stays in the
    # loop). Without recirculation both names are the old quantities (same floats).
    exhaust_flow = diagnostics_scalar.get("exhaust_dry_air_mass_flow_kg_per_s", dry_air_flow)
    fresh_air_humidity = diagnostics_scalar.get("fresh_air_humidity_kg_per_kg", inlet_humidity)

    exhaust_sensible_kw = (
        exhaust_flow
        * (gas.cp_dry_air_j_per_kg_k + outlet_humidity * gas.cp_vapour_j_per_kg_k)
        * (temperature - ambient_temperature)
        / 1000.0
    )
    latent_heat_at_ambient = float(
        backend.to_numpy(
            latent_heat_j_per_kg(backend, backend.asarray(np.array([ambient_temperature])))
        )[0]
    )
    exhaust_latent_kw = (
        exhaust_flow * (outlet_humidity - fresh_air_humidity) * latent_heat_at_ambient / 1000.0
    )

    dew_point_celsius = _solve_dew_point(
        backend=backend,
        outlet_humidity_kg_per_kg=outlet_humidity,
        pressure_pa=gas.pressure_pa,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    reasons: list[str] = []
    u_mf = diagnostics_scalar["minimum_fluidization_velocity_m_per_s"]
    u_terminal = diagnostics_scalar["terminal_velocity_m_per_s"]
    if not gas_velocity > u_mf:
        reasons.append(f"u_gas ({gas_velocity:.4f} m/s) <= u_mf(d50) ({u_mf:.4f} m/s)")
    if not gas_velocity < u_terminal:
        reasons.append(f"u_gas ({gas_velocity:.4f} m/s) >= u_t(d50) ({u_terminal:.4f} m/s)")
    saturation_degree = diagnostics_scalar["saturation_degree"]
    saturation_humidity_at_bed = float(
        backend.to_numpy(
            saturation_humidity_kg_per_kg(backend, gas, backend.asarray(np.array([temperature])))
        )[0]
    )
    drying_regime = inlet_humidity < outlet_humidity < saturation_humidity_at_bed
    if not drying_regime:
        reasons.append(
            f"kein Trocknungsregime: Y_in={inlet_humidity:.6f}, Y_out={outlet_humidity:.6f}, "
            f"Y_s(T_bed)={saturation_humidity_at_bed:.6f} kg/kg"
        )
    if temperature - lower_search_bound <= LOWER_SEARCH_BOUND_TOLERANCE_K:
        reasons.append(
            f"Betttemperatur ({temperature:.6f} °C) auf der unteren Suchgrenze "
            f"({lower_search_bound:.6f} °C)"
        )
    if not saturation_degree <= 1.0 + 1e-9:
        reasons.append(f"Sättigungsgrad ({saturation_degree:.4f}) > 1")
    # Im Beharrungspunkt ist phi_Abluft fest -- anders als bei der Bracket-Grenze oben
    # (vor der Lösung) ist hier die exakte, zustandsabhängige Isotherme ohne
    # Zusatzkosten nutzbar, keine konservative Konstante nötig.
    relative_humidity_at_solution = relative_humidity_from_absolute_humidity(
        backend, gas, backend.asarray(outlet_humidity), backend.asarray(temperature)
    )
    local_equilibrium_moisture = float(
        backend.to_numpy(
            isotherm.equilibrium_moisture(
                backend, relative_humidity_at_solution, backend.asarray(temperature)
            )
        )
    )
    if not (
        local_equilibrium_moisture - 1e-9 <= moisture <= material.critical_moisture_kg_per_kg + 1e-9
    ):
        reasons.append(
            f"Partikelfeuchte ({moisture:.5f} kg/kg) außerhalb des plausiblen Bereichs "
            f"[{local_equilibrium_moisture:.5f}, "
            f"{material.critical_moisture_kg_per_kg}]"
        )

    return SteadyStateSolution(
        state=state,
        diameter_m=diameter_m,
        residual_inf_norm=residual_inf_norm,
        feasible=not reasons,
        infeasibility_reasons=tuple(reasons),
        diagnostics=diagnostics_scalar,
        dry_air_mass_flow_kg_per_s=dry_air_flow,
        outlet_humidity_kg_per_kg=outlet_humidity,
        bed_temperature_celsius=temperature,
        dew_point_celsius=dew_point_celsius,
        exhaust_sensible_kw=exhaust_sensible_kw,
        exhaust_latent_kw=exhaust_latent_kw,
        lower_search_bound_celsius=lower_search_bound,
        bed_temperature_fallback_used=fallback_used,
        drying_regime=drying_regime,
    )


def _bisect_with_payload(
    residual_and_payload: Callable[[float], tuple[float, _T]],
    low: float,
    high: float,
    *,
    max_iterations: int,
    tolerance: float,
    description: str,
    low_value: tuple[float, _T] | None = None,
    high_value: tuple[float, _T] | None = None,
) -> tuple[float, _T]:
    """Bisektion mit Nutzlast (z. B. abgeleitete Größen am jeweiligen Punkt).

    Prüft den Vorzeichenwechsel an den Bracket-Rändern **vor** der Iteration --
    unterschiedliches Vorzeichen ist Vorbedingung, nicht Nebeneffekt. Ohne diese Prüfung
    konvergiert eine Bisektion bei gleichem Vorzeichen still gegen einen der beiden
    Bracket-Ränder, ohne dass das als Fehler sichtbar würde.
    """
    f_low, payload_low = low_value if low_value is not None else residual_and_payload(low)
    f_high, payload_high = high_value if high_value is not None else residual_and_payload(high)
    if f_low == 0.0:
        return low, payload_low
    if f_high == 0.0:
        return high, payload_high
    if f_low * f_high > 0.0:
        raise RuntimeError(
            f"{description}: kein Vorzeichenwechsel im Bracket [{low}, {high}] "
            f"(f({low})={f_low:.6e}, f({high})={f_high:.6e}) -- die Monotonieannahme "
            "ist für diese Konfiguration verletzt."
        )
    a, b, fa = low, high, f_low
    mid, payload = low, payload_low
    for _ in range(max_iterations):
        mid = 0.5 * (a + b)
        f_mid, payload = residual_and_payload(mid)
        if f_mid == 0.0 or (b - a) < tolerance * max(abs(a), abs(b), 1.0):
            break
        if (fa < 0.0) == (f_mid < 0.0):
            a, fa = mid, f_mid
        else:
            b = mid
    return mid, payload


def _solve_dew_point(
    *,
    backend: ArrayBackend,
    outlet_humidity_kg_per_kg: float,
    pressure_pa: float,
    max_iterations: int,
    tolerance: float,
) -> float:
    r"""Taupunkt der Abluft über dieselbe Sättigungsbeziehung wie ``thermo.py``.

    :math:`p_{partial} = Y\,p / (0.622 + Y)` (Umkehrung von
    :func:`~twinlink_ml.simulation.thermo.absolute_humidity_kg_per_kg`), dann
    :math:`T_{dp}` als Nullstelle von :math:`p_s(T) - p_{partial}` über
    :func:`~twinlink_ml.simulation.thermo.saturation_pressure_pa` -- dieselbe
    Buck-Korrelation, die ``_rhs`` für ``saturation_humidity_kg_per_kg`` verwendet. Die
    Buck-Formel ist nicht geschlossen nach ``T`` invertierbar (transzendent), deshalb
    numerisch über Bisektion; ``saturation_pressure_pa`` ist im gültigen Bereich streng
    monoton in ``T``, also mit derselben Bisektionslogik wie die Hauptlösung lösbar.
    """
    p_partial = (
        outlet_humidity_kg_per_kg
        * pressure_pa
        / (_DRY_AIR_GAS_CONSTANT_RATIO + outlet_humidity_kg_per_kg)
    )

    def residual(temperature: float) -> tuple[float, None]:
        value = float(
            backend.to_numpy(
                saturation_pressure_pa(backend, backend.asarray(np.array([temperature])))
            )[0]
        )
        return value - p_partial, None

    root, _ = _bisect_with_payload(
        residual,
        -40.0,
        100.0,
        max_iterations=max_iterations,
        tolerance=tolerance,
        description="Taupunktbestimmung",
    )
    return root
