r"""Psychrometrie und Trocknungskinetik.

Reine Funktionen, backend-agnostisch. Jede Größe trägt ihre Einheit im Namen.

Die Sättigungsfeuchte ist die Stelle, an der die harte Grenze aus
``02_prozess_setup.md`` (Abluftfeuchte :math:`\le` Sättigungsfeuchte) entsteht. Sie wird
hier **nicht** durch eine nachgeschaltete Projektion erzwungen, sondern durch die
Struktur der Trocknungskinetik: der Sättigungsgrad :math:`\nu \le \nu_{max} \le 1`
begrenzt die Verdampfung so, dass die Abluft die Sättigung nie überschreiten kann.
Ein Constraint, das aus der Modellstruktur folgt, kann nicht durch numerische
Ungenauigkeit verletzt werden -- eine Projektion kann das.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import Array, ArrayBackend
from .config import GasConfig, MaterialConfig

if TYPE_CHECKING:
    # Nur für Typannotationen: sorption.py importiert bereits aus thermo.py
    # (absolute_humidity_kg_per_kg, relative_humidity_from_absolute_humidity). Ein
    # Modul-Top-Level-Import hier würde einen Zirkelbezug erzeugen. Die Isotherme wird
    # deshalb injiziert (Dependency Injection, wie ``backend``), nicht intern aus
    # ``material`` konstruiert -- kein Laufzeit-Import nötig.
    from .sorption import SorptionIsotherm

_KELVIN_OFFSET = 273.15
_PICARD_ITERATIONS = 5
_PICARD_DAMPING = 0.7


def kelvin(temperature_celsius: Array) -> Array:
    """Celsius nach Kelvin."""
    return temperature_celsius + _KELVIN_OFFSET


def saturation_pressure_pa(backend: ArrayBackend, temperature_celsius: Array) -> Array:
    r"""Sättigungsdampfdruck nach Buck (1996).

    .. math:: p_s(T) = 611.21\,
        \exp\!\left[\left(18.678 - \frac{T}{234.5}\right)\frac{T}{257.14 + T}\right]

    Genauigkeit besser als 0.2 % im Bereich 0-100 °C.

    Bewusst **nicht** die einfachere Magnus-Formel: die weicht bei 100 °C um 2.6 kPa
    (2.6 %) ab. Weil Zulufttemperaturen bis 120 °C im Stellgrößenbereich liegen und die
    Betttemperatur bei kleiner Sprührate in diese Region läuft, wäre das ein
    systematischer Fehler genau im energetisch interessanten Randbereich -- also dort, wo
    ein RL-Agent später hin optimiert.

    :param temperature_celsius: Temperatur in °C
    :returns: Sättigungsdampfdruck in Pa
    """
    t = backend.clip(temperature_celsius, -40.0, 200.0)
    return 611.21 * backend.exp((18.678 - t / 234.5) * t / (257.14 + t))


def absolute_humidity_kg_per_kg(
    backend: ArrayBackend,
    gas: GasConfig,
    temperature_celsius: Array,
    relative_humidity: Array,
) -> Array:
    r"""Absolute Feuchte aus relativer Feuchte.

    .. math:: Y = 0.622\,\frac{\varphi\, p_s(T)}{p - \varphi\, p_s(T)}

    Einheit: kg Wasser pro kg trockene Luft.
    """
    p_partial = relative_humidity * saturation_pressure_pa(backend, temperature_celsius)
    p_partial = backend.minimum(p_partial, 0.95 * gas.pressure_pa)
    return 0.622 * p_partial / (gas.pressure_pa - p_partial)


def relative_humidity_from_absolute_humidity(
    backend: ArrayBackend,
    gas: GasConfig,
    humidity_kg_per_kg: Array,
    temperature_celsius: Array,
    *,
    saturation_pressure: Array | None = None,
) -> Array:
    r"""Relative Feuchte aus absoluter Feuchte -- Umkehrung von :func:`absolute_humidity_kg_per_kg`.

    .. math:: \varphi = \mathrm{clip}\!\left(\frac{Y\,p}{(0.622+Y)\,p_s(T)},\, 0,\, 1\right)

    Dieselbe Buck-Sättigungsdruckformel wie :func:`saturation_pressure_pa` -- keine zweite
    psychrometrische Formel. ``temperature_celsius`` ist hier nicht notwendig die
    Temperatur, bei der ``humidity_kg_per_kg`` gemessen wurde: die Funktion beantwortet
    "welche relative Feuchte hätte diese absolute Feuchte bei dieser Temperatur", etwa
    für Zuluft, die von :math:`T_{amb}` auf :math:`T_{bed}` erwärmt wird, ohne dass sich
    ihre absolute Feuchte dabei ändert.

    Exakte algebraische Inversion von :func:`absolute_humidity_kg_per_kg`: numerisch
    bestätigt bis auf Float64-Rundung (< 1e-15). Eine Ausnahme: bei sehr hohem
    ``temperature_celsius`` kombiniert mit hoher ``humidity_kg_per_kg`` greift
    :func:`absolute_humidity_kg_per_kg`s eigener, dort schon vorhandener Clip
    (``p_partial = min(p_partial, 0.95*p)``, Sättigungsdampfdruck übersteigt den
    Gesamtdruck ab ca. 95-100 °C) -- das an dieser Stelle bereits geclippte ``Y`` ist dann
    nicht mehr die exakte Umkehrung der ursprünglichen relativen Feuchte, unabhängig davon,
    wie exakt diese Funktion selbst rechnet. Wer diese Funktion bei Temperaturen deutlich
    über dem Sicherheitsraster einsetzt (:func:`~.sorption.sorption_safety_grid`, hier nur
    bis 130 °C ausgewertet), sollte diesen Effekt einkalkulieren.

    :returns: relative Feuchte in [0, 1]
    """
    p_partial = humidity_kg_per_kg * gas.pressure_pa / (0.622 + humidity_kg_per_kg)
    # ``saturation_pressure``: p_s(temperature_celsius) precomputed by a caller that evaluates
    # many humidities at one temperature (recirculated_inlet_humidity); same value, same formula.
    p_saturation = (
        saturation_pressure_pa(backend, temperature_celsius)
        if saturation_pressure is None
        else saturation_pressure
    )
    return backend.clip(p_partial / p_saturation, 0.0, 1.0)


def dew_point_celsius(
    backend: ArrayBackend,
    gas: GasConfig,
    absolute_humidity_kg_per_kg: Array,
    *,
    iterations: int = 60,
) -> Array:
    r"""Taupunkt aus absoluter Feuchte -- Umkehrung von :func:`saturation_pressure_pa`.

    :math:`p_s(T_{dp}) = p_{partial}`, mit :math:`p_{partial} = Y\,p / (0.622 + Y)`
    (Umkehrung von :func:`absolute_humidity_kg_per_kg`, dieselbe Formel wie in
    :func:`relative_humidity_from_absolute_humidity`). Dieselbe Buck-Korrelation
    (:func:`saturation_pressure_pa`) wie überall in diesem Modul -- keine zweite
    psychrometrische Formel. Numerisch über Bisektion, nicht Newton: ``saturation_pressure_pa``
    clippt an den Bracket-Rändern, keine überall differenzierbare Funktion (dieselbe
    Begründung wie bei ``simulation.steady_state._solve_dew_point``, der skalaren, privaten
    Variante für den Beharrungszustands-Löser). Diese Funktion ist die array-fähige,
    öffentliche Entsprechung -- gebraucht, weil eine Auswertung auf ganzen
    Vorhersage-Arrays (viele Fenster/Läufe/Horizonte) läuft, nicht an einem einzelnen
    Beharrungspunkt.

    60 feste Iterationen, keine Abbruchprüfung: bei Startbracket ``[-40, 100]`` °C halbiert
    sich das Intervall auf :math:`140 / 2^{60}` °C -- weit unterhalb jeder
    Float64-Auflösung, ohne bedingten Kontrollfluss über Arrayelemente hinweg (kein früher
    Abbruch bei Konvergenz einzelner Elemente).

    :param absolute_humidity_kg_per_kg: :math:`Y`, kg Wasser pro kg trockene Luft.
    :returns: Taupunkt in °C.
    """
    p_partial = (
        absolute_humidity_kg_per_kg * gas.pressure_pa / (0.622 + absolute_humidity_kg_per_kg)
    )
    low = backend.zeros_like(p_partial) - 40.0
    high = backend.zeros_like(p_partial) + 100.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        residual = saturation_pressure_pa(backend, mid) - p_partial
        too_low = residual < 0.0
        low = backend.where(too_low, mid, low)
        high = backend.where(too_low, high, mid)
    return 0.5 * (low + high)


def saturation_humidity_kg_per_kg(
    backend: ArrayBackend, gas: GasConfig, temperature_celsius: Array
) -> Array:
    """Sättigungsfeuchte :math:`Y_s(T)` in kg/kg trockene Luft."""
    one = backend.zeros_like(temperature_celsius) + 1.0
    return absolute_humidity_kg_per_kg(backend, gas, temperature_celsius, one)


def latent_heat_j_per_kg(backend: ArrayBackend, temperature_celsius: Array) -> Array:
    r"""Verdampfungsenthalpie von Wasser.

    .. math:: \Delta h_v(T) \approx 2.5008\cdot 10^{6} - 2358.6\,T \quad [\mathrm{J/kg}]
    """
    t = backend.clip(temperature_celsius, 0.0, 150.0)
    return 2.5008e6 - 2358.6 * t


def gas_density_kg_per_m3(
    backend: ArrayBackend,  # noqa: ARG001 - gemeinsame Signatur aller Korrelationen
    gas: GasConfig,
    temperature_celsius: Array,
    humidity_kg_per_kg: Array,
) -> Array:
    r"""Dichte feuchter Luft, ideales Gasgemisch.

    .. math:: \rho = \frac{p}{R_{L} T}\cdot\frac{1 + Y}{1 + Y/0.622}
    """
    t_kelvin = kelvin(temperature_celsius)
    dry = gas.pressure_pa / (gas.r_specific_dry_air_j_per_kg_k * t_kelvin)
    return dry * (1.0 + humidity_kg_per_kg) / (1.0 + humidity_kg_per_kg / 0.622)


def gas_viscosity_pa_s(backend: ArrayBackend, gas: GasConfig, temperature_celsius: Array) -> Array:
    r"""Dynamische Viskosität nach Sutherland.

    .. math:: \mu(T) = \mu_0
        \left(\frac{T}{T_0}\right)^{3/2}\frac{T_0 + S}{T + S}
    """
    t_kelvin = kelvin(temperature_celsius)
    ratio = t_kelvin / gas.viscosity_ref_temperature_k
    return (
        gas.viscosity_ref_pa_s
        * ratio
        * backend.sqrt(ratio)
        * (gas.viscosity_ref_temperature_k + gas.sutherland_constant_k)
        / (t_kelvin + gas.sutherland_constant_k)
    )


def normalised_moisture(
    backend: ArrayBackend,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    moisture_kg_per_kg: Array,
    relative_humidity: Array,
    temperature_celsius: Array,
) -> Array:
    r"""Normierte Partikelfeuchte :math:`\eta` der Trocknungskurve.

    .. math:: \eta = \mathrm{clip}\!\left(
        \frac{X_p - X_{eq}(\varphi, T)}{X_{cr} - X_{eq}(\varphi, T)},\, 0,\, 1\right)

    :math:`\eta = 1` entspricht dem ersten Trocknungsabschnitt (konstante Rate),
    :math:`\eta < 1` dem zweiten (fallende Rate). Der Nulldurchgang bei
    :math:`X_p = X_{eq}` garantiert strukturell die zweite harte Grenze aus
    ``02_prozess_setup.md``: die Partikelfeuchte kann die Gleichgewichtsfeuchte der
    Sorptionsisotherme nicht unterschreiten, weil die Verdampfungsrate dort verschwindet.

    :math:`X_{eq}(\varphi, T)` kommt aus der injizierten Sorptionsisotherme
    (``simulation.sorption``) -- ``ConstantIsotherm`` bildet den historischen
    Skalarfall exakt nach, ``GabIsotherm`` macht :math:`X_{eq}` von der relativen
    Abluftfeuchte und der Betttemperatur abhängig. Welche Isotherme aktiv ist, ist
    Sache des Aufrufers (Dependency Injection, wie ``backend``) -- diese Funktion
    kennt nur das Isothermen-Interface, nicht ``MaterialConfig.sorption_isotherm_kind``.

    :param relative_humidity: relative Feuchte der Abluft in [0, 1] -- bei
        ``evaporation_rate_kg_per_s`` das Ergebnis der Picard-Iteration über
        :math:`Y_{out}`, nicht direkt gegeben (siehe dort).
    """
    equilibrium_moisture = isotherm.equilibrium_moisture(
        backend, relative_humidity, temperature_celsius
    )
    span = material.critical_moisture_kg_per_kg - equilibrium_moisture
    excess = (moisture_kg_per_kg - equilibrium_moisture) / span
    return backend.clip(excess, 0.0, 1.0)


def evaporation_rate_kg_per_s(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    dry_air_mass_flow_kg_per_s: Array,
    bed_temperature_celsius: Array,
    inlet_humidity_kg_per_kg: Array,
    particle_moisture_kg_per_kg: Array,
) -> tuple[Array, Array, Array]:
    r"""Verdampfungsrate über die normierte Trocknungskurve.

    .. math::
        \dot m_{evap} = \underbrace{\nu_{max}\,f(\eta)}_{\nu}\;
        \dot m_{L}\,\bigl(Y_s(T_{bed}) - Y_{in}\bigr),
        \qquad f(\eta) = \eta

    Annahme: ideal durchmischtes Bett, Abluft auf Betttemperatur. Der Sättigungsgrad
    :math:`\nu` beschreibt, wie weit die Abluft an die Sättigung herankommt. Für
    Wirbelschichten liegt er im ersten Trocknungsabschnitt typisch bei 0.4-0.8.

    Fallstrick: mit :math:`f(\eta) = \eta` ist die Feuchtedynamik ein
    Verzögerungsglied erster Ordnung mit sehr kleiner Zeitkonstante, weil die
    Trocknungskapazität der Luft die eingesprühte Wassermenge bei
    Granulationsbedingungen um eine Größenordnung übersteigt. Das ist physikalisch
    korrekt (das Bett ist trocken), macht die Feuchte aber zur schnellsten Mode des
    Systems -- relevant für die Wahl von :math:`\Delta t`.

    Kondensation
    ------------
    Ist die Zuluft feuchter als die Sättigung bei Betttemperatur
    (:math:`Y_{in} > Y_s(T_{bed})`, z. B. beim Kaltstart oder bei sehr feuchter
    Umgebungsluft), kehrt sich der Stofftransport um: Wasser kondensiert auf die Partikel,
    bis die Abluft gesättigt ist. Dieser Zweig ist nicht kinetisch begrenzt.

    Er ist nicht nur Vollständigkeit: **ohne** ihn gilt die harte Grenze
    "Abluftfeuchte :math:`\le` Sättigungsfeuchte" nur im Trocknungsregime. Fehlt der
    Zweig, liefert das Modell bei kaltem Bett stillschweigend übersättigte Abluft -- also
    genau in dem Betriebspunkt, in den ein untrainierter RL-Agent zuerst hineinläuft.

    Gedämpfte Picard-Iteration für :math:`Y_{out}`
    ------------------------------------------------
    Mit einer zustandsabhängigen Isotherme entsteht eine algebraische Schleife:
    :math:`\eta` (und damit die Verdampfung) hängt über :math:`X_{eq}(\varphi_{Abluft}, T)`
    von :math:`\varphi_{Abluft}` ab, das seinerseits :math:`Y_{out}` braucht -- den
    Rückgabewert dieser Funktion. :math:`Y_{in}` ist die natürliche nullte Iterierte
    (kein Zirkelbezug, sofort verfügbar). Ungedämpfte Fixpunktiteration oszilliert
    nachweisbar nahe der Knickstelle :math:`X_p \approx X_{eq}` (dort ist :math:`\eta`
    extrem empfindlich gegenüber kleinen Änderungen von :math:`X_{eq}`, siehe
    ``docs/04_stand_ap1.md`` Befund 34) -- Unterrelaxation mit :math:`\alpha = 0{,}7` über
    :math:`n = 5` Iterationen hält den Fehler zwischen der vorletzten und letzten
    Iterierten an allen geprüften Betriebspunkten (Nominalpunkt, zwei Befund-31-Randpunkte,
    Knickszenario) unter 1 % relativ, sowohl für das NumPy- als auch das Torch-Backend
    (bitgenau identisch, beide float64).

    **Rechenkosten, explizit benannt statt verschwiegen:** :math:`n=5` Iterationen à
    Sättigungsfeuchte- und Isothermenauswertung bedeuten 5 statt 1 zusätzliche
    Backend-Auswertung dieser Funktion pro RHS-Aufruf -- bei RK4 (4 RHS-Aufrufe je
    Integrationsschritt) also 20 statt 4, d.h. eine Verfünffachung des Aufwands dieser
    einen Teilrechnung. Für ``ConstantIsotherm`` ist das reine Verschwendung (das Ergebnis
    ist nach der ersten Iteration exakt), aber eine Fallunterscheidung nach Isothermentyp
    würde die Funktion verzweigen und wurde bewusst nicht eingeführt -- der Mehraufwand ist
    gegenüber den übrigen RHS-Termen (Fluidisierung, Energie) klein.

    :returns: (Verdampfungsrate kg/s, Abluftfeuchte kg/kg, Sättigungsgrad -)
    """
    y_saturation = saturation_humidity_kg_per_kg(backend, gas, bed_temperature_celsius)
    surplus = y_saturation - inlet_humidity_kg_per_kg

    # Obergrenze der Treibkraft: oberhalb ~80 °C Betttemperatur divergiert Y_s und die
    # Trocknung ist real nicht mehr sättigungs-, sondern transportlimitiert. Die
    # normierte Trocknungskurve verliert dort ihre Gültigkeit -- die Begrenzung hält das
    # Modell in einem plausiblen Bereich und markiert die Modellgrenze explizit.
    y_effective = backend.minimum(y_saturation, material.max_driving_humidity_kg_per_kg)

    outlet_humidity = inlet_humidity_kg_per_kg  # nullte Iterierte
    evaporation = backend.zeros_like(bed_temperature_celsius)
    saturation_degree = backend.zeros_like(bed_temperature_celsius)
    for _ in range(_PICARD_ITERATIONS):
        relative_humidity = relative_humidity_from_absolute_humidity(
            backend, gas, outlet_humidity, bed_temperature_celsius
        )
        eta = normalised_moisture(
            backend,
            material,
            isotherm,
            particle_moisture_kg_per_kg,
            relative_humidity,
            bed_temperature_celsius,
        )
        if material.drying_curve_exponent is not None:
            # Optional curve shape f(eta) = eta ** q; unset leaves the linear curve (and the
            # arithmetic) exactly as before.
            eta = eta**material.drying_curve_exponent
        saturation_degree = material.max_saturation_degree * eta
        drying = (
            saturation_degree
            * dry_air_mass_flow_kg_per_s
            * backend.maximum(y_effective - inlet_humidity_kg_per_kg, 0.0)
        )
        condensing = dry_air_mass_flow_kg_per_s * surplus
        evaporation = backend.where(surplus > 0.0, drying, condensing)
        outlet_humidity_proposed = (
            inlet_humidity_kg_per_kg + evaporation / dry_air_mass_flow_kg_per_s
        )
        # Dämpfung nur im Trocknungszweig: dort hängt der Fixpunkt über eta(phi(Y_out))
        # vom vorherigen Y_out ab (die Knickstelle, die die Oszillation verursacht).
        # Im Kondensationszweig ist outlet_humidity_proposed = Y_s(T_bed) exakt, für
        # jede Iteration, unabhängig vom Zustand -- eine Konstante, kein iterativer
        # Fixpunkt. Sie zu dämpfen würde die harte Grenze "Abluftfeuchte <= Y_s"
        # (02_prozess_setup.md) nur näherungsweise statt exakt erfüllen (nachgewiesen:
        # test_outlet_humidity_never_exceeds_saturation schlug mit Residuen bis 4e-5
        # fehl, bevor dieser Zweig von der Dämpfung ausgenommen wurde) -- undamped
        # bleibt sie exakt, weil der Kondensationszweig gar keine Oszillationsgefahr hat.
        outlet_humidity_damped = (
            _PICARD_DAMPING * outlet_humidity_proposed + (1.0 - _PICARD_DAMPING) * outlet_humidity
        )
        outlet_humidity = backend.where(
            surplus > 0.0, outlet_humidity_damped, outlet_humidity_proposed
        )

    return evaporation, outlet_humidity, saturation_degree


_RECIRCULATION_ITERATIONS = 15
"""Fixed step count of the bracketed root search in :func:`recirculated_inlet_humidity`.
Checked in ``tests/test_exhaust_recirculation.py`` against 200 steps at the corners of the study
box (r up to 0.9, GAB isotherm)."""


def recirculated_inlet_humidity(
    backend: ArrayBackend,
    gas: GasConfig,
    material: MaterialConfig,
    isotherm: SorptionIsotherm,
    ambient_humidity_kg_per_kg: Array,
    recirculation_fraction: float,
    bed_temperature_celsius: Array,
    particle_moisture_kg_per_kg: Array,
) -> tuple[Array, Array, Array]:
    r"""Inlet and outlet humidity of a bed whose inlet air is partly recirculated exhaust.

    A fraction :math:`r` of the dry air through the bed is taken from the exhaust (state
    :math:`T_{bed}, Y_{out}`, well-mixed bed) and mixed with fresh air (:math:`Y_a`) ahead of the
    heater. Gas phase quasi-stationary as everywhere in the model, so the loop is algebraic:

    .. math::
        Y_{in} = (1 - r)\,Y_a + r\,Y_{out}, \qquad
        Y_{out} = Y_{in} + \nu\,(Y_{eff} - Y_{in})

    with the saturation degree :math:`\nu = \nu_{max} f(\eta)` of
    :func:`evaporation_rate_kg_per_s` (same drying curve, same isotherm, same driving-force
    cap :math:`Y_{eff} = \min(Y_s(T_{bed}), Y_{max})`). For a given :math:`\nu` the pair is linear
    and solved in closed form,

    .. math:: Y_{in} = \frac{(1 - r)\,Y_a + r\,\nu\,Y_{eff}}{1 - r + r\,\nu},

    so the only fixed point left is the dependence of :math:`\nu` on :math:`\varphi_{out}` through
    the isotherm (as in the fresh-air case). A damped Picard iteration as in
    :func:`evaporation_rate_kg_per_s` does NOT converge here near the kink of the drying curve
    (measured: relative errors above 1 at r = 0.8 after 60 steps; the recirculation multiplies the
    loop gain); the fixed point is therefore found by a bracketed root search on the monotone
    residual (see the code), with a fixed step count. The returned pair is evaluated once more at
    the final :math:`\varphi_{out}`, so the two relations above hold exactly; with them the water
    balance of the whole loop closes exactly,
    :math:`(1 - r)\,(Y_{out} - Y_a) = \nu\,(Y_{eff} - Y_{in}) = \dot m_{evap} / \dot m_L`
    (the water leaves only with the purge, :math:`(1 - r)\,\dot m_L`). The dry-air flow does not
    enter: the humidity rise per kg of dry air is independent of it.

    Condensing branch (:math:`Y_a \ge Y_s(T_{bed})`, fresh air wetter than saturation at bed
    temperature, as in :func:`evaporation_rate_kg_per_s`): :math:`Y_{out} = Y_s(T_{bed})`,
    :math:`Y_{in} = (1 - r)\,Y_a + r\,Y_s`. :math:`r = 0` reproduces :math:`Y_{in} = Y_a`
    exactly in both branches.

    :param recirculation_fraction: :math:`r \in [0, 0.95]` (``OperationConfig``).
    :returns: (inlet humidity :math:`Y_{in}`, outlet humidity :math:`Y_{out}`, saturation degree
        :math:`\nu`), kg/kg dry air and -.
    """
    r = recirculation_fraction
    y_saturation = saturation_humidity_kg_per_kg(backend, gas, bed_temperature_celsius)
    y_effective = backend.minimum(y_saturation, material.max_driving_humidity_kg_per_kg)
    drying_branch = y_saturation > ambient_humidity_kg_per_kg
    driving = y_effective > ambient_humidity_kg_per_kg
    p_saturation = saturation_pressure_pa(backend, bed_temperature_celsius)

    def closed_form(outlet: Array) -> tuple[Array, Array, Array]:
        relative_humidity = relative_humidity_from_absolute_humidity(
            backend, gas, outlet, bed_temperature_celsius, saturation_pressure=p_saturation
        )
        eta = normalised_moisture(
            backend,
            material,
            isotherm,
            particle_moisture_kg_per_kg,
            relative_humidity,
            bed_temperature_celsius,
        )
        if material.drying_curve_exponent is not None:
            eta = eta**material.drying_curve_exponent
        nu = backend.where(driving, material.max_saturation_degree * eta, backend.zeros_like(eta))
        inlet = ((1.0 - r) * ambient_humidity_kg_per_kg + r * nu * y_effective) / (1.0 - r + r * nu)
        return inlet, inlet + nu * (y_effective - inlet), nu

    # g(y) = closed_form(y).outlet is non-increasing in y (phi up -> X_eq up -> eta down -> nu down
    # -> Y_out = Y_a + nu (Y_eff - Y_a) / (1 - r + r nu) down), so h(y) = y - g(y) is increasing
    # with h(Y_a) <= 0 <= h(Y_eff): one root, bracketed. False position with every third step a
    # bisection (the bracket shrinks at least by 2**-(n/3) even where eta has its kink).
    low = ambient_humidity_kg_per_kg
    high = backend.maximum(y_effective, low)
    h_low = low - closed_form(low)[1]
    h_high = high - closed_form(high)[1]

    def false_position(low: Array, high: Array, h_low: Array, h_high: Array) -> Array:
        span = h_high - h_low
        safe_span = backend.where(span > 0.0, span, backend.zeros_like(span) + 1.0)
        return backend.where(
            span > 0.0, (low * h_high - high * h_low) / safe_span, 0.5 * (low + high)
        )

    for step in range(_RECIRCULATION_ITERATIONS):
        trial = 0.5 * (low + high) if step % 3 == 2 else false_position(low, high, h_low, h_high)
        h_trial = trial - closed_form(trial)[1]
        below = h_trial < 0.0
        low = backend.where(below, trial, low)
        h_low = backend.where(below, h_trial, h_low)
        high = backend.where(below, high, trial)
        h_high = backend.where(below, h_high, h_trial)
    inlet, outlet, nu = closed_form(false_position(low, high, h_low, h_high))

    inlet_condensing = (1.0 - r) * ambient_humidity_kg_per_kg + r * y_saturation
    return (
        backend.where(drying_branch, inlet, inlet_condensing),
        backend.where(drying_branch, outlet, y_saturation),
        nu,
    )


def mixing_temperature_celsius(
    gas: GasConfig,
    recirculation_fraction: float,
    ambient_temperature_celsius: Array,
    ambient_humidity_kg_per_kg: Array,
    exhaust_temperature_celsius: Array,
    exhaust_humidity_kg_per_kg: Array,
) -> Array:
    r"""Temperature after adiabatic mixing of fresh air and recirculated exhaust.

    Enthalpy per kg dry air :math:`h = c_{p,L} T + Y (\Delta h_0 + c_{p,D} T)` is linear in the
    mixing ratio, and :math:`Y` mixes linearly, so the latent parts cancel and

    .. math:: T_{mix} = T_a + \frac{r\,c_o\,(T_o - T_a)}{(1 - r)\,c_a + r\,c_o},
        \qquad c = c_{p,L} + Y c_{p,D}

    (``r = 0`` gives :math:`T_a` exactly). Fog in the mixing chamber is not modelled; the caller
    can check :math:`\varphi(Y_{in}, T_{mix}) \le 1`.
    """
    c_ambient = gas.cp_dry_air_j_per_kg_k + ambient_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    c_exhaust = gas.cp_dry_air_j_per_kg_k + exhaust_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
    r = recirculation_fraction
    return ambient_temperature_celsius + r * c_exhaust * (
        exhaust_temperature_celsius - ambient_temperature_celsius
    ) / ((1.0 - r) * c_ambient + r * c_exhaust)
