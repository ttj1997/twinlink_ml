r"""Ein-Schritt-Bandit-Umgebung -- Befund 52 in RL-Form.

Reproduziert ``scripts/study_operating_point_regret.py`` (Befund 52) als kontextuellen
Bandit, keine neue Physik und keine neue Zielfunktion.

Beobachtung
    ``ambient_temperature_celsius``, ``ambient_relative_humidity`` -- gezogen aus
    ``ProcessConfig.disturbances`` (``DisturbanceRange.minimum``/``maximum``, NICHT neu
    gesetzt).
Aktion
    ``inlet_temperature_celsius``, ``gas_velocity_m_per_s`` -- normiert auf ``[-1, 1]``,
    Grenzen aus ``ProcessConfig.actions`` (``ActionBound.minimum``/``maximum``, NICHT neu
    gesetzt).
Fest
    ``nozzle_pressure_bar`` = :data:`FIXED_NOZZLE_PRESSURE_BAR` (Befund 52, fester
    Betriebspunkt), ``spray_rate_kg_per_h`` = ``ActionBound.nominal``,
    ``suspension_solid_fraction`` = ``DisturbanceRange.nominal`` -- beide Begruendungen aus
    Befund 52 unveraendert: die Sprührate legt den Durchsatz fest (sonst ist "Energie
    minimieren" entartet), der Duesendruck variiert über alle Gitteroptima nur um 0.4 %.
Uebergang
    ``solve_steady_state`` MIT DER SUCH-PRAEZISION aus Befund 52
    (``DEFAULT_SEARCH_MAX_ITERATIONS``/``DEFAULT_SEARCH_TOLERANCE``, 30 Bisektionsschritte,
    Toleranz 1e-7), NICHT den Loeser-Defaults (100/1e-13). Kein Kompromiss fuer diesen
    Auftrag, sondern numerisch notwendig: mit den Loeser-Defaults scheitern 6 der 25
    Befund-52-Gitterpunkte an einem Bracket-Vorzeichenfehler in der aeusseren
    Temperatur-Bisektion (empirisch geprueft) -- exakt das Phaenomen, das ``evaluate_point``s
    Docstring in ``study_operating_point_regret.py`` beschreibt: an der Bracket-Untergrenze
    (``T_amb + 10``) liegt die wahre Nullstelle knapp ausserhalb, und ob der dort ausgewertete
    Residuenwert numerisches Rauschen mit positivem oder negativem Vorzeichen zeigt, haengt von
    der Bisektionstiefe der INNEREN Feuchte-Bisektion an genau diesem Punkt ab. Befund 52 loest
    das, indem Such- UND Report-Aufruf dieselbe (Such-)Praezision verwenden -- diese Umgebung
    uebernimmt das unveraendert, fuer JEDEN Schritt (nicht nur fuer den VA2-Orakeltest).
Belohnung
    ``-(energy_target + penalty_weight_kw * normalized_violation)``, wie in Befund 52s
    ``_penalized_energy`` (dort Zielfunktion einer Minimierung, hier ihr Negatives als
    Belohnung).

Nebenbedingungsbehandlung (VA1-Entscheidung, siehe Auftrag und Antwort darauf)
-------------------------------------------------------------------------------
Straffunktion fuer ALLE drei Nebenbedingungen aus Befund 52 (Gutsfeuchte,
Fluidisierungsfenster, Bettemperatur <= 80 C), unveraendert per ``check_constraints``
importiert -- KEINE Projektion.

Befund, nicht nur Implementierungsentscheidung: in diesem Environment ist die
Gutsfeuchte die einzige praktisch wirksame Nebenbedingung. Das
Fluidisierungsfenster bindet strukturell nie, weil der Beharrungsdurchmesser d50 (siehe
``steady_state.py``, Schritt 2) hier nur von ``material.nucleus_diameter_m`` und
``material.nucleation_solid_fraction`` abhaengt -- Spruehrate und Suspensionsfeststoffanteil
kuerzen sich in der Herleitung heraus, und beide sind in diesem Environment ohnehin fest.
d50 ist damit UEBER DEN GESAMTEN AKTIONS- UND STOERGROESSENRAUM eine Konstante (750 um bei
den Platzhalterwerten). Eine Sweep-Rechnung ueber Betttemperaturen 20-140 C (weit ueber das
plausible Beharrungsfenster hinaus) und den vollen Stoergroessenbereich liefert worst-case
u_mf ~= 0.236 m/s, worst-case u_terminal ~= 3.42 m/s -- die ``ActionBound`` fuer
``gas_velocity_m_per_s`` ([0.8, 2.5] m/s) liegt mit >= 0.56 m/s Marge unten und
>= 0.92 m/s Marge oben vollstaendig darin. Eine Projektion haette in diesem Environment
nichts zu tun; ``check_constraints`` bleibt trotzdem die alleinige, unveraenderte Quelle
der Zulaessigkeitspruefung (auch fuer Fluidisierung/Bettemperatur), damit ein spaeterer
Wechsel der Nukleationsparameter (die das oben genannte Ergebnis aufheben wuerde) nicht
still unbemerkt bliebe.

WICHTIG: das ist eine Eigenschaft DIESES VIER-ZUSTANDS-MODELLS, keine Aussage über die
reale Anlage oder ein Dyssol-Modell mit Agglomeration/Bruch/Wachstumskopplung an die
Trocknungsbedingungen -- dort haengt die Partikelgroesse sehr wohl von der Fahrweise ab,
und die zustandsabhaengige Aktionsprojektion
(:meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.project_action`) wird erst
in TRANSIENTEN wirksam, wenn d50 vom stationaeren Wert abweicht (Anfahrvorgaenge,
Produktwechsel) -- im stationaeren Fall hier ist sie ein fester Kasten. Dass im
Anlagenmassstab (separate Untersuchung, RA3) bei selbem d50-Platzhalter 15 Gitterpunkte an
der Fluidisierungsobergrenze scheiterten, ist damit vereinbar, nicht widersprüchlich:
vermutlich hoehere Gasgeschwindigkeiten im dortigen Anlagengitter oder andere
Gaseigenschaften bei anderer Betttemperatur -- nicht in diesem Worktree nachgerechnet,
hier nur zur Einordnung genannt, damit beide Aussagen nebeneinander stehen koennen.

``_PENALTY_WEIGHT_KW`` (Befund 52, 50.0 kW) wird UNVERAENDERT uebernommen -- fuer den
VA2-Orakeltest muss die Belohnung am Befund-52-Optimum bitgleich zu ``-energy_target``
sein, und das Gewicht wurde nie neu kalibriert. Es ist aber ein Hyperparameter mit echter
Wirkung auf einen RL-Agenten (der bei bindender Gutsfeuchte direkt an der Grenze sitzt,
anders als der Optimierer, der sie nur als Suchheuristik durchquert) und für RL nicht
abgestimmt -- siehe Befund-Entwurf (VA5) und die berichtete Zulaessigkeitsrate/mittlere
Feuchteueberschreitung (VA3).

Bewusst NICHT in diesem Environment (siehe Auftrag)
-----------------------------------------------------
Speicher, mehrere Linien, Netto-Zielfunktion, Anlagenmassstab -- spaetere Erweiterungen.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.operating_point import (
    DEFAULT_SEARCH_MAX_ITERATIONS,
    DEFAULT_SEARCH_TOLERANCE,
    PENALTY_WEIGHT_KW,
    SOLVE_FAILURE_PENALTY_KW,
    build_variant_config,
    check_constraints,
    specific_energy_kwh_per_kg,
)
from twinlink_ml.simulation.steady_state import solve_steady_state

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
ANCHOR_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"

EXPECTED_CONFIG_FINGERPRINT = "2acd06f38963c3af"
"""Muss mit dem Fingerprint von :data:`CONFIG_PATH`, so wie geladen, uebereinstimmen

(siehe ``probe_critical_moisture_sensitivity.py``, gleiche Konstante dort). Sonst ist der
VA2-Orakeltest nicht am selben Ausgangspunkt gerechnet wie Befund 52.

Der in :data:`ANCHOR_ARTIFACT_PATH` gespeicherte ``config_fingerprint`` traegt noch den
Wert von VOR dem Pseudonymisierungs-Schnitt 2026-09-25 (``abbe8325f1326d80``) -- das
Artefakt ist ein eingefrorener Provenienz-Record und wird nicht angefasst. Der Vergleich
in ``tests/test_operating_point_bandit_env.py`` nutzt deshalb
:func:`twinlink_ml.utils.fingerprint_history.fingerprints_equivalent` statt ``==``, siehe
``docs/ARCHIV.md`` fuer den Aequivalenznachweis."""

FIXED_NOZZLE_PRESSURE_BAR = 1.002634656115166
"""Befund 52, ``fixed_point.nozzle_pressure_bar`` in :data:`ANCHOR_ARTIFACT_PATH` (der
fester, ueberall zulaessige Betriebspunkt) -- ueber alle 25 Gitteroptima variiert der
Duesendruck nur um 0.4 % der Stellgroessenspanne (``setpoint_spread.nozzle_pressure_bar.
span_fraction_of_bound``), deshalb hier als dritte, in diesem Auftrag nicht gelernte Aktion
fixiert (Auftrag VA1). Voller Gleitkommawert, nicht die im Auftragstext gerundete
Kurzform "1.003 bar" -- fuer den VA2-Bitgleichheitstest zaehlt die Praezision."""

FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS = 90.4128503528912
"""Befund 52, ``fixed_point.inlet_temperature_celsius`` in :data:`ANCHOR_ARTIFACT_PATH` --
zusammen mit :data:`FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S` der feste, ueberall
zulaessige Betriebspunkt, der auch :data:`FIXED_NOZZLE_PRESSURE_BAR` traegt. Fuer BB2s
zentrierte Belohnung (:class:`RewardCentering`) -- NICHT fuer die Umgebung selbst gelernt
oder als Aktionsgrenze verwendet."""

FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S = 0.8001296812578764
"""Siehe :data:`FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS`."""

DEFAULT_REWARD_CENTERING_SCALE_KW = 0.2296627007886406
"""Vorschlag fuer :attr:`RewardCentering.scale_kw`: das 95. Perzentil des Befund-52-Regrets
(``regret.absolute_kw.p95`` in :data:`ANCHOR_ARTIFACT_PATH`, Median 0.174 kW, Maximum 0.241
kW -- p95 als Kompromiss zwischen "typisch" und "fast das Maximum"). Begruendung: eine
Politik, die den GESAMTEN Regret an einem typischen Punkt schliesst, sieht damit eine
Belohnungsdifferenz der Groessenordnung 1 -- dieselbe Grossenordnung wie VA3s "Anteil
geschlossenen Regrets" (0 = fest, 1 = Orakel). Kein Bezug zur Strafskala (siehe
:class:`RewardCentering`, Punkt 3)."""

ACTION_NAMES: tuple[str, str] = ("inlet_temperature_celsius", "gas_velocity_m_per_s")
DISTURBANCE_NAMES: tuple[str, str] = (
    "ambient_temperature_celsius",
    "ambient_relative_humidity",
)


def _normalise(value: float, minimum: float, maximum: float) -> float:
    return 2.0 * (value - minimum) / (maximum - minimum) - 1.0


def _denormalise(value: float, minimum: float, maximum: float) -> float:
    return minimum + 0.5 * (value + 1.0) * (maximum - minimum)


@dataclass(frozen=True, slots=True)
class RewardCentering:
    r"""BB2: affine Belohnungstransformation ``r' = (r - r_ref) / scale_kw``.

    ``r_ref`` ist die ROHE Belohnung (kein Strafanteil, da Befund 52s fester Betriebspunkt
    an allen 25 Gitterpunkten zulaessig ist) des FESTEN Befund-52-Betriebspunkts
    (:data:`FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS`/``..._GAS_VELOCITY_M_PER_S``),
    fuer JEDEN Kontext neu ausgewertet -- ein zusaetzlicher ``solve_steady_state``-Aufruf je
    Schritt, etwa doppelte Kosten gegenueber dem rohen Modus.

    WICHTIG, drei Punkte (Auftrag):

    1. Die affine Abbildung AENDERT DAS OPTIMUM NICHT: fuer ``scale_kw > 0`` ist
       ``a -> (a - c) / scale_kw`` streng monoton steigend, das Argmax einer Politik bleibt
       identisch. Befund 52 bleibt der Massstab -- diese Transformation aendert NUR die
       Belohnungs-SKALA, die der Agent sieht, nicht die zugrunde liegende Optimierungsaufgabe.
    2. Die Umgebung behaelt einen Reproduktionsmodus mit ROHER Belohnung
       (``BanditEnvConfig.reward_centering=None``, Default) -- die VA2-Orakelpruefung laeuft
       ausschliesslich in diesem Modus und bleibt bitgleich (siehe
       ``tests/test_operating_point_bandit_env.py``, unveraendert).
    3. Die in YA2 gemessene Dominanz der Strafe (Faktor ~516 gegenueber dem zu schliessenden
       Regret) AENDERT SICH DADURCH NICHT: Strafe und Energie werden GLEICH skaliert (beide
       durch denselben ``scale_kw`` geteilt), ihr VERHAELTNIS zueinander bleibt exakt
       erhalten. Das ist ein getrenntes Problem (siehe YA2s vier Optionen) und hier
       ausdruecklich NICHT adressiert -- BB2 testet ausschliesslich, ob das Entfernen der
       KONTEXTABHAENGIGEN Rohenergie-Grundskala (YA2 (b): ~2-4 kW Streuung zulaessiger
       Aktionen, ~18-fach des Regret-Signals) dem Kritiker hilft, unabhaengig von der
       Strafdominanz.
    """

    scale_kw: float = DEFAULT_REWARD_CENTERING_SCALE_KW


@dataclass(frozen=True, slots=True)
class BanditEnvConfig:
    """Env-eigene Konfiguration.

    Aktions-/Stoergroessengrenzen stehen bereits in ``ProcessConfig``
    (``config.actions``/``config.disturbances``) und werden hier NICHT dupliziert.
    """

    process_config_path: Path = CONFIG_PATH
    fixed_nozzle_pressure_bar: float = FIXED_NOZZLE_PRESSURE_BAR
    penalty_weight_kw: float | None = None
    """``None`` (Default) uebernimmt ``operating_point.PENALTY_WEIGHT_KW`` (Befund 52, in S0.1
    aus dem Skript verschoben) unveraendert -- Bitgleichheit fuer den VA2-Orakeltest.
    Explizit gesetzt nur fuer spaeteres RL-Tuning ausserhalb dieses Auftrags (siehe
    Moduldocstring)."""
    reward_centering: RewardCentering | None = None
    """``None`` (Default): ROHE Belohnung, bitgleich zu VA2 -- siehe :class:`RewardCentering`
    fuer den BB2-Modus mit affiner Transformation."""
    include_penalty: bool = True
    """DA1: ``True`` (Default, unveraendert) -- Belohnung ist ``-(energy_target +
    penalty_weight_kw * normalized_violation)`` wie in Befund 52 und der VA2-Orakelpruefung.
    ``False``: Belohnung ist NUR ``-energy_target``, die Nebenbedingungen werden weiterhin
    ueber ``check_constraints`` geprueft und vollstaendig in ``info`` protokolliert (nur
    NICHT in die Belohnung eingerechnet) -- trennt die Frage "kollabiert SAC wegen der
    Strafklippe?" von "kollabiert SAC grundsaetzlich, mit oder ohne Strafe?" (DA1). Ein
    Loeserfehlschlag bleibt UNABHAENGIG von diesem Schalter bestraft, siehe
    :attr:`solve_failure_penalty_kw` -- das ist keine Nebenbedingungsstrafe, sondern die
    numerische Notwendigkeit, dass ohne geloeste Energie kein ``-energy_target`` existiert,
    das zurueckgegeben werden koennte."""
    product_moisture_limit_kg_per_kg: float | None = None
    """S0.6: quality limit of the product moisture, overriding the loaded config.

    ``None`` (Default): the config decides (its own ``product_moisture_limit_kg_per_kg`` or,
    if unset, the critical moisture). Set: ``MaterialConfig.product_moisture_limit_kg_per_kg``
    of the loaded config is replaced; the drying kinetics keep ``critical_moisture_kg_per_kg``."""
    solve_failure_penalty_kw: float | None = None
    """EA2: ``None`` (Default, unveraendert) uebernimmt Befund 52s Sentinel
    (``operating_point.SOLVE_FAILURE_PENALTY_KW`` = 1.000.000) -- Bitgleichheit fuer VA2, das
    nur loesbare Orakelaktionen einspeist und den Sentinel nie beruehrt. Explizit gesetzt: ein
    Loeserfehlschlag liefert ``reward = -solve_failure_penalty_kw`` statt des Sentinels.

    KEIN fester, willkuerlich gewaehlter Ersatzwert -- die vorgesehene Berechnung (siehe
    ``scripts/probe_bandit_env_solve_failure_bound.py``) ist MODUSABHAENGIG: die
    schlechteste Belohnung, die eine LOESBARE Aktion im jeweils aktiven Modus
    (``include_penalty``) tatsaechlich erreicht (ueber ein Referenzgitter) -- ein
    Loeserfehlschlag wird damit "so schlecht wie der schlechteste loesbare Betriebspunkt",
    nicht ein Ausreisser mehrere Groessenordnungen darueber. Anlass: EA1 zeigt, dass der
    Sentinel vermutlich die Kritiker-Regression vergiftet (Q-Werte an der numerisch
    fragilen kalten Ecke liegen bei allen beobachteten Checkpoints weit jenseits jeder
    physikalisch moeglichen Belohnung)."""


class OperatingPointBanditEnv(gym.Env[np.ndarray, np.ndarray]):
    """Kontextueller Bandit: eine Episode = ein Schritt. Siehe Moduldocstring."""

    # dict statt ClassVar: gym.Env deklariert ``metadata`` selbst als Instanzattribut
    # (mypy lehnt eine ClassVar-Ueberschreibung davon ab) -- Standardmuster im gesamten
    # Gymnasium-Oekosystem, keine tatsaechliche Mutable-Default-Gefahr.
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(self, config: BanditEnvConfig | None = None) -> None:
        self._env_config = config if config is not None else BanditEnvConfig()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            self._base_config: ProcessConfig = load_process_config(
                self._env_config.process_config_path
            )
        if self._env_config.product_moisture_limit_kg_per_kg is not None:
            self._base_config = dataclasses.replace(
                self._base_config,
                material=dataclasses.replace(
                    self._base_config.material,
                    product_moisture_limit_kg_per_kg=self._env_config.product_moisture_limit_kg_per_kg,
                ),
            )

        self._penalty_weight_kw = (
            self._env_config.penalty_weight_kw
            if self._env_config.penalty_weight_kw is not None
            else PENALTY_WEIGHT_KW
        )

        actions = self._base_config.actions
        self._inlet_bound = actions.inlet_temperature_celsius
        self._gas_bound = actions.gas_velocity_m_per_s

        disturbances = self._base_config.disturbances
        self._t_amb_range = disturbances.ambient_temperature_celsius
        self._phi_range = disturbances.ambient_relative_humidity

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self._t_amb_range.minimum, self._phi_range.minimum], dtype=np.float32),
            high=np.array([self._t_amb_range.maximum, self._phi_range.maximum], dtype=np.float32),
            dtype=np.float32,
        )

        self._disturbance: tuple[float, float] | None = None

    # ------------------------------------------------------------- Gymnasium-Schnittstelle

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Zieht einen neuen Stoergroessenpunkt (oder uebernimmt einen vorgegebenen).

        ``options`` mit ``ambient_temperature_celsius``/``ambient_relative_humidity``
        erzwingt einen bestimmten Punkt statt einer Ziehung -- fuer den VA2-Orakeltest
        (die exakten 25 Befund-52-Gitterpunkte) und die VA3-Metrik (fester Testpunktsatz),
        die keinen Zufall vertragen.
        """
        super().reset(seed=seed)
        options = options or {}
        if "ambient_temperature_celsius" in options or "ambient_relative_humidity" in options:
            t_amb = float(options.get("ambient_temperature_celsius", self._t_amb_range.nominal))
            phi_amb = float(options.get("ambient_relative_humidity", self._phi_range.nominal))
        else:
            t_amb = float(
                self.np_random.uniform(self._t_amb_range.minimum, self._t_amb_range.maximum)
            )
            phi_amb = float(
                self.np_random.uniform(self._phi_range.minimum, self._phi_range.maximum)
            )
        self._disturbance = (t_amb, phi_amb)
        observation = np.array([t_amb, phi_amb], dtype=np.float32)
        info: dict[str, Any] = {
            "ambient_temperature_celsius": t_amb,
            "ambient_relative_humidity": phi_amb,
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._disturbance is None:
            raise RuntimeError("step() vor reset() aufgerufen")
        t_amb, phi_amb = self._disturbance

        requested_normalised = np.asarray(action, dtype=np.float64)
        applied_normalised = np.clip(requested_normalised, -1.0, 1.0)
        t_in = _denormalise(
            float(applied_normalised[0]), self._inlet_bound.minimum, self._inlet_bound.maximum
        )
        u_gas = _denormalise(
            float(applied_normalised[1]), self._gas_bound.minimum, self._gas_bound.maximum
        )

        reward, info = self.evaluate_operating_point(
            inlet_temperature_celsius=t_in,
            gas_velocity_m_per_s=u_gas,
            ambient_temperature_celsius=t_amb,
            ambient_relative_humidity=phi_amb,
        )
        info["action_normalised_requested"] = requested_normalised.tolist()
        info["action_normalised_applied"] = applied_normalised.tolist()

        observation = np.array([t_amb, phi_amb], dtype=np.float32)
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info

    # -------------------------------------------------------------------- Kernberechnung

    def evaluate_operating_point(
        self,
        *,
        inlet_temperature_celsius: float,
        gas_velocity_m_per_s: float,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        """Belohnung (roh oder zentriert, siehe :class:`RewardCentering`) und Info-Dict.

        Kernmethode ohne ``reset()``/``step()`` und ohne Normierungs-Rundtrip -- von
        :meth:`step` UND vom VA2-Orakeltest sowie der VA3-Metrik (``evaluation.py``)
        verwendet, damit Belohnung/Zulaessigkeit an genau einer Stelle berechnet werden.
        VA2 verlangt Bitgleichheit im ROHEN Modus (``reward_centering=None``, Default) --
        dieser Pfad ruft intern exakt dieselbe :meth:`_evaluate_operating_point_raw`
        auf wie zuvor, unveraendert.
        """
        raw_reward, info = self._evaluate_operating_point_raw(
            inlet_temperature_celsius=inlet_temperature_celsius,
            gas_velocity_m_per_s=gas_velocity_m_per_s,
            ambient_temperature_celsius=ambient_temperature_celsius,
            ambient_relative_humidity=ambient_relative_humidity,
        )
        centering = self._env_config.reward_centering
        if centering is None:
            return raw_reward, info

        # BB2: r' = (r - r_ref) / scale_kw -- r_ref ist Befund 52s fester Betriebspunkt am
        # SELBEN Kontext, roh ausgewertet (kein Rekursionsrisiko: dieser Aufruf geht direkt
        # an die rohe Methode, nicht an evaluate_operating_point selbst).
        fixed_reward, fixed_info = self._evaluate_operating_point_raw(
            inlet_temperature_celsius=FIXED_OPERATING_POINT_INLET_TEMPERATURE_CELSIUS,
            gas_velocity_m_per_s=FIXED_OPERATING_POINT_GAS_VELOCITY_M_PER_S,
            ambient_temperature_celsius=ambient_temperature_celsius,
            ambient_relative_humidity=ambient_relative_humidity,
        )
        centered_reward = (raw_reward - fixed_reward) / centering.scale_kw
        info["reward_raw"] = raw_reward
        info["reward_centered"] = centered_reward
        info["reward_centering_scale_kw"] = centering.scale_kw
        info["fixed_point_reward_reference"] = fixed_reward
        info["fixed_point_solve_failed_for_reference"] = bool(fixed_info.get("solve_failed"))
        return centered_reward, info

    def _evaluate_operating_point_raw(
        self,
        *,
        inlet_temperature_celsius: float,
        gas_velocity_m_per_s: float,
        ambient_temperature_celsius: float,
        ambient_relative_humidity: float,
    ) -> tuple[float, dict[str, Any]]:
        """Rohe Belohnung, wie vor BB2 -- unveraendert, damit VA2 bitgleich bleibt."""
        variant_config = build_variant_config(
            self._base_config,
            t_in=inlet_temperature_celsius,
            u_gas=gas_velocity_m_per_s,
            p_nozzle=self._env_config.fixed_nozzle_pressure_bar,
            t_amb=ambient_temperature_celsius,
            phi_amb=ambient_relative_humidity,
        )
        simulator = GreyBoxFluidBedGranulator(variant_config)

        requested_physical = {
            "inlet_temperature_celsius": inlet_temperature_celsius,
            "gas_velocity_m_per_s": gas_velocity_m_per_s,
            "nozzle_pressure_bar": self._env_config.fixed_nozzle_pressure_bar,
            "spray_rate_kg_per_h": self._base_config.actions.spray_rate_kg_per_h.nominal,
        }

        try:
            solution = solve_steady_state(
                simulator,
                max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
                tolerance=DEFAULT_SEARCH_TOLERANCE,
            )
        except (RuntimeError, ValueError) as exc:
            solve_failure_penalty_kw = (
                self._env_config.solve_failure_penalty_kw
                if self._env_config.solve_failure_penalty_kw is not None
                else SOLVE_FAILURE_PENALTY_KW
            )
            reward = -solve_failure_penalty_kw
            info: dict[str, Any] = {
                "feasible": False,
                "solve_failed": True,
                "solve_failure_reason": str(exc),
                "solve_failure_penalty_kw": solve_failure_penalty_kw,
                "action_physical_requested": requested_physical,
                "action_physical_applied": requested_physical,
                "ambient_temperature_celsius": ambient_temperature_celsius,
                "ambient_relative_humidity": ambient_relative_humidity,
            }
            return reward, info

        check = check_constraints(
            solution,
            critical_moisture_kg_per_kg=(
                self._base_config.material.quality_moisture_limit_kg_per_kg
            ),
            gas_velocity_m_per_s=gas_velocity_m_per_s,
        )
        energy_target = float(solution.diagnostics["energy_target"])
        if self._env_config.include_penalty:
            reward = -(energy_target + self._penalty_weight_kw * check.normalized_violation)
        else:
            # DA1: Nebenbedingungen bleiben geprueft/protokolliert (unten in ``info``),
            # gehen aber NICHT in die Belohnung ein.
            reward = -energy_target

        info = {
            "feasible": check.feasible,
            "solve_failed": False,
            "energy_target": energy_target,
            "energy_target_unit": self._base_config.energy.target,
            "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg(solution),
            "constraint_check": check.as_dict(),
            "infeasibility_reasons": list(solution.infeasibility_reasons),
            # Keine Projektionsmechanik in diesem Environment (siehe Moduldocstring) --
            # "requested"/"applied" sind hier immer identisch; die Felder bleiben trotzdem
            # bestehen, damit ein Aufrufer nicht zwischen "kein Projektionsmechanismus" und
            # "Projektion hat nicht gegriffen" unterscheiden muss (Analogie zu
            # ``Trajectory.requested_actions``, siehe ``simulation/base.py``).
            "action_physical_requested": requested_physical,
            "action_physical_applied": requested_physical,
            "ambient_temperature_celsius": ambient_temperature_celsius,
            "ambient_relative_humidity": ambient_relative_humidity,
        }
        return float(reward), info
