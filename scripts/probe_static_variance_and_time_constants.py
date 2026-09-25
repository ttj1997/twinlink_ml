r"""Einmalige Probe fuer den Auftrag "zwei Zahlen beschaffen" (A2, A3).

KEIN Netztraining. Ruft nur bestehenden Code auf:
``simulation.diagnostics.static_explained_variance`` (unveraendert, siehe dort) und
``simulation.diagnostics.time_constant_report`` (unveraendert, ruft
``GreyBoxFluidBedGranulator.time_constants``, greybox.py:662-680, auf).

A2, Schritt 1: der Code, der die Tabelle in ``02_prozess_setup.md:189-194`` erzeugt hat,
ist ``scripts/report_baselines.py`` (CONFIG_PATH=greybox_placeholder.yaml, n_runs=32,
sample_time=25s, run_duration=28800s, Standard-``ExcitationConfig``, Standard-Seed
20260812). Dieses Skript uebernimmt exakt dieselbe Anregung/Config-Ladung, ergaenzt nur
eine Schleife ueber 3 Seeds und einen expliziten Schalter fuer
``ExcitationConfig.randomise_initial_conditions`` (Feld existiert bereits,
excitation.py:140), um "mit"/"ohne Streuung der Anfangszustaende" beide Zweige der
Doku-Tabelle zu treffen -- report_baselines.py ruft nur den Default (True) auf.

A2, Schritt 3: zweite Konfiguration ist ``conf/process/greybox_gab_placeholder.yaml``
(GAB-Sorptionsisotherme statt ``ConstantIsotherm``, siehe Dateikopf und Befund 34) --
verifiziert durch Lesen aller drei vorhandenen ``conf/process/*.yaml``.

A3: ruft ``time_constant_report`` fuer beide Konfigurationen am exakten Beharrungszustand
auf (kein Bezug zu den Seeds/der Anregung oben) und stellt die Eigenwert-Zeitkonstanten der
geschlossenen Form tau_T = C / (m_L*c_pL + UA), C = m_s*c_ps + m_w*c_pw, am selben Punkt
gegenueber. Nur Auswertung vorhandener Funktionen, ``greybox.py`` bleibt unveraendert.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.diagnostics import static_explained_variance, time_constant_report
from twinlink_ml.simulation.steady_state import solve_steady_state
from twinlink_ml.utils.seeding import SeedBundle

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PLACEHOLDER = REPO_ROOT / "conf" / "process" / "greybox_placeholder.yaml"
CONFIG_GAB = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"

N_RUNS = 32
SAMPLE_TIME_S = 25.0
RUN_DURATION_S = 28_800.0
MASTER_SEEDS = (20260812, 20260813, 20260814)  # erster Wert = Default aus report_baselines.py


def _load(config_path: Path) -> GreyBoxFluidBedGranulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(config_path)
        config = dataclasses.replace(
            base,
            numerics=dataclasses.replace(base.numerics, sample_time_s=SAMPLE_TIME_S, substeps=5),
        )
    return GreyBoxFluidBedGranulator(config)


def _rollout_r2(
    simulator: GreyBoxFluidBedGranulator, *, scatter: bool, master_seed: int
) -> dict[str, float]:
    seeds = SeedBundle.from_master(master_seed)
    config = simulator.config
    excitation_config = ExcitationConfig(
        run_duration_s=RUN_DURATION_S, randomise_initial_conditions=scatter
    )
    design = MixedExcitationDesign(
        excitation_config,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    schedule = design.build(N_RUNS, seed=seeds.data_seed)
    initial = None
    if schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial,
    )
    return static_explained_variance(trajectory)


def a2() -> dict:
    out: dict = {}
    for label, path in (("placeholder", CONFIG_PLACEHOLDER), ("gab", CONFIG_GAB)):
        simulator = _load(path)
        out[label] = {"fingerprint": simulator.config.fingerprint, "scatter": {}}
        for scatter in (False, True):
            per_seed = []
            for master_seed in MASTER_SEEDS:
                r2 = _rollout_r2(simulator, scatter=scatter, master_seed=master_seed)
                per_seed.append({"master_seed": master_seed, "r2": r2})
            out[label]["scatter"][str(scatter)] = per_seed
    return out


def a3() -> dict:
    out: dict = {}
    for label, path in (("placeholder", CONFIG_PLACEHOLDER), ("gab", CONFIG_GAB)):
        simulator = _load(path)
        config = simulator.config
        report = time_constant_report(simulator)
        solution = solve_steady_state(simulator)
        c_solid = config.material.cp_solid_j_per_kg_k
        c_water = config.material.cp_water_liquid_j_per_kg_k
        c_dry_air = config.gas.cp_dry_air_j_per_kg_k
        ua = config.geometry.heat_loss_ua_w_per_k
        heat_capacity = solution.solid_mass_kg * c_solid + solution.water_mass_kg * c_water
        denom = solution.dry_air_mass_flow_kg_per_s * c_dry_air + ua
        tau_t_closed_form = heat_capacity / denom
        out[label] = {
            "fingerprint": config.fingerprint,
            "eigenvalue_report": report,
            "closed_form": {
                "tau_T_s": tau_t_closed_form,
                "solid_mass_kg": solution.solid_mass_kg,
                "water_mass_kg": solution.water_mass_kg,
                "dry_air_mass_flow_kg_per_s": solution.dry_air_mass_flow_kg_per_s,
                "cp_solid_j_per_kg_k": c_solid,
                "cp_water_liquid_j_per_kg_k": c_water,
                "cp_dry_air_j_per_kg_k": c_dry_air,
                "heat_loss_ua_w_per_k": ua,
            },
        }
    return out


def main() -> None:
    t0 = time.monotonic()
    result = {"a2": a2(), "a3": a3()}
    result["_elapsed_s"] = time.monotonic() - t0
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_a2_a3_static_variance_time_constants.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"geschrieben: {out_path} ({result['_elapsed_s']:.1f} s)")


if __name__ == "__main__":
    main()
