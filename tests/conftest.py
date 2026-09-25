"""Gemeinsame Fixtures.

Die Tests laufen gegen die Platzhalter-Config, unterdrücken aber die
:class:`PlaceholderConfigWarning` bewusst nur hier -- im Produktivcode soll sie sichtbar
bleiben.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.simulation import (
    DisturbanceDesign,
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    HoldTimeRange,
    InitialConditionDesign,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    ProcessConfig,
    load_process_config,
)
from twinlink_ml.simulation.backend import ArrayBackend, NumpyBackend

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"


@pytest.fixture(autouse=True)
def _restore_torch_global_state():
    """Sichert/stellt globalen Torch-Zustand je Test wieder her.

    Default-Dtype, Threads, Grad-Modus. Anlass: ``torch.set_default_dtype(torch.float64)``
    in zwei Gradiententests (``test_greybox_numerics.py::test_gradient_flows_through_
    full_rk4_step_with_gab_isotherm`` und ``::test_gradient_stays_finite_at_floor_
    active_point``) setzt den Default-Dtype global und dauerhaft, ohne ihn
    zurückzusetzen. Das Leck ist ordnungsabhängig und zündet in der natürlichen
    Suite-Reihenfolge nicht -- gefährlich ist deshalb nicht der Fehlschlag (der wäre
    sofort sichtbar), sondern ein Test, der still in float32 statt float64 grün
    durchläuft, ohne dass es auffällt. Die Projektkonvention "Ruff/pytest vor jedem
    Commit" läuft häufig nur auf Teilmengen (einzelne Testdateien) -- genau dort tritt
    das Leck auf, nicht im vollen Lauf.

    autouse, damit sie nicht pro Testdatei importiert werden muss -- ein künftiger Test,
    der irgendwo global Torch-Zustand setzt, ist sonst automatisch wieder ein Leck.
    ``num_threads``/Grad-Modus gleich mit gesichert, da vom selben Mechanismus betroffen
    (globaler, prozessweiter Torch-Zustand, kein Test-lokaler).
    """
    try:
        import torch
    except ImportError:
        yield
        return

    default_dtype = torch.get_default_dtype()
    default_num_threads = torch.get_num_threads()
    grad_enabled = torch.is_grad_enabled()
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)
        torch.set_num_threads(default_num_threads)
        torch.set_grad_enabled(grad_enabled)


@pytest.fixture(scope="session")
def backend() -> ArrayBackend:
    return NumpyBackend()


@pytest.fixture
def config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture
def simulator(config: ProcessConfig) -> GreyBoxFluidBedGranulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return GreyBoxFluidBedGranulator(config)


def build_design(
    config: ProcessConfig,
    *,
    excitation: ExcitationConfig | None = None,
    disturbance_design: DisturbanceDesign | None = None,
    initial_condition_design: InitialConditionDesign | bool | None = None,
    n_runs_hint: int = 8,
) -> MixedExcitationDesign:
    """Standarddesign für Tests, kurze Läufe.

    ``initial_condition_design=False`` schaltet die Streuung der Anfangszustände ab,
    ``None`` verwendet die Voreinstellung des Designs.
    """
    _ = n_runs_hint
    excitation = (
        excitation
        if excitation is not None
        else ExcitationConfig(
            run_duration_s=6000.0,
            hold_time=HoldTimeRange(60.0, 900.0),
            prbs_clock_s=60.0,
            prbs_register_length=8,
            ramp_segment_time=HoldTimeRange(300.0, 1800.0),
            constant_run_fraction=0.0,
        )
    )
    initial: InitialConditionDesign | None
    if initial_condition_design is False:
        initial = None
        excitation = dataclasses.replace(excitation, randomise_initial_conditions=False)
    elif initial_condition_design is None or initial_condition_design is True:
        initial = None
    else:
        initial = initial_condition_design
    return MixedExcitationDesign(
        excitation,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
        disturbance_design=disturbance_design,
        initial_condition_design=initial,
    )


def constant_excitation(
    simulator: GreyBoxFluidBedGranulator, n_steps: int, n_batch: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Konstante Anregung im nominellen Betriebspunkt.

    Absichtlich hier und nicht in ``src/``: das eigentliche Anregungsdesign
    (Sprungfolgen, PRBS, Rampen, LHS) ist Schritt 2 des Arbeitsplans. Diese Funktion ist
    nur ein Testhilfsmittel und soll nicht zur heimlichen Vorwegnahme werden.
    """
    actions = np.repeat(simulator.nominal_action(n_batch)[:, None, :], n_steps, axis=1)
    disturbances = np.repeat(simulator.nominal_disturbance(n_batch)[:, None, :], n_steps, axis=1)
    return actions, disturbances
