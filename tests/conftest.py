"""Shared fixtures.

The tests run against the placeholder config, but deliberately suppress the
:class:`PlaceholderConfigWarning` only here -- in production code it is meant to stay
visible.
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
    """Saves/restores the global Torch state per test.

    Default dtype, threads, grad mode. Occasion: ``torch.set_default_dtype(torch.float64)``
    in two gradient tests (``test_greybox_numerics.py::test_gradient_flows_through_
    full_rk4_step_with_gab_isotherm`` and ``::test_gradient_stays_finite_at_floor_
    active_point``) sets the default dtype globally and permanently, without resetting
    it. The leak is order-dependent and does not fire in the natural suite order --
    the danger is therefore not the failure (that would be visible immediately), but a
    test that silently passes green in float32 instead of float64 without anyone
    noticing. The project convention "ruff/pytest before every commit" is often run only
    on subsets (single test files) -- exactly there the leak occurs, not in the full run.

    autouse, so that it does not have to be imported per test file -- a future test
    that sets global Torch state somewhere would otherwise automatically be a leak again.
    ``num_threads``/grad mode saved along with it, since affected by the same mechanism
    (global, process-wide Torch state, not test-local).
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
    """Standard design for tests, short runs.

    ``initial_condition_design=False`` switches off the scatter of the initial states,
    ``None`` uses the default of the design.
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
    """Constant excitation at the nominal operating point.

    Deliberately here and not in ``src/``: the actual excitation design
    (step sequences, PRBS, ramps, LHS) is step 2 of the work plan. This function is
    only a test aid and is not meant to become a covert anticipation of it.
    """
    actions = np.repeat(simulator.nominal_action(n_batch)[:, None, :], n_steps, axis=1)
    disturbances = np.repeat(simulator.nominal_disturbance(n_batch)[:, None, :], n_steps, axis=1)
    return actions, disturbances
