"""Tests für ``evaluation/recoverable_power.py`` -- Q_rec aus vorhergesagten Abluft-Größen.

Winzige Größen (``n_train_max=8``), analog zu ``tests/test_exhaust_reconstruction.py`` --
geprüft wird die Verdrahtung, nicht die Vorhersagegüte einer echten Studie (X2/X3 laufen
separat, siehe ``scripts/run_outlet_temperature_gru_protocol.py``).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.recoverable_power import (
    DEW_POINT_MARGIN_K,
    DISTRICT_MASS_FLOW_KG_PER_S,
    compute_recoverable_power_kw,
    evaluate_recoverable_power,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.backend import ArrayBackend, NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.thermo import dew_point_celsius

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CONFIG_PATH = "conf/process/greybox_gab_placeholder.yaml"
TARGET_NAMES = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")


def _tiny_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=4000.0,
        n_train_max=8,
        n_val=4,
        n_test=4,
        history_length=16,
        horizon=8,
        stride=8,
        target_names=TARGET_NAMES,
    )


def test_compute_recoverable_power_matches_manual_formula(backend: ArrayBackend) -> None:
    """Direkter Formel-Abgleich, unabhängig vom Rekonstruktions-/Modell-Pfad."""
    gas = GasConfig()
    outlet_temperature = backend.asarray(np.array([80.0, 60.0]))
    humidity = backend.asarray(np.array([0.02, 0.03]))

    result = compute_recoverable_power_kw(backend, gas, outlet_temperature, humidity)

    dew_point = backend.to_numpy(dew_point_celsius(backend, gas, humidity))
    cp_humid = gas.cp_dry_air_j_per_kg_k + backend.to_numpy(humidity) * gas.cp_vapour_j_per_kg_k
    expected = (
        DISTRICT_MASS_FLOW_KG_PER_S
        * (backend.to_numpy(outlet_temperature) - (dew_point + DEW_POINT_MARGIN_K))
        * cp_humid
        / 1000.0
    )
    assert backend.to_numpy(result) == pytest.approx(expected, rel=1e-10)


def test_compute_recoverable_power_zero_for_saturated_air_at_margin() -> None:
    """T_out genau auf Taupunkt+Marge -- Q_rec muss exakt 0 sein (Grenzfall der Formel)."""
    backend = NumpyBackend()
    gas = GasConfig()
    humidity = backend.asarray(np.array([0.02]))
    dew_point = dew_point_celsius(backend, gas, humidity)
    outlet_temperature = dew_point + DEW_POINT_MARGIN_K

    result = compute_recoverable_power_kw(backend, gas, outlet_temperature, humidity)
    assert backend.to_numpy(result)[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_evaluate_recoverable_power_end_to_end_wiring() -> None:
    """Verdrahtung: reconstruct_exhaust_arrays -> Q_rec -> NRMSE/RMSE liefert endliche Werte.

    Kein echtes GRU-Training hier (X2/X3 laufen separat) -- ``PersistenceModel`` reicht für
    die Verdrahtungsprüfung, ohne eine Vorhersagegüte-Aussage zu treffen.
    """
    from twinlink_ml.models.baselines import PersistenceModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config: ProcessConfig = load_process_config(CONFIG_PATH)

    spec = _tiny_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=1)
    train, validation, test = build_datasets(
        data, spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
    )

    model = PersistenceModel()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)

    result = evaluate_recoverable_power(model, data, test, horizons=(1, spec.horizon))

    assert result.n_windows > 0
    assert result.mass_flow_kg_per_s == DISTRICT_MASS_FLOW_KG_PER_S
    assert result.dew_point_margin_k == DEW_POINT_MARGIN_K
    for horizon in result.horizons:
        assert np.isfinite(result.rmse_run_median_kw[horizon])
        assert result.rmse_run_median_kw[horizon] >= 0.0
    # format() muss ohne Fehler laufen und alle Horizonte enthalten.
    formatted = result.format()
    assert "Q_rec" in formatted


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_evaluate_recoverable_power_rejects_mismatched_data_and_test() -> None:
    """Dieselbe Fehlpaarungsprüfung wie ``evaluate_exhaust_reconstruction`` (Befund 42b).

    ``evaluate_recoverable_power`` baut auf ``reconstruct_exhaust_arrays`` auf, muss also
    denselben Schutz erben, nicht neu implementieren.
    """
    from twinlink_ml.data.splits import SplitName
    from twinlink_ml.data.windows import TrajectoryWindowDataset
    from twinlink_ml.models.baselines import PersistenceModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config: ProcessConfig = load_process_config(CONFIG_PATH)

    spec = _tiny_spec()
    holdout_spec = dataclasses.replace(
        spec, n_train_max=2, n_val=2, n_test=spec.n_test, constant_run_fraction=1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=1)
        holdout = generate_data(config, holdout_spec, master_seed=999)

    train, validation, test = build_datasets(
        data, spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
    )
    test_constant = TrajectoryWindowDataset.from_trajectory(
        holdout.trajectory, holdout.split, SplitName.TEST, spec.window, scalers=train.scalers
    )
    assert test.n_windows == test_constant.n_windows

    model = PersistenceModel()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)

    with pytest.raises(ValueError, match="stimmt nicht mit data.trajectory"):
        evaluate_recoverable_power(model, data, test_constant, horizons=(1, spec.horizon))
