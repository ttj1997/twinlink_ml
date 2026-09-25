"""Tests für :class:`CompositeSequenceModel` (Befund 26, ``docs/04_stand_ap1.md``).

Wie bei den GRU-Tests in ``test_experiments.py``: nur mit PyTorch prüfbar, die Tests
überspringen sich ohne Torch. Geprüft wird nicht, dass die getrennten Netze besser
generalisieren -- das ist Aufgabe des Diagnoseexperiments
(``scripts/diagnose_soft_sensor_interference.py``), nicht dieser Testsuite -- sondern
dass die Verbundklasse die bestehenden Invarianten pro Teilmodell erhält und tatsächlich
unabhängige Netze trainiert, keine Fassade um ein gemeinsames.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import DataSpec, RolloutEvaluator, build_datasets, generate_data
from twinlink_ml.models import PersistenceModel
from twinlink_ml.models.registry import build_factories
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"
HAS_TORCH = importlib.util.find_spec("torch") is not None

SMALL = DataSpec(
    sample_time_s=25.0,
    run_duration_s=4000.0,
    n_train_max=8,
    n_val=4,
    n_test=4,
    history_length=16,
    horizon=8,
    stride=8,
    target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
)


@pytest.fixture(scope="module")
def process_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def data(process_config):
    return generate_data(process_config, SMALL, master_seed=1)


def test_submodel_seed_is_reproducible_and_decorrelated() -> None:
    """Kein Torch nötig -- reine Hash-Ableitung, analog zu ``SeedBundle``.

    Siehe ``test_greybox_numerics.py::test_seed_bundle_is_reproducible_and_decorrelated``.
    Die eigentliche Regression, gegen die dieser Test schützt: ``seed + index`` würde für
    benachbarte Zielgrößen (Position 0, 1, 2) um genau 1 auseinanderliegende Seeds
    erzeugen -- korreliert im selben Sinn, den ``utils/seeding._derive`` für Master-Seeds
    vermeidet.
    """
    from twinlink_ml.models.composite import _submodel_seed

    first = _submodel_seed(5, "energy_target")
    assert first == _submodel_seed(5, "energy_target")

    neighbour = _submodel_seed(5, "product_moisture_kg_per_kg")
    assert abs(first - neighbour) > 1000, "Teilseeds wirken wie seed + index -- Regression"

    assert _submodel_seed(5, "energy_target") != _submodel_seed(6, "energy_target")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_matches_three_independently_fitted_models(data) -> None:
    """Kernzusicherung: das Verbundmodell ist keine Fassade um ein gemeinsames Netz.

    Dieselbe Vorhersage wie drei einzeln mit ``TorchSequenceModel`` aufgerufene Modelle,
    mit derselben Seed-Ableitung (``_submodel_seed``, SHA-256 wie ``SeedBundle`` -- nicht
    ``seed + index``, siehe ``CompositeSequenceModel``-Docstring) und demselben
    je-Zielgröße reduzierten Datensatz. Wäre der Verbund heimlich ein gemeinsames Netz
    (oder liefen die Teilmodelle sich gegenseitig ins Gehege, z.B. durch geteilten
    globalen Zustand), wäre diese Übereinstimmung nicht exakt.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=3, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        from twinlink_ml.models.composite import _select_target

        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_untrained_equals_persistence_per_target(data) -> None:
    """Nullinitialisierter Kopf = Persistenz, jetzt pro Teilmodell geprüft.

    Dieselbe Vorgabe wie ``test_untrained_network_equals_persistence`` in
    ``test_experiments.py``, aber für den Verbund: jedes Teilnetz muss die Invariante
    unabhängig erfüllen, nicht nur "im Mittel" über alle drei Zielgrößen.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0))
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_rollout_does_not_see_future_targets(data) -> None:
    """Dieselbe Zusicherung wie beim Einzelnetz: kein heimliches Fortschreiben der Wahrheit.

    Prüft insbesondere, dass das Umschneiden der Batches in ``predict`` (Kanalauswahl je
    Teilmodell) keinen neuen Pfad öffnet, über den zukünftige Zielwerte einfließen könnten.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8, max_epochs=3, patience=0))
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_shares_the_trainer_across_cores(data) -> None:
    """GRU, LSTM und xLSTM teilen weiterhin denselben Trainer -- jetzt auch im Verbund."""
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    evaluator = RolloutEvaluator(horizons=(1, SMALL.horizon))
    for core in ("gru", "lstm", "xlstm"):
        model = CompositeSequenceModel(
            TorchSequenceConfig(core=core, hidden_size=8, max_epochs=2, patience=0)
        )
        model.fit(train, seed=1, validation=None)
        assert model.n_parameters > 0
        result = evaluator.evaluate(model, test)
        assert np.isfinite(result.nrmse_run_spread["energy_target"][SMALL.horizon][0])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_n_parameters_sums_submodels(data) -> None:
    """``n_parameters`` ist die Summe über alle Teilnetze, nicht die eines einzelnen."""
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=0)

    single = TorchSequenceModel(config)
    single.fit(train, seed=0)

    # Die Teilnetze sind schmaler (ein Ausgang statt drei), aber es sind drei -- die
    # Summe muss größer sein als ein einzelnes Netz mit vollem, geteiltem Kopf.
    assert composite.n_parameters > 0
    assert composite.n_parameters == sum(m.n_parameters for m in composite._submodels.values())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_sequence_describe_actual_device_none_before_fit() -> None:
    """``net_actual_device`` ist ``None``, solange ``_network`` noch nicht gebaut ist.

    Defensive Absicherung (analog zu ``n_parameters``) fuer den Fall, dass ``describe()``
    jemals vor ``fit()`` aufgerufen wird -- aktuell tut das keine Aufrufstelle im Repo,
    aber ein falsch belegtes Feld waere schlimmer als gar keines.
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    model = TorchSequenceModel(TorchSequenceConfig(hidden_size=8))
    assert model.describe()["net_actual_device"] is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_sequence_describe_actual_device_matches_network_after_fit(data) -> None:
    """``net_actual_device`` spiegelt das TATSAECHLICH verwendete Geraet nach ``fit()``.

    Unterscheidet sich vom bereits vorhandenen ``net_device`` (dem konfigurierten Wunsch,
    hier ``None`` -- der Default) -- ``device=None`` loest in ``fit()`` zu ``"cuda"`` auf,
    falls verfuegbar, sonst ``"cpu"`` (Befund/Anlass: ``befund42_tbed_exhaust_sweep.jsonl``,
    CUDA-faehiger Torch-Build ohne geloggtes Geraete-Feld).
    """
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0)

    model = TorchSequenceModel(config)
    model.fit(train, seed=0)

    description = model.describe()
    assert description["net_device"] is None  # konfigurierter Wunsch, unveraendert
    assert description["net_actual_device"] == str(next(model._network.parameters()).device)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_describe_actual_device_none_before_fit() -> None:
    """``net_actual_device`` ist ``None``, solange ``_submodels`` noch leer ist."""
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    composite = CompositeSequenceModel(TorchSequenceConfig(hidden_size=8))
    assert composite.describe()["net_actual_device"] is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_describe_actual_device_matches_first_submodel_after_fit(data) -> None:
    """``CompositeSequenceModel.describe()`` uebernimmt ``net_actual_device`` vom Teilmodell.

    Ohne diese Weiterleitung waere das in ``TorchSequenceModel.describe()`` neu eingefuehrte
    Feld fuer praktisch jedes Produktionsskript toter Code -- ``run_xlstm_reduced_comparison.py``
    u.ae. loggen ``CompositeXLSTMSequenceModel``/``CompositeSequenceModel``, nie ein blankes
    ``TorchSequenceModel``.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(hidden_size=8, max_epochs=0, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=0)

    first_submodel = next(iter(composite._submodels.values()))
    expected = str(next(first_submodel._network.parameters()).device)
    assert composite.describe()["net_actual_device"] == expected


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_gru_label_builds_composite_and_trains(data) -> None:
    """Verdrahtung in ``registry.py``: das Label ``gru`` liefert seit Befund 26 den Verbund.

    Regressionstest für die Umstellung in ``_resolve_torch_factory`` -- ohne ihn wäre ein
    Rückfall auf das alte Einzelnetz (z.B. durch einen Merge-Konflikt) erst wieder in
    einem vollen ``study_architectures.py``-Lauf sichtbar, nicht in der Testsuite.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("gru",))
    model = factories["gru"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


# --------------------------------------------------------------------------- xLSTM
#
# Dieselben Invarianten wie oben für GRU/LSTM, hier für den mLSTM-Kern
# (``twinlink_ml.models.xlstm``, Registrierung unter dem Label ``xlstm``, siehe
# ``docs/04_stand_ap1.md`` Befund 29). Ein eigener Abschnitt statt einer Parametrisierung
# der bestehenden Tests -- die bestehenden Tests bleiben dadurch unverändert, und ein
# roter xLSTM-Test kann keinen bestehenden GRU/LSTM-Test grün oder rot mitziehen.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_xlstm_label_builds_composite_and_trains(data) -> None:
    """Verdrahtung in ``registry.py``: das Label ``xlstm`` liefert den Verbund, wie gru/lstm."""
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("xlstm",))
    model = factories["xlstm"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_untrained_equals_persistence_per_target(data) -> None:
    """Nullinitialisierter Kopf = Persistenz, auch für den mLSTM-Kern, pro Teilmodell.

    Der mLSTM-Kern selbst ist **nicht** nullinitialisiert (nur der Kopf) -- die Zelle
    läuft mit ihrer normalen Zufallsinitialisierung durch die volle Historie. Dass das
    Ergebnis trotzdem exakt der Persistenz entspricht, prüft gleichzeitig, dass der Kern
    für zufällige Gewichte endliche Ausgaben liefert (sonst würde ``0 * inf = nan`` den
    nullinitialisierten Kopf durchbrechen, siehe Modul-Docstring von ``xlstm.py``).
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=0, patience=0)
    )
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_rollout_does_not_see_future_targets(data) -> None:
    """Kein heimliches Fortschreiben der Wahrheit -- auch nicht über den mLSTM-Zustand."""
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=3, patience=0)
    )
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_matches_three_independently_fitted_models(data) -> None:
    """Verbund aus drei mLSTM-Teilnetzen == drei einzeln aufgerufene ``TorchSequenceModel``.

    Dieselbe Kernzusicherung wie ``test_composite_matches_three_independently_fitted_models``
    (GRU), hier für ``core="xlstm"`` -- die mLSTM-Zelle hat mit ``(C, n, m)`` einen
    komplexeren Zustand als GRU/LSTM; dieser Test prüft, dass ``CompositeSequenceModel``
    diesen Zustand pro Teilmodell genauso sauber isoliert wie den von GRU/LSTM.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _select_target,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=3, patience=0)

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_xlstm_prediction_is_reproducible_and_seed_sensitive(data) -> None:
    """Gleicher Seed -> bitgleiche Vorhersage; verschiedener Seed -> andere Vorhersage.

    Die zweite Hälfte ist mit ``max_epochs=0`` nicht prüfbar: der nullinitialisierte Kopf
    macht jede Vorhersage unabhängig vom Seed exakt zur Persistenz (siehe Test oben) --
    Seed-Empfindlichkeit zeigt sich erst, nachdem der Kopf trainiert und von Null
    weggerückt ist.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    batch = test.batch(range(min(16, test.n_windows)))
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=5, patience=0)

    first = CompositeSequenceModel(config)
    first.fit(train, seed=7)
    second = CompositeSequenceModel(config)
    second.fit(train, seed=7)
    assert np.allclose(first.predict(batch), second.predict(batch), atol=1e-6)

    third = CompositeSequenceModel(config)
    third.fit(train, seed=8)
    assert not np.allclose(first.predict(batch), third.predict(batch), atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_stays_finite_over_long_rollout(process_config) -> None:
    """Numerische Stabilität der exponentiellen Gatter -- der Grund für Phase 3 dieses Tests.

    Ungestabilisiert überlaufen die exponentiellen Eingangs-/Vergessens-Gatter des mLSTM
    über viele Schritte (Moduldocstring von ``xlstm.py``). Geprüft an zwei Horizonten
    (24, der Projektstandard; 200, künstlich verlängert) mit **zufälliger**
    Kerninitialisierung (``max_epochs=0`` nullt nur den Kopf, siehe Test oben) über
    mehrere Seeds -- kein NaN/Inf, an keiner Stützstelle.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    for horizon, run_duration_s in ((24, 4000.0), (200, 8000.0)):
        spec = DataSpec(
            sample_time_s=25.0,
            run_duration_s=run_duration_s,
            n_train_max=4,
            n_val=2,
            n_test=2,
            history_length=16,
            horizon=horizon,
            stride=8,
            target_names=("energy_target", "product_moisture_kg_per_kg", "d50_um"),
        )
        long_data = generate_data(process_config, spec, master_seed=3)
        train, _, test = build_datasets(long_data, 4)
        batch = test.full_batch()
        for seed in range(3):
            model = CompositeSequenceModel(
                TorchSequenceConfig(core="xlstm", hidden_size=16, max_epochs=0, patience=0)
            )
            model.fit(train, seed=seed, validation=None)
            prediction = model.predict(batch)
            assert np.all(np.isfinite(prediction)), (
                f"horizon={horizon}, seed={seed}: NaN/Inf im Rollout -- "
                "Max-Stabilisierer m_t hat nicht gehalten"
            )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_xlstm_forward_ignores_teacher_when_teacher_forcing_is_zero(data) -> None:
    """Störungstest: bei ``teacher_forcing=0.0`` darf kein wahrer Zukunftswert einfließen.

    Dieselbe Zusicherung wie bei Transformer/Mamba
    (``test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero``,
    ``test_mamba_forward_ignores_teacher_when_teacher_forcing_is_zero``), hier für die
    mLSTM-Rekurrenz: bei der Voreinstellung ``teacher_forcing=0.0`` ist die Eingabe an
    Schritt ``t > 0`` ausschließlich die eigene Vorhersage aus Schritt ``t - 1`` -- nie der
    wahre Zielwert, auch wenn ein ``teacher``-Tensor an ``forward()`` übergeben wird. Der
    bestehende ``test_composite_xlstm_rollout_does_not_see_future_targets`` prüft das nur
    über ``predict()``, das ``teacher`` ohnehin nie übergibt (siehe Transformer-Docstring)
    -- ein Leck ausschließlich im Trainingspfad wäre dort unsichtbar. Dieser Test ruft
    ``forward()`` direkt im Trainingsmodus auf und ersetzt den ``teacher``-Tensor durch
    Zufallsrauschen -- bitgleich (``torch.equal``), keine Toleranz. Die Gegenprobe mit
    ``teacher_forcing=1.0`` stellt sicher, dass der Test nicht zufällig grün ist, weil
    ``teacher`` im Netz gar nicht ankommt.
    """
    import torch

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(core="xlstm", hidden_size=8, max_epochs=2, patience=0)
    model = TorchSequenceModel(config)
    model.fit(train, seed=0, validation=None)
    network = model._network
    network.train()

    batch = train.batch(range(min(8, train.n_windows)))
    device = next(network.parameters()).device
    tensors = model._to_tensors(batch, device)
    noisy_teacher = torch.randn_like(tensors["targets"]) * 1000.0 + 500.0

    baseline = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=0.0
    )
    perturbed = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=0.0
    )
    assert torch.equal(baseline, perturbed), (
        "Vorhersage aendert sich mit dem teacher-Tensor trotz teacher_forcing=0.0 -- "
        "Exposure-Bias-Leck"
    )

    # Gegenprobe: bei aktivem Teacher-Forcing MUSS sich die Vorhersage aendern, sonst
    # waere der Test oben nur zufaellig gruen (z. B. weil `teacher` gar nicht verdrahtet ist).
    with_forcing = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=1.0
    )
    with_forcing_noisy = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=1.0
    )
    assert not torch.equal(with_forcing, with_forcing_noisy)


# ----------------------------------------------------------------------- Transformer
#
# Dieselben Invarianten wie oben für GRU/LSTM/xLSTM, hier für den Transformer-Kern
# (Encoder über die Historie, sequenzieller Decoder über den Horizont,
# ``torch_sequence._build_transformer_network``, Registrierung unter dem Label
# ``transformer``). Ein eigener Abschnitt statt einer Parametrisierung der bestehenden
# Tests -- ein roter Transformer-Test kann keinen bestehenden GRU/LSTM/xLSTM-Test grün
# oder rot mitziehen. ``hidden_size=16`` statt des Dataclass-Standards 64 durchgängig:
# ein Encoder-Layer bei ``d_model=64`` hat allein rund 50000 Parameter (siehe
# ``CompositeTransformerSequenceModel``-Docstring), bei den kleinen Testdatensätzen hier
# unnötig teuer.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_transformer_label_builds_composite_and_trains(data) -> None:
    """Verdrahtung in ``registry.py``: das Label ``transformer`` liefert den Verbund."""
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("transformer",))
    model = factories["transformer"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_transformer_untrained_equals_persistence_per_target(data) -> None:
    """Nullinitialisierter Kopf = Persistenz, auch für den Transformer-Kern, pro Teilmodell.

    Der Encoder/Decoder selbst ist **nicht** nullinitialisiert (nur der Kopf) -- die
    Attention-Gewichte laufen mit ihrer normalen Zufallsinitialisierung durch die volle
    Historie und den Decoder-Schrittzyklus. Dass das Ergebnis trotzdem exakt der
    Persistenz entspricht, prüft gleichzeitig, dass Encoder und Decoder für zufällige
    Gewichte endliche Ausgaben liefern.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=0, patience=0
        )
    )
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_transformer_rollout_does_not_see_future_targets(data) -> None:
    """Kein heimliches Fortschreiben der Wahrheit im Rollout -- auch nicht über den Decoder.

    Wie bei GRU/LSTM/xLSTM: ``predict()`` übergibt ``teacher`` ohnehin nie an
    ``forward()`` (siehe ``TorchSequenceModel.predict``), Manipulation von
    ``future_targets`` im Batch darf die Vorhersage also nicht verändern. Ein Leck in der
    *Trainingsschleife* selbst (bei ``teacher_forcing=0.0``) prüft dieser Test nicht --
    dafür siehe ``test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero``.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=3, patience=0
        )
    )
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_transformer_matches_three_independently_fitted_models(data) -> None:
    """Verbund aus drei Transformer-Teilnetzen == drei einzeln aufgerufene ``TorchSequenceModel``.

    Dieselbe Kernzusicherung wie bei GRU/LSTM/xLSTM, hier für ``core="transformer"``.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _select_target,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=3, patience=0
    )

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero(data) -> None:
    """Störungstest: bei ``teacher_forcing=0.0`` darf kein wahrer Zukunftswert einfließen.

    Direkte Klärung vor der Implementierung (Klartext-Diskussion): bei der Voreinstellung
    ``teacher_forcing=0.0`` ist die Decoder-Eingabe an Schritt ``t > 0`` ausschließlich die
    eigene Vorhersage aus Schritt ``t - 1`` -- nie der wahre Zielwert, auch wenn ein
    ``teacher``-Tensor an ``forward()`` übergeben wird. Dieser Test ersetzt genau diesen
    Tensor durch Zufallsrauschen und ruft ``forward()`` direkt im Trainingsmodus auf --
    bitgleich (``torch.equal``), keine Toleranz. ``predict()`` prüft das nicht: es übergibt
    ``teacher`` ohnehin nie (siehe ``TorchSequenceModel.predict``), ein Leck in der
    Trainingsschleife selbst wäre über den reinen Rollout-Test unsichtbar. Die Gegenprobe
    mit ``teacher_forcing=1.0`` stellt sicher, dass der Test nicht zufällig grün ist, weil
    ``teacher`` im Netz gar nicht ankommt.
    """
    import torch

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="transformer", hidden_size=8, num_heads=2, num_layers=1, max_epochs=2, patience=0
    )
    model = TorchSequenceModel(config)
    model.fit(train, seed=0, validation=None)
    network = model._network
    network.train()

    batch = train.batch(range(min(8, train.n_windows)))
    device = next(network.parameters()).device
    tensors = model._to_tensors(batch, device)
    noisy_teacher = torch.randn_like(tensors["targets"]) * 1000.0 + 500.0

    baseline = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=0.0
    )
    perturbed = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=0.0
    )
    assert torch.equal(baseline, perturbed), (
        "Vorhersage aendert sich mit dem teacher-Tensor trotz teacher_forcing=0.0 -- "
        "Exposure-Bias-Leck"
    )

    # Gegenprobe: bei aktivem Teacher-Forcing MUSS sich die Vorhersage aendern, sonst
    # waere der Test oben nur zufaellig gruen (z. B. weil `teacher` gar nicht verdrahtet ist).
    with_forcing = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=1.0
    )
    with_forcing_noisy = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=1.0
    )
    assert not torch.equal(with_forcing, with_forcing_noisy)


# ----------------------------------------------------------------------------- Mamba
#
# Dieselben Invarianten wie oben für GRU/LSTM/xLSTM/Transformer, hier für den
# selektiven State-Space-Kern (``twinlink_ml.models.mamba``, Registrierung unter dem
# Label ``mamba``). Anders als beim Transformer-Abschnitt ist ``hidden_size=8`` hier
# keine erzwungene Kapazitätsreduktion (die SSM-Zelle wächst nicht quadratisch mit
# ``hidden_size``) -- schlicht dieselbe kleine Testgröße wie bei allen übrigen Kernen.


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_registry_mamba_label_builds_composite_and_trains(data) -> None:
    """Verdrahtung in ``registry.py``: das Label ``mamba`` liefert den Verbund."""
    from twinlink_ml.models.composite import CompositeSequenceModel

    train, _, test = build_datasets(data, 8)
    factories = build_factories(("mamba",))
    model = factories["mamba"]()
    assert isinstance(model, CompositeSequenceModel)
    model.fit(train, seed=0)
    batch = test.batch(range(min(8, test.n_windows)))
    prediction = model.predict(batch)
    assert prediction.shape == batch.future_targets.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_mamba_untrained_equals_persistence_per_target(data) -> None:
    """Nullinitialisierter Kopf = Persistenz, auch für den Mamba-Kern, pro Teilmodell.

    Die SSM-Zelle selbst ist **nicht** nullinitialisiert (nur der Kopf) -- ``A_log``,
    die Projektionen und das Gate laufen mit ihrer normalen Initialisierung durch die
    volle Historie und den Rekursions-Schrittzyklus. Dass das Ergebnis trotzdem exakt
    der Persistenz entspricht, prüft gleichzeitig, dass die Rekurrenz für zufällige
    Gewichte endliche Ausgaben liefert.
    """
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=0, patience=0
        )
    )
    model.fit(train, seed=1, validation=None)

    reference = PersistenceModel()
    reference.fit(train, seed=1)

    batch = test.full_batch()
    assert np.allclose(model.predict(batch), reference.predict(batch), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_mamba_rollout_does_not_see_future_targets(data) -> None:
    """Kein heimliches Fortschreiben der Wahrheit im Rollout -- auch nicht über die SSM.

    Wie bei GRU/LSTM/xLSTM/Transformer: ``predict()`` übergibt ``teacher`` ohnehin nie an
    ``forward()`` (siehe ``TorchSequenceModel.predict``), Manipulation von
    ``future_targets`` im Batch darf die Vorhersage also nicht verändern. Ein Leck in der
    *Trainingsschleife* selbst (bei ``teacher_forcing=0.0``) prüft dieser Test nicht --
    dafür siehe ``test_mamba_forward_ignores_teacher_when_teacher_forcing_is_zero``.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    model = CompositeSequenceModel(
        TorchSequenceConfig(
            core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=3, patience=0
        )
    )
    model.fit(train, seed=0)
    batch = test.batch(range(min(16, test.n_windows)))
    tampered = dataclasses.replace(batch, future_targets=batch.future_targets * 3.0 + 7.0)
    assert np.allclose(model.predict(batch), model.predict(tampered))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_mamba_matches_three_independently_fitted_models(data) -> None:
    """Verbund aus drei Mamba-Teilnetzen == drei einzeln aufgerufene ``TorchSequenceModel``.

    Dieselbe Kernzusicherung wie bei GRU/LSTM/xLSTM/Transformer, hier für ``core="mamba"``.
    """
    from twinlink_ml.models.composite import (
        CompositeSequenceModel,
        _select_target,
        _SingleTargetView,
        _submodel_seed,
    )
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, validation, test = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=3, patience=0
    )

    composite = CompositeSequenceModel(config)
    composite.fit(train, seed=5, validation=validation)

    independent = {}
    for index, target in enumerate(train.target_names):
        submodel = TorchSequenceModel(config)
        submodel.fit(
            _SingleTargetView(train, index),
            seed=_submodel_seed(5, target),
            validation=None if validation is None else _SingleTargetView(validation, index),
        )
        independent[target] = submodel

    batch = test.batch(range(min(16, test.n_windows)))
    composite_prediction = composite.predict(batch)
    for index, target in enumerate(train.target_names):
        history_columns = [*range(composite._n_exogenous), composite._n_exogenous + index]
        sub_batch = _select_target(batch, history_columns, index)
        expected = independent[target].predict(sub_batch)
        assert np.allclose(composite_prediction[..., [index]], expected, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_composite_with_fourth_target_leaves_existing_submodels_bit_identical(data) -> None:
    """Vier statt drei Ziele (``bed_temperature_celsius`` ergaenzt) -- guenstiger Korrektheitslauf.

    Vor einer moeglichen spaeteren Erweiterung von ``DataSpec.target_names``
    (``docs/04_stand_ap1.md``, T_bed als vierte Zielgroesse, Ableitung der
    Abluft-Austrittsgroessen ueber ``simulation.exhaust``). Drei Kernaussagen, nicht nur
    eine Randnotiz:

    1. Das vierte Teilnetz baut ohne Fehler.
    2. Sein Trainingsverlust ist endlich (kein NaN/Inf).
    3. Die drei bestehenden Teilnetze sind **bitgenau unveraendert** gegenueber einem
       Drei-Ziele-Composite auf denselben Daten/demselben Seed -- das ist die wichtigste
       der drei Aussagen: sie sichert ab, dass die Seed-Isolierung zwischen Teilnetzen
       (``_submodel_seed``, Befund 26) auch bei vier statt drei Zielen haelt und das
       vierte Ziel keinen neuen negativen Transfer in die bestehenden drei einschleppt.
    """
    import dataclasses

    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    four_targets = (*SMALL.target_names, "bed_temperature_celsius")
    data_four = dataclasses.replace(
        data, spec=dataclasses.replace(data.spec, target_names=four_targets)
    )

    train3, val3, _test3 = build_datasets(data, 8)
    train4, val4, test4 = build_datasets(data_four, 8)
    assert train4.target_names == four_targets

    net_config = TorchSequenceConfig(hidden_size=8, max_epochs=15, patience=0)

    composite3 = CompositeSequenceModel(net_config)
    composite3.fit(train3, seed=5, validation=val3)

    composite4 = CompositeSequenceModel(net_config)
    composite4.fit(train4, seed=5, validation=val4)

    # (1) Viertes Teilnetz vorhanden.
    assert set(composite4._submodels.keys()) == set(four_targets)

    # (2) bed_temperature_celsius: Trainingsverlust endlich ueber alle Epochen.
    bed_losses = [r["train_loss"] for r in composite4.training_history["bed_temperature_celsius"]]
    assert len(bed_losses) == 15
    assert all(np.isfinite(v) for v in bed_losses), "NaN/Inf in bed_temperature_celsius-Verlust"

    # (3) Die drei bestehenden Ziele sind bitgenau unveraendert -- wichtigste Aussage,
    #     siehe Docstring. Ein einzelnes ungleiches Paar wuerde bedeuten, dass das vierte
    #     Teilnetz heimlich auf gemeinsamen Zustand/RNG-Verbrauch der anderen einwirkt.
    for target in SMALL.target_names:
        losses3 = np.array([r["train_loss"] for r in composite3.training_history[target]])
        losses4 = np.array([r["train_loss"] for r in composite4.training_history[target]])
        assert np.array_equal(losses3, losses4), (
            f"{target}: Trainingsverlust unterscheidet sich zwischen Drei- und "
            "Vier-Ziele-Composite -- Isolationsannahme (Befund 26) verletzt"
        )

    # Inferenz-Sanity: predict() liefert vier endliche Spalten.
    batch4 = test4.batch(range(min(8, test4.n_windows)))
    prediction4 = composite4.predict(batch4)
    assert prediction4.shape[-1] == 4
    assert np.all(np.isfinite(prediction4))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_mamba_forward_ignores_teacher_when_teacher_forcing_is_zero(data) -> None:
    """Störungstest: bei ``teacher_forcing=0.0`` darf kein wahrer Zukunftswert einfließen.

    Dieselbe Zusicherung wie beim Transformer
    (``test_transformer_forward_ignores_teacher_when_teacher_forcing_is_zero``), hier für
    die SSM-Rekurrenz: bei der Voreinstellung ``teacher_forcing=0.0`` ist die Eingabe an
    Schritt ``t > 0`` ausschließlich die eigene Vorhersage aus Schritt ``t - 1`` -- nie der
    wahre Zielwert, auch wenn ein ``teacher``-Tensor an ``forward()`` übergeben wird.
    Dieser Test ersetzt genau diesen Tensor durch Zufallsrauschen und ruft ``forward()``
    direkt im Trainingsmodus auf -- bitgleich (``torch.equal``), keine Toleranz. Die
    Gegenprobe mit ``teacher_forcing=1.0`` stellt sicher, dass der Test nicht zufällig
    grün ist, weil ``teacher`` im Netz gar nicht ankommt.
    """
    import torch

    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, TorchSequenceModel

    train, _, _ = build_datasets(data, 8)
    config = TorchSequenceConfig(
        core="mamba", hidden_size=8, state_size=8, num_layers=1, max_epochs=2, patience=0
    )
    model = TorchSequenceModel(config)
    model.fit(train, seed=0, validation=None)
    network = model._network
    network.train()

    batch = train.batch(range(min(8, train.n_windows)))
    device = next(network.parameters()).device
    tensors = model._to_tensors(batch, device)
    noisy_teacher = torch.randn_like(tensors["targets"]) * 1000.0 + 500.0

    baseline = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=0.0
    )
    perturbed = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=0.0
    )
    assert torch.equal(baseline, perturbed), (
        "Vorhersage aendert sich mit dem teacher-Tensor trotz teacher_forcing=0.0 -- "
        "Exposure-Bias-Leck"
    )

    # Gegenprobe: bei aktivem Teacher-Forcing MUSS sich die Vorhersage aendern, sonst
    # waere der Test oben nur zufaellig gruen (z. B. weil `teacher` gar nicht verdrahtet ist).
    with_forcing = network(
        tensors["history"], tensors["future"], teacher=tensors["targets"], teacher_forcing=1.0
    )
    with_forcing_noisy = network(
        tensors["history"], tensors["future"], teacher=noisy_teacher, teacher_forcing=1.0
    )
    assert not torch.equal(with_forcing, with_forcing_noisy)
