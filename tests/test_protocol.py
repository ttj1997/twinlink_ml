"""Tests für das Evaluationsprotokoll (Arbeitsplan Schritt 5).

Die teuren Zusicherungen dieses Moduls sind nicht die Zahlen, sondern die
**Orthogonalität** der Versuchsfaktoren: dass die Rausch-Achse wirklich nur das Rauschen
variiert, dass ein neu vermessener Pool bitgleich zu einem neu simulierten ist, und dass
die Validierungs-Betriebsart bei parameterfreien Modellen exakt nichts ändert. Jede
dieser Eigenschaften ist im Ergebnis einer Lernkurve unsichtbar, wenn sie verletzt ist.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import (
    DataPoolCache,
    DataSpec,
    ObservationMode,
    RolloutEvaluator,
    ValidationPolicy,
    estimate_noise_floor,
    generate_data,
    measurement_at_level,
    run_architecture_comparison,
    run_noise_study,
    run_validation_policy_comparison,
    run_variance_decomposition,
    spec_for_mode,
    summarise_policy_gap,
)
from twinlink_ml.evaluation.protocol import _spec_digest
from twinlink_ml.evaluation.resolution import (
    collect_spreads,
    decompose_variance,
    resolution_thresholds,
    select,
)
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.utils.seeding import SeedBundle, SeedPlan, VarianceSource

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

SMALL = DataSpec(
    sample_time_s=25.0,
    run_duration_s=4000.0,
    n_train_max=8,
    n_val=4,
    n_test=4,
    history_length=16,
    horizon=8,
    stride=8,
    target_names=("energy_target", "product_moisture_kg_per_kg"),
)

FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}

FACTORIES_BOTH_MODES = {
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}
"""Ohne ``persistence`` -- die braucht Inline-Messung und bricht in ``SOFT_SENSOR`` ab."""

PLAN = SeedPlan(master_seeds=(1, 2))
EVALUATOR = RolloutEvaluator(horizons=(1, SMALL.horizon))


@pytest.fixture(scope="module")
def process_config():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def pool(process_config):
    return generate_data(process_config, SMALL, seeds=SeedBundle.from_master(1))


# ------------------------------------------------------- Sensormodell entkoppeln


def test_remeasure_reproduces_the_rollout_bit_for_bit(process_config, pool) -> None:
    """Bit für Bit identisch mit dem, was der Rollout selbst gemessen hätte.

    Das ist die Grundlage des ganzen Kostenmodells von Schritt 5: Rauschen neu zu ziehen
    darf keine neue Integration brauchen.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(pool.process_config)
    again = simulator.remeasure(pool.trajectory.outputs, seed=pool.seeds.noise_seed)
    assert np.array_equal(again, pool.trajectory.measurements)


def test_with_measurements_leaves_the_process_untouched(pool) -> None:
    variant = pool.with_measurements(noise_seed=pool.seeds.noise_seed + 1)
    assert np.array_equal(variant.trajectory.outputs, pool.trajectory.outputs)
    assert np.array_equal(variant.trajectory.actions, pool.trajectory.actions)
    assert not np.array_equal(variant.trajectory.measurements, pool.trajectory.measurements)
    assert (
        variant.fingerprints["excitation/fingerprint"]
        == pool.fingerprints["excitation/fingerprint"]
    )


def test_noise_level_enters_the_fingerprint(pool) -> None:
    """Ohne diesen Eintrag wären zwei Rauschniveaus im Log nicht unterscheidbar."""
    louder = pool.with_measurements(
        measurement=measurement_at_level(pool.process_config.measurement, 0.02)
    )
    assert louder.fingerprints["measurement/digest"] != pool.fingerprints["measurement/digest"]
    assert louder.relative_noise_std == pytest.approx(0.02)


def test_zero_noise_is_deterministic(pool) -> None:
    """Bei Rauschniveau Null darf der Rausch-Seed nichts mehr bewirken -- sonst ist der.

    Versuchsfaktor nicht das Rauschen, sondern irgendetwas daneben.
    """
    clean = measurement_at_level(pool.process_config.measurement, 0.0)
    first = pool.with_measurements(noise_seed=11, measurement=clean)
    second = pool.with_measurements(noise_seed=12, measurement=clean)
    assert np.array_equal(first.trajectory.measurements, second.trajectory.measurements)
    assert not np.array_equal(first.trajectory.measurements, first.trajectory.outputs)


def test_sensor_state_does_not_leak_between_rollouts(process_config, pool) -> None:
    """Das Sensor-PT1 trägt Zustand.

    Zwei Rollouts auf derselben Instanz dürfen sich
    trotzdem nicht beeinflussen.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(pool.process_config)
    first = simulator.remeasure(pool.trajectory.outputs, seed=7)
    second = simulator.remeasure(pool.trajectory.outputs, seed=7)
    assert np.array_equal(first, second)


# ------------------------------------------------------------------ Pool-Cache


def test_cache_simulates_once_per_data_seed(process_config) -> None:
    cache = DataPoolCache(process_config, SMALL)
    bundle = PLAN.bundle(VarianceSource.DATA, 0)
    for noise_seed in (1, 2, 3):
        cache.variant(
            data_seed=bundle.data_seed, noise_seed=noise_seed, model_seed=bundle.model_seed
        )
    assert cache.n_simulations == 1
    assert cache.n_remeasurements == 3


def test_cache_variant_equals_direct_generation(process_config) -> None:
    """Der Cache darf keine andere Zahl liefern als der direkte Weg."""
    bundle = SeedBundle.from_master(3)
    cache = DataPoolCache(process_config, SMALL)
    cached = cache.variant(
        data_seed=bundle.data_seed,
        noise_seed=bundle.noise_seed,
        model_seed=bundle.model_seed,
    )
    direct = generate_data(process_config, SMALL, seeds=bundle)
    assert np.array_equal(cached.trajectory.measurements, direct.trajectory.measurements)
    assert cached.fingerprints == direct.fingerprints
    assert cached.seeds == direct.seeds


# ------------------------------------------------------------- Spec-Fingerprint


def test_spec_digest_is_stable_and_distinguishes_observation_mode() -> None:
    """Grundlage der Cache-Invalidierung in ``_run_grid``.

    ``spec_for_mode`` baut bei jedem Aufruf ein neues ``DataSpec``-Objekt -- die
    Invalidierung darf sich deshalb nicht auf Objektidentität verlassen (das würde bei
    *jedem* Punkt einen neuen Cache bauen, nicht nur bei Moduswechseln, ohne dass ein
    Test das je bemerkt -- nur die Laufzeit würde leiden). Wertegleichheit über einen
    Fingerprint ist die Voraussetzung dafür, dass die Invalidierung nur bei echten
    Moduswechseln greift.
    """
    inline_first = spec_for_mode(SMALL, ObservationMode.INLINE)
    inline_second = spec_for_mode(SMALL, ObservationMode.INLINE)
    soft_sensor = spec_for_mode(SMALL, ObservationMode.SOFT_SENSOR)

    assert inline_first is not inline_second, "Testvoraussetzung: neue Objekte je Aufruf"
    assert _spec_digest(inline_first) == _spec_digest(inline_second)
    assert _spec_digest(inline_first) != _spec_digest(soft_sensor)


# ----------------------------------------------------------- Varianzzerlegung


@pytest.fixture(scope="module")
def variance_records(process_config):
    return run_variance_decomposition(
        process_config,
        FACTORIES,
        spec=SMALL,
        plan=PLAN,
        sizes=(4, 8),
        evaluator=EVALUATOR,
    )


def test_variance_records_cover_every_axis(variance_records) -> None:
    sources = {record.params["variance/source"] for record in variance_records}
    assert sources == {source.value for source in VarianceSource}
    assert all("variance/replicate" in record.params for record in variance_records)


def test_model_axis_leaves_parameter_free_baseline_untouched(variance_records) -> None:
    """Persistenz hat keine Parameter.

    Streut sie über den Modell-Seed, ist irgendwo
    Zufall im Spiel, der dort nicht hingehört.
    """
    spreads = collect_spreads(variance_records, target="energy_target", horizon=SMALL.horizon)
    for (key, source), spread in spreads.items():
        if key.label == "persistence" and source is VarianceSource.MODEL:
            assert spread.span == 0.0


def test_data_axis_moves_the_result(variance_records) -> None:
    spreads = collect_spreads(variance_records, target="energy_target", horizon=SMALL.horizon)
    data_spans = [
        spread.span
        for (key, source), spread in spreads.items()
        if source is VarianceSource.DATA and key.label == "persistence"
    ]
    assert data_spans and max(data_spans) > 0.0


def test_decomposition_and_threshold_are_computable(variance_records) -> None:
    decomposition = decompose_variance(
        variance_records, target="energy_target", horizon=SMALL.horizon
    )
    assert decomposition
    thresholds = resolution_thresholds(
        variance_records, target="energy_target", horizon=SMALL.horizon
    )
    assert set(thresholds) == {4, 8}
    assert all(value >= 0.0 for value in thresholds.values())


# ---------------------------------------------------------------- Rauschgrenze


def test_noise_floor_grows_with_the_noise_level(process_config) -> None:
    r"""Die Sättigungshöhe muss mit dem Rauschniveau steigen -- und bei Rauschniveau Null.

    exakt Null sein, weil der perfekte Prädiktor dann die Messung *ist*.
    """
    quiet = generate_data(
        process_config,
        dataclasses.replace(SMALL),
        seeds=SeedBundle.from_master(1),
    )
    floors = {}
    for level in (0.0, 0.005, 0.02):
        variant = quiet.with_measurements(
            measurement=measurement_at_level(quiet.process_config.measurement, level)
        )
        floors[level] = estimate_noise_floor(variant, 8, horizons=(1, SMALL.horizon))

    assert floors[0.0].per_run["energy_target"][SMALL.horizon][0] == pytest.approx(0.0, abs=1e-12)
    assert (
        floors[0.005].per_run["energy_target"][SMALL.horizon][0]
        < floors[0.02].per_run["energy_target"][SMALL.horizon][0]
    )


def test_sensor_lag_floor_is_not_zero_even_without_noise(process_config) -> None:
    """Gegen die *ungefilterte* Wahrheit bleibt ein Sockel: das Sensor-PT1.

    Wer ihn nicht kennt, hält Sensordynamik für Modellfehler.
    """
    data = generate_data(process_config, SMALL, seeds=SeedBundle.from_master(1))
    clean = data.with_measurements(
        measurement=measurement_at_level(data.process_config.measurement, 0.0)
    )
    floor = estimate_noise_floor(clean, 8, horizons=(1, SMALL.horizon))
    assert floor.against_truth["energy_target"][SMALL.horizon][0] > 0.0
    assert floor.format()


def test_noise_study_returns_records_and_floors(process_config) -> None:
    records, floors = run_noise_study(
        process_config,
        {"persistence": PersistenceModel, "arx": LinearARXModel},
        spec=SMALL,
        plan=PLAN,
        sizes=(8,),
        noise_levels=(0.0, 0.02),
        evaluator=EVALUATOR,
    )
    assert set(floors) == {0.0, 0.02}
    levels = {record.params["noise/relative_std"] for record in records}
    assert levels == {0.0, 0.02}
    # Ohne Filter auf ein Niveau wäre die "Seed-Streuung" der Effekt des Rauschniveaus.
    with pytest.raises(ValueError):
        collect_spreads(records, target="energy_target", horizon=SMALL.horizon)
    filtered = select(records, **{"noise/relative_std": 0.02})
    assert collect_spreads(filtered, target="energy_target", horizon=SMALL.horizon)


# ------------------------------------------------------- Validierungsbetriebsart


_TIMING_METRICS = {"inference_us_per_window"}
"""Wall-Uhr-Messungen: bei zwei unabhängigen Aufrufen erwartungsgemäß nie gleich,
unabhängig davon, ob die Implementierung dieselbe ist -- aus dem Äquivalenzvergleich
ausgenommen, nicht weil sie unwichtig wären, sondern weil sie nichts über Äquivalenz
aussagen."""


def _assert_records_equal(old, new) -> None:
    """Inhaltlicher Vergleich zweier ``RunRecord``-Listen.

    Nicht ``==``: ``RunRecord.run_uid``/``created_unix_s`` sind ``field(default_factory=...)``
    und deshalb bei zwei unabhängig erzeugten Records nie gleich, selbst wenn Inhalt
    identisch ist. Verglichen wird deshalb explizit ``group``/``params``/``metrics``/``tags``,
    in derselben Reihenfolge (beide Implementierungen iterieren dieselben Achsen in
    derselben Reihenfolge, ein Sortieren würde eine eigene Fehlerquelle einführen).
    """
    assert len(old) == len(new), f"{len(old)} alte gegen {len(new)} neue Records"
    for index, (left, right) in enumerate(zip(old, new, strict=True)):
        assert left.group == right.group, f"Record {index}: group weicht ab"
        assert dict(left.params) == dict(right.params), f"Record {index}: params weichen ab"
        left_metrics = {k: v for k, v in left.metrics.items() if k not in _TIMING_METRICS}
        right_metrics = {k: v for k, v in right.metrics.items() if k not in _TIMING_METRICS}
        assert left_metrics == right_metrics, f"Record {index}: metrics weichen ab"
        assert dict(left.tags) == dict(right.tags), f"Record {index}: tags weichen ab"


def test_validation_policy_is_a_no_op_for_baselines(process_config) -> None:
    """Regressionsprüfung gegen Leakage: keine Baseline benutzt den Validierungssplit.

    Die Differenz muss deshalb exakt Null sein. Wäre sie es nicht, sickerten
    Validierungsläufe ins Training.
    """
    records = run_validation_policy_comparison(
        process_config,
        FACTORIES,
        spec=SMALL,
        plan=SeedPlan(master_seeds=(1, 2)),
        sizes=(8,),
        evaluator=EVALUATOR,
    )
    gaps = summarise_policy_gap(records, target="energy_target", horizon=SMALL.horizon)
    assert gaps
    for _, _, difference in gaps.values():
        assert difference == pytest.approx(0.0, abs=1e-12)


def test_validation_policy_none_reports_zero_budget(process_config) -> None:
    records = run_validation_policy_comparison(
        process_config,
        {"arx": LinearARXModel},
        spec=SMALL,
        plan=SeedPlan(master_seeds=(1, 2)),
        sizes=(8,),
        policies=(ValidationPolicy.NONE,),
        evaluator=EVALUATOR,
    )
    assert all(record.params["n_val"] == 0 for record in records)
    assert all(record.params["n_budget"] == 8 for record in records)


# ------------------------------------------------------------ Architekturvergleich


def test_architecture_comparison_covers_modes_and_sources(process_config) -> None:
    """Verhaltensregression, unabhängig vom Migrationskandidaten oben.

    ``run_architecture_comparison`` hatte vor diesem Umbau keinen einzigen Test in
    diesem Modul (nur ``scripts/study_architectures.py`` rief es auf) -- diese Lücke
    schließt dieser Test dauerhaft, nicht nur für die Dauer der Migration.
    """
    on_point_calls: list[tuple[str, int, int]] = []
    records = run_architecture_comparison(
        process_config,
        FACTORIES_BOTH_MODES,
        spec=SMALL,
        plan=SeedPlan(master_seeds=(1, 2)),
        sizes=(4, 8),
        modes=(ObservationMode.INLINE, ObservationMode.SOFT_SENSOR),
        sources=(VarianceSource.JOINT, VarianceSource.MODEL),
        evaluator=EVALUATOR,
        on_point=lambda *, source, replicate, n_train: on_point_calls.append(
            (source, replicate, n_train)
        ),
    )
    # on_point behält seine alte Signatur (source=, replicate=, n_train=) über einen
    # internen Adapter -- study_architectures.py ruft die Funktion genauso auf.
    assert on_point_calls
    assert {call[0] for call in on_point_calls} == {"joint", "model"}

    modes = {record.params["observation/mode"] for record in records}
    assert modes == {"inline", "soft_sensor"}
    sources = {record.params["variance/source"] for record in records}
    assert sources == {"joint", "model"}

    soft_sensor_records = select(records, **{"observation/mode": "soft_sensor"})
    assert soft_sensor_records
    assert all(
        set(record.params["observation/unobserved"]) == {"product_moisture_kg_per_kg"}
        for record in soft_sensor_records
    )
    inline_records = select(records, **{"observation/mode": "inline"})
    assert all(record.params["observation/unobserved"] == [] for record in inline_records)


# -------------------------------------------------------------------- Abbildungen


def test_plots_render_without_display(process_config, variance_records, tmp_path) -> None:
    """Smoke-Test der Abbildungen.

    Sie sind Teil des Ergebnisformats von M2 -- eine
    Änderung an den Metriknamen darf nicht erst im Notebook auffallen.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from twinlink_ml.evaluation.plots import (
        plot_learning_curve,
        plot_noise_study,
        plot_variance_decomposition,
    )

    thresholds = resolution_thresholds(
        variance_records, target="energy_target", horizon=SMALL.horizon
    )
    axis = plot_learning_curve(
        variance_records,
        target="energy_target",
        horizon=SMALL.horizon,
        resolution=thresholds,
        noise_floor=0.02,
    )
    axis.figure.savefig(tmp_path / "curve.png")
    plot_variance_decomposition(variance_records, target="energy_target", horizon=SMALL.horizon)

    records, floors = run_noise_study(
        process_config,
        {"arx": LinearARXModel},
        spec=SMALL,
        plan=SeedPlan(master_seeds=(1, 2)),
        sizes=(8,),
        noise_levels=(0.0, 0.02),
        evaluator=EVALUATOR,
    )
    plot_noise_study(
        records,
        floors,
        target="energy_target",
        horizon=SMALL.horizon,
        n_train=8,
    )
    plt.close("all")


def test_jsonl_roundtrip_keeps_the_study_parameters(variance_records, tmp_path) -> None:
    """Das Notebook liest aus JSONL.

    Gehen dabei ``variance/source`` oder ``n_train``
    verloren, ist die Auswertung im Notebook eine andere als im Skript.
    """
    from twinlink_ml.utils import JsonlExperimentLogger, read_jsonl

    path = tmp_path / "records.jsonl"
    logger = JsonlExperimentLogger(path)
    for record in variance_records:
        logger.log(record)
    restored = read_jsonl(path)
    assert len(restored) == len(variance_records)
    assert (
        decompose_variance(restored, target="energy_target", horizon=SMALL.horizon).keys()
        == decompose_variance(
            variance_records, target="energy_target", horizon=SMALL.horizon
        ).keys()
    )
