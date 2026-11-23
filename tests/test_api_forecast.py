"""Forecast service: end to end on synthetic runs, equal to the research code bit for bit."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from _api_common import (
    N_TRAIN,
    SAMPLE_TIME_S,
    TARGETS,
    datasets,
    generated,
    response_matrix,
    window_request,
)
from twinlink_ml.api import (
    API_VERSION,
    SCHEMA_VERSION,
    ForecastResponse,
    ForecastService,
    MissingExtraError,
    ModelBundle,
    ModelFailureError,
    RequestRejectedError,
    build_bundle,
    fit_bundle,
)
from twinlink_ml.api.base import LAB_LINE_SCOPE as LAB
from twinlink_ml.api.bundle import instantiate_model
from twinlink_ml.data.windows import WindowBatch
from twinlink_ml.models import LinearARXModel
from twinlink_ml.models.base import SequenceModel
from twinlink_ml.simulation.backend import FloatArray

HAS_TORCH = importlib.util.find_spec("torch") is not None
SOFT_OBSERVED = ("energy_target", "bed_temperature_celsius")
WINDOWS = (0, 7, 23)


@pytest.fixture(scope="module")
def arx_bundle() -> ModelBundle:
    return fit_bundle(generated(), n_train=N_TRAIN, model_seed=3, kind="linear_arx")


@pytest.fixture(scope="module")
def service(arx_bundle: ModelBundle) -> ForecastService:
    return ForecastService(arx_bundle)


@pytest.fixture(scope="module")
def research_arx() -> LinearARXModel:
    train, _, _ = datasets()
    model = LinearARXModel()
    model.fit(train, seed=0)
    return model


def _direct(model: SequenceModel, batch: WindowBatch) -> FloatArray:
    """The research code called directly: predict, then back to engineering units."""
    _, _, test = datasets()
    return test.target_scaler.inverse(model.predict(batch))[0]


@pytest.mark.parametrize("index", WINDOWS)
def test_end_to_end_equals_the_research_code(
    service: ForecastService, research_arx: LinearARXModel, index: int
) -> None:
    """Engineering units in, engineering units out, bit for bit, also through JSON."""
    data, (_, _, test) = generated(), datasets()
    request, batch = window_request(data, test, index)
    expected = _direct(research_arx, batch)

    response = service.handle(request)
    assert np.array_equal(response_matrix(response), expected)

    through_json = ForecastResponse.model_validate_json(
        service.handle_json(request.model_dump_json())
    )
    assert through_json == response
    assert np.array_equal(response_matrix(through_json), expected)


def test_response_says_what_it_is(service: ForecastService, arx_bundle: ModelBundle) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    response = service.handle(request)
    meta = response.meta
    assert (meta.service, meta.api_version, meta.schema_version) == (
        "forecast",
        API_VERSION,
        SCHEMA_VERSION,
    )
    assert meta.model_fingerprint == arx_bundle.fingerprint
    assert meta.model_label == "arx_p4_q2"
    assert meta.model_scope == service.describe().model_scope == LAB
    assert meta.parameter_status == "placeholder"
    assert meta.advisory_only is True
    assert meta.notices == ()
    assert response.fallback_used is False
    horizon = test.config.horizon
    assert response.time_offset_s == tuple(SAMPLE_TIME_S * (k + 1) for k in range(horizon))


def test_description_matches_the_bundle(service: ForecastService, arx_bundle: ModelBundle) -> None:
    description = service.describe()
    assert description.model_fingerprint == arx_bundle.fingerprint
    assert description.stateless is True and description.advisory_only is True
    assert description.required_extras == ("api",)
    assert description.sample_time_s == SAMPLE_TIME_S
    assert (description.history_length, description.maximum_horizon) == (16, 8)
    assert description.observation_mode == "inline"
    assert description.predicted_fields == (
        "energy_target_kw",
        "bed_temperature_celsius",
        "product_moisture_kg_per_kg",
    )
    assert description.required_measured_fields == description.predicted_fields
    assert "energy_target_kw" in description.valid_input_ranges
    assert "energy_target" not in description.valid_input_ranges


def test_service_is_stateless(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    first, _ = window_request(data, test, 0)
    other, _ = window_request(data, test, 7)
    before = service.handle(first)
    service.handle(other)
    assert service.handle(first) == before


def test_longer_history_uses_the_last_window(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    index = 23
    exact, _ = window_request(data, test, index)
    longer, _ = window_request(data, test, index, extra_history=5)
    assert longer.history.n_samples == exact.history.n_samples + 5
    assert response_matrix(service.handle(longer)).tolist() == (
        response_matrix(service.handle(exact)).tolist()
    )


def test_shorter_horizon_is_a_prefix(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 7)
    planned = {name: values[:3] for name, values in request.planned.model_dump().items()}
    short = request.model_copy(update={"planned": type(request.planned)(**planned)})
    full, part = service.handle(request), service.handle(short)
    assert np.array_equal(response_matrix(part), response_matrix(full)[:3])
    assert len(part.time_offset_s) == 3


def test_wrong_grid_is_rejected_not_resampled(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    minute = request.model_copy(update={"sample_time_s": 60.0})
    with pytest.raises(RequestRejectedError, match="resample before the call") as caught:
        service.handle(minute)
    assert caught.value.code == "request_rejected"


def test_short_history_is_rejected(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    cut = {name: values[1:] for name, values in request.history.model_dump().items()}
    measured = {
        name: None if values is None else values[1:]
        for name, values in request.measured_history.model_dump().items()
    }
    short = request.model_copy(
        update={
            "history": type(request.history)(**cut),
            "measured_history": type(request.measured_history)(**measured),
        }
    )
    with pytest.raises(RequestRejectedError, match="history of 16 samples, got 15"):
        service.handle(short)


def test_horizon_beyond_training_is_rejected(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    longer = {name: (*values, values[-1]) for name, values in request.planned.model_dump().items()}
    with pytest.raises(RequestRejectedError, match="horizon of 8 samples"):
        service.handle(request.model_copy(update={"planned": type(request.planned)(**longer)}))


def test_missing_or_misaligned_measurement_is_rejected(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0, measured=SOFT_OBSERVED)
    with pytest.raises(RequestRejectedError, match="product_moisture_kg_per_kg"):
        service.handle(request)
    complete, _ = window_request(data, test, 0)
    series = complete.measured_history.model_dump()
    series["energy_target_kw"] = series["energy_target_kw"][1:]
    crooked = complete.model_copy(
        update={"measured_history": type(complete.measured_history)(**series)}
    )
    with pytest.raises(RequestRejectedError, match="15 samples, the history has 16"):
        service.handle(crooked)


def test_wrong_observation_mode_is_rejected(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    with pytest.raises(RequestRejectedError, match="'inline' model"):
        service.handle(request.model_copy(update={"observation_mode": "soft_sensor"}))


def test_input_outside_the_training_range_is_remarked(service: ForecastService) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    planned = request.planned.model_dump()
    planned["inlet_temperature_celsius"] = tuple(
        400.0 for _ in planned["inlet_temperature_celsius"]
    )
    hot = request.model_copy(update={"planned": type(request.planned)(**planned)})
    notices = service.handle(hot).meta.notices
    assert [notice.code for notice in notices] == ["input_outside_training_range"]
    assert "planned.inlet_temperature_celsius" in notices[0].message
    assert "history." not in notices[0].message


# ---------------------------------------------------------------- soft sensor


@pytest.fixture(scope="module")
def soft_bundle() -> ModelBundle:
    return fit_bundle(generated(SOFT_OBSERVED), n_train=N_TRAIN, model_seed=3, kind="linear_arx")


def test_soft_sensor_equals_the_research_code(soft_bundle: ModelBundle) -> None:
    """Without a moisture measurement its history is the training mean, as in the data set."""
    data, (train, _, test) = generated(SOFT_OBSERVED), datasets(SOFT_OBSERVED)
    assert test.unobserved_target_names == ("product_moisture_kg_per_kg",)
    research = LinearARXModel()
    research.fit(train, seed=0)
    service = ForecastService(soft_bundle)
    description = service.describe()
    assert description.observation_mode == "soft_sensor"
    assert description.required_measured_fields == ("energy_target_kw", "bed_temperature_celsius")
    assert "product_moisture_kg_per_kg" not in description.valid_input_ranges

    for index in WINDOWS:
        request, batch = window_request(data, test, index)
        assert request.observation_mode == "soft_sensor"
        assert request.measured_history.product_moisture_kg_per_kg is None
        expected = test.target_scaler.inverse(research.predict(batch))[0]
        response = service.handle(request)
        assert np.array_equal(response_matrix(response), expected)
        assert response.meta.notices == ()


def test_soft_sensor_ignores_a_supplied_moisture_history(soft_bundle: ModelBundle) -> None:
    data, (_, _, test) = generated(SOFT_OBSERVED), datasets(SOFT_OBSERVED)
    service = ForecastService(soft_bundle)
    plain, _ = window_request(data, test, 7)
    with_moisture, _ = window_request(data, test, 7, measured=TARGETS)
    assert with_moisture.measured_history.product_moisture_kg_per_kg is not None
    response = service.handle(with_moisture)
    assert [notice.code for notice in response.meta.notices] == ["measurement_ignored"]
    assert "product_moisture_kg_per_kg" in response.meta.notices[0].message
    assert np.array_equal(response_matrix(response), response_matrix(service.handle(plain)))
    with pytest.raises(RequestRejectedError, match="'soft_sensor' model"):
        service.handle(plain.model_copy(update={"observation_mode": "inline"}))


# -------------------------------------------------------------------- fallback


def _loader_without_torch(bundle: ModelBundle) -> SequenceModel:
    """A host without PyTorch: networks cannot be loaded, the ARX can."""
    if bundle.manifest.model_kind == "composite_torch":
        raise MissingExtraError("torch", "torch")
    return instantiate_model(bundle)


def _as_network(bundle: ModelBundle) -> ModelBundle:
    """The same bundle labelled as a network (enough for a loader that refuses networks)."""
    manifest = bundle.manifest.model_copy(
        update={"model_kind": "composite_torch", "model_label": "composite_gru_h64_l1"}
    )
    return ModelBundle(manifest=manifest, weights=bundle.weights)


def test_fallback_answers_when_the_extra_is_missing(arx_bundle: ModelBundle) -> None:
    data, (_, _, test) = generated(), datasets()
    primary = _as_network(arx_bundle)
    with pytest.raises(MissingExtraError, match=r"twinlink-ml\[torch\]"):
        ForecastService(primary, model_loader=_loader_without_torch)

    service = ForecastService(primary, arx_bundle, model_loader=_loader_without_torch)
    request, _ = window_request(data, test, 0)
    response = service.handle(request)
    assert response.fallback_used is True
    assert response.meta.model_fingerprint == arx_bundle.fingerprint
    assert response.meta.model_label == "arx_p4_q2"
    assert [notice.code for notice in response.meta.notices] == ["fallback_model_used"]
    assert np.array_equal(
        response_matrix(response), response_matrix(ForecastService(arx_bundle).handle(request))
    )
    description = service.describe()
    assert description.required_extras == ("api", "torch")
    assert description.fallback_model_fingerprint == arx_bundle.fingerprint


class _NonFinite(SequenceModel):
    """A model that diverged."""

    @property
    def name(self) -> str:
        return "diverged"

    def fit(self, train, *, seed, validation=None) -> None:
        raise NotImplementedError

    def predict(self, batch: WindowBatch) -> FloatArray:
        return np.full((batch.n_windows, batch.future_exogenous.shape[1], len(TARGETS)), np.nan)


def test_non_finite_prediction_goes_to_the_fallback_or_fails(arx_bundle: ModelBundle) -> None:
    data, (_, _, test) = generated(), datasets()
    request, _ = window_request(data, test, 0)
    primary = _as_network(arx_bundle)

    def loader(bundle: ModelBundle) -> SequenceModel:
        return _NonFinite() if bundle is primary else instantiate_model(bundle)

    with pytest.raises(ModelFailureError, match="non-finite") as caught:
        ForecastService(primary, model_loader=loader).handle(request)
    assert caught.value.code == "model_failure"
    response = ForecastService(primary, arx_bundle, model_loader=loader).handle(request)
    assert response.fallback_used is True
    assert np.all(np.isfinite(response_matrix(response)))


def test_fallback_must_share_the_interface(
    arx_bundle: ModelBundle, soft_bundle: ModelBundle
) -> None:
    with pytest.raises(ValueError, match="fallback must share"):
        ForecastService(arx_bundle, soft_bundle)


def test_bundle_outside_the_schema_is_refused(arx_bundle: ModelBundle) -> None:
    manifest = arx_bundle.manifest.model_copy(
        update={"measured_input_names": ("exhaust_humidity_kg_per_kg",)}
    )
    with pytest.raises(ValueError, match="later schema version"):
        ForecastService(ModelBundle(manifest=manifest, weights=arx_bundle.weights))


# --------------------------------------------------------------------- network


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_network_end_to_end_equals_the_trained_model() -> None:
    """Composite GRU: service on the bundle against ``predict`` of the trained object."""
    from twinlink_ml.models.composite import CompositeGruSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    data, (train, validation, test) = generated(), datasets()
    trained = CompositeGruSequenceModel(
        TorchSequenceConfig(core="gru", hidden_size=8, max_epochs=2, patience=0, device="cpu")
    )
    trained.fit(train, seed=5, validation=validation)
    bundle = build_bundle(
        trained, train, sample_time_s=SAMPLE_TIME_S, parameter_status="placeholder", model_scope=LAB
    )
    fallback = fit_bundle(data, n_train=N_TRAIN, model_seed=3, kind="linear_arx")
    service = ForecastService(bundle, fallback)
    assert service.describe().required_extras == ("api", "torch")
    for index in WINDOWS:
        request, batch = window_request(data, test, index)
        response = ForecastResponse.model_validate_json(
            service.handle_json(request.model_dump_json())
        )
        assert np.array_equal(response_matrix(response), _direct(trained, batch))
        assert response.fallback_used is False
        assert response.meta.model_label == "composite_gru_h8_l1"
