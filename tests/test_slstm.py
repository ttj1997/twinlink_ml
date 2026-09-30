"""Tests for the sLSTM cores (``twinlink_ml.models.slstm``), Befund 67 / SX2.

Requirements from the task: (1) forward pass with and without stabilisation numerically equal on
small inputs, (2) no NaN on large inputs, (3) parameter count as computed. Beyond that: memory
mixing exists and stays inside a head, the state pass-through of the autoregressive rollout is
exact, the mixed stack contains the intended layers, and the cores plug into the composite model.
The numerical equivalence with the official CPU implementation is a verification script
(``scripts/verify/sx2_official_equivalence.py``), not a test: needs the ``xlstm`` package.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import DataSpec, build_datasets, generate_data
from twinlink_ml.models import PersistenceModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"
INPUT_SIZE = 8


def _torch():
    import torch

    return torch


def _slstm_layer_parameters(input_size: int, hidden: int, heads: int) -> int:
    """W (with bias) + block-diagonal R + GroupNorm affine."""
    head = hidden // heads
    return input_size * 4 * hidden + 4 * hidden + heads * 4 * head * head + 2 * hidden


def _mlstm_layer_parameters(input_size: int, hidden: int) -> int:
    """q, k, v, o projections + two scalar gates from [q;k;v] + LayerNorm."""
    return 4 * (input_size * hidden + hidden) + 2 * (3 * hidden + 1) + 2 * hidden


def test_stabilised_and_plain_forward_agree_on_small_inputs() -> None:
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class

    torch.manual_seed(0)
    stable = build_slstm_core_class(torch, 4, stabilized=True)(
        INPUT_SIZE, 16, num_layers=2
    ).double()
    plain = build_slstm_core_class(torch, 4, stabilized=False)(
        INPUT_SIZE, 16, num_layers=2
    ).double()
    plain.load_state_dict(stable.state_dict())
    sequence = 0.5 * torch.randn(3, 20, INPUT_SIZE, dtype=torch.float64)
    out_stable, _ = stable(sequence)
    out_plain, _ = plain(sequence)
    assert torch.allclose(out_stable, out_plain, rtol=1e-9, atol=1e-10)


def test_stabilised_forward_is_finite_on_large_inputs_and_plain_form_is_not() -> None:
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class

    torch.manual_seed(1)
    stable = build_slstm_core_class(torch, 4, stabilized=True)(INPUT_SIZE, 16, num_layers=2)
    plain = build_slstm_core_class(torch, 4, stabilized=False)(INPUT_SIZE, 16, num_layers=2)
    plain.load_state_dict(stable.state_dict())
    sequence = 1.0e4 * torch.randn(4, 60, INPUT_SIZE)
    out_stable, _ = stable(sequence)
    out_plain, _ = plain(sequence)
    assert torch.isfinite(out_stable).all()
    assert not torch.isfinite(out_plain).all(), "the plain recurrence should overflow here"
    sequence.requires_grad_(False)
    loss = out_stable.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in stable.parameters() if p.grad is not None)


def test_normaliser_is_at_least_one_and_output_bounded() -> None:
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class

    torch.manual_seed(2)
    core = build_slstm_core_class(torch, 2)(INPUT_SIZE, 8)
    layer = core.layers[0]
    state = layer.init_state(5, torch.zeros(5, INPUT_SIZE))
    for _ in range(40):
        y, state = layer.cell(50.0 * torch.randn(5, INPUT_SIZE), state)
        assert (state[1] >= 1.0 - 1e-6).all()
        assert y.abs().max() <= 1.0 + 1e-6


@pytest.mark.parametrize(("hidden", "heads"), [(112, 4), (16, 2), (64, 1)])
def test_slstm_parameter_count_matches_formula(hidden: int, heads: int) -> None:
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class

    core = build_slstm_core_class(torch, heads)(INPUT_SIZE, hidden)
    counted = sum(p.numel() for p in core.parameters())
    assert counted == _slstm_layer_parameters(INPUT_SIZE, hidden, heads)


def test_mixed_stack_is_mlstm_then_slstm_with_computed_parameters() -> None:
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class, build_xlstm_mixed_core_class
    from twinlink_ml.models.xlstm import build_xlstm_core_class

    hidden, heads = 52, 4
    core = build_xlstm_mixed_core_class(torch, heads)(INPUT_SIZE, hidden, num_layers=2)
    # Every build call creates fresh classes, so compare by name.
    assert type(core.layers[0]).__name__ == build_xlstm_core_class(torch).layer_class.__name__
    assert (
        type(core.layers[1]).__name__ == build_slstm_core_class(torch, heads).layer_class.__name__
    )
    expected = _mlstm_layer_parameters(INPUT_SIZE, hidden) + _slstm_layer_parameters(
        hidden, hidden, heads
    )
    assert sum(p.numel() for p in core.parameters()) == expected
    with pytest.raises(ValueError, match="num_layers"):
        build_xlstm_mixed_core_class(torch, heads)(INPUT_SIZE, hidden, num_layers=1)


def test_recurrent_memory_mixing_exists_and_is_block_diagonal_over_heads() -> None:
    """The output at t+1 depends on y_t only through the head it belongs to."""
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class

    torch.manual_seed(3)
    heads, hidden = 4, 16
    layer = build_slstm_core_class(torch, heads)(INPUT_SIZE, hidden).layers[0]
    y_prev = torch.randn(2, hidden)
    baseline = layer.recurrent_contribution(y_prev).reshape(2, 4, heads, hidden // heads)
    perturbed_input = y_prev.clone()
    perturbed_input[:, 0 : hidden // heads] += 1.0  # head 0 only
    perturbed = layer.recurrent_contribution(perturbed_input).reshape(2, 4, heads, hidden // heads)
    difference = (perturbed - baseline).abs()
    assert difference[:, :, 0, :].max() > 1e-4, "no memory mixing inside head 0"
    assert difference[:, :, 1:, :].max() == 0.0, "heads must not mix"


def test_state_passthrough_reproduces_full_sequence() -> None:
    """History encoded at once, then step by step, equals one long pass (rollout contract)."""
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class, build_xlstm_mixed_core_class

    torch.manual_seed(4)
    for factory in (
        build_slstm_core_class(torch, 4)(INPUT_SIZE, 16, num_layers=2),
        build_xlstm_mixed_core_class(torch, 4)(INPUT_SIZE, 16, num_layers=2),
    ):
        sequence = torch.randn(3, 12, INPUT_SIZE)
        full, _ = factory(sequence)
        head, state = factory(sequence[:, :7])
        steps = [head]
        for index in range(7, 12):
            out, state = factory(sequence[:, index : index + 1], state)
            steps.append(out)
        assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-6)


def test_forget_bias_and_recurrent_init() -> None:
    torch = _torch()
    from twinlink_ml.models.slstm import build_slstm_core_class

    hidden = 16
    layer = build_slstm_core_class(torch, 4)(INPUT_SIZE, hidden).layers[0]
    bias = layer.w_x.bias.detach()
    assert torch.all(bias[hidden : 2 * hidden] == 3.0)
    assert torch.all(bias[:hidden] == 0.0) or bias[:hidden].abs().max() < 1.0  # default Linear bias
    bound = 1.0 / hidden**0.5
    assert layer.recurrent.abs().max() <= bound + 1e-7
    assert layer.recurrent.abs().max() > 0.0


@pytest.fixture(scope="module")
def data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    spec = DataSpec(
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
    return generate_data(config, spec, master_seed=1)


@pytest.mark.parametrize(("core", "hidden", "layers"), [("slstm", 8, 1), ("xlstm_mixed", 8, 2)])
def test_new_cores_are_persistence_untrained_and_train_in_composite(
    data, core: str, hidden: int, layers: int
) -> None:
    from twinlink_ml.models.composite import CompositeSequenceModel
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    train, _, test = build_datasets(data, 8)
    untrained = CompositeSequenceModel(
        TorchSequenceConfig(
            core=core, hidden_size=hidden, num_layers=layers, num_heads=2, max_epochs=0, patience=0
        )
    )
    untrained.fit(train, seed=1, validation=None)
    reference = PersistenceModel()
    reference.fit(train, seed=1)
    batch = test.full_batch()
    assert np.allclose(untrained.predict(batch), reference.predict(batch), atol=1e-5)

    trained = CompositeSequenceModel(
        TorchSequenceConfig(
            core=core, hidden_size=hidden, num_layers=layers, num_heads=2, max_epochs=3, patience=0
        )
    )
    trained.fit(train, seed=1, validation=None)
    prediction = trained.predict(batch)
    assert np.all(np.isfinite(prediction))
    assert not np.allclose(prediction, reference.predict(batch), atol=1e-6)


def test_config_validation_of_new_cores() -> None:
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig

    with pytest.raises(ValueError, match="num_heads"):
        TorchSequenceConfig(core="slstm", hidden_size=10, num_heads=4)
    with pytest.raises(ValueError, match="num_layers"):
        TorchSequenceConfig(core="xlstm_mixed", hidden_size=8, num_heads=2, num_layers=1)
    TorchSequenceConfig(core="xlstm_mixed", hidden_size=8, num_heads=2, num_layers=2)


def test_capacity_matched_configurations_have_the_registered_parameter_counts() -> None:
    """Counts are those of the built three-head composite; the band is 43 000 - 57 000."""
    from twinlink_ml.models.capacity_matched import (
        CAPACITY_MATCHED_CONFIGS,
        CAPACITY_MATCHED_PARAMETERS,
    )
    from twinlink_ml.models.composite import CompositeSequenceModel

    assert set(CAPACITY_MATCHED_CONFIGS) == set(CAPACITY_MATCHED_PARAMETERS)
    torch = _torch()
    from twinlink_ml.models.torch_sequence import _build_network

    for label, config in CAPACITY_MATCHED_CONFIGS.items():
        network = _build_network(torch, n_exogenous=7, n_targets=1, config=config)
        total = 3 * sum(p.numel() for p in network.parameters())
        assert total == CAPACITY_MATCHED_PARAMETERS[label], label
        if label == "lstm_small":
            assert 8000 <= total <= 9000
        else:
            assert 43000 <= total <= 57000, label
        assert CompositeSequenceModel(config).name.startswith("composite_")
