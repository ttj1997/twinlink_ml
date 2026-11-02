r"""Step A diagnosis (session on Befund 37): cost source of the Transformer core.

Where does the ~14x training cost factor of the Transformer core relative to GRU come from
(Befund 36)? Two cheap measurements, NO full training (see the task):

A1) Fwd+Bwd breakdown on a freshly instantiated (untrained) network:
    encoder pass (once per batch) separated from the time of the whole
    36-step decoder loop, plus single-step times (step 1 vs. step 30) --
    evidence for/against O(H^2) due to the missing KV cache reuse in the decoder loop
    (``torch_sequence._build_transformer_network``, ``decoder_embeds`` grows cumulatively,
    ``self.decoder(decoder_sequence, memory)`` runs completely anew per step).

A2) Inference comparison with ``RolloutEvaluator`` (the same function as in Befund 36),
    1-2 training epochs (not 200, the model only has to be runnable), directly against
    the already logged GRU value from ``artifacts/runs/arch_gab_constant_holdout.jsonl``
    (``model/label=gru``, ``test_set=excited``).

Small data volume (``n_train_max=4, n_val=4, n_test=8``) -- enough for correct
tensor shapes and a rough uS/window estimate, is NOT the production size.
No change to ``torch_sequence.py``/``composite.py`` -- pure read access to
already existing network attributes (``network.input_proj``/``.encoder``/``.decoder``/``.head``).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.evaluation.evaluator import RolloutEvaluator
from twinlink_ml.models.composite import CompositeTransformerSequenceModel, _SingleTargetView
from twinlink_ml.models.torch_sequence import (
    TorchSequenceConfig,
    _build_network,
    _sinusoidal_position_encoding,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG_PATH = Path("conf") / "process" / "greybox_gab_placeholder.yaml"
SEED = 1

torch.set_num_threads(1)


def _small_spec() -> DataSpec:
    """Like ``_excited_spec`` in the constant-run holdout worker, but tiny volume.

    Only for correct tensor shapes and a rough inference estimate, no real training.
    """
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=4,
        n_val=4,
        n_test=8,
        history_length=32,
        horizon=36,
        stride=8,
    )


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _small_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)
    train, validation, test = build_datasets(
        data, spec.n_train_max, policy=ValidationPolicy.FIXED_POOL
    )

    print(f"n_exogenous={len(train.exogenous_names)}, targets={train.target_names}")
    print(f"history_length={spec.history_length}, horizon={spec.horizon}")

    # ---------------------------------------------------------------- A1: Fwd+Bwd breakdown
    print("\n=== A1: Encoder- vs. Decoder-Loop-Zeit, frisch initialisiertes Netz ===")
    target_index = 0  # energy_target
    single = _SingleTargetView(train, target_index)
    net_config = TorchSequenceConfig(core="transformer", hidden_size=16, num_heads=2, num_layers=1)
    n_exogenous = len(single.exogenous_names)
    network = _build_network(torch, n_exogenous=n_exogenous, n_targets=1, config=net_config)

    batch = single.batch(list(range(min(256, single.n_windows))))
    history = torch.tensor(np.ascontiguousarray(batch.history_inputs), dtype=torch.float32)
    future = torch.tensor(np.ascontiguousarray(batch.future_exogenous), dtype=torch.float32)
    print(f"Batch: {history.shape[0]} Fenster")

    d_model = net_config.hidden_size

    def encoder_forward() -> torch.Tensor:
        history_positions = torch.arange(history.shape[1])
        embed = network.input_proj(history) + _sinusoidal_position_encoding(
            torch, history_positions, d_model
        )
        return network.encoder(embed)

    def decoder_step(
        memory: torch.Tensor, last: torch.Tensor, decoder_embeds: list, step: int
    ) -> torch.Tensor:
        step_input = torch.cat([future[:, step, :], last], dim=-1).unsqueeze(1)
        step_position = torch.tensor([history.shape[1] + step])
        step_embed = network.input_proj(step_input) + _sinusoidal_position_encoding(
            torch, step_position, d_model
        ).unsqueeze(0)
        decoder_embeds.append(step_embed)
        decoder_sequence = torch.cat(decoder_embeds, dim=1)
        decoded = network.decoder(decoder_sequence, memory)
        return last + network.head(decoded[:, -1, :])

    # Warmup (threading/allocator, not part of the measurement)
    with torch.no_grad():
        _ = encoder_forward()

    with torch.no_grad():
        start = time.perf_counter()
        memory = encoder_forward()
        encoder_time = time.perf_counter() - start

        last = history[:, -1, n_exogenous:]
        decoder_embeds: list = []
        step_times = []
        for step in range(future.shape[1]):
            start = time.perf_counter()
            last = decoder_step(memory, last, decoder_embeds, step)
            step_times.append(time.perf_counter() - start)

    decoder_total = sum(step_times)
    print(f"Encoder (1x pro Batch): {encoder_time * 1e3:.3f} ms")
    print(f"Decoder-Loop gesamt ({future.shape[1]} Schritte): {decoder_total * 1e3:.3f} ms")
    print(f"  Schritt 1:  {step_times[0] * 1e3:.4f} ms")
    print(f"  Schritt 18: {step_times[17] * 1e3:.4f} ms")
    print(f"  Schritt 30: {step_times[29] * 1e3:.4f} ms")
    print(f"  Schritt 36: {step_times[-1] * 1e3:.4f} ms")
    print(f"  Verhaeltnis Schritt36/Schritt1: {step_times[-1] / step_times[0]:.2f}x")

    # Full Fwd+Bwd on the same batch (warmup, then measurement)
    def full_forward_backward() -> float:
        network.zero_grad(set_to_none=True)
        start = time.perf_counter()
        output = network(history, future, teacher=None, teacher_forcing=0.0)
        loss = (output**2).mean()
        loss.backward()
        return time.perf_counter() - start

    _ = full_forward_backward()  # Warmup
    fwd_bwd_time = full_forward_backward()
    print(f"\nVolles Fwd+Bwd (1 Batch, {history.shape[0]} Fenster): {fwd_bwd_time * 1e3:.3f} ms")
    print(
        f"  davon reines Fwd (Encoder+Decoder-Loop, kein Backward): "
        f"{(encoder_time + decoder_total) * 1e3:.3f} ms"
    )

    # ---------------------------------------------------------------- A2: inference comparison
    print("\n=== A2: Inferenzvergleich gegen geloggten GRU-Wert (Befund 36) ===")
    model = CompositeTransformerSequenceModel(
        TorchSequenceConfig(
            core="transformer", hidden_size=16, num_heads=2, num_layers=1, max_epochs=2, patience=0
        )
    )
    fit_start = time.perf_counter()
    model.fit(train, seed=SEED, validation=None, verbose=False)
    fit_time = time.perf_counter() - fit_start
    print(f"(2-Epochen-Fit fuer ein lauffaehiges Modell: {fit_time:.1f}s)")

    evaluator = RolloutEvaluator(horizons=(1, spec.horizon))
    result = evaluator.evaluate(model, test)
    print(f"Transformer inference_us_per_window: {result.inference_us_per_window:.2f} us")
    print("Geloggter GRU-Wert (arch_gab_constant_holdout.jsonl, test_set=excited, 10 Seeds):")
    print("  Bereich 111.1 - 179.7 us/Fenster (siehe vorherige Abfrage)")
    print(
        f"Verhaeltnis (Transformer / GRU-Median~135us): "
        f"{result.inference_us_per_window / 135.0:.2f}x"
    )


if __name__ == "__main__":
    main()
