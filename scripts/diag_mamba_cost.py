r"""Pruefung 1 (Skalierungstest) des Mamba-Arbeitsstrangs.

Bleibt die Pro-Schritt-Zeit O(1) ueber die Sequenz, wie es der ganze Grund fuer diese
Architektur ist -- und liegt die absolute Pro-Schritt-Zeit in der Groessenordnung von
GRU/LSTM, oder frisst
Python-Loop-/Tensor-Dispatch-Overhead einer unfusionierten, selbst geschriebenen
Rekursion den strukturellen Vorteil auf? Zwei unabhaengige Fragen, beide real gemessen,
kein neues Volltraining (analog zu ``diag_transformer_cost.py``, Befund 37).

(a) Wachstum ueber die Sequenz: 68 sequenzielle SSM-Schritte (32 Historie + 36
    Horizont, Produktionsgroesse), Zeit je Einzelschritt bei Position 1, 18, 30, 36
    (dieselben Positionen wie im Transformer-Schritt-A-Test) und zusaetzlich bei
    Position 68 (Sequenzende) protokolliert. Erwartung: flach, nicht wachsend wie beim
    Transformer (dort Faktor 4.89x ueber 36 Schritte).

(b) Absolutwert, zwei unabhaengige Messmethoden:
    (b1) Direkter Pro-Schritt-Vergleich: GRU und LSTM (gleiches hidden_size) in
         DERSELBEN sequenziellen Python-Schleife wie Mamba (kein fused-CuDNN-Vorteil
         durch Verarbeitung der vollen Sequenz in einem Aufruf) -- fair, weil beide
         Seiten denselben Python-Dispatch-Overhead pro Aufruf tragen.
    (b2) Naeherung, wie beauftragt: volle ``Network.forward``-Zeit (alle 68 Schritte
         inkl. Concat/Head, ueber den echten Trainer-Pfad) geteilt durch 68 -- als
         Gegenprobe zu (b1), ausdruecklich als Naeherung gekennzeichnet.

Kleiner Datenumfang (``n_train_max=4, n_val=4, n_test=8``), wie in
``diag_transformer_cost.py`` -- reicht fuer korrekte Tensorformen und eine grobe
uS/Schritt-Schaetzung, ist NICHT die Produktionsgroesse. Keine Aenderung an
``torch_sequence.py``/``composite.py``/``mamba.py`` -- reiner Lesezugriff auf bereits
vorhandene Netz-Attribute (``network.core.layers``).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.models.composite import _SingleTargetView
from twinlink_ml.models.torch_sequence import TorchSequenceConfig, _build_network
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG_PATH = Path("conf") / "process" / "greybox_gab_placeholder.yaml"
SEED = 1
HIDDEN_SIZE = 16
STATE_SIZE = 16
N_STEPS = 68  # 32 Historie + 36 Horizont, Produktionsgroesse (Befund 35)

torch.set_num_threads(1)


def _small_spec() -> DataSpec:
    """Wie ``diag_transformer_cost._small_spec`` -- gleiche Groesse, direkt vergleichbar."""
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


def _report_positions(step_times: list[float]) -> None:
    print(f"  Schritt 1:  {step_times[0] * 1e3:.4f} ms")
    print(f"  Schritt 18: {step_times[17] * 1e3:.4f} ms")
    print(f"  Schritt 30: {step_times[29] * 1e3:.4f} ms")
    print(f"  Schritt 36: {step_times[35] * 1e3:.4f} ms")
    print(f"  Schritt 68: {step_times[67] * 1e3:.4f} ms")
    print(f"  Verhaeltnis Schritt68/Schritt1: {step_times[67] / step_times[0]:.2f}x")
    mean_time = sum(step_times) / len(step_times)
    print(f"  Mittelwert ueber {len(step_times)} Schritte: {mean_time * 1e3:.4f} ms")
    print(f"  Median: {sorted(step_times)[len(step_times) // 2] * 1e3:.4f} ms")


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = _small_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)
    train, _, _ = build_datasets(data, spec.n_train_max, policy=ValidationPolicy.FIXED_POOL)

    target_index = 0  # energy_target
    single = _SingleTargetView(train, target_index)
    n_exogenous = len(single.exogenous_names)
    batch_size = min(256, single.n_windows)
    batch = single.batch(list(range(batch_size)))
    history = torch.tensor(np.ascontiguousarray(batch.history_inputs), dtype=torch.float32)
    future = torch.tensor(np.ascontiguousarray(batch.future_exogenous), dtype=torch.float32)
    print(f"n_exogenous={n_exogenous}, Batch: {history.shape[0]} Fenster")
    print(f"history_length={spec.history_length}, horizon={spec.horizon}, N_STEPS={N_STEPS}")

    # Kombinierte 68-Schritt-Eingabesequenz -- feste Kanalbreite (n_exogenous + 1)
    # ueber alle Schritte, wie im echten Rollout (Historie hat dieselbe Breite wie
    # ``cat([future_exogenous, last])`` im Decoder/Horizont-Schritt).
    last = history[:, -1, n_exogenous:]
    combined_inputs = [history[:, t, :] for t in range(history.shape[1])]
    for step in range(future.shape[1]):
        combined_inputs.append(torch.cat([future[:, step, :], last], dim=-1))
    assert len(combined_inputs) == N_STEPS

    # ------------------------------------------------------------ (a) Mamba, echte Pro-Schritt-Zeit
    print("\n=== (a) Mamba: Pro-Schritt-Zeit ueber 68 sequenzielle SSM-Schritte ===")
    mamba_config = TorchSequenceConfig(
        core="mamba", hidden_size=HIDDEN_SIZE, state_size=STATE_SIZE, num_layers=1
    )
    mamba_net = _build_network(torch, n_exogenous=n_exogenous, n_targets=1, config=mamba_config)
    layer = mamba_net.core.layers[0]

    with torch.no_grad():
        state = layer.init_state(batch_size, history)
        _ = layer.step(combined_inputs[0], state)  # Warmup

        state = layer.init_state(batch_size, history)
        mamba_step_times: list[float] = []
        for step_input in combined_inputs:
            start = time.perf_counter()
            _out, state = layer.step(step_input, state)
            mamba_step_times.append(time.perf_counter() - start)
    _report_positions(mamba_step_times)

    # -------------------------------------------------- (b1) GRU/LSTM, dieselbe Schleifendisziplin
    print("\n=== (b1) GRU/LSTM: dieselbe sequenzielle Schleife, direkter Pro-Schritt-Vergleich ===")
    rnn_means: dict[str, float] = {}
    for core_name, cell_cls in (("gru", torch.nn.GRU), ("lstm", torch.nn.LSTM)):
        cell = cell_cls(
            input_size=n_exogenous + 1, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True
        )
        with torch.no_grad():
            _ = cell(combined_inputs[0].unsqueeze(1))  # Warmup
            hidden = None
            step_times: list[float] = []
            for step_input in combined_inputs:
                start = time.perf_counter()
                _out, hidden = cell(step_input.unsqueeze(1), hidden)
                step_times.append(time.perf_counter() - start)
        print(f"\n{core_name.upper()}:")
        _report_positions(step_times)
        rnn_means[core_name] = sum(step_times) / len(step_times)

    mamba_mean = sum(mamba_step_times) / len(mamba_step_times)
    print("\n--- Verhaeltnis Mamba/GRU bzw. Mamba/LSTM (Mittelwert ueber 68 Schritte) ---")
    for core_name, mean_time in rnn_means.items():
        print(f"  Mamba/{core_name.upper()}: {mamba_mean / mean_time:.2f}x")

    # ------------------------------------------------------- (b2) Naeherung: volles Forward / 68
    print("\n=== (b2) Gegenprobe (Naeherung, wie beauftragt): volle Network.forward-Zeit / 68 ===")

    def timed_full_forward(net: torch.nn.Module) -> float:
        with torch.no_grad():
            _ = net(history, future)  # Warmup
            start = time.perf_counter()
            _ = net(history, future)
            return time.perf_counter() - start

    mamba_full = timed_full_forward(mamba_net)
    print(
        f"Mamba volles Forward (68 Schritte inkl. Concat/Head): {mamba_full * 1e3:.3f} ms total, "
        f"{mamba_full / N_STEPS * 1e3:.4f} ms/Schritt (Naeherung)"
    )
    for core_name in ("gru", "lstm"):
        cfg = TorchSequenceConfig(core=core_name, hidden_size=HIDDEN_SIZE, num_layers=1)
        net = _build_network(torch, n_exogenous=n_exogenous, n_targets=1, config=cfg)
        full_time = timed_full_forward(net)
        print(
            f"{core_name.upper()} volles Forward: {full_time * 1e3:.3f} ms total, "
            f"{full_time / N_STEPS * 1e3:.4f} ms/Schritt (Naeherung)"
        )


if __name__ == "__main__":
    main()
