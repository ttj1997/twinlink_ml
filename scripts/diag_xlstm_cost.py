r"""Schritt 2 (xLSTM-Wiederaufnahme): Skalierungstest + Kostendiagnose, kein Volltraining.

Gleiches Muster wie ``diag_mamba_cost.py``/``diag_transformer_cost.py`` (Befund 37/38):
zwei unabhaengige, real gemessene Fragen, kein neues Volltraining.

(a) Skalierungstest, **ERGEBNISOFFEN** (anders als bei Mamba nicht als O(1) erwartet
    vorgegeben -- die matrixwertige Zelle koennte sich anders verhalten, siehe
    Schritt-0-Antwort d/Paper-Abschnitt 2.5: die Rechenkosten sind O(d^2) pro Schritt
    durch die ``d x d``-Matrixzelle, aber das ist eine Aussage ueber die Kosten PRO
    SCHRITT bei fester Groesse ``d``, nicht ueber das Wachstum UEBER DIE SEQUENZ bei
    fester ``d`` -- das ist die hier gemessene, unabhaengige Frage). 68 sequenzielle
    mLSTM-Schritte (32 Historie + 36 Horizont, Produktionsgroesse), Zeit je Einzelschritt
    bei Position 1, 18, 30, 36 und zusaetzlich bei Position 68 (Sequenzende).

(b) Absolutwert, zwei unabhaengige Messmethoden (wie bei Mamba):
    (b1) Direkter Pro-Schritt-Vergleich: GRU und LSTM (gleiches hidden_size) in
         DERSELBEN sequenziellen Python-Schleife wie xLSTM -- fair, weil beide Seiten
         denselben Python-Dispatch-Overhead pro Aufruf tragen.
    (b2) Naeherung: volle ``Network.forward``-Zeit (alle 68 Schritte inkl. Concat/Head,
         ueber den echten Trainer-Pfad) geteilt durch 68.

(c) Realistische Zeitschaetzung fuer den reduzierten Vergleich (Schritt 3): aus der
    ECHTEN, bereits gemessenen Schrittzeit bei Produktionsgroesse
    (``diag_xlstm_memory.py``, ``hidden_size=16``, ``batch_size=256``: 0.070 s pro
    ``forward``+``backward`` ueber alle 68 Schritte) hochgerechnet auf
    ``batches_per_epoch=9`` (2176 Fenster / 256, aufgerundet), ``max_epochs=200``,
    3 Teilnetze -- keine neue Zeitmessung noetig, nur eine Multiplikation der bereits
    real gemessenen Zahl.

``hidden_size=16``, wie in der Schritt-0-Antwort empfohlen (unproblematisch bei
Produktionsgroesse, siehe ``diag_xlstm_memory.py``). Kleiner Datenumfang
(``n_train_max=4, n_val=4, n_test=8``) fuer die Pro-Schritt-Zeitmessung selbst -- reicht
fuer korrekte Tensorformen und eine grobe uS/Schritt-Schaetzung, ist NICHT die
Produktionsgroesse (die ist bereits in ``diag_xlstm_memory.py`` gemessen). Keine
Aenderung an ``torch_sequence.py``/``composite.py``/``xlstm.py`` -- reiner Lesezugriff
auf bereits vorhandene Netz-Attribute (``network.core.layers``).
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
N_STEPS = 68  # 32 Historie + 36 Horizont, Produktionsgroesse (Befund 35)

# Aus diag_xlstm_memory.py real gemessen (hidden_size=16, batch_size=256,
# n_train=16 -> 2176 Fenster, forward+backward ueber alle 68 Schritte, EIN Teilnetz).
MEASURED_STEP_SECONDS_H16_B256 = 0.070
N_WINDOWS_N_TRAIN_16 = 2176
BATCH_SIZE_PRODUCTION = 256
MAX_EPOCHS_PRODUCTION = 200
N_SUBMODELS = 3

torch.set_num_threads(1)


def _small_spec() -> DataSpec:
    """Wie ``diag_mamba_cost._small_spec``/``diag_transformer_cost._small_spec``."""
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

    last = history[:, -1, n_exogenous:]
    combined_inputs = [history[:, t, :] for t in range(history.shape[1])]
    for step in range(future.shape[1]):
        combined_inputs.append(torch.cat([future[:, step, :], last], dim=-1))
    assert len(combined_inputs) == N_STEPS

    # ------------------------------------------------------- (a) xLSTM, echte Pro-Schritt-Zeit
    print(
        "\n=== (a) xLSTM: Pro-Schritt-Zeit ueber 68 sequenzielle mLSTM-Schritte (ERGEBNISOFFEN) ==="
    )
    xlstm_config = TorchSequenceConfig(core="xlstm", hidden_size=HIDDEN_SIZE, num_layers=1)
    xlstm_net = _build_network(torch, n_exogenous=n_exogenous, n_targets=1, config=xlstm_config)
    layer = xlstm_net.core.layers[0]

    with torch.no_grad():
        state = layer.init_state(batch_size, history)
        _ = layer.step(combined_inputs[0], state)  # Warmup

        state = layer.init_state(batch_size, history)
        xlstm_step_times: list[float] = []
        for step_input in combined_inputs:
            start = time.perf_counter()
            _out, state = layer.step(step_input, state)
            xlstm_step_times.append(time.perf_counter() - start)
    _report_positions(xlstm_step_times)

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

    xlstm_mean = sum(xlstm_step_times) / len(xlstm_step_times)
    print("\n--- Verhaeltnis xLSTM/GRU bzw. xLSTM/LSTM (Mittelwert ueber 68 Schritte) ---")
    for core_name, mean_time in rnn_means.items():
        print(f"  xLSTM/{core_name.upper()}: {xlstm_mean / mean_time:.2f}x")

    # ------------------------------------------------------- (b2) Naeherung: volles Forward / 68
    print("\n=== (b2) Gegenprobe (Naeherung): volle Network.forward-Zeit / 68 ===")

    def timed_full_forward(net: torch.nn.Module) -> float:
        with torch.no_grad():
            _ = net(history, future)  # Warmup
            start = time.perf_counter()
            _ = net(history, future)
            return time.perf_counter() - start

    xlstm_full = timed_full_forward(xlstm_net)
    print(
        f"xLSTM volles Forward (68 Schritte inkl. Concat/Head): {xlstm_full * 1e3:.3f} ms total, "
        f"{xlstm_full / N_STEPS * 1e3:.4f} ms/Schritt (Naeherung)"
    )
    for core_name in ("gru", "lstm"):
        cfg = TorchSequenceConfig(core=core_name, hidden_size=HIDDEN_SIZE, num_layers=1)
        net = _build_network(torch, n_exogenous=n_exogenous, n_targets=1, config=cfg)
        full_time = timed_full_forward(net)
        print(
            f"{core_name.upper()} volles Forward: {full_time * 1e3:.3f} ms total, "
            f"{full_time / N_STEPS * 1e3:.4f} ms/Schritt (Naeherung)"
        )

    # --------------------------------------- (c) Zeitschaetzung fuer den reduzierten Vergleich
    print("\n=== (c) Zeitschaetzung Schritt 3 (reduzierter Vergleich, n_train=16, 1 Seed) ===")
    print(
        f"Grundlage: ECHT gemessener Trainingsschritt bei Produktionsgroesse "
        f"(diag_xlstm_memory.py, hidden_size={HIDDEN_SIZE}, batch_size={BATCH_SIZE_PRODUCTION}, "
        f"n_train=16 -> {N_WINDOWS_N_TRAIN_16} Fenster): "
        f"{MEASURED_STEP_SECONDS_H16_B256:.3f} s pro forward+backward (alle 68 Schritte)."
    )
    batches_per_epoch = -(-N_WINDOWS_N_TRAIN_16 // BATCH_SIZE_PRODUCTION)
    seconds_per_epoch = batches_per_epoch * MEASURED_STEP_SECONDS_H16_B256
    seconds_per_submodel = seconds_per_epoch * MAX_EPOCHS_PRODUCTION
    seconds_total = seconds_per_submodel * N_SUBMODELS
    print(
        f"  batches_per_epoch = ceil({N_WINDOWS_N_TRAIN_16}/{BATCH_SIZE_PRODUCTION}) "
        f"= {batches_per_epoch}"
    )
    print(
        f"  Sekunden/Epoche = {batches_per_epoch} * {MEASURED_STEP_SECONDS_H16_B256:.3f}s "
        f"= {seconds_per_epoch:.2f}s"
    )
    print(
        f"  Sekunden/Teilnetz (max_epochs={MAX_EPOCHS_PRODUCTION}, oberer Rand ohne "
        f"Early-Stopping) "
        f"= {seconds_per_epoch:.2f}s * {MAX_EPOCHS_PRODUCTION} = {seconds_per_submodel:.1f}s "
        f"({seconds_per_submodel / 60.0:.2f} min)"
    )
    print(
        f"  Sekunden gesamt ({N_SUBMODELS} Teilnetze) = {seconds_total:.1f}s "
        f"({seconds_total / 60.0:.2f} min)"
    )
    print(
        "  Hinweis: obere Schranke (patience=20 stoppt frueher, wenn die Validierung "
        "abflacht -- reale Laufzeit typischerweise kuerzer, siehe training_history "
        "anderer Architekturen)."
    )


if __name__ == "__main__":
    main()
