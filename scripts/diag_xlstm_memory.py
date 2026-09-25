r"""Schritt 0 (xLSTM-Wiederaufnahme): echte Speicher-/Zeitmessung der mLSTM-Zelle.

**Warum dieses Skript existiert, obwohl ``models/xlstm.py`` schon fertig ist.** Der
Kern (``XLSTMCore``) ist bereits implementiert und unit-getestet (``tests/test_composite.py``,
sechs gruene Tests) -- aber nur bei winzigem Batch (``n_train_max=4``). Der einzige noch
offene Punkt aus der Session-Wiederaufnahme ist: verhaelt sich der Speicher-/Zeitbedarf bei
**Produktionsgroesse** (``batch_size=256`` Standard, ``n_train=16`` -> 2176 Fenster je
Teilmodell, siehe ``transformer_reduced_comparison.log``) noch handhabbar? Der einzige
belastbare Hinweis dazu im Repo ist ein Satz im Docstring von
``CompositeTransformerSequenceModel`` (``composite.py``): ein fruehrer
kapazitaetsangepasster xLSTM-Versuch (vermutlich ``hidden_size~317``, siehe
``scripts/study_xlstm_wide.py``-Docstring-Beispiel) hat **>12h/Trainingspunkt** gekostet.
Das ist eine reale, aber nur verbal ueberlieferte Zahl -- dieses Skript reproduziert die
Groessenordnung an einer einzelnen Trainingsiteration (ein ``forward`` + ``backward``,
kein Volltraining) bei mehreren ``hidden_size``-Werten, damit die Aussage nachpruefbar wird.

**Warum Speicher UND Zeit, nicht nur Zeit wie bei ``diag_mamba_cost.py``/
``diag_transformer_cost.py``.** Die mLSTM-Zelle haelt pro Schritt einen Matrixzustand
``C_t in R^{batch x H x H}`` (``xlstm.py``, ``_XLSTMLayer.init_state``/``step``). Fuer
``loss.backward()`` durch 68 sequenzielle Schritte muss Autograd **jeden** dieser 68
Zwischenzustaende festhalten (kein Truncated-BPTT, kein Checkpointing in
``torch_sequence.py`` -- absichtlich unveraendert, siehe ``NICHT TUN`` in der Aufgabe).
Erwartung, ERGEBNISOFFEN zu pruefen: Speicher waechst mit ``batch * H^2 * horizon``
(quadratisch in ``H``, nicht nur linear wie bei GRU/LSTM/Mamba, deren Zustand
``R^{batch x H}`` bzw. ``R^{batch x d_inner x state_size}`` mit festem ``state_size`` ist).

**Peak Working Set statt ``tracemalloc``.** ``tracemalloc`` sieht nur Python-Heap-
Allokationen, nicht die C-seitigen Tensor-Buffer von PyTorch. Auf dieser Windows-Maschine
ohne ``psutil`` (nicht installiert, kein ungefragtes ``pip install`` fuer eine
Diagnose) liest :func:`_peak_working_set_mb` das reale ``PeakWorkingSetSize`` direkt
ueber ``GetProcessMemoryInfo`` (``psapi.dll``, Windows-Standardbibliothek, kein neues
Package). Deshalb laeuft jede ``hidden_size``-Messung in einem **eigenen Unterprozess**
(``--hidden-size`` als CLI-Argument, von ``main()`` per ``subprocess`` aufgerufen) --
``PeakWorkingSetSize`` ist monoton fuer die Prozesslaufzeit, mehrere Groessen im selben
Prozess wuerden sich gegenseitig verunreinigen.

Keine Aenderung an ``torch_sequence.py``/``composite.py``/``xlstm.py`` -- reiner
Lesezugriff auf ``_build_network`` (identisch zu ``diag_mamba_cost.py``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, ClassVar

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
SEED = 1
N_TRAIN = 16  # Produktionsgroesse, wie run_transformer_/run_mamba_reduced_comparison.py
DEFAULT_BATCH_SIZE = 256  # TorchSequenceConfig-Standard


def _peak_working_set_mb() -> float:
    """Reales ``PeakWorkingSetSize`` dieses Prozesses in MB (Windows, ``psapi.dll``)."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        error_code = ctypes.get_last_error()
        raise OSError(f"GetProcessMemoryInfo fehlgeschlagen, GetLastError={error_code}")
    return counters.PeakWorkingSetSize / (1024.0 * 1024.0)


def _run_single_point(hidden_size: int, batch_size: int) -> dict:
    """Ein ``forward`` + ``backward`` bei ``hidden_size``/``batch_size``, in DIESEM Prozess.

    Wird nur als Unterprozess aufgerufen (siehe ``main``), damit ``PeakWorkingSetSize``
    ausschliesslich diese eine Messung widerspiegelt.
    """
    baseline_mb = _peak_working_set_mb()

    import numpy as np
    import torch

    from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
    from twinlink_ml.models.composite import _SingleTargetView
    from twinlink_ml.models.torch_sequence import TorchSequenceConfig, _build_network
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

    torch.set_num_threads(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)

    spec = DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)
    train, _, _ = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)

    single = _SingleTargetView(train, 0)  # energy_target, wie diag_mamba_cost.py
    n_exogenous = len(single.exogenous_names)
    n_windows = single.n_windows
    used_batch = min(batch_size, n_windows)

    after_data_mb = _peak_working_set_mb()

    net_config = TorchSequenceConfig(core="xlstm", hidden_size=hidden_size, num_layers=1)
    network = _build_network(torch, n_exogenous=n_exogenous, n_targets=1, config=net_config)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    batch = single.batch(list(range(used_batch)))
    history = torch.tensor(np.ascontiguousarray(batch.history_inputs), dtype=torch.float32)
    future = torch.tensor(np.ascontiguousarray(batch.future_exogenous), dtype=torch.float32)
    targets = torch.tensor(np.ascontiguousarray(batch.future_targets), dtype=torch.float32)

    n_steps = history.shape[1] + future.shape[1]

    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prediction = network(history, future)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    loss.backward()
    optimizer.step()
    elapsed_s = time.perf_counter() - start

    peak_mb = _peak_working_set_mb()
    n_parameters = sum(p.numel() for p in network.parameters())

    return {
        "hidden_size": hidden_size,
        "batch_size": used_batch,
        "n_windows_available": n_windows,
        "n_steps": n_steps,
        "n_parameters": n_parameters,
        "baseline_mb": round(baseline_mb, 1),
        "after_data_mb": round(after_data_mb, 1),
        "peak_mb": round(peak_mb, 1),
        "delta_data_to_peak_mb": round(peak_mb - after_data_mb, 1),
        "elapsed_s": round(elapsed_s, 4),
        "elapsed_per_step_ms": round(elapsed_s / n_steps * 1000.0, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--as-subprocess", action="store_true")
    arguments = parser.parse_args()

    if arguments.as_subprocess:
        assert arguments.hidden_size is not None
        result = _run_single_point(arguments.hidden_size, arguments.batch_size)
        print(json.dumps(result))
        return

    print(f"=== xLSTM: echte Speicher-/Zeitmessung, N_TRAIN={N_TRAIN} (Produktionsgroesse) ===\n")

    points: list[tuple[int, int]] = [
        (4, DEFAULT_BATCH_SIZE),
        (8, DEFAULT_BATCH_SIZE),
        (16, DEFAULT_BATCH_SIZE),
        (32, DEFAULT_BATCH_SIZE),
        (64, DEFAULT_BATCH_SIZE),
        (128, DEFAULT_BATCH_SIZE),
        (16, 32),  # kleinere batch_size bei H=16 -- hilft das dem Speicher?
        (16, 8),
    ]

    results = []
    for hidden_size, batch_size in points:
        print(
            f"-- hidden_size={hidden_size}, batch_size={batch_size}: "
            "laeuft (eigener Unterprozess) --"
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--as-subprocess",
                "--hidden-size",
                str(hidden_size),
                "--batch-size",
                str(batch_size),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            print(f"   FEHLGESCHLAGEN (returncode {proc.returncode}):\n{proc.stderr[-2000:]}")
            results.append(
                {"hidden_size": hidden_size, "batch_size": batch_size, "failed": True}
            )
            continue
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(result)
        print(
            f"   n_parameters={result['n_parameters']}, "
            f"peak_working_set={result['peak_mb']:.1f} MB "
            f"(delta ab Netzaufbau: {result['delta_data_to_peak_mb']:.1f} MB), "
            f"1 Schritt (fwd+bwd)={result['elapsed_s']:.3f}s "
            f"({result['elapsed_per_step_ms']:.4f} ms/Sequenzschritt)"
        )

    print("\n=== Zusammenfassung ===")
    header = (
        f"{'hidden_size':>11} {'batch_size':>10} {'n_params':>9} {'peak_MB':>9} "
        f"{'delta_MB':>9} {'s/Schritt':>10}"
    )
    print(header)
    for result in results:
        if result.get("failed"):
            print(f"{result['hidden_size']:>11} {result['batch_size']:>10}   FEHLGESCHLAGEN")
            continue
        print(
            f"{result['hidden_size']:>11} {result['batch_size']:>10} "
            f"{result['n_parameters']:>9} {result['peak_mb']:>9.1f} "
            f"{result['delta_data_to_peak_mb']:>9.1f} {result['elapsed_s']:>10.3f}"
        )

    output_path = REPO_ROOT / "artifacts" / "runs" / "diag_xlstm_memory.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRohdaten: {output_path}")


if __name__ == "__main__":
    main()
