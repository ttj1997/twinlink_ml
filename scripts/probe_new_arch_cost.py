r"""Kostenprobe (OA2, Transformer/xLSTM/Mamba-Erweiterung der Lernkurve): ein Fit je
Architektur, ein Seed, n_train=8, unter EXAKT demselben Protokoll wie
``run_arch_learning_curve_worker.py`` (drei Zielkanaele, ``enforce_gas_velocity_projection``
explizit an, Fingerprint-Pruefung gegen ``bddca37f716f5cfc`` -- vor dem
Pseudonymisierungs-Schnitt 2026-09-25: ``2d2679eeed716577``, siehe ``docs/ARCHIV.md``).

Kein Sweep, kein Checkpoint, keine Auswertung -- nur ``training/elapsed_s`` und die
Epochenzahl je Teilnetz, als Grundlage fuer die Hochrechnung auf 4 Budgetpunkte x 5 Seeds.
``MKL_NUM_THREADS``/``OMP_NUM_THREADS=1`` muessen als PROZESS-Umgebungsvariablen VOR dem
Start gesetzt sein (nicht nur ``torch.set_num_threads``, siehe
``run_arch_learning_curve_sweep.py``-Docstring) -- dieses Skript prueft das und bricht ab,
falls sie fehlen, statt still mit falscher Thread-Konfiguration zu messen.

Aufruf, ein Prozess je Architektur::

    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/probe_new_arch_cost.py --architecture transformer
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from twinlink_ml.evaluation import DataSpec, ValidationPolicy, build_datasets, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

if TYPE_CHECKING:
    from collections.abc import Callable

    from twinlink_ml.models.composite import CompositeSequenceModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
SEED = 1
N_TRAIN = 8
EXPECTED_FINGERPRINT = "bddca37f716f5cfc"

TARGET_NAMES = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")


def _factory(label: str) -> Callable[[], CompositeSequenceModel]:
    from twinlink_ml.models.composite import (
        CompositeMambaSequenceModel,
        CompositeTransformerSequenceModel,
        CompositeXLSTMSequenceModel,
    )

    return {
        "transformer": CompositeTransformerSequenceModel,
        "xlstm": CompositeXLSTMSequenceModel,
        "mamba": CompositeMambaSequenceModel,
    }[label]


def _spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
        target_names=TARGET_NAMES,
    )


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("transformer", "xlstm", "mamba"), required=True)
    arguments = parser.parse_args()

    for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(var) != "1":
            raise RuntimeError(
                f"{var}={os.environ.get(var)!r}, erwartet '1' als PROZESS-Umgebungsvariable "
                "(vor dem Start gesetzt, nicht ueber torch.set_num_threads) -- siehe "
                "Moduldocstring. Abbruch vor der Messung, sonst ist die Zeit nicht "
                "mit dem Sweep-Protokoll vergleichbar."
            )

    import torch

    torch.set_num_threads(1)

    _log(f"Start: Kostenprobe {arguments.architecture!r}, seed={SEED}, n_train={N_TRAIN}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    config = dataclasses.replace(
        config,
        numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True),
    )

    spec = _spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        data = generate_data(config, spec, master_seed=SEED)

    if not fingerprints_equivalent(data.process_config.fingerprint, EXPECTED_FINGERPRINT):
        raise RuntimeError(
            f"process/fingerprint={data.process_config.fingerprint!r}, erwartet "
            f"{EXPECTED_FINGERPRINT!r} -- Config weicht vom Lernkurven-Protokoll ab, "
            "Abbruch vor dem Training."
        )
    _log(f"Fingerprint bestaetigt: {data.process_config.fingerprint}")

    train, validation, _test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    _log(
        f"Datensaetze gebaut: {train.n_windows} Trainingsfenster, "
        f"{validation.n_windows if validation is not None else 0} Validierungsfenster"
    )

    factory = _factory(arguments.architecture)
    model = factory()
    start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation, verbose=True)
    elapsed = time.perf_counter() - start

    _log(
        f"-- {arguments.architecture!r}: Training fertig ({elapsed:.2f}s, {elapsed / 60.0:.2f} Min) --"
    )
    describe = model.describe()
    _log(f"   net_actual_device={describe.get('net_actual_device')!r}")
    for target, history in model.training_history.items():
        _log(f"   Teilnetz {target!r}: {len(history)}/{model.config.max_epochs} Epochen")

    _log(
        f"ERGEBNIS architecture={arguments.architecture} elapsed_s={elapsed:.3f} "
        f"torch_num_threads={torch.get_num_threads()} "
        f"mkl_num_threads={os.environ.get('MKL_NUM_THREADS')} "
        f"omp_num_threads={os.environ.get('OMP_NUM_THREADS')}"
    )


if __name__ == "__main__":
    main()
