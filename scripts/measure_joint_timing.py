r"""Joint timing measurement on an idle machine: study A (all methods incl. the hybrid's simulator channel) and the
eight architectures of the re-run (Befund 94). Written before the run (committed before it is executed; clean tree).

Design (docs/17, "Timing and the machine"; docs/15 section 6)
------------------------------------------------------------
* ONE thread (``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`` = 1, ``torch.set_num_threads(1)``), warm-up excluded, nothing
  else running: the script aborts unless the machine is idle (machine load <= 5 % and no other Python process above
  5 % of one core, measured over 5 s) at the start, and records the load again between the parts and at the end.
* **Part A** (study A): ``scripts/measure_a1_inference_v2.py --require-idle`` as a subprocess (persistence, ARX,
  simulator only, from scratch, hybrid network, pretrain + fine-tune; the horizon simulation of the hybrid in four
  variants; the hybrid total). Its own artifact is ``a1_inference_time_v2_idle``.
* **Part B** (architectures): the eight methods of the re-run (persistence, static regression, ARX, GRU, LSTM,
  Transformer, xLSTM, Mamba) on the pool of arm N (preset version 2), seed 1, ``n_train`` 64, excited test windows:
  batch of one window (latency) and the whole test set (throughput), as in ``measure_a1_inference.measure``. The
  networks are fitted for ``--fit-epochs`` epochs only (default 2): the weights do not change the cost of
  ``predict``; their accuracy is not used. The wall time of that short fit is recorded per epoch as an indication of
  the training cost (it includes setup; NOT the cost of a training to convergence, which is in the records of the
  re-run under parallel load and is not a timing result).
* The artifact ``joint_timing_idle`` holds part B, the environment (torch / numpy / python versions, logical CPUs,
  processor name, power plan is not read), the machine load at start / between / end, and the name of the artifact of
  part A. Provenance is taken before part A writes its artifact, so ``git/dirty`` refers to the committed script.

    tools/run_here.sh scripts/measure_joint_timing.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the thread variables must be set before numpy / torch are imported
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import dataclasses
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_a1_inference as first
import measure_a1_inference_v2 as part_a
import run_arch_rerun_worker as worker
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.evaluation import ValidationPolicy, build_datasets, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

SEED = 1
N_TRAIN = 64
OUTPUT_NAME = "joint_timing_idle"


def environment() -> dict[str, Any]:
    import numpy
    import torch

    processor = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "logical_cpus": os.cpu_count(),
        "processor": processor,
        "platform": platform.platform(),
        "thread_variables": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def measure_architectures(fit_epochs: int) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(worker.old_worker.CONFIG_PATH)
    config = dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )
    spec = worker.arm_spec("N")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
        data = generate_data(config, spec, master_seed=SEED)
    train, validation, test = build_datasets(data, N_TRAIN, policy=ValidationPolicy.FIXED_POOL)
    results = []
    factories: dict[str, Any] = dict(worker.old_worker.BASELINE_FACTORIES)
    for label in worker.old_worker.ARCHITECTURE_LABELS:
        factories[label] = None
    for label in ("persistence", "static", "arx", "gru", "lstm", "transformer", "xlstm", "mamba"):
        if factories[label] is None:
            model = worker.old_worker._build_model(
                worker.old_worker._architecture_factory(label), fit_epochs
            )
            epochs = fit_epochs
        else:
            model = factories[label]()
            epochs = 0
        started = time.perf_counter()
        model.fit(train, seed=data.seeds.model_seed, validation=validation)
        fit_s = time.perf_counter() - started
        entry = first.measure(label, model, test)
        entry["fit_elapsed_s"] = fit_s
        entry["fit_epochs"] = epochs
        entry["fit_s_per_epoch_indicative"] = fit_s / epochs if epochs else None
        results.append(entry)
        print(
            f"{label}: latency {entry['latency_batch1_us_median']:.0f} us, "
            f"throughput {entry['throughput_us_per_window_median']:.1f} us/window",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-epochs", type=int, default=2)
    arguments = parser.parse_args()
    require_clean_tree()
    start_load = first.machine_load()
    if not part_a.is_idle(start_load):
        print("machine is not idle:", start_load)
        return 4
    meta = provenance()  # before part A writes its artifact
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.time()

    print("part A: study A (subprocess)", flush=True)
    command = [sys.executable, str(Path(part_a.__file__)), "--require-idle"]
    code = subprocess.run(command, check=False).returncode
    if code != 0:
        print(f"part A failed with exit code {code}")
        return code
    between_load = first.machine_load()

    print("part B: eight architectures", flush=True)
    warmup_s = part_a.warm_up()
    results = measure_architectures(arguments.fit_epochs)
    end_load = first.machine_load()
    write_artifact(
        OUTPUT_NAME,
        {
            "_provenance": meta,
            "design": {
                "threads": 1,
                "part_a_artifact": "a1_inference_time_v2_idle",
                "part_b": {
                    "arm": "N (preset version 2)",
                    "seed": SEED,
                    "n_train": N_TRAIN,
                    "test_windows": "excited test set of the pool",
                    "fit_epochs_networks": arguments.fit_epochs,
                    "note": "weights do not change the cost of predict; accuracy unused",
                },
                "idle_criterion": f"machine load <= {part_a.IDLE_MACHINE_LOAD_PCT} % and no other python "
                f"process > {part_a.IDLE_PROCESS_CPU_PCT} % of one core",
                "warmup_s": warmup_s,
                "wall_s_total": time.time() - started,
            },
            "environment": environment(),
            "load_at_start": start_load,
            "load_between_parts": between_load,
            "load_at_end": end_load,
            "idle_at_start": True,
            "idle_between_parts": part_a.is_idle(between_load),
            "idle_at_end": part_a.is_idle(end_load),
            "results": results,
        },
    )
    print("joint timing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
