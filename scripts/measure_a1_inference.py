r"""Inference time per method of the A1 main run, measured once, apart from the run.

The times recorded in the main run (``a1_main_records.json``) were taken under parallel load and carry
``timing_context``; they are not results. This script measures the inference time of every method of the main
run separately: ONE thread (torch, BLAS and numpy), warm-up excluded, repeated, on the 32 excited test runs of the
plant P0, seed 1 (the data of the main run).

Written before the run (committed before it is executed; clean tree, ``git/dirty`` is logged).

What is measured
----------------
* Per method the cost of ``predict`` on the windows of the test set: (a) THROUGHPUT -- all test windows in one
  batch (the way the evaluator calls the model), microseconds per window; (b) LATENCY -- one window per call
  (batch of 1, the real-time case), microseconds per call. Median and minimum over ``REPEATS`` repeats after
  ``WARMUP`` discarded calls.
* The neural methods are measured with models of the architecture of the main run (one LSTM, hidden 64, one
  layer, three targets): from scratch and pretraining + fine-tuning share the architecture (7 exogenous
  channels), the hybrid has 10. The weights do not change the cost of ``predict``; the models are therefore
  trained for 2 epochs only (the fine-tuned one starts from the pretrained model of seed 1 of the main run). The
  accuracy of these models is NOT used for anything.
* The machine is NOT guaranteed idle: the machine load (all logical processors, in per cent) and the CPU use of
  every Python process are sampled before and after each method and recorded; the artifact says what ran
  beside. The measurement is to be repeated (new artifact name) when the machine is free of other runs.

    tools/run_here.sh scripts/measure_a1_inference.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the thread variables must be set before numpy / torch are imported
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact
from run_a1_main import FINETUNE_LEARNING_RATE, make_plan
from run_a1_pilot_pp import generate, load_config, warm_up

from twinlink_ml.data.splits import SplitName
from twinlink_ml.evaluation.experiments import build_datasets
from twinlink_ml.models.baselines import LinearARXModel, PersistenceModel
from twinlink_ml.models.hybrid import SimulatorOnlyModel, with_simulator_channels
from twinlink_ml.models.torch_sequence import TorchSequenceConfig
from twinlink_ml.models.transfer import (
    FreezeSpec,
    TransferSequenceModel,
    build_datasets_with_scalers,
    load_pretrained,
)

OUTPUT_NAME = "a1_inference_time_1thread"
CHECKPOINT = Path("C:/Users/ctj1637/tl_a1_main/work/ckpt/pretrained_s1.pt")
SEED = 1
WARMUP = 3
REPEATS_BATCH = 15
REPEATS_SINGLE = 300
TRAIN_EPOCHS = 2


def machine_load() -> dict[str, Any]:
    """Load of all logical processors and the CPU share of the busy Python processes over 5 s."""
    command = (
        "$a = Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU;"
        "$l = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average;"
        "Start-Sleep -Seconds 5;"
        "$b = Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU;"
        "$rows = foreach ($x in $b) { $o = $a | Where-Object Id -eq $x.Id;"
        " if ($o) { '{0}:{1}' -f $x.Id, [math]::Round(($x.CPU-$o.CPU)/5*100,0) } };"
        "'LOAD=' + $l; 'PY=' + ($rows -join ',')"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    load = next((float(line[5:]) for line in out if line.startswith("LOAD=")), float("nan"))
    python = next((line[3:] for line in out if line.startswith("PY=")), "")
    busy = {
        item.split(":")[0]: int(item.split(":")[1])
        for item in python.split(",")
        if ":" in item and int(item.split(":")[1]) >= 5 and int(item.split(":")[0]) != os.getpid()
    }
    return {"machine_load_pct": load, "other_python_processes_cpu_pct_of_one_core": busy}


def time_calls(call: Any, repeats: int) -> dict[str, float]:
    for _ in range(WARMUP):
        call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append(time.perf_counter() - started)
    return {"median_s": statistics.median(times), "min_s": min(times), "repeats": repeats}


def measure(name: str, model: Any, test: Any) -> dict[str, Any]:
    before = machine_load()
    batch = test.full_batch()
    n_windows = batch.n_windows
    single = test.batch([0])
    throughput = time_calls(lambda: model.predict(batch), REPEATS_BATCH)
    latency = time_calls(lambda: model.predict(single), REPEATS_SINGLE)
    after = machine_load()
    return {
        "method": name,
        "n_test_windows": int(n_windows),
        "throughput_us_per_window_median": 1e6 * throughput["median_s"] / n_windows,
        "throughput_us_per_window_min": 1e6 * throughput["min_s"] / n_windows,
        "latency_batch1_us_median": 1e6 * latency["median_s"],
        "latency_batch1_us_min": 1e6 * latency["min_s"],
        "repeats": {"batch": REPEATS_BATCH, "single": REPEATS_SINGLE, "warmup": WARMUP},
        "load_before": before,
        "load_after": after,
        "n_parameters": int(model.n_parameters),
    }


def main() -> None:
    require_clean_tree()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warmup_s = warm_up()
    plan = make_plan(False)
    config = load_config()
    data = generate(config, "P0", plan.pool_spec, SEED)
    hybrid = with_simulator_channels(data)
    n = plan.sizes[-1]

    results = []
    train, _, test = build_datasets(data, n)
    persistence = PersistenceModel()
    persistence.fit(train, seed=0)
    results.append(measure("persistence", persistence, test))

    arx = LinearARXModel()
    arx.fit(train, seed=SEED)
    results.append(measure("arx", arx, test))

    h_train, _, h_test = build_datasets(hybrid, n)
    simulator = SimulatorOnlyModel()
    simulator.fit(h_train, seed=0)
    results.append(measure("simulator_only", simulator, h_test))

    config_lstm = TorchSequenceConfig(core="lstm", max_epochs=TRAIN_EPOCHS, patience=0)
    scratch = TransferSequenceModel(config_lstm)
    scratch.fit(train, seed=SEED)
    results.append(measure("from_scratch", scratch, test))

    hybrid_model = TransferSequenceModel(config_lstm)
    hybrid_model.fit(h_train, seed=SEED)
    results.append(measure("hybrid", hybrid_model, h_test))

    finetuned = load_pretrained(CHECKPOINT)
    assert finetuned.pretrained_scalers is not None
    ft_train, _, ft_test = build_datasets_with_scalers(data, n, finetuned.pretrained_scalers)
    finetuned.fine_tune(
        ft_train,
        seed=SEED,
        learning_rate=FINETUNE_LEARNING_RATE,
        max_epochs=TRAIN_EPOCHS,
        freeze=FreezeSpec(),
    )
    results.append(measure("pretrain_finetune", finetuned, ft_test))

    write_artifact(
        OUTPUT_NAME,
        {
            "_provenance": provenance(),
            "design": {
                "threads": 1,
                "plant": "P0",
                "seed": SEED,
                "test_set": "excited test runs of the plant's own pool (data of the main run)",
                "split": SplitName.TEST.value,
                "n_train_runs_for_the_models": n,
                "models_trained_epochs": TRAIN_EPOCHS,
                "note": "weights do not change the cost of predict; accuracy of these models unused",
                "warmup_s": warmup_s,
                "torch": torch.__version__,
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
