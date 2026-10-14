r"""Inference time per method incl. the cost of the simulator channel of the hybrid (A1, second measurement).

Repeats the measurement of ``scripts/measure_a1_inference.py`` (same data, same method set, same conditions: ONE
thread, warm-up excluded, machine load recorded) and adds what that measurement left out: the hybrid model
needs the open P0 simulation as an input channel, so in operation the simulator has to be run ahead over the
prediction horizon. Written before the run (committed before it is executed; clean tree).

Cost of the simulator channel (``GreyBoxFluidBedGranulator.rollout`` of the P0 simulator with the noise-free sensor
path, the config of the main run, the safety projection included)
-------------------------------------------------------------------------------------------------------------
* ``horizon_batch1``: the horizon of one prediction (36 steps of 25 s, 5 RK4 sub-steps each) for ONE window, from a
  state of a test run -- the real-time case; microseconds per prediction.
* ``horizon_batched``: the horizons of all test windows at once (the vectorised case, as the evaluator calls a
  model); microseconds per window.
* ``one_step_batch1``: one sample step, the incremental cost when the simulator runs along with the plant.
* ``full_run_batch1``: a whole 8 h run (1152 steps), the cost when the channel is computed for the planned inputs
  of the whole run in advance; also per prediction if a prediction is made at every sample.
* ``steady_state_start``: the steady-state solve at the start of one run (once per run).
The TOTAL cost of the hybrid per prediction (real time) is the LSTM latency (batch of 1) plus the horizon
simulation of one window; the amortised total uses the batched figures. Everything is reported next to the other
methods so that the totals can be compared.

``--require-idle`` aborts unless no other process loads the machine (machine load <= 5 % and no other Python process
above 5 % of one core); the artifact name then is ``a1_inference_time_v2_idle``, otherwise
``a1_inference_time_v2_background`` (the first one, taken while a foreign process ran, as in the first measurement).

    tools/run_here.sh scripts/measure_a1_inference_v2.py                 # same conditions as the first measurement
    tools/run_here.sh scripts/measure_a1_inference_v2.py --require-idle  # only on a free machine
"""

from __future__ import annotations

# ruff: noqa: E402 -- the thread variables must be set before numpy / torch are imported
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

import measure_a1_inference as first
from _common import provenance, require_clean_tree, write_artifact
from run_a1_main import FINETUNE_LEARNING_RATE, make_plan
from run_a1_pilot_pp import generate, load_config, warm_up

from twinlink_ml.evaluation.experiments import build_datasets
from twinlink_ml.models import hybrid
from twinlink_ml.models.baselines import LinearARXModel, PersistenceModel
from twinlink_ml.models.hybrid import SimulatorOnlyModel, with_simulator_channels
from twinlink_ml.models.torch_sequence import TorchSequenceConfig
from twinlink_ml.models.transfer import (
    FreezeSpec,
    TransferSequenceModel,
    build_datasets_with_scalers,
    load_pretrained,
)
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator

IDLE_MACHINE_LOAD_PCT = 5.0
IDLE_PROCESS_CPU_PCT = 5
REPEATS = {"horizon_batch1": 20, "horizon_batched": 5, "one_step_batch1": 300, "full_run_batch1": 3}
WARMUP = 2
SEED = 1


def timing(call: Any, repeats: int) -> dict[str, float]:
    for _ in range(WARMUP):
        call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append(time.perf_counter() - started)
    return {"median_s": float(np.median(times)), "min_s": float(min(times)), "repeats": repeats}


def simulation_costs(data: Any, plan: Any) -> dict[str, Any]:
    """Costs of the simulator channel, see the module docstring."""
    import warnings

    base = data.base_config
    quiet = dataclasses.replace(
        base,
        measurement=dataclasses.replace(
            base.measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        simulator = GreyBoxFluidBedGranulator(quiet)
    backend = simulator.backend
    trajectory = data.trajectory
    horizon = plan.pool_spec.horizon
    history = plan.pool_spec.history_length
    set_points = (
        trajectory.requested_actions
        if trajectory.requested_actions is not None
        else trajectory.actions
    )
    test_runs = [trajectory.run_id.index(r) for r in data.split.test]
    run = test_runs[0]
    start = history  # a window start inside the run
    state = backend.asarray(np.asarray(trajectory.states[run : run + 1, start - 1, :]))
    actions = np.asarray(set_points[run : run + 1, start : start + horizon, :])
    disturbances = np.asarray(trajectory.disturbances[run : run + 1, start : start + horizon, :])

    # all test windows of the pool (the windows start every ``stride`` samples)
    stride = plan.pool_spec.stride
    n_steps = trajectory.n_steps
    starts = list(range(history, n_steps - horizon + 1, stride))
    rows = [(r, s) for r in test_runs for s in starts]
    states_all = backend.asarray(
        np.stack([trajectory.states[r, s - 1, :] for r, s in rows], axis=0)
    )
    actions_all = np.stack([set_points[r, s : s + horizon, :] for r, s in rows], axis=0)
    disturbances_all = np.stack(
        [trajectory.disturbances[r, s : s + horizon, :] for r, s in rows], axis=0
    )

    out: dict[str, Any] = {}
    before = first.machine_load()
    h1 = timing(
        lambda: simulator.rollout(actions, disturbances, seed=0, initial_state=state),
        REPEATS["horizon_batch1"],
    )
    out["horizon_batch1"] = {
        "us_per_prediction_median": 1e6 * h1["median_s"],
        "us_per_prediction_min": 1e6 * h1["min_s"],
        "steps": horizon,
        "repeats": h1["repeats"],
    }
    hb = timing(
        lambda: simulator.rollout(actions_all, disturbances_all, seed=0, initial_state=states_all),
        REPEATS["horizon_batched"],
    )
    out["horizon_batched"] = {
        "n_windows": len(rows),
        "us_per_window_median": 1e6 * hb["median_s"] / len(rows),
        "us_per_window_min": 1e6 * hb["min_s"] / len(rows),
        "repeats": hb["repeats"],
    }
    step = timing(
        lambda: simulator.rollout(
            actions[:, :1, :], disturbances[:, :1, :], seed=0, initial_state=state
        ),
        REPEATS["one_step_batch1"],
    )
    out["one_step_batch1"] = {
        "us_median": 1e6 * step["median_s"],
        "us_min": 1e6 * step["min_s"],
        "repeats": step["repeats"],
        "note": "rollout of a single step, includes the rollout bookkeeping",
    }
    full_actions = np.asarray(set_points[run : run + 1, :, :])
    full_disturbances = np.asarray(trajectory.disturbances[run : run + 1, :, :])
    full = timing(
        lambda: simulator.rollout(full_actions, full_disturbances, seed=0),
        REPEATS["full_run_batch1"],
    )
    out["full_run_batch1"] = {
        "steps": int(n_steps),
        "s_median": full["median_s"],
        "s_min": full["min_s"],
        "us_per_prediction_if_one_prediction_per_sample_median": 1e6 * full["median_s"] / n_steps,
        "repeats": full["repeats"],
    }
    solve_times = []
    for r in test_runs[:5]:
        t0 = time.perf_counter()
        hybrid.stationary_initial_states(
            base, set_points[r : r + 1, 0, :], trajectory.disturbances[r : r + 1, 0, :]
        )
        solve_times.append(time.perf_counter() - t0)
    out["steady_state_start"] = {
        "s_median_per_run": float(np.median(solve_times)),
        "s_all": solve_times,
        "note": "once per run, includes the construction of the solver simulator",
    }
    out["load_before"] = before
    out["load_after"] = first.machine_load()
    return out


def is_idle(load: dict[str, Any]) -> bool:
    busy = load["other_python_processes_cpu_pct_of_one_core"]
    return bool(
        load["machine_load_pct"] <= IDLE_MACHINE_LOAD_PCT
        and not any(v > IDLE_PROCESS_CPU_PCT for v in busy.values())
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-idle", action="store_true")
    arguments = parser.parse_args()
    require_clean_tree()
    start_load = first.machine_load()
    if arguments.require_idle and not is_idle(start_load):
        print("machine is not idle:", start_load)
        return 4
    name = (
        "a1_inference_time_v2_idle" if arguments.require_idle else "a1_inference_time_v2_background"
    )

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warmup_s = warm_up()
    plan = make_plan(False)
    config = load_config()
    data = generate(config, "P0", plan.pool_spec, SEED)
    hybrid_data = with_simulator_channels(data)
    n = plan.sizes[-1]

    results = []
    train, _, test = build_datasets(data, n)
    persistence = PersistenceModel()
    persistence.fit(train, seed=0)
    results.append(first.measure("persistence", persistence, test))
    arx = LinearARXModel()
    arx.fit(train, seed=SEED)
    results.append(first.measure("arx", arx, test))
    h_train, _, h_test = build_datasets(hybrid_data, n)
    simulator_only = SimulatorOnlyModel()
    simulator_only.fit(h_train, seed=0)
    results.append(first.measure("simulator_only", simulator_only, h_test))
    cfg = TorchSequenceConfig(core="lstm", max_epochs=first.TRAIN_EPOCHS, patience=0)
    scratch = TransferSequenceModel(cfg)
    scratch.fit(train, seed=SEED)
    results.append(first.measure("from_scratch", scratch, test))
    hybrid_model = TransferSequenceModel(cfg)
    hybrid_model.fit(h_train, seed=SEED)
    results.append(first.measure("hybrid_network_only", hybrid_model, h_test))
    tuned = load_pretrained(first.CHECKPOINT)
    assert tuned.pretrained_scalers is not None
    ft_train, _, ft_test = build_datasets_with_scalers(data, n, tuned.pretrained_scalers)
    tuned.fine_tune(
        ft_train,
        seed=SEED,
        learning_rate=FINETUNE_LEARNING_RATE,
        max_epochs=first.TRAIN_EPOCHS,
        freeze=FreezeSpec(),
    )
    results.append(first.measure("pretrain_finetune", tuned, ft_test))

    simulation = simulation_costs(data, plan)
    network = next(r for r in results if r["method"] == "hybrid_network_only")
    total = {
        "realtime_batch1_us_per_prediction": network["latency_batch1_us_median"]
        + simulation["horizon_batch1"]["us_per_prediction_median"],
        "network_part_us": network["latency_batch1_us_median"],
        "simulation_part_us": simulation["horizon_batch1"]["us_per_prediction_median"],
        "simulation_share": simulation["horizon_batch1"]["us_per_prediction_median"]
        / (
            network["latency_batch1_us_median"]
            + simulation["horizon_batch1"]["us_per_prediction_median"]
        ),
        "batched_us_per_window": network["throughput_us_per_window_median"]
        + simulation["horizon_batched"]["us_per_window_median"],
        "network_batched_part_us": network["throughput_us_per_window_median"],
        "simulation_batched_part_us": simulation["horizon_batched"]["us_per_window_median"],
        "channel_precomputed_per_run_us_per_prediction": network["latency_batch1_us_median"]
        + simulation["full_run_batch1"]["us_per_prediction_if_one_prediction_per_sample_median"],
        "note": (
            "total of the hybrid = network + simulator channel; the 'hybrid' figure of the first measurement "
            "(network only) does not contain the simulator"
        ),
    }
    write_artifact(
        name,
        {
            "_provenance": provenance(),
            "design": {
                "threads": 1,
                "require_idle": arguments.require_idle,
                "idle_criterion": f"machine load <= {IDLE_MACHINE_LOAD_PCT} % and no other python process > "
                f"{IDLE_PROCESS_CPU_PCT} % of one core",
                "load_at_start": start_load,
                "plant": "P0",
                "seed": SEED,
                "models_trained_epochs": first.TRAIN_EPOCHS,
                "note": "weights do not change the cost of predict; accuracy of these models unused",
                "warmup_s": warmup_s,
                "torch": torch.__version__,
            },
            "results": results,
            "simulator_channel_costs": simulation,
            "hybrid_total": total,
        },
    )
    print(
        "hybrid total (real time, batch 1): "
        f"{total['realtime_batch1_us_per_prediction']:.0f} us "
        f"(network {total['network_part_us']:.0f}, simulation {total['simulation_part_us']:.0f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
