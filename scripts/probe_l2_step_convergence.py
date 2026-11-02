r"""L2: step-size convergence (``numerics.substeps`` 5 -> 10 -> 20) of the RK4 integrator.

NO network training. Only calls existing code (``GreyBoxFluidBedGranulator.rollout``,
unchanged). NO change to ``src/``.

Question: is the step size ``substeps=5`` (default in ``greybox_gab_placeholder.yaml``)
fine enough, or does the trajectory change relevantly on refinement? Of particular
interest are runs with high ``bed_temperature_celsius`` peaks, because there the
safety net from ``greybox.py`` (state floor, kink of the evaporation rate at the
clipping point, see ``thermo.evaporation_rate_kg_per_s``) is most likely to act and
RK4 integrates right-hand sides that are not differentiable everywhere.

Procedure: simulate a standard excitation pool (128 runs, ``DataSpec`` default, Δt=25s,
training pool seed 1, as in ``probe_e2_e3_e4_lag_depth.py``) at ``substeps=5``,
select the three runs with the highest and the three with the lowest
``bed_temperature_celsius`` peak ("hot"/"control"), and integrate each of these
six runs again -- same excitation, same initial state -- at ``substeps`` in
{5, 10, 20}. Compared is the true, noise-free bed temperature
(``trajectory.outputs[..., 1]``, see ``greybox.step`` -- formed there without sensor
filter/noise, in contrast to ``trajectory.measurements``), not the measurement: the
sensor model superimposes 0.2 K absolute noise, which would mask the integration errors
sought here (< 0.002 K) by two orders of magnitude.

Reported per run is the maximum absolute deviation of the trajectory between
5 and 10 and between 10 and 20 substeps.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import DataSpec
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.excitation import ExcitationConfig, MixedExcitationDesign
from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator
from twinlink_ml.utils.seeding import SeedBundle

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
POOL_MASTER_SEED = 1
SUBSTEPS_VALUES = (5, 10, 20)
N_SELECTED_PER_GROUP = 3


def _load_config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


def _simulator(config: ProcessConfig, *, substeps: int) -> GreyBoxFluidBedGranulator:
    numerics = dataclasses.replace(config.numerics, substeps=substeps)
    return GreyBoxFluidBedGranulator(dataclasses.replace(config, numerics=numerics))


def main() -> None:
    t0 = time.monotonic()
    base_config = _load_config()
    spec = DataSpec(
        sample_time_s=25.0, run_duration_s=28_800.0, n_train_max=64, n_val=32, n_test=32
    )
    seeds = SeedBundle.from_master(POOL_MASTER_SEED)

    config = dataclasses.replace(
        base_config,
        numerics=dataclasses.replace(base_config.numerics, sample_time_s=spec.sample_time_s),
    )
    excitation_config = ExcitationConfig(
        run_duration_s=spec.run_duration_s, constant_run_fraction=spec.constant_run_fraction
    )
    design = MixedExcitationDesign(
        excitation_config,
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    schedule = design.build(spec.pool_size, seed=seeds.data_seed)
    assert schedule.initial_states is not None

    sim_baseline = _simulator(config, substeps=5)
    initial_baseline = sim_baseline.state_from_physical(
        schedule.initial_states[:, 0],
        schedule.initial_states[:, 1],
        schedule.initial_states[:, 2],
        schedule.initial_states[:, 3],
    )
    baseline_trajectory = sim_baseline.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial_baseline,
    )
    output_names = baseline_trajectory.metadata["output_names"]
    bed_temp_idx = output_names.index("bed_temperature_celsius")
    peaks = baseline_trajectory.outputs[:, :, bed_temp_idx].max(axis=1)
    order = np.argsort(peaks)
    selected = {
        "heiss": order[-N_SELECTED_PER_GROUP:][::-1],
        "kontrolle": order[:N_SELECTED_PER_GROUP],
    }

    results: dict[str, list[dict]] = {}
    for group, indices in selected.items():
        results[group] = []
        for index in indices:
            index = int(index)
            action = schedule.actions[index : index + 1]
            disturbance = schedule.disturbances[index : index + 1]
            initial_physical = schedule.initial_states[index : index + 1]
            run_id = schedule.run_id[index : index + 1]

            traces: dict[int, np.ndarray] = {}
            for substeps in SUBSTEPS_VALUES:
                sim = _simulator(config, substeps=substeps)
                initial = sim.state_from_physical(
                    initial_physical[:, 0],
                    initial_physical[:, 1],
                    initial_physical[:, 2],
                    initial_physical[:, 3],
                )
                trajectory = sim.rollout(
                    action, disturbance, run_id=run_id, seed=seeds.noise_seed, initial_state=initial
                )
                traces[substeps] = trajectory.outputs[0, :, bed_temp_idx]

            max_dev_5_10 = float(np.max(np.abs(traces[10] - traces[5])))
            max_dev_10_20 = float(np.max(np.abs(traces[20] - traces[10])))
            results[group].append(
                {
                    "run_index": index,
                    "peak_bed_temperature_celsius": float(peaks[index]),
                    "max_deviation_5_to_10_substeps_k": max_dev_5_10,
                    "max_deviation_10_to_20_substeps_k": max_dev_10_20,
                }
            )
            print(
                f"{group} run={index}: peak={peaks[index]:6.2f} C, "
                f"|dev(5,10)|={max_dev_5_10:.6f} K, |dev(10,20)|={max_dev_10_20:.6f} K"
            )

    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_l2_step_convergence.json"
    out_path.write_text(
        json.dumps({"results": results, "_elapsed_s": time.monotonic() - t0}, indent=2),
        encoding="utf-8",
    )
    print(f"geschrieben: {out_path} ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
