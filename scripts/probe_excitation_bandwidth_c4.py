r"""Probe C4: V0 mit Δt = 5 s, getrennt von C0-C3 (Δt = 25 s) gehalten.

KEIN Netztraining, KEINE Aenderung an ``src/``. Reine Zusatzauswertung zu
``scripts/probe_excitation_bandwidth.py`` (dort C0-C3, unveraendert). Zwei Werte:

  (i)  R2_stat/R2_lag1 auf dem nativen 5-s-Raster (andere Stichprobenzahl,
       andere Rauschmittelung -- NICHT mit den 25-s-Werten vergleichbar).
  (ii) dieselben Kennzahlen nach Unterabtastung des 5-s-Datensatzes zurueck auf 25 s
       (jeder fuenfte Punkt, beginnend bei Index 4) -- das ist die mit V0-V3
       vergleichbare Zahl.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np
from probe_excitation_bandwidth import (  # type: ignore[import-not-found]
    _aggregate_lag1,
    static_explained_variance_with_lag1,
)

from twinlink_ml.simulation import (
    ExcitationConfig,
    GreyBoxFluidBedGranulator,
    MixedExcitationDesign,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.utils.seeding import SeedBundle

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_GAB = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"

N_RUNS = 32
RUN_DURATION_S = 28_800.0
MASTER_SEEDS = (20260812, 20260813, 20260814)
FINE_DT_S = 5.0
COARSE_DT_S = 25.0
DOWNSAMPLE_FACTOR = round(COARSE_DT_S / FINE_DT_S)


def _load(config_path: Path, *, sample_time_s: float) -> GreyBoxFluidBedGranulator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(config_path)
        numerics = dataclasses.replace(base.numerics, sample_time_s=sample_time_s, substeps=5)
        config = dataclasses.replace(base, numerics=numerics)
    return GreyBoxFluidBedGranulator(config)


def _downsample(trajectory: Trajectory, factor: int) -> Trajectory:
    start = factor - 1
    idx = np.arange(start, trajectory.n_steps, factor)
    return Trajectory(
        run_id=trajectory.run_id,
        time_s=trajectory.time_s[idx],
        actions=trajectory.actions[:, idx, :],
        disturbances=trajectory.disturbances[:, idx, :],
        states=trajectory.states[:, idx, :],
        outputs=trajectory.outputs[:, idx, :],
        measurements=trajectory.measurements[:, idx, :],
        diagnostics={k: v[:, idx, ...] for k, v in trajectory.diagnostics.items()},
        metadata=dict(trajectory.metadata),
    )


def main() -> None:
    t0 = time.monotonic()
    simulator = _load(CONFIG_GAB, sample_time_s=FINE_DT_S)
    config = simulator.config
    print(f"config fingerprint (dt=5s): {config.fingerprint}")
    print(f"n_steps native: {round(RUN_DURATION_S / FINE_DT_S)}")

    excitation_config = ExcitationConfig(
        run_duration_s=RUN_DURATION_S, randomise_initial_conditions=True
    )

    native_lag1 = []
    downsampled_lag1 = []
    for master_seed in MASTER_SEEDS:
        t_start = time.monotonic()
        seeds = SeedBundle.from_master(master_seed)
        design = MixedExcitationDesign(
            excitation_config,
            config.actions,
            config.disturbances,
            sample_time_s=config.numerics.sample_time_s,
            geometry=config.geometry,
            material=config.material,
            operation=config.operation,
        )
        schedule = design.build(N_RUNS, seed=seeds.data_seed)
        initial = None
        if schedule.initial_states is not None:
            columns = schedule.initial_states
            initial = simulator.state_from_physical(
                columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
            )
        trajectory = simulator.rollout(
            schedule.actions,
            schedule.disturbances,
            run_id=schedule.run_id,
            seed=seeds.noise_seed,
            initial_state=initial,
        )
        rollout_elapsed = time.monotonic() - t_start
        print(f"  seed={master_seed}: rollout {rollout_elapsed:.1f}s, n_steps={trajectory.n_steps}")

        native_lag1.append(static_explained_variance_with_lag1(trajectory))
        coarse = _downsample(trajectory, DOWNSAMPLE_FACTOR)
        downsampled_lag1.append(static_explained_variance_with_lag1(coarse))

    result = {
        "config_fingerprint": config.fingerprint,
        "fine_dt_s": FINE_DT_S,
        "coarse_dt_s": COARSE_DT_S,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "i_native_5s": _aggregate_lag1(native_lag1),
        "ii_downsampled_to_25s": _aggregate_lag1(downsampled_lag1),
        "_elapsed_s": time.monotonic() - t0,
    }
    print(json.dumps(result, indent=2))

    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_excitation_bandwidth_c4.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ngeschrieben: {out_path} ({result['_elapsed_s']:.1f} s)")


if __name__ == "__main__":
    main()
