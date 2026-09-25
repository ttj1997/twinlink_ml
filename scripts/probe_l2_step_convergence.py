r"""L2: Schrittweitenkonvergenz (``numerics.substeps`` 5 -> 10 -> 20) am RK4-Integrator.

KEIN Netztraining. Ruft nur bestehenden Code auf (``GreyBoxFluidBedGranulator.rollout``,
unveraendert). KEINE Aenderung an ``src/``.

Frage: ist die Schrittweite ``substeps=5`` (Standard in ``greybox_gab_placeholder.yaml``)
fein genug, oder aendert sich die Trajektorie relevant bei Verfeinerung? Besonders
interessant sind Laeufe mit hohen ``bed_temperature_celsius``-Spitzen, weil dort das
Sicherheitsnetz aus ``greybox.py`` (Zustandsboden, Knick der Verdampfungsrate am
Klippungspunkt, siehe ``thermo.evaporation_rate_kg_per_s``) am ehesten greift und
RK4 nicht ueberall differenzierbare rechte Seiten integriert.

Vorgehen: ein Standard-Anregungspool (128 Laeufe, ``DataSpec``-Default, Δt=25s,
Trainingspool-Seed 1, wie in ``probe_e2_e3_e4_lag_depth.py``) bei ``substeps=5``
simulieren, je die drei Laeufe mit der hoechsten und die drei mit der niedrigsten
``bed_temperature_celsius``-Spitze auswaehlen ("heiss"/"Kontrolle"), und jeden dieser
sechs Laeufe -- gleiche Anregung, gleicher Anfangszustand -- bei ``substeps`` in
{5, 10, 20} erneut integrieren. Verglichen wird die wahre, unverrauschte Betttemperatur
(``trajectory.outputs[..., 1]``, siehe ``greybox.step`` -- dort ohne Sensorfilter/Rauschen
gebildet, im Gegensatz zu ``trajectory.measurements``), nicht die Messung: das
Sensormodell ueberlagert mit 0.2 K absolutem Rauschen, das die hier gesuchten
Integrationsfehler (< 0.002 K) um zwei Grossenordnungen verdecken wuerde.

Berichtet wird je Lauf die maximale absolute Abweichung der Trajektorie zwischen
5 und 10 sowie zwischen 10 und 20 Substeps.
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
