r"""Study settings and set-up of docs/39_hybrid_model.md (``conf/hybrid_model/hybrid_model.yaml``).

Configs of the study:

* **data config** -- ``conf/process/greybox_gab_placeholder.yaml`` with the gas-velocity projection
  enforced, as the Befund 94 pipeline (arm N) generates its data; the truth of a draw replaces the
  drawn material parameters.
* **steady config** -- the config of the bandit of Befund 52/96/113 (the same file, quality limit
  0.15 as ``product_moisture_limit_kg_per_kg``); truth, prior knowledge and refit replace material
  parameters only. ``wide`` additionally widens the ambient ranges for the extrapolation contexts
  (a); only the bracket of the inner bisection of the steady solver depends on them.

Draws: exactly the 20 draws of Befund 111 -- ``sensor_value.draws.parameter_draws`` with the
settings of ``conf/sensor_value/sensor_value.yaml`` and the nominal of its headroom study for the
validity check, as ``scripts/run_sensor_value.py`` calls it; the drawn VALUES are then applied to
the configs above.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config

from .grid_rto import BisectionSettings, GridSettings
from .refit import RefitSettings

__all__ = [
    "Case",
    "HybridStudyConfig",
    "StudySetup",
    "build_setup",
    "load_hybrid_config",
    "widen",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "hybrid_model" / "hybrid_model.yaml"


@dataclass(frozen=True, slots=True)
class HybridStudyConfig:
    """Settings of the study (``conf/hybrid_model/hybrid_model.yaml``)."""

    sensor_value_config: Path
    process_config: Path
    quality_limit_kg_per_kg: float
    master_seeds: tuple[int, ...]
    n_train: tuple[int, ...]
    max_epochs: int
    ensemble_members: int
    ensemble_seed_label: str
    refit: RefitSettings
    grid: GridSettings
    holding_share: float
    margin_constant: BisectionSettings
    margin_k: BisectionSettings
    outside_violation_limit: float
    coverage_threshold: float
    interior_grid_indices: tuple[tuple[int, ...], ...]
    extrapolation_contexts: tuple[tuple[float, float], ...]
    extrapolation_per_draw: tuple[tuple[int, ...], ...]
    widened_ambient_temperature_celsius: tuple[float, float]
    widened_ambient_relative_humidity: tuple[float, float]
    discretisation_n_cases: int
    discretisation_energy_threshold: float
    error_horizons: tuple[int, ...]
    holdout_seed: int
    raw: dict[str, Any]


def load_hybrid_config(path: Path | str = DEFAULT_CONFIG_PATH) -> HybridStudyConfig:
    """Read and check the study config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError("unknown config version")
    r, m, c = raw["refit"], raw["margin"], raw["cases"]
    config = HybridStudyConfig(
        sensor_value_config=REPO_ROOT / raw["sensor_value_config"],
        process_config=REPO_ROOT / raw["process_config"],
        quality_limit_kg_per_kg=float(raw["quality_limit_kg_per_kg"]),
        master_seeds=tuple(int(s) for s in raw["data"]["master_seeds"]),
        n_train=tuple(int(n) for n in raw["data"]["n_train"]),
        max_epochs=int(raw["data"]["max_epochs"]),
        ensemble_members=int(raw["models"]["ensemble_members"]),
        ensemble_seed_label=str(raw["models"]["ensemble_seed_label"]),
        refit=RefitSettings(
            parameters=tuple(r["parameters"]),
            lower=tuple(float(v) for v in r["lower"]),
            upper=tuple(float(v) for v in r["upper"]),
            max_nfev=int(r["max_nfev"]),
            diff_step=float(r["diff_step"]),
        ),
        grid=GridSettings(**{k: int(v) for k, v in raw["rto"].items()}),
        holding_share=float(m["holding_share"]),
        margin_constant=BisectionSettings(**{k: float(v) for k, v in m["constant"].items()}),
        margin_k=BisectionSettings(**{k: float(v) for k, v in m["ensemble_k"].items()}),
        outside_violation_limit=float(m["outside_violation_limit"]),
        coverage_threshold=float(raw["coverage"]["threshold"]),
        interior_grid_indices=tuple(
            tuple(int(i) for i in row) for row in c["interior_grid_indices"]
        ),
        extrapolation_contexts=tuple((float(a), float(b)) for a, b in c["extrapolation_contexts"]),
        extrapolation_per_draw=tuple(
            tuple(int(i) for i in row) for row in c["extrapolation_per_draw"]
        ),
        widened_ambient_temperature_celsius=tuple(c["widened_ambient_temperature_celsius"]),  # type: ignore[arg-type]
        widened_ambient_relative_humidity=tuple(c["widened_ambient_relative_humidity"]),  # type: ignore[arg-type]
        discretisation_n_cases=int(raw["discretisation_check"]["n_cases"]),
        discretisation_energy_threshold=float(
            raw["discretisation_check"]["energy_report_threshold"]
        ),
        error_horizons=tuple(int(h) for h in raw["errors"]["horizons"]),
        holdout_seed=int(raw["errors"]["holdout_seed"]),
        raw=raw,
    )
    if len(config.interior_grid_indices) != len(config.extrapolation_per_draw):
        raise ValueError("one interior and one extrapolation row per draw")
    return config


@dataclass(frozen=True, slots=True)
class Case:
    """One test case: a context of one draw. ``kind`` is ``interior`` or ``a`` (extrapolation)."""

    draw: int
    slot: int
    kind: str
    ambient_temperature_celsius: float
    ambient_relative_humidity: float
    source_index: int

    @property
    def wide(self) -> bool:
        return self.kind == "a"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def widen(
    config: ProcessConfig, t_range: tuple[float, float], phi_range: tuple[float, float]
) -> ProcessConfig:
    """``config`` with the ambient ranges replaced (extrapolation contexts (a))."""
    d = config.disturbances
    return dataclasses.replace(
        config,
        disturbances=dataclasses.replace(
            d,
            ambient_temperature_celsius=dataclasses.replace(
                d.ambient_temperature_celsius, minimum=float(t_range[0]), maximum=float(t_range[1])
            ),
            ambient_relative_humidity=dataclasses.replace(
                d.ambient_relative_humidity,
                minimum=float(phi_range[0]),
                maximum=float(phi_range[1]),
            ),
        ),
    )


@dataclass(frozen=True)
class StudySetup:
    """Configs, draws, cases, action box and fixed inputs of the study."""

    config: HybridStudyConfig
    data_nominal: ProcessConfig
    steady_nominal: ProcessConfig
    draws: tuple[dict[str, Any], ...]
    """``as_dict()`` of every draw (index, sequence position, unit, values)."""
    rejected_positions: tuple[int, ...]
    cases: tuple[Case, ...]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    nozzle_pressure_bar: float
    fixed_point: tuple[float, float]

    def draw_values(self, draw: int) -> dict[str, float]:
        return dict(self.draws[draw]["values"])

    def data_truth(self, draw: int) -> ProcessConfig:
        from twinlink_ml.sensor_value.draws import truth_config

        return truth_config(self.data_nominal, self.draw_values(draw))

    def steady_truth(self, draw: int) -> ProcessConfig:
        from twinlink_ml.sensor_value.draws import truth_config

        return truth_config(self.steady_nominal, self.draw_values(draw))

    def wide(self, config: ProcessConfig) -> ProcessConfig:
        return widen(
            config,
            self.config.widened_ambient_temperature_celsius,
            self.config.widened_ambient_relative_humidity,
        )

    def draw_cases(self, draw: int) -> list[Case]:
        return [c for c in self.cases if c.draw == draw]


def build_setup(config: HybridStudyConfig | None = None) -> StudySetup:
    """Configs, draws and cases of the study (reads the configs named in the study config)."""
    from twinlink_ml.headroom.config import load_headroom_config, study_process_config
    from twinlink_ml.rl.env import (
        FIXED_NOZZLE_PRESSURE_BAR,
        BanditEnvConfig,
        OperatingPointBanditEnv,
    )
    from twinlink_ml.rl.evaluation import load_test_points
    from twinlink_ml.sensitivity.config import load_sensitivity_config
    from twinlink_ml.sensor_value.config import load_sensor_value_config
    from twinlink_ml.sensor_value.draws import parameter_draws

    cfg = config if config is not None else load_hybrid_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        sv = load_sensor_value_config(cfg.sensor_value_config)
        headroom_nominal = study_process_config(load_headroom_config(sv.headroom_config))
        draws, rejected = parameter_draws(
            sv.draws, load_sensitivity_config(sv.sensitivity_config), headroom_nominal
        )
        data_nominal = load_process_config(cfg.process_config)
        data_nominal = dataclasses.replace(
            data_nominal,
            numerics=dataclasses.replace(
                data_nominal.numerics, enforce_gas_velocity_projection=True
            ),
        )
        env = OperatingPointBanditEnv(
            BanditEnvConfig(product_moisture_limit_kg_per_kg=cfg.quality_limit_kg_per_kg)
        )
    if len(draws) != len(cfg.interior_grid_indices):
        raise ValueError(f"{len(draws)} draws but {len(cfg.interior_grid_indices)} case rows")
    steady_nominal = env._base_config
    points = load_test_points(
        REPO_ROOT / "artifacts" / "runs" / "s05_limit015_oracle_and_fixed.json"
    )
    cases: list[Case] = []
    for d in range(len(draws)):
        slot = 0
        for index in cfg.interior_grid_indices[d]:
            p = points[index]
            cases.append(
                Case(
                    d,
                    slot,
                    "interior",
                    p.ambient_temperature_celsius,
                    p.ambient_relative_humidity,
                    index,
                )
            )
            slot += 1
        for index in cfg.extrapolation_per_draw[d]:
            t, phi = cfg.extrapolation_contexts[index]
            cases.append(Case(d, slot, "a", t, phi, index))
            slot += 1
    inlet = env._inlet_bound
    gas = env._gas_bound
    reference = REPO_ROOT / "artifacts" / "runs" / "s05_limit015_oracle_and_fixed.json"
    import json

    fixed = json.loads(reference.read_text(encoding="utf-8"))["fixed_point"]
    return StudySetup(
        config=cfg,
        data_nominal=data_nominal,
        steady_nominal=steady_nominal,
        draws=tuple(d.as_dict() for d in draws),
        rejected_positions=tuple(rejected),
        cases=tuple(cases),
        bounds=((inlet.minimum, inlet.maximum), (gas.minimum, gas.maximum)),
        nozzle_pressure_bar=FIXED_NOZZLE_PRESSURE_BAR,
        fixed_point=(
            float(fixed["inlet_temperature_celsius"]),
            float(fixed["gas_velocity_m_per_s"]),
        ),
    )


def case_arrays(cases: list[Case]) -> tuple[np.ndarray, np.ndarray]:
    """``contexts (c, 2)`` and ``wide (c,)`` of a list of cases."""
    contexts = np.array(
        [[c.ambient_temperature_celsius, c.ambient_relative_humidity] for c in cases]
    )
    return contexts, np.array([c.wide for c in cases])
