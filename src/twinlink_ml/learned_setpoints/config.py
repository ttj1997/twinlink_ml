"""Settings of the learned-set-points study, ``conf/learned_setpoints/learned_setpoints.yaml``.

The study reads three existing configurations unchanged: the load-following study (task, truth,
days, margin search; Befund 126), and the two sensor-value studies (feedback grid, bias filter and
pre-roll of Befund 111; delay, error and gain of the laboratory sample of Befund 112). The
fingerprint covers the own YAML, the fingerprint of the load-following config and the feedback
values read, so that a change in any of them is visible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twinlink_ml.load_following.config import LoadFollowingConfig, load_load_following_config

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "REPO_ROOT",
    "FeedbackSettings",
    "LearnedSetpointsConfig",
    "PolicySettings",
    "TableSettings",
    "load_learned_setpoints_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "learned_setpoints" / "learned_setpoints.yaml"


@dataclass(frozen=True, slots=True)
class TableSettings:
    """Grid of the tabulated steady response and of the optimisation on it."""

    inlet_temperature_points: int
    gas_velocity_points: int
    spray_rate_points: int
    ambient_temperature_points: int
    relative_humidity_points: int
    fine_inlet_temperature_step_k: float
    fine_gas_velocity_step_m_per_s: float
    chunk_points: int


@dataclass(frozen=True, slots=True)
class FeedbackSettings:
    """Feedback grid, bias filter and pre-roll (Befund 111); laboratory sample (Befund 112)."""

    measurement_interval_s: float
    bias_time_constant_s: float
    pre_roll_s: float
    lab_sampling_interval_s: float
    lab_delay_s: float
    lab_error_std_kg_per_kg: float
    lab_gain: float
    noise_seed: int


@dataclass(frozen=True, slots=True)
class PolicySettings:
    """Family P: budgets, label margins and the settings of the three arms."""

    budgets_days: tuple[int, ...]
    seeds: tuple[int, ...]
    label_margins: tuple[float, ...]
    strata: int
    validation_fraction: float
    ridge: float
    hidden_size: int
    gru_hidden_size: int
    learning_rate: float
    max_epochs: int
    patience: int


@dataclass(frozen=True, slots=True)
class LearnedSetpointsConfig:
    """All settings of the study (see the YAML for reasons and placeholder status)."""

    load_following: LoadFollowingConfig
    table: TableSettings
    feedback: FeedbackSettings
    policy: PolicySettings
    n_train: tuple[int, ...]
    seeds: tuple[int, ...]
    max_epochs: int
    befund94_records: str
    process_fingerprint: str
    gru_mpc_n_train: tuple[int, ...]
    gru_mpc_fd_step: float
    gru_mpc_max_iterations: int
    gru_mpc_measurement_seed: int
    stage0_artifact: str
    main_artifact: str
    analysis_artifact: str
    twin_margin_grid: tuple[float, ...]
    short_horizon_intervals: int
    tuning_seed: int
    tuning_per_month_by_tercile: tuple[int, int, int]
    probe_days: int
    within_twin: float
    within_twin_every_seed: float
    admissible_seeds_minimum: int
    budget_wall_clock_h: float
    extended_wall_clock_h: float
    cut_order: tuple[str, ...]
    extrapolation_safety: float
    raw: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        """Hash of the own YAML, the load-following fingerprint and the feedback values read."""
        fb = self.feedback
        text = json.dumps(
            {
                "own": self.raw,
                "load_following": self.load_following.fingerprint,
                "feedback": [
                    fb.measurement_interval_s,
                    fb.bias_time_constant_s,
                    fb.pre_roll_s,
                    fb.lab_delay_s,
                    fb.lab_error_std_kg_per_kg,
                    fb.lab_gain,
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_learned_setpoints_config(
    path: Path | str = DEFAULT_CONFIG_PATH, *, day_hours: int | None = None
) -> LearnedSetpointsConfig:
    """Load the study settings; ``day_hours`` shortens the day (dry run and tests only)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError(f"unsupported learned-setpoints config version {raw.get('version')!r}")
    load_following = load_load_following_config(
        REPO_ROOT / raw["load_following_config"], day_hours=day_hours
    )
    sensor = yaml.safe_load((REPO_ROOT / raw["sensor_value_config"]).read_text(encoding="utf-8"))
    sensor_2 = yaml.safe_load(
        (REPO_ROOT / raw["sensor_value_2_config"]).read_text(encoding="utf-8")
    )
    loop, lab = sensor["loop"], sensor_2["F4"]
    ref, models, table, fb = raw["reference"], raw["models"], raw["table"], raw["feedback"]
    mpc, pol, days = raw["gru_mpc"], raw["policy"], raw["days"]
    reading, budget = raw["reading"], raw["budget"]
    a, b, c = (int(v) for v in days["tuning_per_month_by_tercile"])
    return LearnedSetpointsConfig(
        load_following=load_following,
        table=TableSettings(
            inlet_temperature_points=int(table["inlet_temperature_points"]),
            gas_velocity_points=int(table["gas_velocity_points"]),
            spray_rate_points=int(table["spray_rate_points"]),
            ambient_temperature_points=int(table["ambient_temperature_points"]),
            relative_humidity_points=int(table["relative_humidity_points"]),
            fine_inlet_temperature_step_k=float(table["fine_inlet_temperature_step_k"]),
            fine_gas_velocity_step_m_per_s=float(table["fine_gas_velocity_step_m_per_s"]),
            chunk_points=int(table["chunk_points"]),
        ),
        feedback=FeedbackSettings(
            measurement_interval_s=float(loop["measurement_interval_s"]),
            bias_time_constant_s=float(loop["bias_filter_time_constant_s"]),
            pre_roll_s=float(loop["pre_roll_s"]),
            lab_sampling_interval_s=3600.0 * float(fb["lab_sampling_interval_h"]),
            lab_delay_s=float(lab["delay_s"]),
            lab_error_std_kg_per_kg=float(lab["error_std_kg_per_kg"]),
            lab_gain=float(lab["gain"]),
            noise_seed=int(fb["noise_seed"]),
        ),
        policy=PolicySettings(
            budgets_days=tuple(int(k) for k in pol["budgets_days"]),
            seeds=tuple(int(s) for s in pol["seeds"]),
            label_margins=tuple(float(m) for m in pol["label_margins_kg_per_kg"]),
            strata=int(pol["strata"]),
            validation_fraction=float(pol["validation_fraction"]),
            ridge=float(pol["ridge"]),
            hidden_size=int(pol["hidden_size"]),
            gru_hidden_size=int(pol["gru_hidden_size"]),
            learning_rate=float(pol["learning_rate"]),
            max_epochs=int(pol["max_epochs"]),
            patience=int(pol["patience"]),
        ),
        n_train=tuple(int(n) for n in models["n_train"]),
        seeds=tuple(int(s) for s in models["seeds"]),
        max_epochs=int(models["max_epochs"]),
        befund94_records=str(models["befund94_records"]),
        process_fingerprint=str(models["process_fingerprint"]),
        gru_mpc_n_train=tuple(int(n) for n in mpc["n_train"]),
        gru_mpc_fd_step=float(mpc["finite_difference_step"]),
        gru_mpc_max_iterations=int(mpc["max_iterations"]),
        gru_mpc_measurement_seed=int(mpc["measurement_seed"]),
        stage0_artifact=str(ref["stage0_artifact"]),
        main_artifact=str(ref["main_artifact"]),
        analysis_artifact=str(ref["analysis_artifact"]),
        twin_margin_grid=tuple(float(m) for m in ref["twin_margin_grid"]),
        short_horizon_intervals=int(ref["short_horizon_intervals"]),
        tuning_seed=int(days["tuning_seed"]),
        tuning_per_month_by_tercile=(a, b, c),
        probe_days=int(days["probe_days"]),
        within_twin=float(reading["within_twin"]),
        within_twin_every_seed=float(reading["within_twin_every_seed"]),
        admissible_seeds_minimum=int(reading["admissible_seeds_minimum"]),
        budget_wall_clock_h=float(budget["wall_clock_h"]),
        extended_wall_clock_h=float(budget["extended_wall_clock_h"]),
        cut_order=tuple(str(c) for c in budget["cut_order"]),
        extrapolation_safety=float(budget["extrapolation_safety"]),
        raw=raw,
    )
