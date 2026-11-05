"""Units of the static transfer study: pretraining, pilot cells, main units.

* :func:`pretraining_task` -- one pretrained MLP per seed on balance points of the pretraining box
  (admissible for the SOURCE only: the plants are unknown to it).
* :func:`pilot_task` -- one (method, setting) on the pilot plant PP-st: every pilot seed and pilot
  budget, scored on the pilot's own validation set (512 points in the interpolation box, own
  scramble, never a test set of the main run).
* :func:`run_unit` -- one (plant, condition, seed) of the main run: all methods at every N, scored
  against the noise-free plant truth (at the TRUE inputs) on the four named test sets.

Workers get a :class:`StudyContext` built once in the parent (test sets, designs, models).
"""

from __future__ import annotations

import ctypes
import itertools
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .balance import DryerPlantModel, plant_model, source_model
from .config import TARGET_NAMES, StaticTransferConfig
from .design import (
    TEST_SET_NAMES,
    AdmissibleSample,
    admissible_points,
    input_offsets,
    noise_draws,
    noisy,
    test_sets,
    training_design,
)
from .surrogates import (
    BalanceSurrogate,
    BiasCorrectedBalance,
    CalibratedBalance,
    FineTunedMlp,
    LinearRidge,
    MlpSettings,
    PretrainedMlp,
    ResidualMlp,
    ScratchMlp,
    StaticSurrogate,
    TargetScaling,
    make_net,
    pretrain,
)

__all__ = [
    "FITTED_METHODS",
    "FIXED_METHODS",
    "StudyContext",
    "build_context",
    "criterion_scale",
    "errors",
    "extend_grid",
    "grid_settings",
    "peak_memory_mb",
    "pilot_task",
    "pretraining_task",
    "run_unit",
    "select_setting",
]

FIXED_METHODS: tuple[str, ...] = ("L0", "L0_v412", "L3_0")
"""Methods without plant data (scored once per unit)."""
FITTED_METHODS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "B1", "B2")
"""Methods fitted at every N."""


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (Windows: peak working set)."""
    if os.name == "nt":

        class _Counters(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 - ctypes layout
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        ok = get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return float(counters.PeakWorkingSetSize) / 1e6 if ok else math.nan
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined,unused-ignore]
    return float(usage.ru_maxrss) / 1e3


# ------------------------------------------------------------------------------ context
@dataclass(frozen=True)
class StudyContext:
    """Everything a worker needs that is the same for every unit."""

    config: StaticTransferConfig
    tests: dict[str, AdmissibleSample]
    pilot_validation: AdmissibleSample
    designs: dict[int, AdmissibleSample]
    """Training design per seed (main and pilot seeds)."""


def admissibility_models(config: StaticTransferConfig) -> list[DryerPlantModel]:
    """Source, every main plant and the pilot plant (test sets are shared by all)."""
    models = [source_model(config.source)]
    models += [plant_model(config.source, p) for p in (*config.plants, config.pilot_plant)]
    return models


def build_context(
    config: StaticTransferConfig, seeds: Sequence[int], test_count: int | None = None
) -> StudyContext:
    """Test sets, pilot validation set and the training designs of ``seeds`` (built once)."""
    models = admissibility_models(config)
    n = config.test_count if test_count is None else test_count
    return StudyContext(
        config=config,
        tests=test_sets(config, models, n),
        pilot_validation=admissible_points(
            config.boxes["interpolation"],
            n,
            config.test_seed + 11,
            models,
            config.max_relative_humidity,
        ),
        designs={s: training_design(config, s, models) for s in seeds},
    )


def scaling(config: StaticTransferConfig) -> TargetScaling:
    return TargetScaling(config.boxes["pretraining"], config.target_centre, config.target_scale)


# ------------------------------------------------------------------------------ metrics
def errors(pred: np.ndarray, truth: np.ndarray) -> dict[str, list[float]]:
    """RMSE, largest absolute error and mean error (prediction - truth) per target."""
    e = np.asarray(pred) - np.asarray(truth)
    return {
        "rmse": np.sqrt(np.mean(e**2, axis=0)).tolist(),
        "max_abs": np.max(np.abs(e), axis=0).tolist(),
        "bias": np.mean(e, axis=0).tolist(),
    }


def criterion_scale(config: StaticTransferConfig, truth: np.ndarray) -> np.ndarray:
    """Primary noise per target on a point set (power: relative noise x mean truth)."""
    n = config.noise
    return np.asarray(
        [
            n.outlet_temperature_k,
            n.outlet_humidity_g_per_kg,
            n.heater_power_relative * float(np.mean(np.abs(truth[:, 2]))),
        ]
    )


# ------------------------------------------------------------------------------ pretraining
def pretraining_task(
    config: StaticTransferConfig, seed: int, *, smoke: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Pretrain one MLP on balance points (source-admissible) of the pretraining box."""
    t0 = time.perf_counter()
    pre = config.pretraining
    source = source_model(config.source)
    n_train = 256 if smoke else pre.n_train
    n_val = 64 if smoke else pre.n_validation
    box = config.boxes["pretraining"]
    rh = config.max_relative_humidity
    train = admissible_points(box, n_train, pre.seed_offset + seed, [source], rh)
    val = admissible_points(box, n_val, pre.seed_offset + 100 + seed, [source], rh)
    net, record = pretrain(
        train.x,
        source.predict(train.x),
        val.x,
        source.predict(val.x),
        scaling(config),
        config.hidden,
        learning_rate=pre.learning_rate,
        batch_size=pre.batch_size,
        max_epochs=2 if smoke else pre.max_epochs,
        patience=pre.patience,
        seed=pre.seed_offset + seed,
    )
    record.update(
        {
            "seed": seed,
            "n_train": n_train,
            "n_validation": n_val,
            "train_candidates": train.candidates,
            "train_admissible_share": train.admissible / train.candidates,
            "seconds": time.perf_counter() - t0,
        }
    )
    return net, record


# ------------------------------------------------------------------------------ methods
def mlp_settings(values: Mapping[str, float], penalty_key: str) -> MlpSettings:
    return MlpSettings(
        learning_rate=float(values["learning_rate"]),
        steps=int(values["steps"]),
        penalty=float(values[penalty_key]),
    )


def make_method(
    name: str,
    config: StaticTransferConfig,
    settings: Mapping[str, float],
    pretrained: Any,
    seed: int,
    n: int,
) -> StaticSurrogate:
    """One fitted method with its setting; the net seed depends on (seed, method, N)."""
    source = source_model(config.source)
    net_seed = 100_003 * seed + 1009 * FITTED_METHODS.index(name) + n
    if name == "L1":
        return CalibratedBalance(
            config.source,
            config.calibration_parameters,
            config.noise,
            float(settings["prior_scale"]),
        )
    if name == "L2":
        return ResidualMlp(
            source,
            config.boxes["pretraining"],
            config.residual_scale,
            config.hidden,
            mlp_settings(settings, "weight_decay"),
            net_seed,
        )
    if name == "L3":
        return FineTunedMlp(scaling(config), pretrained, mlp_settings(settings, "sp_lambda"))
    if name == "L4":
        return ScratchMlp(
            scaling(config), config.hidden, mlp_settings(settings, "weight_decay"), net_seed
        )
    if name == "B1":
        return BiasCorrectedBalance(source)
    if name == "B2":
        return LinearRidge(config.boxes["pretraining"], config.ridge_alphas)
    raise ValueError(name)


def _plant_data(
    config: StaticTransferConfig,
    plant: DryerPlantModel,
    condition: str,
    seed: int,
    design: np.ndarray,
) -> tuple[np.ndarray, Any]:
    cond = config.condition(condition)
    offsets = input_offsets(config.input_errors, seed, cond.input_errors)
    truth = plant.predict(offsets.true_inputs(design))
    y = noisy(truth, config.noise_for(cond), noise_draws(seed, len(design)))
    return y, offsets


# ------------------------------------------------------------------------------ pilot
def grid_settings(grid: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    """Every combination of a grid (keys sorted)."""
    keys = sorted(grid)
    return [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))
    ]


def pilot_task(
    ctx: StudyContext,
    method: str,
    setting: Mapping[str, float],
    pretrained: Mapping[int, Any],
    *,
    steps_cap: int | None = None,
) -> dict[str, Any]:
    """One setting on the pilot plant: RMSE per (seed, N) on the pilot validation set."""
    config = ctx.config
    pilot_cfg = config.raw["pilot"]
    plant = plant_model(config.source, config.pilot_plant)
    x_val = ctx.pilot_validation.x
    truth_val = plant.predict(x_val)
    scale = criterion_scale(config, truth_val)
    t0 = time.perf_counter()
    cells = []
    for seed in pilot_cfg["seeds"]:
        design = ctx.designs[int(seed)].x
        y, _ = _plant_data(config, plant, "clean_low", int(seed), design)
        for n in pilot_cfg["budgets"]:
            eff = dict(setting)
            if steps_cap is not None and "steps" in eff:
                eff["steps"] = min(int(eff["steps"]), steps_cap)
            model = make_method(method, config, eff, pretrained.get(int(seed)), int(seed), int(n))
            model.fit(design[: int(n)], y[: int(n)])
            err = errors(model.predict(x_val), truth_val)
            cells.append(
                {
                    "seed": int(seed),
                    "n": int(n),
                    **err,
                    "criterion": float(np.mean(np.asarray(err["rmse"]) / scale)),
                }
            )
    return {
        "method": method,
        "setting": dict(setting),
        "cells": cells,
        "criterion": float(np.mean([c["criterion"] for c in cells])),
        "seconds": time.perf_counter() - t0,
    }


def select_setting(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Lowest mean criterion C; ties broken by the order of the results (first wins)."""
    return min(results, key=lambda r: r["criterion"])


def _round_like(value: float, like: Sequence[float]) -> float:
    if all(float(v).is_integer() for v in like):
        return float(max(1, round(value)))
    return float(value)


def extend_grid(
    grid: Mapping[str, Sequence[float]],
    best: Mapping[str, float],
    extensions: Mapping[str, int],
    max_extensions: int,
) -> tuple[dict[str, list[float]], dict[str, int], list[str]]:
    """Extend every dimension whose selected value lies at its edge (geometric continuation).

    A dimension with one value, one that has used ``max_extensions``, or an edge at 0 (a natural
    bound, e.g. no L2-SP penalty) is not extended. Returns the new grid, the extension counts and
    the names of the extended dimensions (empty: the selection is final).
    """
    new = {k: sorted(float(v) for v in vals) for k, vals in grid.items()}
    counts = dict(extensions)
    extended: list[str] = []
    for key, vals in new.items():
        if len(vals) < 2 or counts.get(key, 0) >= max_extensions:
            continue
        chosen = float(best[key])
        if chosen == vals[0] and vals[0] > 0.0 and vals[1] > 0.0:
            vals.insert(0, _round_like(vals[0] ** 2 / vals[1], vals))
        elif chosen == vals[-1] and vals[-2] > 0.0:
            vals.append(_round_like(vals[-1] ** 2 / vals[-2], vals))
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
        extended.append(key)
    return new, counts, extended


# ------------------------------------------------------------------------------ main unit
def run_unit(
    ctx: StudyContext,
    plant_name: str,
    condition: str,
    seed: int,
    pretrained: Any,
    *,
    steps_cap: int | None = None,
    l1_max_nfev: int | None = None,
) -> dict[str, Any]:
    """One (plant, condition, seed): every method at every N on the four test sets."""
    config = ctx.config
    t0 = time.perf_counter()
    plant = plant_model(config.source, config.plant(plant_name))
    design = ctx.designs[seed].x
    y, offsets = _plant_data(config, plant, condition, seed, design)
    truths = {k: plant.predict(offsets.true_inputs(ctx.tests[k].x)) for k in TEST_SET_NAMES}
    true_rh = {
        k: float(np.max(plant.solve(offsets.true_inputs(ctx.tests[k].x))[1].relative_humidity))
        for k in TEST_SET_NAMES
    }

    def score(model: StaticSurrogate) -> dict[str, Any]:
        return {k: errors(model.predict(ctx.tests[k].x), truths[k]) for k in TEST_SET_NAMES}

    fixed: dict[str, Any] = {
        "L0": score(BalanceSurrogate(source_model(config.source))),
        "L0_v412": score(BalanceSurrogate(source_model(config.sensitivity_source), "L0_v412")),
        "L3_0": score(PretrainedMlp(scaling(config), pretrained)),
    }
    fitted: dict[str, dict[str, Any]] = {}
    for n in config.budgets:
        per_method: dict[str, Any] = {}
        for name in FITTED_METHODS:
            setting = dict(config.selected(name) or {})
            if steps_cap is not None and "steps" in setting:
                setting["steps"] = min(int(setting["steps"]), steps_cap)
            model = make_method(name, config, setting, pretrained, seed, n)
            if isinstance(model, CalibratedBalance) and l1_max_nfev is not None:
                model.max_nfev = l1_max_nfev  # dry run only: cost, never a result
            t_fit = time.perf_counter()
            model.fit(design[:n], y[:n])
            fit_seconds = time.perf_counter() - t_fit
            per_method[name] = {
                "test": score(model),
                "info": model.info(),
                "fit_seconds": fit_seconds,
            }
        fitted[str(n)] = per_method
    return {
        "plant": plant_name,
        "condition": condition,
        "seed": seed,
        "offsets": offsets.as_dict(),
        "max_true_relative_humidity": true_rh,
        "targets": list(TARGET_NAMES),
        "fixed": fixed,
        "fitted": fitted,
        "seconds": time.perf_counter() - t0,
        "peak_memory_mb": peak_memory_mb(),
    }


def fresh_net(config: StaticTransferConfig, seed: int) -> Any:
    """An untrained net of the study's architecture (dry runs before any pretraining)."""
    return make_net(config.hidden, config.pretraining.seed_offset + seed)
