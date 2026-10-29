r"""S1.2 / S1.3: identifiability, calibration, validation and daily balance of storage A1.

Product 1 and product 2 are the two plant days. Calibration on one day, validation on the other,
then the reverse; splits are by day only. Everything is written to one artifact (never
overwritten). Raw data come from the environment variables of the plant-validation scripts
(``TWINLINK_PLANT_DATA_PROD1`` = product 1, ``TWINLINK_PLANT_DATA_PROD2`` = product 2), set at the call
site; the channel assignment from ``conf/plant/plant_local.yaml`` (git-ignored). Call::

    TWINLINK_PLANT_DATA_PROD1=... TWINLINK_PLANT_DATA_PROD2=... \
        tools/run_here.sh scripts/study_storage_a1.py --name study_storage_a1

The artifact stores derived numbers, hashes of the raw files, the config fingerprint and the
provenance block -- no raw data and no paths.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import REPO_ROOT, provenance, require_clean_tree, write_artifact

from twinlink_ml.storage_a1.balance import air_heating_kw, daily_balance, steam_share
from twinlink_ml.storage_a1.calibration import (
    DERIVATIVE_WINDOW,
    arx_free_run,
    error_metrics,
    fit_arx,
    fit_conductances,
    mean_q_in_kw,
    oracle_q_in_kw,
    persistence,
    predict_extraction,
    repair_glitches,
    require_different_days,
    rollout_storage,
)
from twinlink_ml.storage_a1.config import (
    PLACEHOLDERS,
    AirConfig,
    SteamHeaterConfig,
    StorageConfig,
    StorageSystemConfig,
    Topology,
)
from twinlink_ml.storage_a1.identifiability import (
    heat_out_kw,
    implied_conductances,
    q_in_curve,
    volume_bounds,
)
from twinlink_ml.storage_a1.roles import PlantDay, load_day, load_role_map

PRODUCT_ENV = {"product_1": "TWINLINK_PLANT_DATA_PROD1", "product_2": "TWINLINK_PLANT_DATA_PROD2"}
ROLE_FILE = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
TOPOLOGIES: tuple[Topology, ...] = ("storage_then_exchanger", "exchanger_then_storage")
BOUND_WINDOWS = (1, 15, 30, 60, 120)
BOUND_TOLERANCES = (0.0, 0.01, 0.05)
UA_STORAGE_GRID = (0.0, 0.02, 0.2, 2.0)
VOLUME_GRID_M3 = (5.0, 10.0, 20.0, 30.0)
Q_MAX_GRID_KW = (1200.0, 700.0)
# exogenous inputs of the ARX baselines (the same measured inputs the physical model gets)
ARX_INPUTS = {
    "supply_temperature": (
        "storage_temperature",
        "ambient_temperature",
        "preheater_loop_flow",
        "process_air_flow",
    ),
    "q_out": (
        "storage_temperature",
        "ambient_temperature",
        "preheater_loop_flow",
        "process_air_flow",
    ),
    "storage_temperature": (
        "ambient_temperature",
        "storage_loop_flow",
        "preheater_loop_flow",
        "process_air_flow",
        "steam_valve_position",
    ),
}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exog(day: PlantDay, names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([day[n] for n in names])


def _metrics_block(
    cfg: StorageSystemConfig, fit_cfg: StorageSystemConfig, cal: PlantDay, val: PlantDay
) -> dict[str, Any]:
    """Extraction-part metrics on the validation day (+ in-sample on the calibration day)."""
    out: dict[str, Any] = {}
    for tag, day in (("in_sample", cal), ("validation", val)):
        index, res = predict_extraction(fit_cfg, day)
        ref_sup = day["preheater_supply_temperature"][index]
        ref_ret = day["preheater_return_temperature"][index]
        ref_q = heat_out_kw(day, cfg)[index]
        q_air = air_heating_kw(day, cfg)[index]
        ref_share = steam_share(q_air, ref_q)
        block: dict[str, Any] = {
            "model": {
                "supply_temperature_k": error_metrics(res.t_supply_celsius, ref_sup),
                "return_temperature_k": error_metrics(res.t_return_celsius, ref_ret),
                "q_out_kw": error_metrics(res.q_out_kw, ref_q),
                "steam_share": error_metrics(steam_share(q_air, res.q_out_kw), ref_share),
                "infeasible_minutes": int(np.count_nonzero(~res.feasible)),
                "min_approach_exchanger_k": float(np.min(res.approach_exchanger_k)),
                "min_approach_preheater_k": float(np.min(res.approach_preheater_k)),
            }
        }
        if tag == "validation":
            persist_sup = persistence(day["preheater_supply_temperature"])[index]
            persist_q = persistence(heat_out_kw(day, cfg))[index]
            arx_sup = arx_free_run(
                fit_arx(
                    cal["preheater_supply_temperature"],
                    _exog(cal, ARX_INPUTS["supply_temperature"]),
                ),
                val["preheater_supply_temperature"][0],
                _exog(val, ARX_INPUTS["supply_temperature"]),
            )[index]
            arx_q = arx_free_run(
                fit_arx(heat_out_kw(cal, cfg), _exog(cal, ARX_INPUTS["q_out"])),
                float(heat_out_kw(val, cfg)[0]),
                _exog(val, ARX_INPUTS["q_out"]),
            )[index]
            block["persistence"] = {
                "supply_temperature_k": error_metrics(persist_sup, ref_sup),
                "q_out_kw": error_metrics(persist_q, ref_q),
                "steam_share": error_metrics(steam_share(q_air, persist_q), ref_share),
            }
            block["arx"] = {
                "supply_temperature_k": error_metrics(arx_sup, ref_sup),
                "q_out_kw": error_metrics(arx_q, ref_q),
                "steam_share": error_metrics(steam_share(q_air, arx_q), ref_share),
            }
        out[tag] = block
    return out


def _storage_rollouts(fit_cfg: StorageSystemConfig, cal: PlantDay, val: PlantDay) -> dict[str, Any]:
    """24 h rollouts of T_S on the validation day for the volume grid."""
    ref = val["storage_temperature"]
    out: dict[str, Any] = {
        "persistence": error_metrics(persistence(ref), ref),
        "arx": error_metrics(
            arx_free_run(
                fit_arx(cal["storage_temperature"], _exog(cal, ARX_INPUTS["storage_temperature"])),
                ref[0],
                _exog(val, ARX_INPUTS["storage_temperature"]),
            ),
            ref,
        ),
        "by_volume_m3": {},
    }
    for volume in VOLUME_GRID_M3:
        cfg_v = dataclasses.replace(fit_cfg, storage=StorageConfig(volume_m3=volume))
        q_const = mean_q_in_kw(cfg_v, cal)
        t_const, _ = rollout_storage(cfg_v, val, q_const)
        t_oracle, _ = rollout_storage(cfg_v, val, oracle_q_in_kw(cfg_v, val))
        out["by_volume_m3"][str(volume)] = {
            "constant_q_in_from_calibration_day": {
                "q_in_kw": q_const,
                **error_metrics(t_const, ref),
            },
            "oracle_q_in_from_validation_day": error_metrics(t_oracle, ref),
        }
    return out


def _identifiability(days: dict[str, PlantDay], cfg: StorageSystemConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"volume_bounds": {}, "q_in_band": {}, "implied_conductances": {}}
    for label, day in days.items():
        rows = []
        for ua_s in UA_STORAGE_GRID:
            cfg_u = dataclasses.replace(cfg, storage=StorageConfig(volume_m3=1.0, ua_kw_per_k=ua_s))
            for q_max in Q_MAX_GRID_KW:
                for window in BOUND_WINDOWS:
                    for tol in BOUND_TOLERANCES:
                        b = volume_bounds(day, cfg_u, window, tol, q_max)
                        rows.append(
                            {
                                "ua_storage_kw_per_k": ua_s,
                                "q_max_kw": q_max,
                                "window_min": window,
                                "tolerance": tol,
                                "lower_m3": b.lower_m3,
                                "upper_m3": b.upper_m3,
                                "upper_from_nonnegative_m3": b.upper_from_nonnegative_m3,
                                "upper_from_q_max_m3": b.upper_from_q_max_m3,
                                "infeasible_samples": b.infeasible_samples,
                            }
                        )
        out["volume_bounds"][label] = rows
        strict = volume_bounds(day, cfg, 15, 0.0)
        out["q_in_band"][label] = {}
        for frac in (0.0, 0.25, 0.5, 1.0):
            volume = frac * strict.upper_m3
            _, q_in = q_in_curve(day, cfg, volume, 15)
            out["q_in_band"][label][f"{frac:g}_of_upper_bound"] = {
                "volume_m3": volume,
                "q_in_min_kw": float(q_in.min()),
                "q_in_mean_kw": float(q_in.mean()),
                "q_in_max_kw": float(q_in.max()),
            }
        out["implied_conductances"][label] = {}
        for topology in TOPOLOGIES:
            grid = VOLUME_GRID_M3 if topology == "exchanger_then_storage" else (1.0,)
            entry = {}
            for volume in grid:
                ua1, ua2 = implied_conductances(day, cfg, topology, volume, DERIVATIVE_WINDOW)
                entry[str(volume)] = {
                    "ua_exchanger_median": float(np.nanmedian(ua1)),
                    "ua_exchanger_cv": float(np.nanstd(ua1) / np.nanmedian(ua1)),
                    "ua_preheater_median": float(np.nanmedian(ua2)),
                    "ua_preheater_cv": float(np.nanstd(ua2) / np.nanmedian(ua2)),
                    "invalid_fraction_exchanger": float(np.mean(np.isnan(ua1))),
                }
            out["implied_conductances"][label][topology] = entry
    return out


def _balance_bands(day: PlantDay, cfg: StorageSystemConfig, q_out: np.ndarray) -> dict[str, Any]:
    """Daily balance with measured Q_out, bands over the placeholders."""
    central = daily_balance(day, cfg, q_out).to_dict()
    results = []
    for fan in (0.5, cfg.steam.fan_heat_fraction, 1.0):
        for rel_h in (0.3, cfg.air.relative_humidity, 1.0):
            for eta in (0.8, cfg.steam.boiler_efficiency, 1.0):
                cfg_b = dataclasses.replace(
                    cfg,
                    air=AirConfig(
                        gas=cfg.air.gas,
                        normal_density_kg_per_nm3=cfg.air.normal_density_kg_per_nm3,
                        relative_humidity=rel_h,
                    ),
                    steam=SteamHeaterConfig(boiler_efficiency=eta, fan_heat_fraction=fan),
                )
                results.append(daily_balance(day, cfg_b, q_out).to_dict())
    keys = (
        "heating_demand_kwh",
        "waste_heat_share",
        "steam_share",
        "gas_kwh",
        "co2_kg",
        "co2_without_recovery_kg",
    )
    return {
        "central": central,
        "band_over_placeholders": {
            k: [min(r[k] for r in results), max(r[k] for r in results)] for k in keys
        },
        "mean_heating_demand_kw": float(np.mean(air_heating_kw(day, cfg))),
        "mean_q_out_kw": float(np.mean(q_out)),
        "max_q_out_kw": float(np.max(q_out)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="artifact name (never overwritten)")
    args = parser.parse_args()
    require_clean_tree()
    role_map = load_role_map(ROLE_FILE)
    paths = {label: os.environ.get(var) for label, var in PRODUCT_ENV.items()}
    missing = [PRODUCT_ENV[k] for k, v in paths.items() if not v]
    if missing:
        raise SystemExit("set " + ", ".join(missing))
    raw_days = {label: load_day(str(path), role_map, label) for label, path in paths.items()}
    repaired = {label: repair_glitches(day) for label, day in raw_days.items()}
    days = {label: day for label, (day, _) in repaired.items()}
    base = StorageSystemConfig()
    base.warn_if_placeholder()

    payload: dict[str, Any] = {
        "_provenance": provenance(),
        "config/fingerprint": base.fingerprint,
        "config/section_fingerprints": base.section_fingerprints,
        "placeholders": PLACEHOLDERS,
        "data": {
            label: {
                "sha256_of_export": _sha256(str(path)),
                "samples": len(days[label].time_s),
                "step_s": days[label].step_s,
                "role_map_sha256": hashlib.sha256(ROLE_FILE.read_bytes()).hexdigest(),
            }
            for label, path in paths.items()
        },
        "glitches_repaired_per_role": {label: counts for label, (_, counts) in repaired.items()},
        "external_snapshot_note": (
            "three height sensors of the storage, two control-system snapshots on other days "
            "(stated by the project lead, not in the exports): spread 0.1 K on both"
        ),
    }
    payload["identifiability"] = _identifiability(days, base)

    payload["calibration"] = {}
    for topology in TOPOLOGIES:
        volumes = (
            VOLUME_GRID_M3 if topology == "exchanger_then_storage" else (base.storage.volume_m3,)
        )
        for cal_label, val_label in (("product_1", "product_2"), ("product_2", "product_1")):
            cal, val = days[cal_label], days[val_label]
            require_different_days(cal, val)
            for volume in volumes:
                cfg_t = dataclasses.replace(
                    base, topology=topology, storage=StorageConfig(volume_m3=volume)
                )
                fit = fit_conductances(cfg_t, cal)
                key = f"{topology}|cal={cal_label}|val={val_label}|volume_m3={volume:g}"
                payload["calibration"][key] = {
                    "ua_exchanger_kw_per_k": fit.ua_exchanger_kw_per_k,
                    "ua_preheater_kw_per_k": fit.ua_preheater_kw_per_k,
                    "fit_rmse_supply_k": fit.rmse_supply_k,
                    "fit_rmse_return_k": fit.rmse_return_k,
                    "fit_success": fit.success,
                    "extraction": _metrics_block(cfg_t, fit.config, cal, val),
                    "storage_rollout": _storage_rollouts(fit.config, cal, val),
                }

    payload["daily_balance"] = {
        label: _balance_bands(day, base, heat_out_kw(day, base)) for label, day in days.items()
    }
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()
