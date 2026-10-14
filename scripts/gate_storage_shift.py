r"""S2 gate: potential of better storage operation to save steam and CO2.

Question: how much steam / CO2 could better operation of storage A1 save at all?

THIS DOCSTRING IS THE PRE-REGISTRATION. It was committed before the script was run; nothing in it
may be changed after seeing results (corrections only as dated addenda in the lab book).

Scope: a SCENARIO BAND on a model whose volume is not identifiable (S1: only an upper bound of
about 30 m3 from the two days), whose exchanger conductances differ between the two plant days,
and whose compressor-heat availability is unmeasured. No number below is a plant statement; every
result carries its scenario and says so. No reinforcement learning, no controller design.

Policies (all on the same discrete model, see ``twinlink_ml.storage_a1.scheduling``; 5 min steps)
    R0   replay of the logged preheater-loop flow, no extra compressor heat taken.
    R1   rule: maximum preheater-loop flow while T_S > T_S,min (minimum flow otherwise); requests
         all discarded compressor heat (taken only as far as it fits below T_S,max).
    R*   perfect-foresight optimum over the day, backward dynamic programming on a T_S grid
         (exact up to the 0.25 K grid; chosen over an LP because the extraction duty is non-linear
         in state and flow). Objective: steam energy of the day (CO2 follows at fixed boiler
         efficiency). Constraints: T_S in [T_S,min, T_S,max], supply-air set point kept
         (Q_out <= Q_air), exchanger approach floors, terminal T_S >= T_S(end) of R0.
    All steam energies are corrected to the stored energy at the end of R0
    (``corrected_steam_kwh``, credit factor 1: optimistic for the policy that ends hotter).

R0 check (per product day, volume 10 m3, no shift, no extra heat): R0 "reproduces the measured
day" if the steam energy deviates by <= 3 % from the air-balance reconstruction of the measured
day and the T_S track error is <= 2 K RMSE. Otherwise the deviation is reported and savings are
read relative to the model's R0 with that deviation as an uncertainty. Conductances: fitted on the
same day (in-sample, topology B); the other day's fit is a sensitivity case.

Scenario grid (all placeholders)
    a) discarded compressor heat Q_dump(t) available to the storage loop, kW:
       none (0); constant 100; constant 300; day profile (500 from 06:00 to 18:00, 100 otherwise;
       mean 300); constant 800 (partner statements: about 1200 kW total, up to about 400 kW used).
    b) storage volume, m3: 5, 10, 20, 30, 50. Reference volume V0 = 10 (placeholder). 5 and 30 are
       the ends of the S1 band, 20 = 2 x V0, 50 = 5 x V0 (outside the S1 band: hypothetical new
       storage). The compressor heat taken today, Q_in_base(t), is reconstructed once for V0 and
       does not change with the volume under study.
    c) weather: both measured days, each with the ambient temperature shifted by -5, 0, +5 K
       (set-point of the supply air as logged).
    Fixed placeholders: T_S,min 40 degC, T_S,max 90 degC, V2 range 5 .. 1.5 x median logged flow,
    topology B.

Gate criterion (pre-registered)
    Realistic scenario: Q_dump > 0 AND volume <= 20 m3 (x2). For one scenario:
    saving G = (E_R0 - E_R*) / E_R0 and gap = (E_R1 - E_R*) / E_R0, E = steam energy of the day.
    L1  RL set-up justified:   some realistic scenario has G >= 5 % AND gap >= 2 %.
    L2  a rule suffices:       some realistic scenario has G >= 5 %, but none of them has gap >= 2 %.
    L3  no shifting potential: no realistic scenario has G >= 5 %.
    Also reported (same rule, not replacing it): the conservative subset Q_dump <= 300 kW, and the
    verdict per product day.

Decomposition of the saving E_R0 - E_R* (all in kWh/d of steam, same terminal storage energy):
    exchanger use      E_R0 - E_R1,nodump   (maximum flow, no extra heat)
    timing / level     E_R1,nodump - E_R*,nodump   (optimised flow, no extra heat)
    extra heat used    E_R*,nodump - E_R*   (otherwise discarded heat taken into the storage)
    Without discarded heat a storage can only move heat in time, not add energy; the third line
    is the only energy that is new. The split is order-dependent (stated, not hidden).

Sensitivity (reference: Q_dump 300 kW constant, V = 10 m3, no shift, both days): one placeholder
at a time; ranking by the change of the R* saving in kWh/d and in percentage points of E_R0.

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... \
        tools/run_here.sh scripts/gate_storage_shift.py --name gate_storage_shift
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

from twinlink_ml.storage_a1.calibration import fit_conductances, repair_glitches
from twinlink_ml.storage_a1.config import (
    AirConfig,
    ExchangerConfig,
    SteamHeaterConfig,
    StorageSystemConfig,
    Topology,
)
from twinlink_ml.storage_a1.identifiability import heat_out_kw
from twinlink_ml.storage_a1.roles import PlantDay, load_day, load_role_map
from twinlink_ml.storage_a1.scheduling import (
    ShiftProblem,
    ShiftResult,
    build_problem,
    co2_kg,
    corrected_steam_kwh,
    dp_policy,
    replay,
    rule_policy,
    simulate,
)

PRODUCT_ENV = {"product_1": "TWINLINK_PLANT_DATA_ACE", "product_2": "TWINLINK_PLANT_DATA_RID"}
ROLE_FILE = REPO_ROOT / "conf" / "plant" / "plant_local.yaml"
BLOCK = 5
V0_M3 = 10.0
VOLUMES_M3 = (5.0, 10.0, 20.0, 30.0, 50.0)
SHIFTS_K = (-5.0, 0.0, 5.0)
DAY_PROFILE_KW = (100.0, 500.0)  # (night, day), day = 06:00 .. 18:00
DUMP_SCENARIOS = ("none", "const_100", "const_300", "day_profile", "const_800")
GATE_G, GATE_GAP = 0.05, 0.02
R0_STEAM_TOLERANCE, R0_TRACK_TOLERANCE_K = 0.03, 2.0
REALISTIC_VOLUME_MAX_M3 = 2.0 * V0_M3


def dump_profile(name: str, n_steps: int, dt_s: float) -> np.ndarray:
    hours = np.arange(n_steps) * dt_s / 3600.0
    if name == "none":
        return np.zeros(n_steps)
    if name == "day_profile":
        night, day = DAY_PROFILE_KW
        return np.where((hours >= 6.0) & (hours < 18.0), day, night)
    return np.full(n_steps, float(name.split("_")[1]))


def mean_dump_kw(name: str) -> float:
    return float(np.mean(dump_profile(name, 288, 300.0)))


def with_volume(cfg: StorageSystemConfig, volume: float) -> StorageSystemConfig:
    s = cfg.storage
    return dataclasses.replace(cfg, storage=dataclasses.replace(s, volume_m3=volume))


def make_problem(
    day: PlantDay, cfg: StorageSystemConfig, volume: float, dump: str, shift: float, **kw: Any
) -> ShiftProblem:
    n_steps = len(day.time_s) // BLOCK
    return build_problem(
        day,
        with_volume(cfg, volume),
        reference_volume_m3=kw.pop("reference_volume_m3", V0_M3),
        q_dump_kw=dump_profile(dump, n_steps, day.step_s * BLOCK),
        ambient_shift_k=shift,
        block=BLOCK,
        **kw,
    )


def run_policies(problem: ShiftProblem, *, with_nodump: bool) -> dict[str, ShiftResult]:
    r0 = replay(problem)
    out = {
        "R0": r0,
        "R1": simulate(problem, rule_policy(problem, use_dump=True)),
        "R*": simulate(problem, dp_policy(problem, allow_dump=True, t_end_reference=r0.t_end)),
    }
    if with_nodump:
        out["R1_nodump"] = simulate(problem, rule_policy(problem, use_dump=False))
        out["R*_nodump"] = simulate(
            problem, dp_policy(problem, allow_dump=False, t_end_reference=r0.t_end)
        )
    return out


def energies(runs: dict[str, ShiftResult], cfg: StorageSystemConfig) -> dict[str, Any]:
    ref = runs["R0"].t_end
    e = {name: corrected_steam_kwh(r, ref) for name, r in runs.items()}
    out: dict[str, Any] = {
        "steam_kwh_per_day": e,
        "co2_kg_per_day": {name: co2_kg(v, cfg) for name, v in e.items()},
        "t_end_celsius": {name: r.t_end for name, r in runs.items()},
        "t_max_reached_celsius": {name: float(np.max(r.t_s)) for name, r in runs.items()},
        "feasible": {name: r.feasible for name, r in runs.items()},
        "extra_heat_taken_kwh": {name: r.dump_used_kwh for name, r in runs.items()},
        "saving_fraction_R*": (e["R0"] - e["R*"]) / e["R0"],
        "gap_fraction_R1_minus_R*": (e["R1"] - e["R*"]) / e["R0"],
    }
    if "R1_nodump" in e:
        out["decomposition_kwh_per_day"] = {
            "exchanger_use": e["R0"] - e["R1_nodump"],
            "timing_and_level": e["R1_nodump"] - e["R*_nodump"],
            "extra_heat_used": e["R*_nodump"] - e["R*"],
            "total": e["R0"] - e["R*"],
        }
    return out


def r0_check(day: PlantDay, cfg: StorageSystemConfig) -> dict[str, Any]:
    problem = make_problem(day, cfg, V0_M3, "none", 0.0)
    r0 = replay(problem)
    n = problem.n_steps
    q_meas = heat_out_kw(day, cfg)[: n * BLOCK].reshape(-1, BLOCK).mean(axis=1)
    steam_meas = float(np.sum(problem.q_air_kw - q_meas) * problem.dt_s / 3600.0)
    t_meas = day["storage_temperature"][: n * BLOCK : BLOCK]
    track = r0.t_s[:-1] - t_meas
    deviation = (r0.steam_kwh - steam_meas) / steam_meas
    rmse = float(np.sqrt(np.mean(track**2)))
    return {
        "steam_measured_reconstruction_kwh": steam_meas,
        "steam_r0_kwh": r0.steam_kwh,
        "steam_deviation_fraction": deviation,
        "storage_temperature_rmse_k": rmse,
        "storage_temperature_max_abs_k": float(np.max(np.abs(track))),
        "reproduces_measured_day": bool(
            abs(deviation) <= R0_STEAM_TOLERANCE and rmse <= R0_TRACK_TOLERANCE_K
        ),
    }


def gate_verdict(cases: list[dict[str, Any]], dump_max_kw: float | None = None) -> dict[str, Any]:
    def realistic(c: dict[str, Any]) -> bool:
        ok = c["mean_dump_kw"] > 0.0 and c["volume_m3"] <= REALISTIC_VOLUME_MAX_M3
        return ok and (dump_max_kw is None or c["mean_dump_kw"] <= dump_max_kw)

    real = [c for c in cases if realistic(c)]
    hit = [c for c in real if c["saving_fraction_R*"] >= GATE_G]
    both = [c for c in hit if c["gap_fraction_R1_minus_R*"] >= GATE_GAP]
    reading = "L1" if both else ("L2" if hit else "L3")
    return {
        "reading": reading,
        "n_realistic": len(real),
        "n_saving_at_least_5_percent": len(hit),
        "n_saving_and_gap": len(both),
        "max_saving_fraction": max((c["saving_fraction_R*"] for c in real), default=None),
        "max_gap_fraction": max((c["gap_fraction_R1_minus_R*"] for c in real), default=None),
    }


def sensitivity(days: dict[str, PlantDay], base: StorageSystemConfig) -> list[dict[str, Any]]:
    """One placeholder at a time around the reference scenario, per product day."""

    def variants(
        label: str, day: PlantDay
    ) -> list[tuple[str, StorageSystemConfig, dict[str, Any]]]:
        def refit(cfg: StorageSystemConfig) -> StorageSystemConfig:
            return fit_conductances(with_volume(cfg, V0_M3), day).config

        out: list[tuple[str, StorageSystemConfig, dict[str, Any]]] = []
        fitted = refit(base)
        out.append(("reference", fitted, {}))
        for t_max in (75.0, 90.0, 100.0):
            if t_max != 90.0:
                out.append(
                    (
                        f"T_S,max={t_max:g}",
                        dataclasses.replace(
                            fitted,
                            storage=dataclasses.replace(
                                fitted.storage, max_temperature_celsius=t_max
                            ),
                        ),
                        {},
                    )
                )
        for t_min in (35.0, 45.0):
            out.append((f"T_S,min={t_min:g}", fitted, {"t_min": t_min}))
        for f in (1.0, 2.0):
            out.append((f"V2_max_factor={f:g}", fitted, {"v2_max_factor": f}))
        other = "product_2" if label == "product_1" else "product_1"
        out.append(
            (
                f"conductances_of_{other}",
                fit_conductances(with_volume(base, V0_M3), days[other]).config,
                {},
            )
        )
        for scale in (0.8, 1.25):
            out.append(
                (
                    f"conductances_x{scale:g}",
                    dataclasses.replace(
                        fitted,
                        storage_exchanger=ExchangerConfig(
                            fitted.storage_exchanger.ua_kw_per_k * scale,
                            fitted.storage_exchanger.minimum_approach_k,
                        ),
                        preheater=ExchangerConfig(
                            fitted.preheater.ua_kw_per_k * scale,
                            fitted.preheater.minimum_approach_k,
                        ),
                    ),
                    {},
                )
            )
        topo: Topology = "exchanger_then_storage"
        out.append(("topology_A", refit(dataclasses.replace(base, topology=topo)), {}))
        for ua in (0.0, 0.2):
            out.append(
                (
                    f"UA_S={ua:g}",
                    dataclasses.replace(
                        fitted, storage=dataclasses.replace(fitted.storage, ua_kw_per_k=ua)
                    ),
                    {},
                )
            )
        for fan in (0.5, 1.0):
            out.append(
                (
                    f"fan_heat_fraction={fan:g}",
                    dataclasses.replace(
                        fitted, steam=SteamHeaterConfig(fitted.steam.boiler_efficiency, fan)
                    ),
                    {},
                )
            )
        for eta in (0.8, 1.0):
            out.append(
                (
                    f"boiler_efficiency={eta:g}",
                    dataclasses.replace(
                        fitted, steam=SteamHeaterConfig(eta, fitted.steam.fan_heat_fraction)
                    ),
                    {},
                )
            )
        for rh in (0.3, 1.0):
            out.append(
                (
                    f"relative_humidity={rh:g}",
                    dataclasses.replace(
                        fitted,
                        air=AirConfig(fitted.air.gas, fitted.air.normal_density_kg_per_nm3, rh),
                    ),
                    {},
                )
            )
        for vref in (5.0, 20.0):
            out.append(
                (f"reference_volume_of_Q_in_base={vref:g}", fitted, {"reference_volume_m3": vref})
            )
        return out

    rows: list[dict[str, Any]] = []
    for label, day in days.items():
        ref_saving: tuple[float, float, float] | None = None
        for name, cfg, kw in variants(label, day):
            problem = make_problem(day, cfg, V0_M3, "const_300", 0.0, **kw)
            runs = run_policies(problem, with_nodump=False)
            e = energies(runs, cfg)
            saving_kwh = e["steam_kwh_per_day"]["R0"] - e["steam_kwh_per_day"]["R*"]
            if name == "reference":
                ref_saving = (
                    saving_kwh,
                    e["saving_fraction_R*"],
                    e["co2_kg_per_day"]["R0"] - e["co2_kg_per_day"]["R*"],
                )
            assert ref_saving is not None
            rows.append(
                {
                    "product_day": label,
                    "variant": name,
                    "saving_kwh_per_day": saving_kwh,
                    "saving_fraction": e["saving_fraction_R*"],
                    "saving_co2_kg_per_day": e["co2_kg_per_day"]["R0"] - e["co2_kg_per_day"]["R*"],
                    "delta_saving_kwh": saving_kwh - ref_saving[0],
                    "delta_saving_percentage_points": 100.0
                    * (e["saving_fraction_R*"] - ref_saving[1]),
                    "delta_saving_co2_kg": e["co2_kg_per_day"]["R0"]
                    - e["co2_kg_per_day"]["R*"]
                    - ref_saving[2],
                    "feasible": e["feasible"],
                }
            )
    return rows


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
    days: dict[str, PlantDay] = {}
    repaired: dict[str, dict[str, int]] = {}
    for label, path in paths.items():
        days[label], repaired[label] = repair_glitches(load_day(str(path), role_map, label))
    base = StorageSystemConfig()
    base.warn_if_placeholder()
    fitted = {
        label: fit_conductances(with_volume(base, V0_M3), day).config for label, day in days.items()
    }

    payload: dict[str, Any] = {
        "_provenance": provenance(),
        "interpretation": (
            "SCENARIO BAND on a model with unidentified volume, day-dependent conductances and "
            "unmeasured compressor-heat availability; not a plant statement"
        ),
        "config/fingerprint_base": base.fingerprint,
        "config/fingerprint_fitted": {k: c.fingerprint for k, c in fitted.items()},
        "fitted_conductances_kw_per_k": {
            k: {"exchanger": c.storage_exchanger.ua_kw_per_k, "preheater": c.preheater.ua_kw_per_k}
            for k, c in fitted.items()
        },
        "data": {
            label: {"sha256_of_export": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for label, p in paths.items()
        },
        "glitches_repaired_per_role": repaired,
        "preregistration": {
            "gate_saving_fraction": GATE_G,
            "gate_gap_fraction": GATE_GAP,
            "realistic_volume_max_m3": REALISTIC_VOLUME_MAX_M3,
            "r0_steam_tolerance": R0_STEAM_TOLERANCE,
            "r0_track_tolerance_k": R0_TRACK_TOLERANCE_K,
        },
        "r0_check": {label: r0_check(day, fitted[label]) for label, day in days.items()},
    }

    cases: list[dict[str, Any]] = []
    for label, day in days.items():
        cfg = fitted[label]
        for shift in SHIFTS_K:
            for volume in VOLUMES_M3:
                nodump_runs: dict[str, ShiftResult] = {}
                for dump in DUMP_SCENARIOS:
                    problem = make_problem(day, cfg, volume, dump, shift)
                    runs = run_policies(problem, with_nodump=dump == "none")
                    if dump == "none":
                        nodump_runs = {k: runs[k] for k in ("R1_nodump", "R*_nodump")}
                    else:
                        runs["R1_nodump"] = nodump_runs["R1_nodump"]
                        runs["R*_nodump"] = nodump_runs["R*_nodump"]
                    cases.append(
                        {
                            "product_day": label,
                            "ambient_shift_k": shift,
                            "volume_m3": volume,
                            "dump_scenario": dump,
                            "mean_dump_kw": mean_dump_kw(dump),
                            **energies(runs, cfg),
                        }
                    )
                print(label, shift, volume, "done", flush=True)
    payload["cases"] = cases
    payload["gate"] = {
        "all_products": gate_verdict(cases),
        "conservative_dump_le_300": gate_verdict(cases, 300.0),
        **{label: gate_verdict([c for c in cases if c["product_day"] == label]) for label in days},
    }
    payload["sensitivity"] = sensitivity(days, base)
    write_artifact(args.name, payload)


if __name__ == "__main__":
    main()
