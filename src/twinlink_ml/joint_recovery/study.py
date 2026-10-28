"""Settings, problem families and the four strategies of the joint-recovery study.

Strategies per test state ``w`` (decisions: inlet temperature, air, medium flow; a medium flow of
0 is the bypass of :mod:`.problem`):

S0 :func:`strategy_s0`
    no recovery (bypass); inlet temperature and air optimal. The same procedure as ``u*`` of
    Befund 107 (:func:`~twinlink_ml.surrogate_opt.optimize.minimize_box` with the ``truth`` and
    ``truth_check`` settings of the base config, the better of the two on the truth). S0 does not
    depend on ``dT_min`` or the medium (tested).
S1 :func:`medium_optimum` at S0's operating point
    one after the other: inlet temperature and air from S0, then the medium flow optimal over
    the bypass and ``[medium lower, upper]``.
S2 :func:`strategy_s2`
    jointly: ``u*`` of :func:`~twinlink_ml.surrogate_opt.study.reference` (read only; multistart,
    main run and independent check) over all three decisions with the medium in its box, or the
    bypass at S0's operating point, whichever is better.
S3 :func:`medium_optimum` at ``u0``
    the robust fixed operating point ``u0`` of :func:`~twinlink_ml.surrogate_opt.study.reference`
    (least energy at the nominal disturbance, feasible in every test state, as in Befund 107),
    inlet temperature and air fixed, the medium flow optimal.

"Better" everywhere: feasible first, then the lower energy (:func:`better`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from twinlink_ml.surrogate_opt.config import (
    REPO_ROOT,
    SurrogateStudyConfig,
    build_problems,
    load_study_config,
)
from twinlink_ml.surrogate_opt.line_problem import LineProblemFamily
from twinlink_ml.surrogate_opt.optimize import batch_from_problem, minimize_box
from twinlink_ml.surrogate_opt.study import make_test_states

from .problem import MEDIUM_INDEX, JointLineFamily, JointLineProblem, RestrictedProblem

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "JointRecoveryConfig",
    "assemble",
    "better",
    "build_families",
    "joint_test_states",
    "load_joint_config",
    "medium_optimum",
    "record",
    "strategy_s0",
    "strategy_s2",
]

DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "joint_recovery" / "joint_recovery.yaml"


@dataclass(frozen=True)
class JointRecoveryConfig:
    """The study YAML and the base config of P1 it reads."""

    version: int
    dt_min_k: tuple[float, ...]
    media: tuple[str, ...]
    pressurised: Mapping[str, bool]
    medium_main: Mapping[str, int]
    medium_check: Mapping[str, int]
    recovery_used_kw: float
    condensate_kg_per_h: float
    active_normalised: float
    optimality_rtol: float
    base: SurrogateStudyConfig = field(compare=False, repr=False)
    base_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Hash of this YAML and of the base config's fingerprint."""
        blob = json.dumps({"own": self.raw, "base": self.base.fingerprint}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def combos(self) -> list[tuple[float, str]]:
        return [(dt, m) for dt in self.dt_min_k for m in self.media]


def load_joint_config(path: Path | str = DEFAULT_CONFIG_PATH) -> JointRecoveryConfig:
    """Read the study YAML and the base config it names."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base_path = str(raw["base_config"])
    base = load_study_config(REPO_ROOT / base_path)
    media = raw["factors"]["media"]
    th = raw["thresholds"]
    return JointRecoveryConfig(
        version=int(raw["version"]),
        dt_min_k=tuple(float(v) for v in raw["factors"]["dt_min_k"]),
        media=tuple(media),
        pressurised={k: bool(v["pressurised"]) for k, v in media.items()},
        medium_main={k: int(v) for k, v in raw["medium_optimizer"]["main"].items()},
        medium_check={k: int(v) for k, v in raw["medium_optimizer"]["check"].items()},
        recovery_used_kw=float(th["recovery_used_kw"]),
        condensate_kg_per_h=float(th["condensate_kg_per_h"]),
        active_normalised=float(th["active_normalised"]),
        optimality_rtol=float(th["optimality_rtol"]),
        base=base,
        base_path=base_path,
        raw=raw,
    )


def build_families(
    config: JointRecoveryConfig, dt_values: Sequence[float] | None = None
) -> dict[tuple[float, str], JointLineFamily]:
    """One :class:`JointLineFamily` per ``(dT_min, medium)``, from P1 of the base config."""
    p1 = build_problems(config.base)["P1"]
    assert isinstance(p1, LineProblemFamily)
    dts = config.dt_min_k if dt_values is None else tuple(dt_values)
    return {
        (dt, m): JointLineFamily(
            p1.settings, p1.parameters, dt_min_k=dt, pressurised=config.pressurised[m]
        )
        for dt in dts
        for m in config.media
    }


def joint_test_states(config: JointRecoveryConfig) -> np.ndarray:
    """The 50 test disturbances of Befund 107 (same function, same base config)."""
    p1 = build_problems(config.base)["P1"]
    return make_test_states(p1, config.base)


# --------------------------------------------------------------------------- records
def record(
    problem: JointLineProblem, u: np.ndarray, w: np.ndarray, config: JointRecoveryConfig
) -> dict[str, Any]:
    """Energy, constraints, active constraints and side metrics of ``u`` on ``problem``."""
    ev = problem.evaluate(u, w)
    g = np.asarray(ev.constraints, dtype=np.float64)
    norm = g / problem.scales
    return {
        "u": [float(v) for v in u],
        "energy": float(ev.energy_kw),
        "feasible": problem.feasible(ev),
        "constraints": {c.name: float(v) for c, v in zip(problem.constraints, g, strict=True)},
        "active": [
            c.name
            for c, v in zip(problem.constraints, norm, strict=True)
            if v >= -config.active_normalised
        ],
        "side": {k: float(v) for k, v in ev.side.items()},
    }


def better(a: Mapping[str, Any], b: Mapping[str, Any]) -> Mapping[str, Any]:
    """Feasible first, then lower energy; ties keep ``a``."""
    ka = (not a["feasible"], a["energy"])
    kb = (not b["feasible"], b["energy"])
    return b if kb < ka else a


def _best_of_runs(
    problem: RestrictedProblem,
    w: np.ndarray,
    settings: Sequence[Mapping[str, int]],
    feasibility_tolerance: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run :func:`minimize_box` once per setting; the best end point on the problem itself."""
    runs, best_u, best_key = [], None, None
    for s in settings:
        r = minimize_box(
            batch_from_problem(problem, w), problem.decisions,
            candidates=int(s["candidates"]), polish=int(s["polish"]), seed=int(s.get("seed", 0)),
            feasibility_tolerance=feasibility_tolerance,
        )  # fmt: skip
        ev = problem.evaluate(r.u, w)
        key = (not problem.feasible(ev), ev.energy_kw)
        runs.append(
            {"u": problem.full(r.u).tolist(), "energy": float(ev.energy_kw),
             "feasible": problem.feasible(ev), "starts": r.starts, "at_best": r.starts_at_best,
             "spread": r.spread, "evaluations": r.evaluations}
        )  # fmt: skip
        if best_key is None or key < best_key:
            best_u, best_key = r.u, key
    assert best_u is not None
    return problem.full(best_u), runs


# --------------------------------------------------------------------------- strategies
def strategy_s0(
    family: JointLineFamily, w: np.ndarray, config: JointRecoveryConfig
) -> dict[str, Any]:
    """S0: bypass, inlet temperature and air optimal (``truth`` and ``truth_check`` runs)."""
    truth = family.truth()
    fixed = [np.nan, np.nan, 0.0]
    problem = RestrictedProblem(truth, (0, 1), fixed)
    base = config.base
    u, runs = _best_of_runs(
        problem, w, (base.optimizer("truth"), base.optimizer("truth_check")),
        base.feasibility_tolerance,
    )  # fmt: skip
    return {"best": record(truth, u, w, config), "runs": runs}


def medium_optimum(
    family: JointLineFamily,
    w: np.ndarray,
    operating_point: Sequence[float],
    config: JointRecoveryConfig,
) -> dict[str, Any]:
    """Inlet temperature and air fixed; the medium flow optimal over the bypass and its box."""
    truth = family.truth()
    t_in, air = float(operating_point[0]), float(operating_point[1])
    problem = RestrictedProblem(truth, (MEDIUM_INDEX,), [t_in, air, np.nan])
    u_on, runs = _best_of_runs(
        problem, w, (config.medium_main, config.medium_check), config.base.feasibility_tolerance
    )
    on = record(truth, u_on, w, config)
    off = record(truth, np.array([t_in, air, 0.0]), w, config)
    best = better(off, on)
    return {"best": best, "on": on, "bypass": off, "runs": runs}


def strategy_s2(
    family: JointLineFamily,
    w: np.ndarray,
    u_star: Sequence[float],
    s0_u: Sequence[float],
    config: JointRecoveryConfig,
) -> dict[str, Any]:
    """S2: the joint optimum ``u*`` with the medium in its box, or the bypass at S0's point."""
    truth = family.truth()
    on = record(truth, np.asarray(u_star, dtype=np.float64), w, config)
    off = record(truth, np.array([float(s0_u[0]), float(s0_u[1]), 0.0]), w, config)
    return {"best": better(off, on), "on": on, "bypass": off}


def assemble(
    config: JointRecoveryConfig,
    families: Mapping[tuple[float, str], JointLineFamily],
    tw: np.ndarray,
    s0: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[float, str], Mapping[str, Any]],
    media_results: Mapping[tuple[float, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Per ``(dT_min, medium)`` and state: the records of S0-S3, ``V_J`` and the checks.

    ``s0[i]``: :func:`strategy_s0` of state ``i``; ``references[(dt, m)]``: the result of
    :func:`~twinlink_ml.surrogate_opt.study.reference`; ``media_results[(dt, m, i)]``:
    ``{"S1": medium_optimum at S0, "S3": medium_optimum at u0}``.
    """
    rtol = config.optimality_rtol
    out: dict[str, Any] = {}
    for (dt, m), family in families.items():
        ref = references[(dt, m)]
        truth = family.truth()
        states = []
        for i, w in enumerate(tw):
            s0_best = s0[i]["best"]
            # S0 on this combo's problem (the bypass does not depend on dT_min or the medium,
            # except for the medium constraint, which is dropped when pressurised)
            s0_rec = record(truth, np.asarray(s0_best["u"]), w, config)
            s2 = strategy_s2(family, w, ref["u_star"][i], s0_best["u"], config)
            mr = media_results[(dt, m, i)]
            e0, e1 = s0_rec["energy"], mr["S1"]["best"]["energy"]
            e2, e3 = s2["best"]["energy"], mr["S3"]["best"]["energy"]
            checks = {
                "S1_le_S0": e1 <= e0 * (1 + rtol),
                "S2_le_S1": e2 <= e1 * (1 + rtol),
                "S2_le_S3": e2 <= e3 * (1 + rtol),
                "S2on_le_S1on": s2["on"]["energy"] <= mr["S1"]["on"]["energy"] * (1 + rtol),
                "S2on_le_S3on": s2["on"]["energy"] <= mr["S3"]["on"]["energy"] * (1 + rtol),
            }
            states.append(
                {
                    "w": [float(v) for v in w],
                    "S0": s0_rec,
                    "S1": mr["S1"]["best"],
                    "S1_on": mr["S1"]["on"],
                    "S2": s2["best"],
                    "S2_on": s2["on"],
                    "S3": mr["S3"]["best"],
                    "S3_on": mr["S3"]["on"],
                    "V_J": (e1 - e2) / e1,
                    "checks": checks,
                    "all_feasible": all(
                        r["feasible"]
                        for r in (s0_rec, mr["S1"]["best"], s2["best"], mr["S3"]["best"])
                    ),
                }
            )
        out[f"{dt:g}K/{m}"] = {
            "dt_min_k": dt,
            "medium": m,
            "pressurised": config.pressurised[m],
            "u0": [float(v) for v in ref["u0"]],
            "u0_model_feasible": bool(ref["u0_model_feasible"]),
            "states": states,
        }
    return out
