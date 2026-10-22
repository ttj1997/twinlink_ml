"""The study: designs, reference optima, the four model arms, the adaptive arm and the scoring.

Designs
-------
* Test states: ``count`` scrambled Sobol points of the disturbance box with the fixed
  ``test_states.seed`` (first ``count`` of ``2^m``) -- an own sequence, never a training design.
* Training design of seed ``s``: scrambled Sobol over the JOINT box ``(u, w)`` (seed ``s``), the
  first ``N`` points for budget ``N`` (nested designs).
* Adaptive pool of seed ``s``: scrambled Sobol over the disturbance box (seed ``s + offset``).

:func:`check_disjoint` reports the smallest distance between test and training disturbances (unit
coordinates); the run aborts on an exact coincidence.

Budget
------
Every training point is one evaluation of a :class:`BudgetedTruth`, which refuses to go beyond its
budget and counts. The scoring (true model at the decision a model found) uses a separate instance
and is not part of the budget -- it is the evaluation of the study, not information for a model.

Reference
---------
``u*`` per test state: P2 the R1 NLP (IPOPT) from the configured starts, cross-checked by
:func:`~.optimize.minimize_box` on the true model; P1 :func:`~.optimize.minimize_box` (``truth``
settings), checked by a second independent run (``truth_check``). ``u*`` is the better of the two
on the truth (feasible first, then energy); the source of each is recorded.

``u0`` (decision of the project lead, 2026-10-22): the robust fixed point,
:func:`robust_fixed_point` -- least energy at the nominal disturbance subject to feasibility in
every test state. It is determined ON the 50 test states, so it knows the disturbance envelope:
that favours the baseline and makes ``eta`` conservative.

Secondary, the R0-style point of docs/24: the optimum of the true model at the nominal
disturbance with the constraints that are ACTIVE there (``g / scale >= -1e-3``; P2: the moisture,
exactly as R0) tightened by one normalised margin ``m`` (``g / scale + m <= 0``), ``m`` the
smallest one (bisection, :func:`~twinlink_ml.headroom.margin.bisect_margin`) with which the point
is feasible in every test state. Tightening inactive constraints as well would only make it worse
(on P2 the bed-temperature limit tightened by the same margin makes the problem infeasible); with
two active constraints (P1) one shared margin depends on their scales.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from twinlink_ml.headroom.margin import bisect_margin

from .config import SurrogateStudyConfig
from .greybox_problem import GreyBoxProblem
from .optimize import BatchModel, OptResult, minimize_box
from .problem import Box, ProblemFamily, SteadyProblem
from .surrogates import Surrogate, make_surrogate

__all__ = [
    "BudgetExceededError",
    "BudgetedTruth",
    "adaptive_arm",
    "blackbox_arm",
    "check_disjoint",
    "fit_greybox",
    "greybox_arm",
    "make_test_states",
    "reference",
    "robust_fixed_point",
    "score",
    "surrogate_batch",
    "training_design",
]


ACTIVE_TOLERANCE = 1e-3
"""A constraint is active at the nominal optimum when ``g / scale >= -ACTIVE_TOLERANCE``."""


class BudgetExceededError(RuntimeError):
    """A model asked for more true evaluations than its budget."""


class BudgetedTruth:
    """The true model behind a counter and a hard budget."""

    def __init__(self, problem: SteadyProblem, budget: int) -> None:
        self.problem, self.budget = problem, budget
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.calls >= self.budget:
            raise BudgetExceededError(f"budget of {self.budget} true evaluations exhausted")
        self.calls += 1
        t0 = time.perf_counter()
        y = self.problem.evaluate(u, w).outputs
        self.seconds += time.perf_counter() - t0
        return y


# --------------------------------------------------------------------------- designs
def _sobol(dim: int, n: int, seed: int) -> np.ndarray:
    m = int(np.ceil(np.log2(max(n, 2))))
    out: np.ndarray = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)[:n]
    return out


def make_test_states(family: ProblemFamily, config: SurrogateStudyConfig) -> np.ndarray:
    """The test disturbances (own Sobol sequence of the disturbance box, fixed seed)."""
    box = family.truth().disturbances
    return box.from_unit(_sobol(box.dim, config.test_count, config.test_seed))


def joint_box(problem: SteadyProblem) -> Box:
    u, w = problem.decisions, problem.disturbances
    return Box(u.names + w.names, u.lower + w.lower, u.upper + w.upper)


def training_design(problem: SteadyProblem, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(u, w)`` of the first ``n`` points of the seed's joint Sobol design."""
    pts = joint_box(problem).from_unit(_sobol(joint_box(problem).dim, n, seed))
    du = problem.decisions.dim
    return pts[:, :du], pts[:, du:]


def adaptive_pool(problem: SteadyProblem, seed: int, n: int, offset: int) -> np.ndarray:
    box = problem.disturbances
    return box.from_unit(_sobol(box.dim, n, seed + offset))


def check_disjoint(box: Box, test_w: np.ndarray, train_w: np.ndarray) -> float:
    """Smallest max-norm distance (unit coordinates) between a test and a training disturbance."""
    a, b = box.to_unit(test_w), box.to_unit(train_w)
    d = float(np.min(np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)))
    if d <= 0.0:
        raise ValueError("a test state coincides with a training disturbance")
    return d


# --------------------------------------------------------------------------- optimisation
def surrogate_batch(
    surrogate: Surrogate, problem: SteadyProblem, w: np.ndarray, kappa: float
) -> BatchModel:
    """The surrogate at fixed ``w`` as a :data:`BatchModel`: ``(mean + kappa sigma) / scale``."""
    box, scales = joint_box(problem), problem.scales

    def model(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = box.to_unit(np.hstack((u, np.repeat(w[None, :], len(u), axis=0))))
        mean, sigma = surrogate.predict(x)
        return mean[:, 0], (mean[:, 1:] + kappa * sigma[:, 1:]) / scales

    return model


def _problem_batch(
    problem: SteadyProblem, w: np.ndarray, margin: float | np.ndarray = 0.0
) -> BatchModel:
    scales = problem.scales

    def model(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = problem.outputs(u, w)
        return y[:, 0], y[:, 1:] / scales + margin

    return model


def optimize_problem(
    problem: SteadyProblem,
    w: np.ndarray,
    config: SurrogateStudyConfig,
    kind: str,
    *,
    margin: float | np.ndarray = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Optimum of a (true or fitted) physics model at ``w``: R1 NLP for P2, else minimize_box."""
    if isinstance(problem, GreyBoxProblem):
        res = problem.optimum(w, np.array(config.raw["optimizer"]["r1_starts"]), margin=margin)
        return {
            "u": res.u, "energy": res.energy_kw, "feasible": res.feasible, "starts": res.starts,
            "converged": res.converged_starts, "at_best": res.starts_at_best,
            "spread": res.spread_kw,
        }  # fmt: skip
    opt = config.optimizer(kind)
    r = minimize_box(
        _problem_batch(problem, w, margin), problem.decisions,
        candidates=int(opt["candidates"]), polish=int(opt["polish"]),
        seed=int(opt.get("seed", seed)), feasibility_tolerance=config.feasibility_tolerance,
    )  # fmt: skip
    return {
        "u": r.u, "energy": r.energy, "feasible": r.feasible, "starts": r.starts,
        "converged": r.starts, "at_best": r.starts_at_best, "spread": r.spread,
    }  # fmt: skip


# --------------------------------------------------------------------------- scoring
def score(truth: SteadyProblem, test_w: np.ndarray, decisions: np.ndarray) -> dict[str, list[Any]]:
    """True energy, constraints and side metrics at one decision per test state."""
    out: dict[str, list[Any]] = {"energy": [], "constraints": [], "feasible": [], "side": []}
    for u, w in zip(decisions, test_w, strict=True):
        ev = truth.evaluate(u, w)
        side = dict(ev.side)
        if isinstance(truth, GreyBoxProblem):
            side.update(truth.side_metrics(u, w))
        out["energy"].append(ev.energy_kw)
        out["constraints"].append(ev.constraints.tolist())
        out["feasible"].append(truth.feasible(ev))
        out["side"].append({k: float(v) for k, v in side.items()})
    return out


def reference(family: ProblemFamily, config: SurrogateStudyConfig) -> dict[str, Any]:
    """Test states, ``u*`` with its convergence check, ``u0`` with its margin, timing."""
    truth = family.truth()
    tw = make_test_states(family, config)
    star, check = [], []
    t0 = time.perf_counter()
    for w in tw:
        star.append(optimize_problem(truth, w, config, "truth"))
        if isinstance(truth, GreyBoxProblem):
            r = minimize_box(
                _problem_batch(truth, w), truth.decisions,
                candidates=int(config.optimizer("truth")["candidates"]),
                polish=int(config.optimizer("truth")["polish"]),
                feasibility_tolerance=config.feasibility_tolerance,
            )  # fmt: skip
            check.append({"u": r.u, "energy": r.energy, "feasible": r.feasible})
        else:
            check.append(optimize_problem(truth, w, config, "truth_check"))
    seconds_star = time.perf_counter() - t0
    # u* = the better of the two runs on the truth (feasible first, then energy); the source is kept
    u_star, star_source = [], []
    for s, c, w in zip(star, check, tw, strict=True):
        ranked = []
        for name, raw in (("main", s["u"]), ("check", c["u"])):
            u = np.asarray(raw, dtype=np.float64)
            if np.all(np.isfinite(u)):
                ev = truth.evaluate(u, w)
                ranked.append((not truth.feasible(ev), ev.energy_kw, name, u))
        best = min(ranked, key=lambda t: (t[0], t[1]))
        u_star.append(best[3])
        star_source.append(best[2])
    u_star_arr = np.array(u_star)
    w_nom = family.nominal_disturbance  # type: ignore[attr-defined]

    nominal = optimize_problem(truth, w_nom, config, "truth")
    g_nom = truth.evaluate(nominal["u"], w_nom).constraints / truth.scales
    active = (g_nom >= -ACTIVE_TOLERANCE).astype(np.float64)

    def u0_at(m: float) -> np.ndarray:
        return np.asarray(optimize_problem(truth, w_nom, config, "truth", margin=m * active)["u"])

    def feasible(m: float) -> bool:
        u0 = u0_at(m)
        if not np.all(np.isfinite(u0)):
            return False  # no fixed point at this margin (the tightened problem is infeasible)
        return all(truth.feasible(truth.evaluate(u0, w)) for w in tw)

    mc = config.raw["margin"]
    search = bisect_margin(
        feasible, lower=0.0, upper=float(mc["upper"]), tolerance=float(mc["tolerance"]),
        max_upper=float(mc["max_upper"]),
    )  # fmt: skip
    u0_margin = u0_at(search.margin)
    extra = u0_margin if np.all(np.isfinite(u0_margin)) else None
    robust = robust_fixed_point(truth, tw, w_nom, config, extra=extra)
    u0 = robust.u
    return {
        "test_w": tw,
        "u_star": u_star_arr,
        "star_source": star_source,
        "star": star,
        "star_check": check,
        "score_star": score(truth, tw, u_star_arr),
        "u0": u0,
        "u0_model_feasible": robust.feasible,
        "u0_starts_at_best": robust.starts_at_best,
        "score_u0": score(truth, tw, np.repeat(u0[None, :], len(tw), axis=0)),
        "u0_margin": u0_margin,
        "margin": search.margin,
        "margin_constraints": [c.name for c, a in zip(truth.constraints, active, strict=True) if a],
        "margin_converged": search.converged,
        "margin_evaluations": search.evaluations,
        "score_u0_margin": score(truth, tw, np.repeat(u0_margin[None, :], len(tw), axis=0)),
        "seconds_star": seconds_star,
    }


def robust_fixed_point(
    truth: SteadyProblem,
    test_w: np.ndarray,
    w_nom: np.ndarray,
    config: SurrogateStudyConfig,
    *,
    extra: np.ndarray | None = None,
) -> OptResult:
    """``u0``: least energy at ``w_nom`` subject to feasibility in every test state.

    Each constraint gets exactly the margin it needs; with one active constraint this is R0 (on
    P2 the R0-style point and this one coincide). ``extra`` (the R0-style point) joins the
    candidates, so this point is never worse than it.
    """
    scales = truth.scales

    def model(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = truth.outputs(u, w_nom)[:, 0]
        g = np.full((len(u), len(scales)), -np.inf)
        for w in test_w:
            g = np.maximum(g, truth.outputs(u, w)[:, 1:] / scales)
        return e, g

    opt = config.optimizer("model")
    return minimize_box(
        model, truth.decisions, candidates=int(opt["candidates"]), polish=int(opt["polish"]),
        feasibility_tolerance=config.feasibility_tolerance, extra=extra,
    )  # fmt: skip


# --------------------------------------------------------------------------- arms
def _decisions(
    surrogate: Surrogate, problem: SteadyProblem, test_w: np.ndarray, kappa: float,
    config: SurrogateStudyConfig, seed: int,
) -> tuple[np.ndarray, list[bool]]:  # fmt: skip
    opt = config.optimizer("surrogate")
    us, feas = [], []
    for i, w in enumerate(test_w):
        r = minimize_box(
            surrogate_batch(surrogate, problem, w, kappa), problem.decisions,
            candidates=int(opt["candidates"]), polish=int(opt["polish"]), seed=seed * 1000 + i,
            feasibility_tolerance=config.feasibility_tolerance,
        )  # fmt: skip
        us.append(r.u)
        feas.append(r.feasible)
    return np.array(us), feas


def _timing(surrogate: Surrogate, problem: SteadyProblem, n: int = 512) -> dict[str, float]:
    x = np.random.default_rng(0).random((n, joint_box(problem).dim))
    t0 = time.perf_counter()
    for row in x[:32]:
        surrogate.predict(row[None, :])
    single = (time.perf_counter() - t0) / 32
    t0 = time.perf_counter()
    surrogate.predict(x)
    return {"predict_single_s": single, "predict_batch_s_per_point": (time.perf_counter() - t0) / n}


def blackbox_arm(
    family: ProblemFamily, key: str, seed: int, config: SurrogateStudyConfig,
    test_w: np.ndarray, log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:  # fmt: skip
    """M1/M2/M3 at every budget and kappa for one seed."""
    truth, scorer = family.truth(), family.truth()
    n_max = max(config.budgets)
    budget = BudgetedTruth(truth, n_max)
    u, w = training_design(truth, seed, n_max)
    disjoint = check_disjoint(truth.disturbances, test_w, w)
    y = np.array([budget(a, b) for a, b in zip(u, w, strict=True)])
    x = joint_box(truth).to_unit(np.hstack((u, w)))
    rows = []
    for n in config.budgets:
        t0 = time.perf_counter()
        model = make_surrogate(key, config.model(key.lower()), seed).fit(x[:n], y[:n])
        fit_s = time.perf_counter() - t0
        timing = _timing(model, truth)
        for kappa in config.kappas:
            t0 = time.perf_counter()
            dec, model_feasible = _decisions(model, truth, test_w, kappa, config, seed)
            rows.append({
                "model": key, "kappa": kappa, "seed": seed, "N": n, "true_evaluations": n,
                "u_hat": dec, "model_feasible": model_feasible, "score": score(scorer, test_w, dec),
                "fit_s": fit_s, "optimize_s": time.perf_counter() - t0,
                "truth_s_per_eval": budget.seconds / budget.calls,
                "disjoint_min_distance": disjoint,
                **timing,
            })  # fmt: skip
            log(f"{family.truth().name} {key} seed {seed} N {n} kappa {kappa} done")
    return rows


def fit_greybox(
    family: ProblemFamily, u: np.ndarray, w: np.ndarray, y: np.ndarray, max_nfev: int
) -> dict[str, Any]:
    """Least squares of the fitted parameters on all outputs, each scaled by its spread in ``y``."""
    specs = family.fitted
    scale = np.maximum(y.std(axis=0), 1e-6 * np.maximum(np.abs(y).mean(axis=0), 1e-9))
    failures = 0

    def residual(theta: np.ndarray) -> np.ndarray:
        nonlocal failures
        model = family.model(theta)
        try:
            pred = model.outputs(u, w)
        except RuntimeError:
            failures += 1
            return np.full(y.size, 1e3)
        out: np.ndarray = ((pred - y) / scale).ravel()
        return out

    x0 = np.array([p.nominal for p in specs])
    res = least_squares(
        residual, x0, bounds=([p.lower for p in specs], [p.upper for p in specs]),
        x_scale="jac", diff_step=1e-4, max_nfev=max_nfev,
    )  # fmt: skip
    return {
        "theta": res.x, "names": [p.name for p in specs], "cost": float(res.cost),
        "status": int(res.status), "nfev": int(res.nfev), "failures": failures,
        "rms_scaled_residual": float(np.sqrt(np.mean(res.fun**2))),
    }  # fmt: skip


def greybox_arm(
    family: ProblemFamily, seed: int, config: SurrogateStudyConfig, test_w: np.ndarray,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:  # fmt: skip
    """M4 at every budget for one seed (kappa 0 only)."""
    truth, scorer = family.truth(), family.truth()
    n_max = max(config.budgets)
    budget = BudgetedTruth(truth, n_max)
    u, w = training_design(truth, seed, n_max)
    disjoint = check_disjoint(truth.disturbances, test_w, w)
    y = np.array([budget(a, b) for a, b in zip(u, w, strict=True)])
    rows = []
    for n in config.budgets:
        t0 = time.perf_counter()
        fit = fit_greybox(family, u[:n], w[:n], y[:n], int(config.model("m4")["max_nfev"]))
        fit_s = time.perf_counter() - t0
        model = family.model(fit["theta"])
        t0 = time.perf_counter()
        opts = [
            optimize_problem(model, tw, config, "model", seed=seed * 1000 + i)
            for i, tw in enumerate(test_w)
        ]
        dec = np.array([o["u"] for o in opts])
        t1 = time.perf_counter()
        model.outputs(dec[:8], test_w[:8])
        model_s = (time.perf_counter() - t1) / 8
        rows.append({
            "model": "M4", "kappa": 0.0, "seed": seed, "N": n, "true_evaluations": n,
            "u_hat": dec, "model_feasible": [bool(o["feasible"]) for o in opts],
            "score": score(scorer, test_w, dec), "fit": fit, "fit_s": fit_s,
            "optimize_s": time.perf_counter() - t0,
            "truth_s_per_eval": budget.seconds / budget.calls,
            "disjoint_min_distance": disjoint, "predict_single_s": model_s,
        })  # fmt: skip
        log(f"{truth.name} M4 seed {seed} N {n} theta {np.round(fit['theta'], 4)} done")
    return rows


def adaptive_batches(initial: int, budgets: Sequence[int], max_batch: int) -> list[int]:
    """Sizes of the sequential batches from ``initial`` to ``max(budgets)``; hits every budget."""
    n, out = initial, []
    for target in sorted(budgets):
        while n < target:
            b = max(1, min(max_batch, n // 2, target - n))
            out.append(b)
            n += b
    return out


def adaptive_arm(
    family: ProblemFamily, kappa: float, seed: int, config: SurrogateStudyConfig,
    test_w: np.ndarray, log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:  # fmt: skip
    """M2 with sequential design (see the module doc); one row per budget."""
    truth, scorer = family.truth(), family.truth()
    n_max = max(config.budgets)
    budget = BudgetedTruth(truth, n_max)
    u0, w0 = training_design(truth, seed, config.adaptive_initial)
    pool = adaptive_pool(truth, seed, n_max, config.adaptive_pool_seed_offset)
    disjoint = check_disjoint(truth.disturbances, test_w, np.vstack((w0, pool)))
    us, ws = list(u0), list(w0)
    ys = [budget(a, b) for a, b in zip(u0, w0, strict=True)]
    box = joint_box(truth)
    settings = config.model("m2")
    opt = config.optimizer("surrogate")
    next_pool, refits, seq_s = 0, 0, 0.0
    batches = iter(
        adaptive_batches(config.adaptive_initial, config.budgets, config.adaptive_max_batch)
    )
    rows = []
    for target in config.budgets:
        while len(ys) < target:
            t0 = time.perf_counter()
            b = next(batches)
            x = box.to_unit(np.hstack((np.array(us), np.array(ws))))
            gp = make_surrogate("M2", settings, seed).fit(x, np.array(ys))
            refits += 1
            for _ in range(b):
                w = pool[next_pool]
                next_pool += 1
                r = minimize_box(
                    surrogate_batch(gp, truth, w, kappa), truth.decisions,
                    candidates=int(opt["candidates"]), polish=int(opt["polish"]),
                    seed=seed * 7919 + next_pool,
                    feasibility_tolerance=config.feasibility_tolerance,
                )  # fmt: skip
                us.append(r.u)
                ws.append(w)
                ys.append(budget(r.u, w))
            seq_s += time.perf_counter() - t0
        if budget.calls != target:
            raise AssertionError(f"budget count {budget.calls} != {target}")
        x = box.to_unit(np.hstack((np.array(us), np.array(ws))))
        t0 = time.perf_counter()
        gp = make_surrogate("M2", settings, seed).fit(x, np.array(ys))
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        dec, model_feasible = _decisions(gp, truth, test_w, kappa, config, seed)
        rows.append({
            "model": "M2a", "kappa": kappa, "seed": seed, "N": target,
            "true_evaluations": budget.calls, "refits": refits, "u_hat": dec,
            "model_feasible": model_feasible, "score": score(scorer, test_w, dec), "fit_s": fit_s,
            "sequential_s": seq_s, "optimize_s": time.perf_counter() - t0,
            "truth_s_per_eval": budget.seconds / budget.calls, "disjoint_min_distance": disjoint,
        })  # fmt: skip
        if target == n_max:  # the designs are nested: the last one holds all of them
            rows[-1]["design_u"], rows[-1]["design_w"] = np.array(us), np.array(ws)
        log(f"{truth.name} M2a kappa {kappa} seed {seed} N {target} done")
    return rows
