r"""Case-ranking study: analysis of a run of ``scripts/run_case_ranking.py`` (readings pre-registered).

docs/38_case_ranking.md. Reads ``artifacts/runs/<run>.json`` and ``<run>.npz``; writes
``artifacts/runs/<name>.json`` and the figures ``docs/figures/case_ranking/<name>_*.png``. Never
overwrites. Clean tree required.

Readings fixed BEFORE the main run (committed with this docstring; the numbers cannot choose them)
---------------------------------------------------------------------------------------------------

Target quantity: total fuel of both heaters after recovery, kg/h, corrected gas formula; per case
1-6 and case 0 (no recovery). Every statement is made **per mode** (partner, physical unpressurised
95 degC, physical pressurised; docs/11 section 13.6) and **per variant**:

* ``fixed`` -- circuit flows of the workbook (40 / 30 kg/h): "does the statement of the workbook
  hold" (main computation, decision of the project lead F6);
* ``optimised`` -- both circuit flows chosen per case and point within 20-150 kg/h (multistart):
  every topology at its own best operating point.

For a recommendation to the application partner the **physical unpressurised** mode is the main
variant (decision of the project lead F2, as Befund 114); the partner mode shows whether the
workbook says the same. "The Excel-best case" = case 1 (partner mode, workbook point).

Test sets. Primary: the admissible points of the Sobol set ``full`` (the whole box). Secondary:
``workbook_neighbourhood`` (workbook point +- 10 % of the box width per input, cut at the box) and
``plant_like`` (dryer 2 supply air 75-95 degC). ``rule_test``: own Sobol stream over the box, only
the test set of the rule (Q2). The optimised variant uses the first 4096 / 1024 / 1024 / 2048 points
of the same streams (nested prefixes).

Admissibility (fixed before the run): a point is admissible when both exhausts are at or below
90 % relative humidity (IAPWS-IF97 saturation) in every mode and no mode failed at it. Inadmissible
points are counted (by dryer), never dropped silently. Counted and reported, not excluded: stages
without driving force, stages with negative recovery, and (partner mode) a medium above 99.974 degC
(the boiling point at 1 atm, docs/11 section 13.5; flagged, not excluded).

Ties (fixed before the run; decision of the project lead F4):

* exact: two fuels differ by less than 1e-9 kg/h (docs/11 section 13);
* practical: gap < 1 % of the saving of the pointwise best case against case 0 (placeholder; never
  below the exact tie); sensitivity 0.5 % and 2 %.

Per point: **strict** = case 1 lower than every other case by at least 1e-9 kg/h; **co-best** =
case 1 within the exact tie of the best; **practical** = case 1 best or practically equal to the
best. The share ``p`` of admissible points with the flag carries a 95 % Clopper-Pearson interval.

Readings of "case 1", each computed twice, on ``p_strict`` and on ``p_practical`` (lead F4: "case
1 best or practically equal" is counted in addition to "case 1 strictly best"), per mode and
variant, on the primary set:

* **"case 1 robust"**: ``p >= 0.99``;
* **"conditional"**: ``0.50 <= p < 0.99`` (then with the rule of Q2);
* **"not robust"**: ``p < 0.50``.

The reading is taken from the point estimate; when the 95 % interval contains the threshold the
reading is reported with the flag "at the threshold". The secondary sets get the same reading,
labelled secondary.

Q1 (per mode and variant): "a case is best in the whole admissible space" when one case is co-best
at every admissible point of the primary set (``p_co-best = 1``); additionally with the practical
tie. Otherwise the case with the largest co-best share is named with its share.

Q2: shallow decision tree (depth <= 3, at least 50 points per leaf), target "case 1 strictly best"
(and a second tree for "case 1 practically best"), on the eight inputs and derived quantities
(:mod:`twinlink_ml.case_ranking.rule`); trained on the admissible points of ``full`` (fixed: 32768
candidates; optimised: 4096), tested on the admissible points of ``rule_test``. Reading: the rule
"describes the boundary" when its test accuracy is at least 0.95 (placeholder); the accuracy
outside the practical tie is reported as well. A description, not physics. Not fitted when the
training target has a single class (then Q1 already answers).

Q3: regret of always choosing case 1 instead of the pointwise best case, ``F_1 - min_k F_k``,
over the admissible points: median, 95th percentile and maximum; absolute (kg/h), relative to the
best fuel and relative to the best case's saving against case 0. In the optimised variant also the
regret of "case 1 at the workbook's flows" against "the best case at its optimised flows" (same
points: the prefix of the fixed set).

Also reported: saving of every case against case 0, the ranking of all six cases (distribution of
the rank of each case, the most frequent orders), the gap between places 1 and 2, the convergence
of the shares over nested prefixes of ``full`` (fixed variant), and the 1D and 2D cuts through the
workbook point as figures (no share or rank claim is read from the cuts).

The computation is deterministic and noise-free; ``paired_resolution`` does not apply. The only
uncertainty is the sample of the space (binomial intervals, convergence).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    RUNS_DIR,
    _json_default,
    provenance,
    require_clean_tree,
)

from twinlink_ml.case_ranking import ranking as rk  # noqa: E402
from twinlink_ml.case_ranking.config import CaseRankingConfig as Cfg  # noqa: E402
from twinlink_ml.case_ranking.config import load_config  # noqa: E402
from twinlink_ml.case_ranking.evaluator import MODES  # noqa: E402
from twinlink_ml.case_ranking.rule import FEATURE_NAMES, features, fit_rule  # noqa: E402
from twinlink_ml.case_ranking.space import INPUT_NAMES  # noqa: E402
from twinlink_ml.case_ranking.workbook import WORKBOOK_POINT  # noqa: E402

REPO_ROOT = SCRIPTS.parent
FIG_DIR = REPO_ROOT / "docs" / "figures" / "case_ranking"
SETS = ("full", "workbook_neighbourhood", "plant_like")
VARIANTS = ("fixed", "optimised")
THRESHOLDS = (0.99, 0.50)
CASE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
"""Categorical slots 1-6 of the validated default palette (dataviz skill), case 1..6."""
INADMISSIBLE_GREY = "#d9d8d4"


def reading(est: rk.ShareEstimate) -> dict[str, Any]:
    p, (lo, hi) = est.share, est.ci95
    label = "robust" if p >= 0.99 else "conditional" if p >= 0.50 else "not robust"
    at = [t for t in THRESHOLDS if lo <= t <= hi]
    return {"reading": label, "at_threshold": at, **est.as_dict()}


def _q(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan"), "n": 0}
    return {
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        "n": int(v.size),
    }


class Run:
    """The arrays of one run artifact."""

    def __init__(self, arrays: Any) -> None:
        self.a = arrays

    def get(self, variant: str, label: str, mode: str | None, q: str) -> np.ndarray:
        key = f"{variant}/{label}/{q}" if mode is None else f"{variant}/{label}/{mode}/{q}"
        return self.a[key]

    def admissible(self, variant: str, label: str, max_rh: float) -> dict[str, Any]:
        n = len(self.get(variant, label, None, "points"))
        ok = np.ones(n, dtype=bool)
        by_dryer = np.zeros(2, dtype=int)
        errors = np.zeros(n, dtype=bool)
        rh_ok_per_mode = {}
        for m in MODES:
            err = self.get(variant, label, m, "error")
            rh = self.get(variant, label, m, "exh_rh")
            rh_ok = (rh <= max_rh).all(axis=1) & ~err
            rh_ok_per_mode[m] = rh_ok
            errors |= err
            ok &= rh_ok
        rh_p = self.get(variant, label, "partner", "exh_rh")
        by_dryer = [int((rh_p[:, i] > max_rh).sum()) for i in (0, 1)]
        disagree = int(sum((rh_ok_per_mode[m] != rh_ok_per_mode["partner"]).sum() for m in MODES))
        return {
            "mask": ok,
            "candidates": n,
            "admissible": int(ok.sum()),
            "inadmissible": int((~ok).sum()),
            "inadmissible_exhaust_1": by_dryer[0],
            "inadmissible_exhaust_2": by_dryer[1],
            "points_with_errors": int(errors.sum()),
            "mode_disagreements": disagree,
        }


def _flags_block(run: Run, variant: str, label: str, mode: str, mask: np.ndarray, cfg: Cfg) -> dict:
    fuel = run.get(variant, label, mode, "fuel")[mask]
    out: dict[str, Any] = {}
    f = rk.case_one_flags(fuel, cfg.tie_exact_kg_per_h, cfg.tie_practical_fraction)
    out["case_1"] = {
        "strict": reading(rk.share(f["strict"])),
        "co_best": rk.share(f["co_best"]).as_dict(),
        "practical": reading(rk.share(f["practical"])),
        "practical_sensitivity": {
            str(s): reading(
                rk.share(rk.case_one_flags(fuel, cfg.tie_exact_kg_per_h, s)["practical"])
            )
            for s in cfg.tie_practical_sensitivity
        },
    }
    if len(fuel) == 0:
        return out
    labels = rk.winner_labels(fuel, cfg.tie_exact_kg_per_h)
    ranks = rk.competition_ranks(fuel, cfg.tie_exact_kg_per_h)
    best = rk.best_fuel(fuel)
    tau = rk.practical_threshold(fuel, cfg.tie_practical_fraction, cfg.tie_exact_kg_per_h)
    cases = fuel[:, 1:]
    co_best_share = {
        str(k + 1): float((cases[:, k] - best < cfg.tie_exact_kg_per_h).mean()) for k in range(6)
    }
    prac_share = {str(k + 1): float((cases[:, k] - best < tau).mean()) for k in range(6)}
    everywhere = [k for k, v in co_best_share.items() if v == 1.0]
    everywhere_p = [k for k, v in prac_share.items() if v == 1.0]
    orders = Counter(
        " ".join(str(c + 1) for c in np.argsort(row, kind="stable")) for row in cases
    ).most_common(5)
    r = rk.regret(fuel)
    saving = fuel[:, [0]] - cases
    out.update(
        {
            "q1": {
                "co_best_share_per_case": co_best_share,
                "practical_share_per_case": prac_share,
                "best_everywhere_exact": everywhere,
                "best_everywhere_practical": everywhere_p,
                "largest_co_best": max(co_best_share, key=co_best_share.get),
            },
            "winners": dict(Counter(labels.tolist()).most_common()),
            "rank_distribution": {
                str(k + 1): {str(r_): int((ranks[:, k] == r_).sum()) for r_ in range(1, 7)}
                for k in range(6)
            },
            "mean_rank": {str(k + 1): float(ranks[:, k].mean()) for k in range(6)},
            "most_frequent_orders": [[o, c] for o, c in orders],
            "top_two_gap_kg_per_h": _q(rk.top_two_gap(fuel)),
            "regret": {
                "kg_per_h": _q(r["kg_per_h"]),
                "relative_to_best_percent": _q(100 * r["relative_to_best"]),
                "relative_to_best_saving_percent": _q(100 * r["relative_to_best_saving"]),
            },
            "saving_against_case_0": {
                str(k + 1): {
                    "kg_per_h": _q(saving[:, k]),
                    "percent": _q(100 * saving[:, k] / fuel[:, 0]),
                }
                for k in range(6)
            },
            "case_0_fuel_kg_per_h": _q(fuel[:, 0]),
            "counted": _counted(run, variant, label, mode, mask, cfg),
        }
    )
    return out


def _counted(
    run: Run, variant: str, label: str, mode: str, mask: np.ndarray, cfg: Cfg
) -> dict[str, Any]:
    ndf = run.get(variant, label, mode, "ndf")[mask]
    neg = run.get(variant, label, mode, "neg")[mask]
    out = {
        "points_with_stage_without_driving_force": {
            str(k): int((ndf[:, k] > 0).sum()) for k in range(1, 7)
        },
        "points_with_negative_recovery": {str(k): int((neg[:, k] > 0).sum()) for k in range(1, 7)},
    }
    if mode == "partner":
        med = run.get(variant, label, mode, "medium_max")[mask]
        out["points_medium_above_boiling"] = {
            str(k): int((med[:, k] > cfg.partner_medium_boiling_celsius).sum()) for k in range(1, 7)
        }
    return out


def _convergence(run: Run, mode: str, cfg: Cfg, max_rh: float) -> list[dict[str, Any]]:
    adm = run.admissible("fixed", "full", max_rh)["mask"]
    fuel = run.get("fixed", "full", mode, "fuel")
    out = []
    for n in cfg.convergence_prefixes:
        if n > len(adm):
            continue
        m = adm[:n]
        f = rk.case_one_flags(fuel[:n][m], cfg.tie_exact_kg_per_h, cfg.tie_practical_fraction)
        out.append(
            {
                "candidates": n,
                "admissible": int(m.sum()),
                "strict": rk.share(f["strict"]).as_dict(),
                "practical": rk.share(f["practical"]).as_dict(),
            }
        )
    return out


def _cross_regret(run: Run, mode: str, max_rh: float) -> dict[str, Any]:
    """Case 1 at the workbook's flows against the best case at its optimised flows."""
    opt_pts = run.get("optimised", "full", None, "points")
    fix_pts = run.get("fixed", "full", None, "points")[: len(opt_pts)]
    if not np.array_equal(opt_pts, fix_pts):
        return {"error": "points of the optimised set are not the prefix of the fixed set"}
    adm = run.admissible("optimised", "full", max_rh)["mask"]
    adm &= run.admissible("fixed", "full", max_rh)["mask"][: len(opt_pts)]
    f1_fixed = run.get("fixed", "full", mode, "fuel")[: len(opt_pts)][adm][:, 1]
    best_opt = rk.best_fuel(run.get("optimised", "full", mode, "fuel")[adm])
    f0 = run.get("fixed", "full", mode, "fuel")[: len(opt_pts)][adm][:, 0]
    r = f1_fixed - best_opt
    return {
        "kg_per_h": _q(r),
        "relative_to_best_percent": _q(100 * r / best_opt),
        "relative_to_best_saving_percent": _q(100 * r / (f0 - best_opt)),
    }


def _rule(run: Run, variant: str, mode: str, cfg: Cfg, max_rh: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sets = {}
    for label in ("full", "rule_test"):
        adm = run.admissible(variant, label, max_rh)["mask"]
        pts = run.get(variant, label, None, "points")[adm]
        fuel = run.get(variant, label, mode, "fuel")[adm]
        x = features(
            pts,
            run.get(variant, label, mode, "exh_T")[adm],
            run.get(variant, label, mode, "exh_Y")[adm],
        )
        flags = rk.case_one_flags(fuel, cfg.tie_exact_kg_per_h, cfg.tie_practical_fraction)
        gap_others = fuel[:, 2:].min(axis=1) - fuel[:, 1]
        tau = rk.practical_threshold(fuel, cfg.tie_practical_fraction, cfg.tie_exact_kg_per_h)
        sets[label] = (x, flags, np.abs(gap_others) >= tau)
    for target in ("strict", "practical"):
        x_tr, f_tr, _ = sets["full"]
        x_te, f_te, clear = sets["rule_test"]
        if len(np.unique(f_tr[target])) < 2:
            out[target] = {"fitted": False, "reason": "single class in the training target"}
            continue
        fit = fit_rule(
            x_tr,
            f_tr[target],
            x_te,
            f_te[target],
            clear,
            cfg.rule_max_depth,
            cfg.rule_min_samples_leaf,
        )
        d = fit.as_dict()
        d["fitted"] = True
        d["reading"] = (
            "describes the boundary"
            if fit.test_accuracy >= cfg.rule_accuracy_reading
            else "does not describe the boundary"
        )
        out[target] = d
    return out


# ----------------------------------------------------------------------------- figures
def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _figures(run: Run, cfg: Cfg, name: str, max_rh: float) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    labels_short = {
        "ambient_celsius": "ambient [degC]",
        "ambient_relative_humidity": "ambient RH [-]",
        "inlet_1_celsius": "supply air dryer 1 [degC]",
        "inlet_2_celsius": "supply air dryer 2 [degC]",
        "spray_load_1_kg_per_kg": "spray load 1 [kg/kg]",
        "spray_load_2_kg_per_kg": "spray load 2 [kg/kg]",
        "solid_fraction_1": "solid fraction 1 [-]",
        "solid_fraction_2": "solid fraction 2 [-]",
    }
    wb = dict(zip(INPUT_NAMES, WORKBOOK_POINT.as_array(), strict=True))
    for variant in VARIANTS:
        for mode in MODES:
            fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=False)
            for ax, name_in in zip(axes.ravel(), INPUT_NAMES, strict=True):
                label = f"cut1d_{name_in}"
                pts = run.get(variant, label, None, "points")
                if len(pts) == 0:
                    continue
                xv = pts[:, INPUT_NAMES.index(name_in)]
                fuel = run.get(variant, label, mode, "fuel")
                rh = np.concatenate(
                    [run.get(variant, label, m, "exh_rh").max(axis=1)[:, None] for m in MODES],
                    axis=1,
                ).max(axis=1)
                bad = ~(rh <= max_rh)
                gap = fuel[:, 1:] - np.nanmin(fuel[:, 1:], axis=1, keepdims=True)
                for k in range(6):
                    ax.plot(xv, gap[:, k], color=CASE_COLOURS[k], lw=2, label=f"case {k + 1}")
                if bad.any():
                    ax.fill_between(
                        xv, 0.0, np.nanmax(gap), where=bad, color=INADMISSIBLE_GREY, alpha=0.6, lw=0
                    )
                ax.axvline(wb[name_in], color="#52514e", lw=0.8, ls=":")
                ax.set_xlabel(labels_short[name_in])
                ax.set_ylabel("fuel minus best case [kg/h]")
                ax.grid(alpha=0.25, lw=0.5)
            axes[0, 0].legend(fontsize=7, frameon=False)
            fig.suptitle(
                f"1D cuts through the workbook point -- {mode}, {variant} circuit flows "
                "(grey: inadmissible, dotted: workbook value)"
            )
            fig.tight_layout()
            path = FIG_DIR / f"{name}_cut1d_{variant}_{mode}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(_rel(path))
    cmap = ListedColormap([INADMISSIBLE_GREY, *CASE_COLOURS])
    for p, (a, b) in enumerate(cfg.two_d_pairs):
        label = f"cut2d_{p}"
        pts = run.get("fixed", label, None, "points")
        if len(pts) == 0:
            continue
        ia, ib = INPUT_NAMES.index(a), INPUT_NAMES.index(b)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        adm = run.admissible("fixed", label, max_rh)["mask"]
        for ax, mode in zip(axes, MODES, strict=True):
            fuel = run.get("fixed", label, mode, "fuel")
            cases = fuel[:, 1:]
            win = np.argmin(cases, axis=1) + 1
            gap = rk.top_two_gap(fuel)
            tau = rk.practical_threshold(fuel, cfg.tie_practical_fraction, cfg.tie_exact_kg_per_h)
            code = np.where(adm, win, 0)
            ax.scatter(
                pts[:, ia],
                pts[:, ib],
                c=code,
                cmap=cmap,
                vmin=-0.5,
                vmax=6.5,
                s=9,
                marker="s",
                linewidths=0,
            )
            near = adm & (gap < tau)
            ax.scatter(pts[near, ia], pts[near, ib], s=2, c="#0b0b0b", linewidths=0)
            ax.plot(wb[a], wb[b], marker="*", ms=14, color="#0b0b0b", mec="#fcfcfb")
            ax.set_xlabel(labels_short[a])
            ax.set_ylabel(labels_short[b])
            ax.set_title(mode)
        handles = [
            plt.Line2D([], [], marker="s", ls="", color=c, label=f"case {k + 1} best")
            for k, c in enumerate(CASE_COLOURS)
        ]
        handles.append(
            plt.Line2D([], [], marker="s", ls="", color=INADMISSIBLE_GREY, label="inadmissible")
        )
        handles.append(
            plt.Line2D(
                [], [], marker=".", ls="", color="#0b0b0b", label="places 1-2 practically equal"
            )
        )
        fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False)
        fig.suptitle("Best case, 2D cut through the workbook point (star), fixed circuit flows")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = FIG_DIR / f"{name}_cut2d_{p}_{a}_{b}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(_rel(path))
    return written


def _figures_summary(result: dict[str, Any], name: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    mode_colours = dict(zip(MODES, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ("strict", "practical"), strict=True):
        for mode, conv in result["convergence"].items():
            n = [c["admissible"] for c in conv]
            p = [c[key]["share"] for c in conv]
            lo = [c[key]["ci95"][0] for c in conv]
            hi = [c[key]["ci95"][1] for c in conv]
            ax.plot(n, p, marker="o", ms=4, lw=2, color=mode_colours[mode], label=mode)
            ax.fill_between(n, lo, hi, color=mode_colours[mode], alpha=0.15, lw=0)
        for t in THRESHOLDS:
            ax.axhline(t, color="#52514e", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("admissible points (nested Sobol prefixes of 'full')")
        ax.set_ylabel(f"share case 1 {key}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Convergence of the share of case 1 (fixed circuit flows, 95 % interval)")
    fig.tight_layout()
    path = FIG_DIR / f"{name}_convergence.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(_rel(path))
    return written


def _regret_ecdf(run: Run, name: str, max_rh: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_colours = dict(zip(MODES, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, variant in zip(axes, VARIANTS, strict=True):
        adm = run.admissible(variant, "full", max_rh)["mask"]
        for mode in MODES:
            r = rk.regret(run.get(variant, "full", mode, "fuel")[adm])["kg_per_h"]
            r = np.sort(r)
            ax.plot(
                r, np.arange(1, len(r) + 1) / len(r), lw=2, color=mode_colours[mode], label=mode
            )
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("regret of always choosing case 1 [kg/h]")
        ax.set_title(f"{variant} circuit flows")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("share of admissible points (ECDF)")
    fig.tight_layout()
    path = FIG_DIR / f"{name}_regret_ecdf.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return _rel(path)


def main() -> None:
    global FIG_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="name of the run artifact (without suffix)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR, help="only to test on a dry run")
    args = ap.parse_args()
    FIG_DIR = args.fig_dir.resolve()
    out_path = RUNS_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    if FIG_DIR.exists() and any(FIG_DIR.glob(f"{args.name}_*")):
        raise SystemExit("figures of this name exist -- use a new name")
    require_clean_tree()
    prov = provenance(NO_HEAT_NETWORK)
    record = json.loads((RUNS_DIR / f"{args.run}.json").read_text(encoding="utf-8"))
    cfg = load_config()
    if record["config_fingerprint"] != cfg.fingerprint:
        raise SystemExit("the run was made with another config")
    run = Run(np.load(RUNS_DIR / record["arrays"]))
    max_rh = cfg.max_exhaust_relative_humidity
    result: dict[str, Any] = {
        "provenance": prov,
        "run": args.run,
        "run_commit": record["provenance"]["git/commit"],
        "run_dirty": record["provenance"]["git/dirty"],
        "run_failures": len(record["failures"]),
        "run_point_errors": record["point_errors"],
        "config_fingerprint": cfg.fingerprint,
        "workbook_point": dict(zip(INPUT_NAMES, WORKBOOK_POINT.as_array(), strict=True)),
        "feature_names": list(FEATURE_NAMES),
        "admissibility": {},
        "results": {},
    }
    for variant in VARIANTS:
        for label in (*SETS, "rule_test"):
            adm = run.admissible(variant, label, max_rh)
            result["admissibility"][f"{variant}/{label}"] = {
                k: v for k, v in adm.items() if k != "mask"
            }
            if label == "rule_test":
                continue
            for mode in MODES:
                result["results"][f"{variant}/{label}/{mode}"] = _flags_block(
                    run, variant, label, mode, adm["mask"], cfg
                )
    result["convergence"] = {m: _convergence(run, m, cfg, max_rh) for m in MODES}
    result["cross_regret_case1_workbook_flows_vs_best_optimised"] = {
        m: _cross_regret(run, m, max_rh) for m in MODES
    }
    result["rule"] = {f"{v}/{m}": _rule(run, v, m, cfg, max_rh) for v in VARIANTS for m in MODES}
    figs = _figures(run, cfg, args.name, max_rh)
    figs += _figures_summary(result, args.name)
    figs.append(_regret_ecdf(run, args.name, max_rh))
    result["figures"] = figs
    out_path.write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path} and {len(figs)} figures")


if __name__ == "__main__":
    main()
