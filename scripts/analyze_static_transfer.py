"""Analysis of the static transfer study with the PRE-REGISTERED reading (docs/NN_static_transfer.md).

This docstring and the code below were committed BEFORE the main run.

Data: the main-run artifact ``artifacts/runs/<run>.json.gz`` (``scripts/run_static_transfer.py``).
Metric: RMSE in physical units (K, g/kg, kW per kg/s) against the noise-free plant truth on a named
test set, one value per seed (5 seeds). Primary condition of every hypothesis: ``clean_low``
(primary noise, no input errors). Test sets: ``interpolation`` (the plant's operating range),
``extrapolation`` (outer box minus that range), ``plant_like`` (supply air 75-95 degC, plant-like
load and solid fraction), ``product_change`` (solid fraction 0.58-0.62 instead of 0.38-0.42).

Statistics (decision of the project lead): paired t-test over the seeds, Holm per hypothesis
family (family sizes reported), "resolved better" = Holm-rejected AND the same sign in at least 4
of 5 seeds; the uncorrected verdict is reported beside it; effect size = median of the paired
differences in physical units. "L3 beats X in a cell (plant, N)" = resolved better on EVERY
affected target of the plant (``affected_targets`` of the YAML: P-loss T_out; P-air T_out, Y_out;
P-eta power; P-moist T_out, Y_out; P-comb all three).

Families relative to L1 (fixed before the run): P-loss within, P-air partial, P-eta outside,
P-moist outside, P-comb outside.

Hypotheses (each its own Holm family)
-------------------------------------
H1 (control, P0, interpolation): none of L1, L2, L3, L4, B1, B2 is resolved better than L0 on any
   target at any N (6 x 3 x 6 = 108 tests). On P0 the noise-free truth IS L0, so H1 can only fail
   through an error of the pipeline; the informative part is the reported cost (RMSE) of each
   method where there is nothing to learn, and the zero-shot error of the pretrained net (L3_0).
H2 (within, P-loss, all four test sets, T_out): L3 is resolved better than L1 in no cell
   (4 x 6 = 24 tests). Supported if no such cell.
H3 (outside: P-eta, P-moist, P-comb; interpolation): per plant N* = smallest N from which L3 beats
   L1 at that and every larger N (6 targets-of-plants x 6 N = 36 tests). Supported if N* exists
   for all three plants; partly supported if for one or two (named); not supported if for none.
H4 (interpolation, the five deviating plants): L3 beats L2 in no cell (affected targets: 9 x 6 =
   54 tests). Supported if no cell in which L3 is resolved better than L2 on an affected target.
H5 (extrapolation): (a) L3 resolved better than L4 on every affected target of every deviating
   plant at every N <= 16 (9 x 4 = 36 tests); (b) on P-loss and P-air L1 resolved better than each
   of L0, L2, L3, L4, B1, B2 on every affected target at every N <= 16 (3 x 4 x 6 = 72 tests).
   One family of 108. Supported if (a) and (b); the parts are reported separately.

Reading (interpolation, clean_low; family "reading": L3 against L0, L1, L2, L4 and B1 on the five
deviating plants, affected targets, every N: 5 x 9 x 6 = 270 tests)
------------------------------------------------------------------------------------------------
R1 if some cell (plant, N) has L3 beating all five of L0, L1, L2, L4, B1: pretraining is worth it
   there (the map of such cells is the result).
R2 else, if some cell has L3 beating L0, L1 and L4 (whether L3 beats L2 anywhere is reported):
   with the balance available at run time a residual model does the job; pretraining is not needed.
R3 else, if some cell has L3 beating L4: pretraining only carries the physics where the balance
   itself cannot be used.
none else (L3 beats not even L4 in any cell).
R4 (flag, independent): L3 beats L4 in no cell with N <= 8 on any deviating plant -- a
   contradiction of A1 (Befund 88) for the static case.
The same reading is computed for the other three test sets and for ``clean_high`` (descriptive,
own families).

Core figures per test set (change 4 of the approval)
----------------------------------------------------
Family per test set: L2 against L1 and L3 against L1, five deviating plants, affected targets,
every N (2 x 9 x 6 = 108 tests). Reported: per plant the smallest N from which L2 (resp. L3)
beats L1 (at that and every larger N; ``None`` = not within 64), and the residual error of L1 at
N = 64 (median over seeds, every target).

Descriptive (no hypothesis)
---------------------------
* Input-error condition ``inputerr_low``: per plant, test set, N and target the order of L1, L2,
  B1 by median RMSE, against ``clean_low``; number of cells in which the order changes; pairwise
  verdicts (L1-L2, L1-B1, L2-B1) under input errors, Holm per plant over its test sets, N and
  affected targets.
* ``clean_high``: learning curves and the reading.
* L0 sensitivity to the input set of Model v4.1_2 (``L0_v412``).
* Deviation table (L0 against each plant in units of the primary noise): a plant below 2 sigma on
  an affected target is NOT dropped and nothing is changed; it is marked in the report.
* Learning curves: median and range over seeds per plant, condition, method, N, test set, target.

Output: ``artifacts/runs/<run>_analysis.json`` (never overwritten) and the figures
``docs/figures/<run>_<plant>_<test set>.png``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, _json_default, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.static_transfer.analysis import (  # noqa: E402
    Family,
    MetricTable,
    affected,
    compare,
    smallest_n_from_which,
)
from twinlink_ml.static_transfer.config import TARGET_NAMES, TARGET_UNITS  # noqa: E402
from twinlink_ml.static_transfer.design import TEST_SET_NAMES  # noqa: E402

REPO_ROOT = SCRIPTS.parent
FIGURES = REPO_ROOT / "docs" / "figures"
DEVIATING = ("P-loss", "P-air", "P-eta", "P-moist", "P-comb")
OUTSIDE = ("P-eta", "P-moist", "P-comb")
ALL_METHODS = ("L0", "L0_v412", "L1", "L2", "L3", "L3_0", "L4", "B1", "B2")
PRIMARY = "clean_low"


class Study:
    """Verdicts and summaries over the records of one main run."""

    def __init__(self, run: dict[str, Any]) -> None:
        self.run = run
        self.table = MetricTable(run["units"])
        cfg = run["config"]
        self.budgets = [int(n) for n in cfg["budgets"]]
        self.affected = {k: affected(v) for k, v in cfg["affected_targets"].items()}
        self.alpha = float(cfg["statistics"]["alpha"])
        self.min_sign = int(cfg["statistics"]["min_sign_agreement"])
        self.conditions = run["conditions"]

    def family(self, name: str, specs: list[tuple[str, str, str, str, int, str, str]]) -> Family:
        """specs: (candidate, reference, plant, condition, n, test_set, target)."""
        comps = [compare(self.table, c, r, p, cond, n, ts, t) for c, r, p, cond, n, ts, t in specs]
        return Family(name, comps, self.alpha, self.min_sign)

    def beats(
        self, fam: Family, cand: str, ref: str, plant: str, n: int, ts: str, cond: str
    ) -> bool:
        """Candidate resolved better on every affected target of the plant at this cell."""
        for t in self.affected[plant]:
            cells = fam.get(
                candidate=cand,
                reference=ref,
                plant=plant,
                n=n,
                test_set=ts,
                target=t,
                condition=cond,
            )
            if not cells or cells[0].verdict != "better":
                return False
        return True

    # ------------------------------------------------------------------ hypotheses
    def h1(self) -> dict[str, Any]:
        specs = [
            (m, "L0", "P0", PRIMARY, n, "interpolation", t)
            for m in ("L1", "L2", "L3", "L4", "B1", "B2")
            for n in self.budgets
            for t in TARGET_NAMES
        ]
        fam = self.family("H1", specs)
        cost = {
            m: {
                str(n): {
                    t: self.table.summary("P0", PRIMARY, m, n, "interpolation", t)
                    for t in TARGET_NAMES
                }
                for n in self.budgets
            }
            for m in ("L1", "L2", "L3", "L3_0", "L4", "B1", "B2")
        }
        return {
            "supported": not fam.any_verdict("better"),
            "family": fam.as_dict(),
            "cost_on_P0_interpolation": cost,
        }

    def h2(self) -> dict[str, Any]:
        specs = [
            ("L3", "L1", "P-loss", PRIMARY, n, ts, "outlet_temperature")
            for ts in TEST_SET_NAMES
            for n in self.budgets
        ]
        fam = self.family("H2", specs)
        return {"supported": not fam.any_verdict("better"), "family": fam.as_dict()}

    def h3(self) -> dict[str, Any]:
        specs = [
            ("L3", "L1", p, PRIMARY, n, "interpolation", t)
            for p in OUTSIDE
            for n in self.budgets
            for t in self.affected[p]
        ]
        fam = self.family("H3", specs)
        n_star = {
            p: smallest_n_from_which(
                self.budgets,
                {
                    n: self.beats(fam, "L3", "L1", p, n, "interpolation", PRIMARY)
                    for n in self.budgets
                },
            )
            for p in OUTSIDE
        }
        found = [p for p, v in n_star.items() if v is not None]
        status = (
            "supported" if len(found) == 3 else ("partly supported" if found else "not supported")
        )
        return {"status": status, "n_star": n_star, "family": fam.as_dict()}

    def h4(self) -> dict[str, Any]:
        specs = [
            ("L3", "L2", p, PRIMARY, n, "interpolation", t)
            for p in DEVIATING
            for n in self.budgets
            for t in self.affected[p]
        ]
        fam = self.family("H4", specs)
        cells = [c.as_dict() for c in fam.comparisons if c.verdict == "better"]
        return {"supported": not cells, "cells_l3_better": cells, "family": fam.as_dict()}

    def h5(self) -> dict[str, Any]:
        small = [n for n in self.budgets if n <= 16]
        specs_a = [
            ("L3", "L4", p, PRIMARY, n, "extrapolation", t)
            for p in DEVIATING
            for n in small
            for t in self.affected[p]
        ]
        specs_b = [
            ("L1", ref, p, PRIMARY, n, "extrapolation", t)
            for p in ("P-loss", "P-air")
            for n in small
            for t in self.affected[p]
            for ref in ("L0", "L2", "L3", "L4", "B1", "B2")
        ]
        fam = self.family("H5", specs_a + specs_b)
        part_a = all(fam.all_better(candidate="L3", reference="L4", plant=p) for p in DEVIATING)
        part_b = all(
            fam.all_better(candidate="L1", reference=r, plant=p)
            for p in ("P-loss", "P-air")
            for r in ("L0", "L2", "L3", "L4", "B1", "B2")
        )
        return {
            "supported": part_a and part_b,
            "part_a_L3_better_than_L4": part_a,
            "part_b_L1_best_on_loss_and_air": part_b,
            "part_a_failing_cells": [
                c.as_dict()
                for c in fam.get(candidate="L3", reference="L4")
                if c.verdict != "better"
            ],
            "part_b_failing_cells": [
                c.as_dict() for c in fam.get(candidate="L1") if c.verdict != "better"
            ],
            "family": fam.as_dict(),
        }

    # ------------------------------------------------------------------ reading
    def reading(self, test_set: str, condition: str) -> dict[str, Any]:
        refs = ("L0", "L1", "L2", "L4", "B1")
        specs = [
            ("L3", r, p, condition, n, test_set, t)
            for r in refs
            for p in DEVIATING
            for n in self.budgets
            for t in self.affected[p]
        ]
        fam = self.family(f"reading/{test_set}/{condition}", specs)
        cells = {
            f"{p}/N{n}": {r: self.beats(fam, "L3", r, p, n, test_set, condition) for r in refs}
            for p in DEVIATING
            for n in self.budgets
        }
        r1 = [k for k, v in cells.items() if all(v.values())]
        r2 = [k for k, v in cells.items() if v["L0"] and v["L1"] and v["L4"]]
        r3 = [k for k, v in cells.items() if v["L4"]]
        beats_l2 = [k for k, v in cells.items() if v["L2"]]
        r4 = not any(v["L4"] for k, v in cells.items() if int(k.split("/N")[1]) <= 8)
        reading = "R1" if r1 else "R2" if r2 else "R3" if r3 else "none"
        return {
            "reading": reading,
            "R4_flag": r4,
            "cells_R1": r1,
            "cells_L3_beats_L0_L1_L4": r2,
            "cells_L3_beats_L4": r3,
            "cells_L3_beats_L2": beats_l2,
            "cells": cells,
            "family": fam.as_dict(),
        }

    # ------------------------------------------------------------------ core figures
    def core(self, test_set: str) -> dict[str, Any]:
        specs = [
            (c, "L1", p, PRIMARY, n, test_set, t)
            for c in ("L2", "L3")
            for p in DEVIATING
            for n in self.budgets
            for t in self.affected[p]
        ]
        fam = self.family(f"core/{test_set}", specs)
        smallest = {
            c: {
                p: smallest_n_from_which(
                    self.budgets,
                    {n: self.beats(fam, c, "L1", p, n, test_set, PRIMARY) for n in self.budgets},
                )
                for p in DEVIATING
            }
            for c in ("L2", "L3")
        }
        residual = {
            p: {t: self.table.summary(p, PRIMARY, "L1", 64, test_set, t) for t in TARGET_NAMES}
            for p in ("P0", *DEVIATING)
        }
        return {
            "smallest_n_beating_L1": smallest,
            "L1_residual_at_64": residual,
            "family": fam.as_dict(),
        }

    # ------------------------------------------------------------------ descriptive
    def input_errors(self) -> dict[str, Any]:
        if "inputerr_low" not in self.conditions:
            return {"available": False}
        trio = ("L1", "L2", "B1")
        changes: dict[str, Any] = {}
        families: dict[str, Any] = {}
        for p in ("P0", *DEVIATING):
            changed = []
            total = 0
            for ts in TEST_SET_NAMES:
                for n in self.budgets:
                    for t in TARGET_NAMES:
                        orders = {}
                        for cond in (PRIMARY, "inputerr_low"):
                            med = {
                                m: self.table.summary(p, cond, m, n, ts, t)["median"] for m in trio
                            }
                            orders[cond] = sorted(trio, key=lambda m: med[m])
                        total += 1
                        if orders[PRIMARY] != orders["inputerr_low"]:
                            changed.append({"test_set": ts, "n": n, "target": t, **orders})
            targets = self.affected.get(p, list(TARGET_NAMES))
            specs = [
                (a, b, p, "inputerr_low", n, ts, t)
                for a, b in (("L1", "L2"), ("L1", "B1"), ("L2", "B1"))
                for ts in TEST_SET_NAMES
                for n in self.budgets
                for t in targets
            ]
            fam = self.family(f"inputerr/{p}", specs)
            specs_clean = [(a, b, p, PRIMARY, n, ts, t) for a, b, _, _, n, ts, t in specs]
            fam_clean = self.family(f"inputerr-reference-clean/{p}", specs_clean)
            flips = [
                {"input_errors": ci.as_dict(), "clean_verdict": cc.verdict}
                for ci, cc in zip(fam.comparisons, fam_clean.comparisons, strict=True)
                if ci.verdict != cc.verdict
            ]
            changes[p] = {
                "cells": total,
                "order_changed": len(changed),
                "order_changes": changed,
                "pairwise_verdict_changes": flips,
            }
            families[p] = fam.as_dict()
        return {"available": True, "changes": changes, "families": families}

    def learning_curves(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in ("P0", *DEVIATING):
            for cond in self.conditions:
                for ts in TEST_SET_NAMES:
                    for m in ALL_METHODS:
                        for t in TARGET_NAMES:
                            key = f"{p}|{cond}|{ts}|{m}|{t}"
                            out[key] = {
                                str(n): self.table.summary(p, cond, m, n, ts, t)
                                for n in self.budgets
                            }
        return out

    def l1_parameters(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for u in self.run["units"]:
            if u["condition"] != PRIMARY:
                continue
            for n, methods in u["fitted"].items():
                key = f"{u['plant']}|N{n}"
                out.setdefault(key, []).append(
                    {"seed": u["seed"], **methods["L1"]["info"]["parameters"]}
                )
        return out


def figures(study: Study, name: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    colors = {
        "L0": "#777777",
        "L1": "#1f77b4",
        "L2": "#2ca02c",
        "L3": "#d62728",
        "L4": "#9467bd",
        "B1": "#8c564b",
        "B2": "#e377c2",
    }
    for p in DEVIATING:
        for ts in ("interpolation", "extrapolation", "plant_like", "product_change"):
            targets = study.affected[p]
            fig, axes = plt.subplots(
                1, len(targets), figsize=(4.2 * len(targets), 3.4), squeeze=False
            )
            for ax, t in zip(axes[0], targets, strict=True):
                j = TARGET_NAMES.index(t)
                for m, col in colors.items():
                    med = [study.table.summary(p, PRIMARY, m, n, ts, t) for n in study.budgets]
                    ax.plot(
                        study.budgets,
                        [s["median"] for s in med],
                        marker="o",
                        ms=3,
                        color=col,
                        label=m,
                    )
                    ax.fill_between(
                        study.budgets,
                        [s["min"] for s in med],
                        [s["max"] for s in med],
                        color=col,
                        alpha=0.12,
                    )
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xlabel("plant points N")
                ax.set_ylabel(f"RMSE {t} [{TARGET_UNITS[j]}]")
                ax.set_title(f"{p}, {ts}", fontsize=9)
            axes[0][0].legend(fontsize=7, ncol=2)
            fig.tight_layout()
            path = FIGURES / f"{name}_{p}_{ts}.png"
            if path.exists():
                raise SystemExit(f"{path} exists -- figures are never overwritten")
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(str(path.relative_to(REPO_ROOT)))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="name of the main-run artifact (without .json.gz)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out_path = RUNS_DIR / f"{args.run}_analysis.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists -- artifacts are never overwritten")
    require_clean_tree()
    with gzip.open(RUNS_DIR / f"{args.run}.json.gz", "rt", encoding="utf-8") as fh:
        run = json.load(fh)
    if run.get("dry_run"):
        raise SystemExit("a dry run is never analysed")
    if run["failures"]:
        print(f"WARNING: {len(run['failures'])} failed units in the run", flush=True)
    s = Study(run)
    readings = {
        f"{ts}/{cond}": s.reading(ts, cond)
        for cond in s.conditions
        if cond != "inputerr_low"
        for ts in TEST_SET_NAMES
    }
    result = {
        "provenance": provenance(),
        "run": args.run,
        "run_commit": run["provenance"]["git/commit"],
        "primary_condition": PRIMARY,
        "H1": s.h1(),
        "H2": s.h2(),
        "H3": s.h3(),
        "H4": s.h4(),
        "H5": s.h5(),
        "reading": readings,
        "core": {ts: s.core(ts) for ts in TEST_SET_NAMES},
        "input_errors": s.input_errors(),
        "deviation_table": run["deviation_table"],
        "coverage": run["coverage"],
        "pretraining": run["pretraining"],
        "learning_curves": s.learning_curves(),
        "l1_parameters": s.l1_parameters(),
        "failures": run["failures"],
    }
    result["figures"] = [] if args.no_figures else figures(s, args.run)
    out_path.write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {out_path}")
    primary = readings["interpolation/clean_low"]
    print(
        json.dumps(
            {
                "H1": result["H1"]["supported"],
                "H2": result["H2"]["supported"],
                "H3": [result["H3"]["status"], result["H3"]["n_star"]],
                "H4": result["H4"]["supported"],
                "H5": [
                    result["H5"]["part_a_L3_better_than_L4"],
                    result["H5"]["part_b_L1_best_on_loss_and_air"],
                ],
                "reading": [primary["reading"], primary["R4_flag"]],
                "family_sizes": {
                    k: result[k]["family"]["size"] for k in ("H1", "H2", "H3", "H4", "H5")
                },
            },
            indent=1,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
