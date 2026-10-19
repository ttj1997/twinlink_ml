r"""Convergence check of Befund 94 (epoch cap): does a budget of 600 epochs change order or gaps.

Addendum to Befund 94: energy and bed temperature, GRU / LSTM / Transformer.

PRE-REGISTRATION. Committed together with ``run_convergence_check_94.py`` BEFORE any record of the run
exists; no rule below is changed after the records are seen.

Question. The addendum to Befund 94 on the epoch cap counted that the fixed budget of 200 epochs binds for a
large part of the heads of GRU, LSTM and Transformer, mostly on energy and bed temperature (at ``n_train``
64: Transformer 10 of 15 heads, LSTM 5, GRU 4). Does the order GRU >= LSTM > Transformer of Befund 94 c
on energy and bed temperature, and do the gaps, survive a budget of 600 epochs?

Run (driver ``run_convergence_check_94.py``): arm N of Befund 94, the same data, seeds 1-5, splits,
holdout and configuration; the ONLY change is ``max_epochs`` 600 instead of 200, patience unchanged (20).
Models GRU, LSTM, Transformer; ``n_train`` 16 and 64. Every model trains all three heads (one network per
head, as in Befund 94); the moisture head is evaluated only at ``n_train`` 64 and only as a control.
Records ``conv600_<run>_N.jsonl.gz``; the 200-epoch side is ``rerun_full_N.jsonl`` of Befund 94.

Metric and cells, as Befund 94: ``rmse_run_median_physical``, primary evaluation variant of arm N (own pool,
planned windows), horizons 1 and 36, test sets excited and constant-run holdout (always named). Paired t
test over the five seeds (``paired_resolution`` via ``compare``; 4 degrees of freedom); verdict per cell
"candidate better" / "candidate worse" / "unresolved", uncorrected.

* PAIRS: GRU against LSTM and Transformer against LSTM (the rank cells of Befund 94) and, for the gap of
  GRU to Transformer, Transformer against GRU (not a cell of Befund 94).
* PRIMARY CELLS: energy and bed temperature, ``n_train`` 16 and 64, both test sets, h = 1 and 36:
  16 per pair. CONTROL CELLS: moisture at ``n_train`` 64 (4 per pair), not part of any reading.
* Next to every verdict: the effect size of Befund 98 addendum (ratio of the medians over the seeds,
  candidate / reference, > 1 = candidate worse), the sign consistency (seeds in which the candidate is
  worse), the relative difference of the medians and the p value.

Multiplicity (Holm 5 %; a cell is resolved after Holm if its uncorrected verdict is resolved and its
adjusted p value is below 0.05):

* MAIN FAMILY = the family of Befund 94/98 for arm N: all 336 rank cells (seven candidates against LSTM,
  ``n_train`` 8-64, both test sets, three channels, two horizons), recomputed from ``rerun_full_N.jsonl``;
  the script ASSERTS the Befund 94 counts (260 resolved uncorrected, 70 after Holm). For the 600-epoch side
  the 40 cells covered by this run (GRU and Transformer against LSTM; primary and control cells) are
  replaced by their 600-epoch values, the other 296 cells keep their Befund 94 values, and Holm is applied
  to the 336 again. Same family, same size: the Holm verdicts of the 40 cells are compared old against new.
* SENSITIVITY FAMILIES (reported, not used for a reading): the 40 cells alone, at 200 and at 600; the 20
  Transformer-against-GRU cells alone, at 200 and at 600.

Training budget used: per head-training (network, seed, ``n_train``, head; de-duplicated over the
evaluation records as in the epoch-cap addendum): epochs run, best epoch; counts per network, budget and
head of the head-trainings that reach 600, that stop before 600, and of the heads that reached 200 in
Befund 94 that now stop before 600; for the cap hits at 600 whether the best epoch lies in the last 20.

Reproduction gate (checked first; if it fails, no reading is assigned and the script exits with 1):
every 600-epoch record carries ``net_max_epochs`` 600, ``git/dirty`` False, preset version 2, zero projected
steps, and the same process, excitation, split and measurement fingerprints and model parameters
(except ``net_max_epochs``) as its Befund 94 record; for every head, the validation-loss history of the
first min(E200, E600) epochs equals the Befund 94 history (relative tolerance 1e-6), and a head that stopped
before 200 in Befund 94 stops at the same epoch; where all three heads of a network stopped before 200, all
its metrics equal Befund 94 (relative 1e-6). There is no learning-rate schedule, so these must hold.

Readings (fixed here; evaluated on the 16 primary cells per pair, uncorrected verdicts unless stated):

* "Against the order" = GRU worse than LSTM, Transformer better than LSTM, Transformer better than GRU.
* K3 (the order flips -> revision of the synthesis needed): in at least one primary cell the 600-epoch
  verdict is against the order and the 200-epoch verdict was not.
* K2 (the gap of GRU/LSTM to the Transformer shrinks below the resolution -> weaken the statement on
  energy/bed temperature): not K3, and for LSTM or for GRU as reference the number of primary cells with
  Transformer resolvably worse at 600 is at most half of that number at 200 (which must be > 0).
* K1 (budget restriction defused): not K3, not K2, and the Holm verdict in the main family of every
  primary cell of GRU and Transformer against LSTM is unchanged between 200 and 600.
* Otherwise NO reading: not K3, not K2, but at least one Holm verdict changes; the changed cells are listed
  and the consequence is left to the project lead.

Also reported, descriptive: the same network at 600 against 200 epochs per cell (median over seeds at both
budgets, ratio, seeds improved; paired verdict, not part of any family); the GRU-better-than-LSTM counts;
the control cells (moisture, ``n_train`` 64); the wall times of the run (timing file of the driver).

Not claimed: convergence of the networks in general (only the budget 600 with patience 20 is tested), any
statement for xLSTM, Mamba or the baselines (not re-trained; their cells keep the 200-epoch reference),
``n_train`` 8 and 32, or the old data A0/A1. Five seeds, four degrees of freedom; cells of a channel are not
independent.

    tools/run_here.sh scripts/analyze_convergence_check_94.py --run-name main
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from analyze_83_addendum import holm
from analyze_paired_reevaluation import compare
from analyze_rerun import correction, rank_cells, series

ALPHA = 0.05
REL_TOL = 1e-6
OLD_EPOCHS, NEW_EPOCHS = 200, 600
LAST = 20
SEEDS = (1, 2, 3, 4, 5)
NETWORKS = ("gru", "lstm", "transformer")
SIZES = (16, 64)
ALL_SIZES = (8, 16, 32, 64)
HEADS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
PRIMARY_CHANNELS = ("energy_target", "bed_temperature_celsius")
CONTROL_CHANNEL = "product_moisture_kg_per_kg"
CONTROL_SIZE = 64
TEST_SETS = ("excited", "constant_holdout")
HORIZONS = (1, 36)
PAIRS = (("gru", "lstm"), ("transformer", "lstm"), ("transformer", "gru"))
BETTER, WORSE, UNRESOLVED = "candidate better", "candidate worse", "unresolved"
AGAINST_ORDER = {
    ("gru", "lstm"): WORSE,
    ("transformer", "lstm"): BETTER,
    ("transformer", "gru"): BETTER,
}
EXPECTED_MAIN_FAMILY = {"tests": 336, "resolved_uncorrected": 260, "resolved_holm": 70}
FINGERPRINT_KEYS = (
    "process/fingerprint",
    "excitation/fingerprint",
    "split/digest",
    "measurement/digest",
    "eval/pool_excitation_fingerprint",
    "preset/new_study_version",
)
Records = list[dict[str, Any]]
Cell = tuple[int, str, str, int]  # n_train, test_set, channel, horizon


def load_jsonl(path: Path) -> Records:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cell_list(*, control: bool) -> list[Cell]:
    """Primary cells (energy, bed temperature at n_train 16 and 64) and optionally the control cells."""
    out = [
        (n, t, c, h) for n in SIZES for t in TEST_SETS for c in PRIMARY_CHANNELS for h in HORIZONS
    ]
    if control:
        out += [(CONTROL_SIZE, t, CONTROL_CHANNEL, h) for t in TEST_SETS for h in HORIZONS]
    return out


def is_primary(cell: Cell) -> bool:
    return cell[2] in PRIMARY_CHANNELS


# ----------------------------------------------------------------------------- reproduction gate


def _record_key(p: dict[str, Any]) -> tuple:
    return (p["model/label"], int(p["n_train"]), int(p["master_seed"]), p["test_set"],
            p["eval/pool"], p["eval/window"])  # fmt: skip


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0) or a == b


def trainings(records: Records) -> dict[tuple[str, int, int], dict[str, Any]]:
    """One params dict per (network, n_train, seed); the evaluation records repeat the training."""
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for r in records:
        p = r["params"]
        if p["model/label"] not in NETWORKS or int(p["n_train"]) not in SIZES:
            continue
        out.setdefault((p["model/label"], int(p["n_train"]), int(p["master_seed"])), p)
    return out


def gate(old: Records, new: Records) -> dict[str, Any]:
    failures: list[str] = []
    index_old = {_record_key(r["params"]): r for r in old}
    expected = {(a, n, s) for a in NETWORKS for n in SIZES for s in SEEDS}
    for r in new:
        p = r["params"]
        key = _record_key(p)
        if key not in index_old:
            failures.append(f"{key}: no Befund 94 record")
            continue
        q = index_old[key]["params"]
        if (
            int(p["model/net_max_epochs"]) != NEW_EPOCHS
            or int(q["model/net_max_epochs"]) != OLD_EPOCHS
        ):
            failures.append(
                f"{key}: net_max_epochs {p['model/net_max_epochs']} / {q['model/net_max_epochs']}"
            )
        if p.get("git/dirty") is not False:
            failures.append(f"{key}: git/dirty {p.get('git/dirty')}")
        if p.get("pool/projected_steps") != 0 or p.get("holdout_pool/projected_steps") != 0:
            failures.append(f"{key}: projected steps")
        for k in FINGERPRINT_KEYS:
            if p.get(k) != q.get(k):
                failures.append(f"{key}: {k} {p.get(k)!r} != {q.get(k)!r}")
        for k in q:
            if k.startswith("model/") and k != "model/net_max_epochs" and p.get(k) != q[k]:
                failures.append(f"{key}: {k} {p.get(k)!r} != {q[k]!r}")
    found = {
        (p["model/label"], int(p["n_train"]), int(p["master_seed"]))
        for p in (r["params"] for r in new)
    }
    if found != expected:
        failures.append(f"trainings {sorted(found ^ expected)} missing or unexpected")
    old_t, new_t = trainings(old), trainings(new)
    histories_checked = 0
    for key in sorted(expected & set(new_t)):
        p, q = new_t[key], old_t[key]
        for head in HEADS:
            e_new = int(p[f"training/epochs_run/{head}"])
            e_old = int(q[f"training/epochs_run/{head}"])
            if e_old < OLD_EPOCHS and (
                e_new != e_old
                or p[f"training/best_epoch/{head}"] != q[f"training/best_epoch/{head}"]
            ):
                failures.append(f"{key} {head}: stopped at {e_old} in Befund 94, now {e_new}")
            h_new = p[f"training/loss_history_val/{head}"]
            h_old = q[f"training/loss_history_val/{head}"]
            m = min(e_old, e_new)
            if len(h_old) < m or len(h_new) < m:
                failures.append(f"{key} {head}: history shorter than {m}")
                continue
            bad = [i for i in range(m) if not _close(float(h_new[i]), float(h_old[i]))]
            if bad:
                failures.append(f"{key} {head}: validation history differs from epoch {bad[0]}")
            histories_checked += 1
    metrics_checked = 0
    for r in new:
        p = r["params"]
        key = (p["model/label"], int(p["n_train"]), int(p["master_seed"]))
        q = old_t.get(key)
        if q is None or any(int(q[f"training/epochs_run/{h}"]) >= OLD_EPOCHS for h in HEADS):
            continue
        reference = index_old[_record_key(p)]["metrics"]
        for k, v in r["metrics"].items():
            if (
                isinstance(v, int | float)
                and k in reference
                and not _close(float(v), float(reference[k]))
            ):
                failures.append(f"{_record_key(p)}: metric {k} {v} != {reference[k]}")
        metrics_checked += 1
    return {
        "passed": not failures,
        "records": len(new),
        "head_histories_checked": histories_checked,
        "records_with_metrics_checked": metrics_checked,
        "failures": failures[:200],
        "n_failures": len(failures),
    }


# ----------------------------------------------------------------------------- epochs


def epoch_table(old: Records, new: Records) -> dict[str, Any]:
    old_t, new_t = trainings(old), trainings(new)
    rows = []
    for (net, n, seed), p in sorted(new_t.items()):
        q = old_t[(net, n, seed)]
        for head in HEADS:
            e_new = int(p[f"training/epochs_run/{head}"])
            rows.append(
                {
                    "network": net,
                    "n_train": n,
                    "seed": seed,
                    "head": head,
                    "epochs_200": int(q[f"training/epochs_run/{head}"]),
                    "epochs_600": e_new,
                    "best_epoch_600": int(p[f"training/best_epoch/{head}"]),
                    "elapsed_s_600": float(p["training/elapsed_s"]),
                    "elapsed_s_200": float(q["training/elapsed_s"]),
                }
            )
    summary: dict[str, Any] = {}
    for net in NETWORKS:
        for n in SIZES:
            for head in (*HEADS, "all", "energy_and_bed"):
                sub = [
                    r
                    for r in rows
                    if r["network"] == net
                    and r["n_train"] == n
                    and (
                        head == "all"
                        or r["head"] == head
                        or (head == "energy_and_bed" and r["head"] in PRIMARY_CHANNELS)
                    )
                ]
                if not sub:
                    continue
                capped_200 = [r for r in sub if r["epochs_200"] >= OLD_EPOCHS]
                capped_600 = [r for r in sub if r["epochs_600"] >= NEW_EPOCHS]
                summary[f"{net}/n{n}/{head}"] = {
                    "head_trainings": len(sub),
                    "reached_200_in_befund94": len(capped_200),
                    "reached_600": len(capped_600),
                    "stopped_before_600": len(sub) - len(capped_600),
                    "share_stopped_before_600": (len(sub) - len(capped_600)) / len(sub),
                    "of_reached_200_now_stopped_before_600": sum(
                        r["epochs_600"] < NEW_EPOCHS for r in capped_200
                    ),
                    "cap_hits_600_best_in_last_20": sum(
                        r["best_epoch_600"] >= NEW_EPOCHS - LAST for r in capped_600
                    ),
                    "epochs_600_median": statistics.median(r["epochs_600"] for r in sub),
                    "epochs_600_max": max(r["epochs_600"] for r in sub),
                }
    return {"rows": rows, "summary": summary}


# ----------------------------------------------------------------------------- cells


def pair_cell(records: Records, candidate: str, reference: str, cell: Cell) -> dict[str, Any]:
    n, test_set, channel, h = cell
    kw = {"n_train": n, "test_set": test_set, "channel": channel, "horizon": h}
    row = compare(
        series(records, label=candidate, **kw), series(records, label=reference, **kw),
        candidate, reference,
    )  # fmt: skip
    row.update(kw)
    row["ratio_of_medians"] = row["median_candidate"] / row["median_reference"]
    return row


def holm_verdicts(rows: list[dict[str, Any]]) -> list[str]:
    p = np.array([1.0 if math.isnan(r["p_value"]) else r["p_value"] for r in rows])
    adjusted = holm(p)
    out = []
    for row, adj in zip(rows, adjusted, strict=True):
        resolved = row["verdict"] in (BETTER, WORSE) and adj < ALPHA
        out.append(row["verdict"] if resolved else UNRESOLVED)
    return out


def _main_key(row: dict[str, Any]) -> tuple:
    return (row["candidate"], row["n_train"], row["test_set"], row["channel"], row["horizon"])


def main_family(old: Records, new: Records) -> dict[str, Any]:
    family_200 = rank_cells(old, list(ALL_SIZES))
    counts = correction(family_200)
    for k, v in EXPECTED_MAIN_FAMILY.items():
        if counts[k] != v:
            raise SystemExit(f"main family of Befund 94: {k} = {counts[k]}, expected {v}")
    covered = {}
    for candidate in ("gru", "transformer"):
        for cell in cell_list(control=True):
            row = pair_cell(new, candidate, "lstm", cell)
            covered[_main_key(row)] = row
    family_600 = [covered.get(_main_key(r), r) for r in family_200]
    if sum(_main_key(r) in covered for r in family_200) != len(covered):
        raise SystemExit("covered cells not all found in the main family")
    holm_200, holm_600 = holm_verdicts(family_200), holm_verdicts(family_600)
    cells = []
    for r200, _r600, h200, h600 in zip(family_200, family_600, holm_200, holm_600, strict=True):
        if _main_key(r200) not in covered:
            continue
        cells.append({"key": _main_key(r200), "holm_200": h200, "holm_600": h600})
    return {
        "counts_200": counts,
        "counts_600": correction(family_600),
        "substituted_cells": len(covered),
        "cells": cells,
    }


def compare_budgets(old: Records, new: Records) -> list[dict[str, Any]]:
    rows = []
    for candidate, reference in PAIRS:
        for cell in cell_list(control=True):
            a = pair_cell(old, candidate, reference, cell)
            b = pair_cell(new, candidate, reference, cell)
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "n_train": cell[0],
                    "test_set": cell[1],
                    "channel": cell[2],
                    "horizon": cell[3],
                    "primary": is_primary(cell),
                    **{f"{k}_200": a[k] for k in _FIELDS},
                    **{f"{k}_600": b[k] for k in _FIELDS},
                }
            )
    return rows


_FIELDS = (
    "verdict",
    "p_value",
    "ratio_of_medians",
    "relative_difference_of_medians",
    "seeds_candidate_worse",
    "median_candidate",
    "median_reference",
)


def same_network(old: Records, new: Records) -> list[dict[str, Any]]:
    """The same network at 600 against 200 epochs (descriptive)."""
    rows = []
    for net in NETWORKS:
        for n, test_set, channel, h in cell_list(control=True):
            kw = {"n_train": n, "test_set": test_set, "channel": channel, "horizon": h}
            a, b = series(old, label=net, **kw), series(new, label=net, **kw)
            row = compare(b, a, f"{net}@600", f"{net}@200")
            rows.append(
                {
                    "network": net,
                    **kw,
                    "median_200": row["median_reference"],
                    "median_600": row["median_candidate"],
                    "ratio_600_over_200": row["median_candidate"] / row["median_reference"],
                    "seeds_improved": sum(b[s] < a[s] for s in SEEDS),
                    "verdict_600_vs_200": row["verdict"],
                    "p_value": row["p_value"],
                }
            )
    return rows


def sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, select in (
        ("against_lstm_40", lambda r: r["reference"] == "lstm"),
        ("transformer_against_gru_20", lambda r: r["reference"] == "gru"),
    ):
        sub = [r for r in rows if select(r)]
        for budget in ("200", "600"):
            verdicts = holm_verdicts(
                [
                    {"verdict": r[f"verdict_{budget}"], "p_value": r[f"p_value_{budget}"]}
                    for r in sub
                ]
            )
            out[f"{name}/{budget}"] = {
                "tests": len(sub),
                "better_holm": verdicts.count(BETTER),
                "worse_holm": verdicts.count(WORSE),
            }
    return out


def tally(rows: list[dict[str, Any]], budget: str) -> dict[str, dict[str, int]]:
    out = {}
    for candidate, reference in PAIRS:
        sub = [
            r
            for r in rows
            if r["primary"] and (r["candidate"], r["reference"]) == (candidate, reference)
        ]
        out[f"{candidate}_vs_{reference}"] = {
            "cells": len(sub),
            "better": sum(r[f"verdict_{budget}"] == BETTER for r in sub),
            "worse": sum(r[f"verdict_{budget}"] == WORSE for r in sub),
            "unresolved": sum(r[f"verdict_{budget}"] == UNRESOLVED for r in sub),
            "all_seeds_worse": sum(r[f"seeds_candidate_worse_{budget}"] == len(SEEDS) for r in sub),
            "all_seeds_better": sum(r[f"seeds_candidate_worse_{budget}"] == 0 for r in sub),
            "ratio_of_medians_median": statistics.median(
                r[f"ratio_of_medians_{budget}"] for r in sub
            ),
        }
    return out


def reading(rows: list[dict[str, Any]], family: dict[str, Any]) -> dict[str, Any]:
    primary = [r for r in rows if r["primary"]]
    flips = [
        r
        for r in primary
        if r["verdict_600"] == AGAINST_ORDER[(r["candidate"], r["reference"])]
        and r["verdict_200"] != r["verdict_600"]
    ]
    shrink = {}
    for reference in ("lstm", "gru"):
        sub = [r for r in primary if (r["candidate"], r["reference"]) == ("transformer", reference)]
        w200 = sum(r["verdict_200"] == WORSE for r in sub)
        w600 = sum(r["verdict_600"] == WORSE for r in sub)
        shrink[reference] = {
            "worse_200": w200,
            "worse_600": w600,
            "halved": w200 > 0 and w600 <= w200 / 2,
        }
    holm_changed = [
        c
        for c in family["cells"]
        if c["key"][3] in PRIMARY_CHANNELS and c["holm_200"] != c["holm_600"]
    ]
    if flips:
        label = "K3"
    elif any(v["halved"] for v in shrink.values()):
        label = "K2"
    elif not holm_changed:
        label = "K1"
    else:
        label = "none"
    return {
        "reading": label,
        "cells_against_order_new_at_600": [
            {
                k: r[k]
                for k in (
                    "candidate",
                    "reference",
                    "n_train",
                    "test_set",
                    "channel",
                    "horizon",
                    "verdict_200",
                    "verdict_600",
                )
            }
            for r in flips
        ],
        "transformer_worse_counts": shrink,
        "primary_holm_verdicts_changed": holm_changed,
    }


def evaluate(old: Records, new: Records) -> dict[str, Any]:
    result: dict[str, Any] = {"gate": gate(old, new)}
    result["epochs"] = epoch_table(old, new)
    if not result["gate"]["passed"]:
        result["reading"] = None
        return result
    rows = compare_budgets(old, new)
    family = main_family(old, new)
    result.update(
        {
            "cells": rows,
            "verdict_counts_primary": {"200": tally(rows, "200"), "600": tally(rows, "600")},
            "main_family": family,
            "sensitivity_families": sensitivity(rows),
            "same_network_600_vs_200": same_network(old, new),
            "reading": reading(rows, family),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--old-prefix", default="rerun_full")
    arguments = parser.parse_args()
    require_clean_tree()
    prefix = f"conv600_{arguments.run_name}"
    old = load_jsonl(RUNS_DIR / f"{arguments.old_prefix}_N.jsonl")
    new = load_jsonl(RUNS_DIR / f"{prefix}_N.jsonl.gz")
    timing_path = RUNS_DIR / f"{prefix}_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
    result = evaluate(old, new)
    if timing is not None:
        result["timing"] = {
            "sessions": timing["sessions"],
            "job_wall_s": {k: v["wall_s"] for k, v in timing["jobs"].items()},
        }
    write_artifact(
        f"convergence_check_94_{prefix}",
        {
            "_provenance": provenance(),
            "prefix": prefix,
            "old_prefix": arguments.old_prefix,
            **result,
        },
    )
    print(f"gate passed: {result['gate']['passed']} ({result['gate']['n_failures']} failures)")
    if result["reading"] is None:
        return 1
    print(f"reading: {result['reading']['reading']}")
    for budget, counts in result["verdict_counts_primary"].items():
        print(budget, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
