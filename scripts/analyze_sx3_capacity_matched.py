r"""SX3 (Befund 67): evaluation of the capacity-matched comparison at n_train = 64.

Pure evaluation of committed records. Verdict rule and readings are the ones fixed in the SX3
pre-registration of ``docs/04_stand_ap1.md`` (Befund 67), written before any run:

* per variant and per cell (3 channels x 2 test sets, ``rmse_run_median_physical``, ``h = 36``,
  5 seeds) the variant is compared with the LSTM of Befund 51/58 (57 027 parameters) with the
  two-sample t threshold of the SX1 addendum (``analyze_sx1_resolution_t._cell``: Welch, range
  estimator with ``nu_eff = 3.76``);
* reading A (capacity effect): not resolvably worse in all 6 cells -> reaches LSTM level;
  resolvably worse in at most 2 cells -> closes most of the gap; reading C (statement holds):
  resolvably worse in at least 4 cells; otherwise mixed;
* reading B (memory mixing matters): sLSTM-only (2 layers) resolvably better than mLSTM-only in at
  least 4 cells and resolvably worse in none (sLSTM 1 layer and mixed are reported against the mLSTM
  as supplementary rows);
* control: small LSTM against LSTM (57 027) and against the small xLSTM (8 649) and small Mamba
  (8 139) of Befund 51/58;
* epoch-cap hits (fits that ran the full ``max_epochs``) are counted per variant.

    python scripts/analyze_sx3_capacity_matched.py --logs <new records .jsonl> [...]
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact
from analyze_m2_resolution_verdicts import CHANNELS, EXTENDED_JSONL, PROJECTED_JSONL, TEST_SETS
from analyze_sx1_resolution_t import _cell, range_estimator_dof

from twinlink_ml.utils.tracking import read_jsonl

METRIC = "rmse_run_median_physical"
HORIZON = 36
N_TRAIN = "64"
VARIANTS = ("mlstm_cm", "slstm_cm", "slstm_l1_cm", "mixed_cm", "mamba_cm")
CONTROL = "lstm_small"
NEW_LABELS = (*VARIANTS, CONTROL)
EXPECTED_SEEDS = 5


def _values(records: list[Any], model: str, test_set: str, channel: str) -> list[float]:
    key = f"{METRIC}/{channel}/h{HORIZON}"
    return sorted(
        r.metrics[key]
        for r in records
        if r.tags.get("model") == model
        and r.tags.get("test_set") == test_set
        and r.tags.get("n_train") == N_TRAIN
        and key in r.metrics
    )


def _compare(
    records: list[Any], candidate: str, reference: str, nu_eff: float
) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        for test_set in TEST_SETS:
            cand = _values(records, candidate, test_set, channel)
            ref = _values(records, reference, test_set, channel)
            if len(cand) != EXPECTED_SEEDS or len(ref) != EXPECTED_SEEDS:
                raise SystemExit(
                    f"{candidate} vs {reference} {channel}/{test_set}: "
                    f"{len(cand)} / {len(ref)} seeds, expected {EXPECTED_SEEDS}"
                )
            cells[f"{channel}/{test_set}"] = _cell(
                candidate=candidate,
                candidate_values=cand,
                reference=reference,
                reference_values=ref,
                nu_eff=nu_eff,
            )
    return cells


def _tally(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    worse = better = 0
    for cell in cells.values():
        verdict = cell["t_range_dof"]
        if verdict["resolved"]:
            if cell["improvement_relative"] < 0:
                worse += 1
            else:
                better += 1
    return {
        "resolvably_worse": worse,
        "resolvably_better": better,
        "unresolved": len(cells) - worse - better,
        "median_improvement_relative": statistics.median(
            c["improvement_relative"] for c in cells.values()
        ),
    }


def _reading_a_c(tally: dict[str, Any]) -> str:
    worse = tally["resolvably_worse"]
    if worse == 0:
        return "A: reaches LSTM level (not resolvably worse in any cell)"
    if worse <= 2:
        return "A': closes most of the gap (resolvably worse in at most 2 cells)"
    if worse >= 4:
        return "C: statement holds (resolvably worse in at least 4 cells)"
    return "mixed (resolvably worse in 3 cells)"


def _epoch_caps(records: list[Any], label: str) -> dict[str, Any]:
    capped = total = 0
    for r in records:
        if (
            r.tags.get("model") == label
            and r.tags.get("test_set") == "excited"
            and r.tags.get("n_train") == N_TRAIN
        ):
            limit = r.params["model/net_max_epochs"]
            for key, value in r.params.items():
                if key.startswith("training/epochs_run/"):
                    total += 1
                    capped += int(value >= limit)
    return {"heads_at_epoch_cap": capped, "heads_total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--name", default="sx3_capacity_matched")
    arguments = parser.parse_args()
    require_clean_tree()

    reference_records = read_jsonl(PROJECTED_JSONL) + read_jsonl(EXTENDED_JSONL)
    new_records = [r for path in arguments.logs for r in read_jsonl(path)]
    records = reference_records + new_records
    nu_eff = range_estimator_dof(EXPECTED_SEEDS)["nu_eff_range"]

    out: dict[str, Any] = {
        "_provenance": provenance(),
        "nu_eff": nu_eff,
        "logs": [str(p) for p in arguments.logs],
        "commits_in_new_records": sorted({r.params.get("git/commit") for r in new_records}),
        "dirty_in_new_records": sorted({bool(r.params.get("git/dirty")) for r in new_records}),
        "parameters": {
            label: next(
                (r.params["model/n_parameters"] for r in records if r.tags.get("model") == label),
                None,
            )
            for label in ("lstm", "gru", "xlstm", "mamba", *NEW_LABELS)
        },
        "versus_lstm": {},
        "epoch_caps": {},
        "medians": {},
    }
    for label in ("gru", "xlstm", "mamba", *NEW_LABELS):
        cells = _compare(records, label, "lstm", nu_eff)
        tally = _tally(cells)
        out["versus_lstm"][label] = {"cells": cells, "tally": tally, "reading": _reading_a_c(tally)}
    for label in ("lstm", "gru", "xlstm", "mamba", *NEW_LABELS):
        out["epoch_caps"][label] = _epoch_caps(records, label)
        out["medians"][label] = {
            f"{channel}/{test_set}": statistics.median(_values(records, label, test_set, channel))
            for channel in CHANNELS
            for test_set in TEST_SETS
        }

    supplementary = {}
    for label in ("slstm_cm", "slstm_l1_cm", "mixed_cm"):
        cells = _compare(records, label, "mlstm_cm", nu_eff)
        supplementary[f"{label}_vs_mlstm_cm"] = {"cells": cells, "tally": _tally(cells)}
    reading_b = supplementary["slstm_cm_vs_mlstm_cm"]["tally"]
    out["reading_b_slstm_vs_mlstm"] = {
        **supplementary,
        "holds": reading_b["resolvably_better"] >= 4 and reading_b["resolvably_worse"] == 0,
    }

    control: dict[str, Any] = {}
    for reference in ("lstm", "xlstm", "mamba"):
        cells = _compare(records, CONTROL, reference, nu_eff)
        control[f"{CONTROL}_vs_{reference}"] = {"cells": cells, "tally": _tally(cells)}
    versus_lstm = control[f"{CONTROL}_vs_lstm"]["tally"]["resolvably_worse"]
    better_than_small = sum(
        control[f"{CONTROL}_vs_{ref}"]["tally"]["resolvably_better"] for ref in ("xlstm", "mamba")
    )
    out["control"] = {
        **control,
        "size_effect_criterion_met": versus_lstm >= 4 and better_than_small == 0,
        "worse_than_lstm_cells": versus_lstm,
        "better_than_small_xlstm_or_mamba_cells": better_than_small,
    }

    for label, entry in out["versus_lstm"].items():
        tally = entry["tally"]
        print(
            f"{label:12s} worse {tally['resolvably_worse']}  better {tally['resolvably_better']}  "
            f"unresolved {tally['unresolved']}  median diff {tally['median_improvement_relative']:+.1%}"
            f"  -> {entry['reading']}"
        )
    print("reading B holds:", out["reading_b_slstm_vs_mlstm"]["holds"])
    print("control size-effect criterion met:", out["control"]["size_effect_criterion_met"])
    if any(math.isnan(v) for m in out["medians"].values() for v in m.values()):
        raise SystemExit("NaN median in the result")
    write_artifact(arguments.name, out)


if __name__ == "__main__":
    main()
