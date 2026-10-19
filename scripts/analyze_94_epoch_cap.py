r"""Epoch-cap counts of the training in arm N (addendum to Befund 94).

Pure re-analysis of the committed records ``rerun_full_N.jsonl``; no training, no new data.  Rule fixed
here, before the run:

* Unit: ONE training of one head (target) of one network: records are de-duplicated over their
  evaluation variants (``eval/pool``, ``eval/window``, ``test_set``, which repeat the same training) by
  ``(model/label, master_seed, n_train)``; the trainings are the keys ``training/epochs_run/<head>`` of the
  networks (GRU, LSTM, Transformer, xLSTM, Mamba; one head per target: energy, moisture, bed temperature).
* The cap is the recorded ``model/net_max_epochs`` (200).  A training "reaches the cap" if
  ``epochs_run == net_max_epochs`` (early stopping with patience 20 did not stop it earlier).
* Output per architecture: cap hits and trainings in total, per head, per ``n_train``, and per
  head and ``n_train``; and, as a second figure, the trainings whose best epoch lies within the last
  ``--last`` epochs (default 20, the patience) of the run (``training/best_epoch/<head>``), which
  separates "ran to the cap while still improving" from "ran to the cap on a flat curve".
* No threshold, no test: counts only.  The statement is a limitation of the fixed training budget, not a
  result about the architectures.

Call::

    tools/run_here.sh scripts/analyze_94_epoch_cap.py --prefix rerun_full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact

NETWORKS = ("gru", "lstm", "transformer", "xlstm", "mamba")
EPOCHS = "training/epochs_run/"
BEST = "training/best_epoch/"
COUNTS = ("trainings", "at_cap", "best_in_last", "at_cap_and_best_in_last")


def trainings(records: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    """One params dict per (label, seed, n_train); the evaluation variants repeat the same training."""
    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in records:
        params = record["params"]
        if params["model/label"] not in NETWORKS:
            continue
        key = (params["model/label"], int(params["master_seed"]), int(params["n_train"]))
        unique.setdefault(key, params)
    return unique


def _bucket() -> dict[str, int]:
    return dict.fromkeys(COUNTS, 0)


def analyse(records: list[dict[str, Any]], last: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {}
    unique = trainings(records)
    for network in NETWORKS:
        rows = [(k, p) for k, p in unique.items() if k[0] == network]
        if not rows:
            continue
        caps = {int(p["model/net_max_epochs"]) for _, p in rows}
        if len(caps) != 1:
            raise ValueError(f"{network}: several epoch caps {sorted(caps)}")
        cap_value = caps.pop()
        total = _bucket()
        by_head: dict[str, dict[str, int]] = defaultdict(_bucket)
        by_size: dict[str, dict[str, int]] = defaultdict(_bucket)
        by_cell: dict[str, dict[str, int]] = defaultdict(_bucket)
        for (_, _, n_train), params in rows:
            for key, run in params.items():
                if not key.startswith(EPOCHS):
                    continue
                head = key[len(EPOCHS) :]
                at_cap = int(run) == cap_value
                best = params.get(BEST + head)
                in_last = best is not None and int(best) >= int(run) - last
                for bucket in (
                    total,
                    by_head[head],
                    by_size[str(n_train)],
                    by_cell[f"{head}/n{n_train}"],
                ):
                    bucket["trainings"] += 1
                    bucket["at_cap"] += int(at_cap)
                    bucket["best_in_last"] += int(in_last)
                    bucket["at_cap_and_best_in_last"] += int(at_cap and in_last)
        result[network] = {
            "epoch_cap": cap_value,
            "total": total,
            "by_head": dict(sorted(by_head.items())),
            "by_n_train": dict(sorted(by_size.items(), key=lambda kv: int(kv[0]))),
            "by_head_and_n_train": dict(sorted(by_cell.items())),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--last", type=int, default=20)
    arguments = parser.parse_args()
    require_clean_tree()
    source = RUNS_DIR / f"{arguments.prefix}_N.jsonl"
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    result = analyse(records, arguments.last)
    write_artifact(
        f"befund94_addendum_epoch_cap_{arguments.prefix}",
        {"_provenance": provenance(), "source": source.name, "last": arguments.last, **result},
    )
    for network, row in result.items():
        print(network, "all budgets", row["total"], "n_train 64", row["by_n_train"].get("64", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
