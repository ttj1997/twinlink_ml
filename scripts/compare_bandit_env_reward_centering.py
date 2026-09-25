r"""BB2/CB2: Vergleich -- Zufall (YA1) neben XA3, BB2 (zentriert), CB2 (gamma=0).

Reine Auswertung, kein Training -- liest die bereits geschriebenen Artefakte
(``rl_bandit_random_baseline.jsonl``, ``rl_bandit_sac_learning_curve_xa3.jsonl``,
``rl_bandit_sac_learning_curve_bb2_centered.jsonl``,
``rl_bandit_sac_learning_curve_cb2_gamma0.jsonl``) und druckt (a) Zulaessigkeit und
(b) gepoolten Anteil geschlossenen Regrets nebeneinander, je Checkpoint. Fehlende Artefakte
werden uebersprungen (kein Fehler), damit dasselbe Skript vor UND nach CB2 laeuft.

Aufruf::

    python scripts/compare_bandit_env_reward_centering.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_BASELINE_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_random_baseline.jsonl"
XA3_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_sac_learning_curve_xa3.jsonl"
BB2_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_sac_learning_curve_bb2_centered.jsonl"
CB2_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_sac_learning_curve_cb2_gamma0.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _random_summary(path: Path) -> tuple[float, float, float, float, float, float]:
    records = _load_jsonl(path)
    feasibility = np.array([r["metrics"]["feasibility_rate"] for r in records])
    pooled = np.array([r["metrics"]["pooled_closed_regret_fraction"] for r in records])
    pooled_finite = pooled[np.isfinite(pooled)]
    return (
        float(np.median(feasibility)),
        float(np.min(feasibility)),
        float(np.max(feasibility)),
        float(np.median(pooled_finite)) if pooled_finite.size else float("nan"),
        float(np.min(pooled_finite)) if pooled_finite.size else float("nan"),
        float(np.max(pooled_finite)) if pooled_finite.size else float("nan"),
    )


def _by_checkpoint(path: Path) -> dict[str, dict]:
    records = _load_jsonl(path)
    return {r["tags"]["checkpoint"]: r["metrics"] for r in records}


def _format_cell(metrics: dict | None) -> tuple[str, str]:
    if metrics is None:
        return "--", "--"
    feasibility = f"{metrics['feasibility_rate']:.2f}"
    pooled = metrics["pooled_closed_regret_fraction"]
    pooled_text = f"{pooled:.4f}" if np.isfinite(pooled) else "n. def."
    return feasibility, pooled_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-baseline", type=Path, default=RANDOM_BASELINE_PATH)
    parser.add_argument("--xa3", type=Path, default=XA3_PATH)
    parser.add_argument("--bb2-centered", type=Path, default=BB2_PATH)
    parser.add_argument("--cb2-gamma0", type=Path, default=CB2_PATH)
    arguments = parser.parse_args()

    print("=" * 110)
    print("Zufall (YA1) vs. XA3 (unzentriert) vs. BB2 (zentriert) vs. CB2 (gamma=0), je Checkpoint")
    print("=" * 110)

    if arguments.random_baseline.exists():
        feas_med, feas_min, feas_max, pool_med, pool_min, pool_max = _random_summary(
            arguments.random_baseline
        )
        print(
            f"Zufall (Median [Min,Max], n=20): "
            f"(a)={feas_med:.2f} [{feas_min:.2f}, {feas_max:.2f}]  "
            f"(b)={pool_med:.4f} [{pool_min:.4f}, {pool_max:.4f}]"
        )
    else:
        print(f"(Zufallsbaseline {arguments.random_baseline} nicht gefunden)")

    runs = {
        "XA3": arguments.xa3,
        "BB2": arguments.bb2_centered,
        "CB2": arguments.cb2_gamma0,
    }
    by_run = {label: _by_checkpoint(path) if path.exists() else {} for label, path in runs.items()}
    checkpoints = sorted({c for run in by_run.values() for c in run}, key=int)

    header = f"{'Schritt':>8s}"
    for label in runs:
        header += f"  {label + ' (a)':>8s}  {label + ' (b)':>12s}"
    print(f"\n{header}")
    for checkpoint in checkpoints:
        row = f"{checkpoint:>8s}"
        for label in runs:
            feasibility, pooled = _format_cell(by_run[label].get(checkpoint))
            row += f"  {feasibility:>8s}  {pooled:>12s}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
