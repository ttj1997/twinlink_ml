"""Compare the retrained F3 soft sensors with the training record of Befund 111 (step 0, docs/29).

The checkpoints of ``sensor_value_soft_sensor_v1`` (Befund 111) were lost with the worktree that
held them. They were retrained with the unchanged committed script
``scripts/train_sensor_value_soft_sensor.py`` (seeds 1-3) under the name
``sensor_value_soft_sensor_v1_retrain``. This script compares the two records field by field, every
field of every seed except those that differ by construction (checkpoint path and SHA-256 -- the
checkpoint stores its run name and the git commit --, and the wall times ``fit_s`` and ``total_s``),
and records whether the code the training uses changed between the two commits. The weights of the
original checkpoints are lost, so the comparison is on the record's figures (training diagnostics,
test errors, data fingerprints, input ranges), not on the weights.

Writes ``artifacts/runs/sensor_value_soft_sensor_v1_retrain_comparison.json`` (never overwritten).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, provenance, require_clean_tree  # noqa: E402

ORIGINAL = "sensor_value_soft_sensor_v1"
RETRAINED = "sensor_value_soft_sensor_v1_retrain"
BY_CONSTRUCTION = ("checkpoint", "checkpoint_sha256", "fit_s", "total_s")
TRAINING_CODE = (
    "scripts/train_sensor_value_soft_sensor.py",
    "scripts/run_arch_rerun_worker.py",
    "scripts/run_arch_learning_curve_worker.py",
    "scripts/run_outlier_robustness_worker.py",
    "src/twinlink_ml/models",
    "src/twinlink_ml/data",
    "src/twinlink_ml/evaluation/experiments.py",
    "src/twinlink_ml/evaluation/protocol.py",
    "src/twinlink_ml/simulation",
    "src/twinlink_ml/utils",
    "conf/process/greybox_gab_placeholder.yaml",
    "conf/sensor_value/sensor_value.yaml",
)


def compare(a: Any, b: Any, path: str, out: list[dict[str, Any]]) -> int:
    """Append every differing leaf to ``out``; return the number of compared leaves."""
    if isinstance(a, dict) and isinstance(b, dict):
        n = 0
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append({"path": f"{path}/{key}", "difference": "missing"})
                n += 1
            else:
                n += compare(a[key], b[key], f"{path}/{key}", out)
        return n
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum(
            compare(x, y, f"{path}[{i}]", out) for i, (x, y) in enumerate(zip(a, b, strict=False))
        )
    if a != b:
        entry: dict[str, Any] = {"path": path, "original": a, "retrained": b}
        if isinstance(a, int | float) and isinstance(b, int | float):
            entry["abs_difference"] = abs(float(a) - float(b))
            entry["rel_difference"] = abs(float(a) - float(b)) / max(abs(float(a)), 1e-300)
        out.append(entry)
    return 1


def main() -> int:
    output = RUNS_DIR / f"{RETRAINED}_comparison.json"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten")
    require_clean_tree()
    original = json.loads((RUNS_DIR / f"{ORIGINAL}.json").read_text(encoding="utf-8"))
    retrained = json.loads((RUNS_DIR / f"{RETRAINED}.json").read_text(encoding="utf-8"))
    commits = {
        "original": original["provenance"]["git/commit"],
        "retrained": retrained["provenance"]["git/commit"],
    }
    diff = subprocess.run(
        ["git", "diff", "--stat", commits["original"], commits["retrained"], "--", *TRAINING_CODE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    seeds = []
    for a, b in zip(original["seeds"], retrained["seeds"], strict=True):
        differing: list[dict[str, Any]] = []
        fields = {k: v for k, v in a.items() if k not in BY_CONSTRUCTION}
        other = {k: v for k, v in b.items() if k not in BY_CONSTRUCTION}
        n = compare(fields, other, "", differing)
        numeric = [d["abs_difference"] for d in differing if "abs_difference" in d]
        seeds.append(
            {
                "seed": a["seed"],
                "compared_fields": n,
                "differing_fields": len(differing),
                "identical": not differing,
                "max_abs_difference": max(numeric) if numeric else 0.0,
                "differing": differing,
                "checkpoint_sha256": {
                    "original": a["checkpoint_sha256"],
                    "retrained": b["checkpoint_sha256"],
                },
                "fit_s": {"original": a["fit_s"], "retrained": b["fit_s"]},
            }
        )
    top = {k: v for k, v in original.items() if k not in ("seeds", "provenance", "runtime_s")}
    top_retrained = {
        k: v for k, v in retrained.items() if k not in ("seeds", "provenance", "runtime_s")
    }
    top_diff: list[dict[str, Any]] = []
    compare(top, top_retrained, "", top_diff)
    payload = {
        "provenance": provenance("not used"),
        "study": "step 0 of docs/29: retraining of the F3 soft sensors of Befund 111",
        "original": ORIGINAL,
        "retrained": RETRAINED,
        "commits": commits,
        "training_code_diff_between_commits": diff or "none",
        "excluded_by_construction": list(BY_CONSTRUCTION),
        "top_level_differences": top_diff,
        "seeds": seeds,
        "all_identical": all(s["identical"] for s in seeds) and not top_diff,
    }
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for s in seeds:
        print(
            f"seed {s['seed']}: {s['compared_fields']} fields, {s['differing_fields']} differ, "
            f"max |diff| {s['max_abs_difference']}"
        )
    print(f"code diff: {payload['training_code_diff_between_commits']}")
    print(f"all identical: {payload['all_identical']}; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
