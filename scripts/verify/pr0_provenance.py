"""PR0 -- trace every abstract number to artifact, script, config, commit and git/dirty.

Reads the committed artifacts with plain json and asks git (this repository and the
read-only archive of the pre-cut history, see docs/ARCHIV.md) whether each recorded commit
exists and whether the generating script existed at that commit. Writes nothing to the
archive; no project code is imported.

    python scripts/verify/pr0_provenance.py
"""

from __future__ import annotations

import collections
import json
from typing import Any

from _common import (
    ARCHIVE_ROOT,
    REPO_ROOT,
    RUNS_DIR,
    git,
    provenance,
    require_clean_tree,
    write_artifact,
)

# claim id -> (abstract wording, Befund, artifacts (relative to artifacts/runs), scripts)
CLAIMS: dict[str, dict[str, Any]] = {
    "C1_split_29_68": {
        "claim": "window-wise vs run-wise split: 29 to 68 % lower error",
        "befund": "39",
        "artifacts": ["extra_trees_split_comparison.jsonl"],
        "scripts": ["run_extra_trees_split_comparison.py"],
    },
    "C2_k5_factors": {
        "claim": "factor vs persistence on moisture: 1.31/1.70 constant, 4.4-24 excited",
        "befund": "51 (K5)",
        "artifacts": ["arch_learning_curve_projected.jsonl"],
        "scripts": ["run_arch_learning_curve_sweep.py", "run_arch_learning_curve_worker.py"],
    },
    "C3_shift_63_14": {
        "claim": "persistence shifts 6.3x, static regression 1.4x between test sets",
        "befund": "58 OA5a / 59a",
        "artifacts": ["arch_learning_curve_projected.jsonl"],
        "scripts": ["analyze_energy_target_static_factor.py"],
    },
    "C4_energy_r2": {
        "claim": "energy R2 0.972 static vs >=0.999 / >=0.9989, factor 5.2-7.8 in RMSE",
        "befund": "59",
        "artifacts": [
            "arch_learning_curve_projected.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "scripts": ["analyze_energy_target_static_factor.py", "report_baselines.py"],
        "note": "R2_static=0.972 is quoted from docs/02_prozess_setup.md (fp 82f65f576a2e5253,"
        " report_baselines.py, seeds 20260812-14, 32 runs), NOT from the Befund 51/58 data.",
    },
    "C5_resolution": {
        "claim": "resolution with 5 seeds: 3.9-11 % / 35-91 %, ~40 seeds for 12.5 %",
        "befund": "63",
        "artifacts": [
            "analyze_m2_resolution_verdicts.json",
            "arch_learning_curve_projected.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "scripts": ["analyze_m2_resolution_verdicts.py"],
    },
    "C6_architectures": {
        "claim": "GRU/LSTM 6/6 indistinguishable; Transformer 15.8 %, parity 1.000, 2.6 %;"
        " xLSTM/Mamba behind in 10/12",
        "befund": "58, 63",
        "artifacts": [
            "analyze_m2_resolution_verdicts.json",
            "arch_epoch_budget_probe600.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "scripts": ["analyze_m2_resolution_verdicts.py", "analyze_epoch_budget_probe600.py"],
    },
    "C7_inference": {
        "claim": "inference 183 us LSTM, 323 us GRU, ~3.1 ms Transformer; training ~10x",
        "befund": "63a",
        "artifacts": [
            "arch_learning_curve_projected.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "scripts": ["analyze_m2_resolution_verdicts.py"],
    },
    "C8_moisture_abs": {
        "claim": "moisture 0.00149 vs 0.00054 kg/kg (factor 2.7); persistence never beaten;"
        " early stopping ~24 vs ~170 epochs",
        "befund": "55, 62 HA1",
        "artifacts": [
            "arch_learning_curve_constant_trained.jsonl",
            "arch_learning_curve_projected.jsonl",
        ],
        "scripts": [
            "run_arch_constant_trained_learning_curve_sweep.py",
            "run_arch_constant_trained_learning_curve_worker.py",
            "probe_bandit_env_ha1_margin_vs_prediction_error.py",
        ],
    },
}


def _resolve(commit: str) -> dict[str, Any]:
    """Which repositories know this commit."""
    known_here = bool(git("cat-file", "-t", commit)) if commit else False
    known_archive = bool(git("cat-file", "-t", commit, cwd=ARCHIVE_ROOT)) if commit else False
    return {"this_repo": known_here, "archive": known_archive}


def _script_at_commit(script: str, commit: str) -> dict[str, bool]:
    """Did ``scripts/<script>`` exist at ``commit`` in either repository."""
    spec = f"{commit}:scripts/{script}"
    return {
        "this_repo": bool(git("cat-file", "-t", spec)),
        "archive": bool(git("cat-file", "-t", spec, cwd=ARCHIVE_ROOT)),
    }


def _summarise_records(name: str, scripts: list[str]) -> dict[str, Any]:
    path = RUNS_DIR / name
    if not path.exists():
        return {"artifact": name, "exists": False}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    counter: collections.Counter[tuple[Any, ...]] = collections.Counter()
    for row in rows:
        p = row["params"]
        counter[
            (
                p.get("git/commit"),
                p.get("git/dirty"),
                p.get("platform"),
                p.get("python"),
                p.get("version/torch"),
                p.get("process/fingerprint"),
                p.get("numerics/enforce_gas_velocity_projection"),
                p.get("model/net_max_epochs"),
            )
        ] += 1
    groups = []
    for (commit, dirty, plat, py, torch_v, fp, proj, epochs), n in counter.items():
        groups.append(
            {
                "n_records": n,
                "git/commit": commit,
                "git/dirty": dirty,
                "platform": plat,
                "python": py,
                "torch": torch_v,
                "process/fingerprint": fp,
                "projection": proj,
                "net_max_epochs_param": epochs,
                "commit_resolvable": _resolve(commit) if commit and commit != "unknown" else None,
                "scripts_at_commit": {s: _script_at_commit(s, commit) for s in scripts if commit},
                "traceable_to_clean_commit": bool(
                    commit and commit != "unknown" and dirty is False
                ),
            }
        )
    return {"artifact": name, "exists": True, "n_records": len(rows), "groups": groups}


def _summarise_json(name: str) -> dict[str, Any]:
    path = RUNS_DIR / name
    data = json.loads(path.read_text(encoding="utf-8"))
    env = data.get("environment") or data.get("provenance") or {}
    added = git("log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", str(path))
    return {
        "artifact": name,
        "exists": True,
        "has_git_fields": bool(env and "git/commit" in env),
        "git/commit": env.get("git/commit"),
        "git/dirty": env.get("git/dirty"),
        "added_in_this_repo": added,
        "traceable_to_clean_commit": bool(env and env.get("git/dirty") is False),
    }


def main() -> None:
    require_clean_tree()
    report: dict[str, Any] = {}
    for claim_id, spec in CLAIMS.items():
        entries = []
        for artifact in spec["artifacts"]:
            if artifact.endswith(".jsonl"):
                entries.append(_summarise_records(artifact, spec["scripts"]))
            else:
                entries.append(_summarise_json(artifact))
        script_state = {
            s: {
                "exists_in_worktree": (REPO_ROOT / "scripts" / s).exists(),
                "last_commit_this_repo": git("log", "-1", "--format=%h", "--", f"scripts/{s}"),
            }
            for s in spec["scripts"]
        }
        untraceable = [
            e["artifact"]
            for e in entries
            if e.get("exists")
            and (
                not e.get("traceable_to_clean_commit", False)
                if "groups" not in e
                else not all(g["traceable_to_clean_commit"] for g in e["groups"])
            )
        ]
        report[claim_id] = {
            "claim": spec["claim"],
            "befund": spec["befund"],
            "note": spec.get("note"),
            "artifacts": entries,
            "scripts": script_state,
            "artifacts_not_traceable_to_clean_commit": untraceable,
        }
    write_artifact(
        "verify_pr0_provenance",
        {"_provenance": provenance(), "archive_root": str(ARCHIVE_ROOT), "claims": report},
    )
    for claim_id, entry in report.items():
        print(claim_id, "-> untraceable:", entry["artifacts_not_traceable_to_clean_commit"])


if __name__ == "__main__":
    main()
