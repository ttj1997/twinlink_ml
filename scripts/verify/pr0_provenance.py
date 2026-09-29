"""PR0 -- trace every abstract number to artifact, script, config, commit and git/dirty.

Reads the committed artifacts with plain json and asks git (this repository and the
read-only archive of the pre-cut history, see docs/ARCHIV.md) whether each recorded commit
exists, whether it was logged dirty (``git/dirty`` field or a ``-dirty`` suffix on the commit
string) and whether the generating script existed at that commit. Analysis-only artifacts
(no records, no git fields) are compared with their last archived version instead. Writes
nothing to the archive; no project code is imported.

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

# claim id -> abstract wording, Befund, artifacts (relative to artifacts/runs),
# generators (scripts that produce the records), analysis (scripts that only aggregate)
CLAIMS: dict[str, dict[str, Any]] = {
    "C1_split_29_68": {
        "claim": "window-wise vs run-wise split: 29 to 68 % lower error",
        "befund": "39",
        "artifacts": ["extra_trees_split_comparison.jsonl"],
        "generators": ["run_extra_trees_split_comparison.py"],
        "analysis": [],
    },
    "C2_k5_factors": {
        "claim": "factor vs persistence on moisture: 1.31/1.70 constant, 4.4-24 excited",
        "befund": "51 (K5)",
        "artifacts": ["arch_learning_curve_projected.jsonl"],
        "generators": ["run_arch_learning_curve_worker.py"],
        "analysis": [],
    },
    "C3_shift_63_14": {
        "claim": "persistence shifts 6.3x, static regression 1.4x between test sets",
        "befund": "58 OA5a / 59a",
        "artifacts": ["arch_learning_curve_projected.jsonl"],
        "generators": ["run_arch_learning_curve_worker.py"],
        "analysis": ["analyze_energy_target_static_factor.py"],
    },
    "C4_energy_r2": {
        "claim": "energy R2 0.972 static vs >=0.999 / >=0.9989, factor 5.2-7.8 in RMSE",
        "befund": "59",
        "artifacts": [
            "arch_learning_curve_projected.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "generators": ["run_arch_learning_curve_worker.py"],
        "analysis": ["analyze_energy_target_static_factor.py", "report_baselines.py"],
        "note": "R2_static=0.972 is quoted from docs/02_prozess_setup.md (fp 82f65f576a2e5253,"
        " report_baselines.py, seeds 20260812-14, 32 runs), NOT from the Befund 51/58 data;"
        " the >=0.999 values are derived as 1-(1-0.972)/factor^2 (Befund 59b), not measured.",
    },
    "C5_resolution": {
        "claim": "resolution with 5 seeds: 3.9-11 % / 35-91 %, ~40 seeds for 12.5 %",
        "befund": "63",
        "artifacts": [
            "analyze_m2_resolution_verdicts.json",
            "arch_learning_curve_projected.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "generators": ["run_arch_learning_curve_worker.py"],
        "analysis": ["analyze_m2_resolution_verdicts.py"],
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
        "generators": ["run_arch_learning_curve_worker.py"],
        "analysis": ["analyze_m2_resolution_verdicts.py", "analyze_epoch_budget_probe600.py"],
    },
    "C7_inference": {
        "claim": "inference 183 us LSTM, 323 us GRU, ~3.1 ms Transformer; training ~10x",
        "befund": "63a",
        "artifacts": [
            "arch_learning_curve_projected.jsonl",
            "arch_learning_curve_projected_extended.jsonl",
        ],
        "generators": ["run_arch_learning_curve_worker.py"],
        "analysis": ["analyze_m2_resolution_verdicts.py"],
    },
    "C8_moisture_abs": {
        "claim": "moisture 0.00149 vs 0.00054 kg/kg (factor 2.7); persistence never beaten;"
        " early stopping ~24 vs ~170 epochs",
        "befund": "55, 62 HA1",
        "artifacts": [
            "arch_learning_curve_constant_trained.jsonl",
            "arch_learning_curve_projected.jsonl",
        ],
        "generators": [
            "run_arch_constant_trained_learning_curve_worker.py",
            "run_arch_learning_curve_worker.py",
        ],
        "analysis": ["probe_bandit_env_ha1_margin_vs_prediction_error.py"],
    },
}


def _exists(spec: str, *, archive: bool) -> bool:
    return bool(git("cat-file", "-t", spec, cwd=ARCHIVE_ROOT if archive else REPO_ROOT))


def _clean_commit(raw: str | None) -> tuple[str | None, bool]:
    """Split ``<sha>-dirty`` into (sha, dirty_suffix)."""
    if not raw:
        return None, False
    if raw.endswith("-dirty"):
        return raw[: -len("-dirty")], True
    return raw, False


def _summarise_records(name: str, generators: list[str]) -> dict[str, Any]:
    path = RUNS_DIR / name
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
            )
        ] += 1
    groups = []
    for (raw, dirty_field, plat, py, torch_v, fp, proj), n in counter.items():
        sha, dirty_suffix = _clean_commit(raw)
        dirty = True if dirty_suffix else dirty_field  # None = never logged
        known = (
            {"this_repo": _exists(sha, archive=False), "archive": _exists(sha, archive=True)}
            if sha and sha != "unknown"
            else None
        )
        groups.append(
            {
                "n_records": n,
                "git/commit_raw": raw,
                "git/dirty_field": dirty_field,
                "dirty_suffix_in_commit": dirty_suffix,
                "platform": plat,
                "python": py,
                "torch": torch_v,
                "process/fingerprint": fp,
                "projection": proj,
                "commit_resolvable": known,
                "generators_at_commit": {
                    s: {
                        "this_repo": _exists(f"{sha}:scripts/{s}", archive=False),
                        "archive": _exists(f"{sha}:scripts/{s}", archive=True),
                    }
                    for s in generators
                }
                if sha
                else None,
                "traceable_to_clean_commit": bool(
                    known and (known["this_repo"] or known["archive"]) and dirty is False
                ),
            }
        )
    return {"artifact": name, "kind": "records", "n_records": len(rows), "groups": groups}


def _summarise_json(name: str) -> dict[str, Any]:
    """Analysis artifact: git fields if present, else compare with the archived version."""
    rel = f"artifacts/runs/{name}"
    data = json.loads((RUNS_DIR / name).read_text(encoding="utf-8"))
    env = data.get("environment") or data.get("provenance") or {}
    archived_commit = git("log", "-1", "--format=%h", "--", rel, cwd=ARCHIVE_ROOT)
    identical = None
    if archived_commit:
        archived = git("show", f"{archived_commit}:{rel}", cwd=ARCHIVE_ROOT)
        identical = json.loads(archived) == data if archived else None
    return {
        "artifact": name,
        "kind": "analysis_json",
        "has_git_fields": "git/commit" in env,
        "git/commit": env.get("git/commit"),
        "git/dirty": env.get("git/dirty"),
        "last_archive_commit_touching_file": archived_commit or None,
        "content_equals_archived_version": identical,
        "added_in_this_repo": git("log", "--diff-filter=A", "--format=%h", "--", rel),
        "traceable_to_clean_commit": env.get("git/dirty") is False,
    }


def main() -> None:
    require_clean_tree()
    report: dict[str, Any] = {}
    for claim_id, spec in CLAIMS.items():
        entries = [
            _summarise_records(a, spec["generators"])
            if a.endswith(".jsonl")
            else _summarise_json(a)
            for a in spec["artifacts"]
        ]
        analysis = {
            s: {
                "exists_in_worktree": (REPO_ROOT / "scripts" / s).exists(),
                "last_commit_archive": git(
                    "log", "-1", "--format=%h", "--", f"scripts/{s}", cwd=ARCHIVE_ROOT
                ),
            }
            for s in spec["analysis"]
        }
        untraceable = [
            e["artifact"]
            for e in entries
            if (
                not all(g["traceable_to_clean_commit"] for g in e["groups"])
                if e["kind"] == "records"
                else not e["traceable_to_clean_commit"]
            )
        ]
        report[claim_id] = {
            "claim": spec["claim"],
            "befund": spec["befund"],
            "note": spec.get("note"),
            "generators": spec["generators"],
            "analysis_scripts": analysis,
            "artifacts": entries,
            "artifacts_not_traceable_to_clean_commit": untraceable,
        }
    write_artifact(
        "verify_pr0_provenance",
        {"_provenance": provenance(), "archive_root": str(ARCHIVE_ROOT), "claims": report},
    )
    for claim_id, entry in report.items():
        print(claim_id, "-> not clean-traceable:", entry["artifacts_not_traceable_to_clean_commit"])


if __name__ == "__main__":
    main()
