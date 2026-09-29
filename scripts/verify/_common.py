"""Shared plumbing for the independent verification scripts (``scripts/verify/``).

Deliberately contains NO evaluation logic and imports nothing from the project's evaluation
code (``twinlink_ml.evaluation``, ``scripts/analyze_*``): only git provenance, JSON output
and the data-generation entry point that the audit is allowed to reuse (the simulator and
``generate_data``, which are themselves the object of the PR1 checks).

Independence rule: a verification script reads the *documented protocol* (Befund 51 text and
the record parameters) and re-implements every metric from raw arrays.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
ARCHIVE_ROOT = REPO_ROOT.parent / "twinlink_ml"
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    """Run a git command and return stripped stdout ('' on failure)."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def working_tree_dirty() -> bool:
    """True if there are uncommitted changes (tracked or untracked, not ignored)."""
    return bool(git("status", "--porcelain"))


def provenance() -> dict[str, Any]:
    """Provenance block written into every verification artifact."""
    return {
        "git/commit": git("rev-parse", "HEAD"),
        "git/dirty": working_tree_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def require_clean_tree() -> None:
    """Abort if the working tree is dirty (CLAUDE.md: clean tree before artifact runs)."""
    if working_tree_dirty():
        raise SystemExit("working tree is dirty -- commit the script first, then run it")


def write_artifact(name: str, payload: dict[str, Any]) -> Path:
    """Write ``artifacts/runs/<name>.json``; never overwrite an existing artifact."""
    path = RUNS_DIR / f"{name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists -- artifacts are never overwritten; use a new name")
    payload = {"provenance": payload.pop("_provenance"), **payload}
    path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {path}")
    return path


def _json_default(value: object) -> object:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serialisable: {type(value)}")


def read_jsonl(name: str) -> list[dict[str, Any]]:
    """Read ``artifacts/runs/<name>.jsonl`` with plain json (no project loader)."""
    with (RUNS_DIR / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
