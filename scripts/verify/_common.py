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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
ARCHIVE_ROOT = REPO_ROOT.parent / "twinlink_ml"
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"

TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")


def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command (in ``REPO_ROOT`` by default) and return stripped stdout ('' on failure)."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd if cwd is not None else REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def working_tree_dirty(ignore_paths: Sequence[Path | str] = ()) -> bool:
    """True if there are uncommitted changes (tracked or untracked, not ignored).

    ``ignore_paths``: UNTRACKED files that do not count, typically the output file a driver is about
    to write into (``twinlink_ml.utils.tracking.status_is_dirty``); a modified tracked file always
    counts.
    """
    from twinlink_ml.utils.tracking import status_is_dirty

    if not ignore_paths:
        return bool(git("status", "--porcelain"))
    return status_is_dirty(
        git("status", "--porcelain", "--untracked-files=all"), ignore_paths, REPO_ROOT
    )


NO_HEAT_NETWORK = "not used"
"""Value of ``heat_network`` in :func:`provenance` for a run without a heat network."""


def provenance(heat_network: Any = None, ignore_paths: Sequence[Path | str] = ()) -> dict[str, Any]:
    """Provenance block written into every verification artifact.

    Since Befund 79 every block also names the search bound of the steady-state solver and, for
    the heat network, the check of ``dT_min`` and the fingerprint of the network settings:
    pass the ``SiteSpec`` or ``HeatNetworkConfig`` of the run (anything with ``approach_check``
    and ``fingerprint``), or :data:`NO_HEAT_NETWORK` for a run without one. ``None`` (the
    default, scripts written before Befund 79) records both as ``None``, i.e. "not recorded".
    """
    from twinlink_ml.evaluation.experiments import NEW_STUDY_PRESET_VERSION
    from twinlink_ml.simulation.steady_state import BED_TEMPERATURE_BRACKET_RULE
    from twinlink_ml.utils.code_location import require_code_location

    # Abort if twinlink_ml is not loaded from this repository/worktree (the one whose commit
    # is recorded below); fields: code/package_dir, code/interpreter, code/root_checked.
    location = require_code_location(REPO_ROOT)

    if heat_network is None:
        approach_check = fingerprint = None
    elif heat_network == NO_HEAT_NETWORK:
        approach_check = fingerprint = NO_HEAT_NETWORK
    else:
        approach_check = heat_network.approach_check
        fingerprint = heat_network.fingerprint
    block = {
        "git/commit": git("rev-parse", "HEAD"),
        "git/dirty": working_tree_dirty(ignore_paths),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **location,
        "preset/new_study_version": NEW_STUDY_PRESET_VERSION,
        "solver/bed_temperature_bracket": BED_TEMPERATURE_BRACKET_RULE,
        "heatnetwork/approach_check": approach_check,
        "heatnetwork/fingerprint": fingerprint,
    }
    if ignore_paths:
        # only when an exemption was used: the record says which untracked files did not count
        block["git/dirty_ignored_paths"] = sorted(str(p) for p in ignore_paths)
    return block


def require_clean_tree(ignore_paths: Sequence[Path | str] = ()) -> None:
    """Abort if the working tree is dirty (CLAUDE.md: clean tree before artifact runs)."""
    if working_tree_dirty(ignore_paths):
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
