r"""Redaction of the plant-near run record before it enters the repository (docs/NN_plant_scale.md).

``scripts/run_plant_scale.py`` (commit b4ce48d) wrote two per-segment absolute role values into the
balances of the repository record: the segment mean of the intake temperature (``surroundings_celsius``
of the ``ambient`` surroundings, repeated under ``heat_loss`` and ``false_air``) and the segment mean
of the process-gas flow (``gas_flow_kg_per_s``). The study request allows only relative or aggregated
values in the repository and keeps the per-role means local. This script reads the full record (kept
git-ignored under ``local/reports/<name>_full.json``), removes exactly these fields under
``balances.per_segment`` and writes ``artifacts/runs/<name>.json`` with a ``redaction`` entry
(SHA-256 of the full record, removed field names and count, commit of this script). Every other value
is copied unchanged; ``scripts/analyze_plant_scale.py`` then runs on the redacted record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import REPO_ROOT, RUNS_DIR, git  # noqa: E402

REMOVED_FIELDS = ("gas_flow_kg_per_s", "surroundings_celsius")
"""Per-segment absolute role values removed from ``balances.per_segment``."""


def _strip(node: Any) -> int:
    """Remove :data:`REMOVED_FIELDS` from every dict below ``node``; return the count."""
    count = 0
    if isinstance(node, dict):
        for key in REMOVED_FIELDS:
            if key in node:
                del node[key]
                count += 1
        for value in node.values():
            count += _strip(value)
    elif isinstance(node, list):
        for value in node:
            count += _strip(value)
    return count


def redact_record(record: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Copy of ``record`` without :data:`REMOVED_FIELDS` under ``balances.per_segment``."""
    redacted = copy.deepcopy(record)
    count = _strip(redacted["balances"]["per_segment"])
    return redacted, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    source = REPO_ROOT / "local" / "reports" / f"{args.name}_full.json"
    out_path = RUNS_DIR / f"{args.name}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; artifacts are never overwritten")
    raw = source.read_bytes()
    record = json.loads(raw)
    redacted, count = redact_record(record)
    redacted["redaction"] = {
        "source": f"local/reports/{args.name}_full.json (git-ignored)",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "removed_fields": list(REMOVED_FIELDS),
        "removed_scope": "balances.per_segment",
        "removed_count": count,
        "redaction_commit": git("rev-parse", "HEAD"),
        "redaction_dirty": bool(git("status", "--porcelain")),
    }
    out_path.write_text(json.dumps(redacted, indent=1), encoding="utf-8")
    print(f"wrote {out_path} ({count} fields removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
