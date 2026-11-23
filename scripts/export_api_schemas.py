"""Write the JSON Schemas of ``twinlink_ml.api`` to ``docs/api/``.

One file per request, response and self-description of every service and one for the manifest of
the model bundle. The files are documentation for the consortium partners, generated from the
pydantic models; ``tests/test_api_schemas.py`` fails when they are stale.

    tools/run_here.sh scripts/export_api_schemas.py            # write docs/api/*.schema.json
    tools/run_here.sh scripts/export_api_schemas.py --check    # report differences, write nothing

Existing files are replaced only when their content changes; a file that is no longer generated is
reported and left in place (remove it by hand in the same commit).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from twinlink_ml.api.schema_export import collect_schemas, render_schema
from twinlink_ml.utils.code_location import require_code_location

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare only, write nothing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_code_location(ROOT)
    expected = {name: render_schema(schema) for name, schema in collect_schemas().items()}
    changed = []
    for name, text in sorted(expected.items()):
        path = OUTPUT_DIR / name
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            continue
        changed.append(name)
        if not args.check:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    stray = sorted(path.name for path in OUTPUT_DIR.glob("*.json") if path.name not in expected)
    verb = "differs" if args.check else "written"
    for name in changed:
        print(f"{verb}: docs/api/{name}")
    for name in stray:
        print(f"not generated any more: docs/api/{name}")
    print(f"{len(expected)} schemas, {len(changed)} {verb}, {len(stray)} stray")
    return 1 if (args.check and changed) or stray else 0


if __name__ == "__main__":
    sys.exit(main())
