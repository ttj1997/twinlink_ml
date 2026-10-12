#!/usr/bin/env bash
# Start a Python command so that it loads the code of THIS checkout/worktree.
#
#   tools/run_here.sh scripts/run_m4r_test.py --dt 20 --resume
#   tools/run_here.sh -m pytest
#
# Why: an editable install points at one directory only. In a git worktree the script would
# otherwise run the code of the main checkout (or of whatever the interpreter was installed
# from) while the record names the worktree's commit. This script
#   * changes into the root of the checkout/worktree this file belongs to,
#   * puts <root>/src first on PYTHONPATH,
#   * uses the .venv of this root, else of the main checkout (override: TWINLINK_PYTHON).
# twinlink_ml.utils.code_location aborts any artifact run for which this is not true.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)"
MAIN="$(dirname "$COMMON")"
PY="${TWINLINK_PYTHON:-}"
for candidate in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" \
                 "$MAIN/.venv/Scripts/python.exe" "$MAIN/.venv/bin/python"; do
  if [ -z "$PY" ] && [ -x "$candidate" ]; then PY="$candidate"; fi
done
if [ -z "$PY" ]; then
  echo "run_here.sh: no .venv in $ROOT or $MAIN; set TWINLINK_PYTHON" >&2
  exit 2
fi
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
exec "$PY" -u "$@"
