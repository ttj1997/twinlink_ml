"""TE1 -- is every third-party package imported in src/, scripts/ and tests/ declared.

Parses all git-tracked ``.py`` files with ``ast`` (static ``import`` statements; the two dynamic
imports in the repository are ``importlib`` loads of repository scripts and a ``__import__`` in
``utils/tracking.py`` for the optional MLflow logger), maps each top-level module to its
distribution and checks it against ``pyproject.toml`` (base dependencies and every extra).

    python scripts/verify/te1_declared_dependencies.py
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections import defaultdict
from importlib.metadata import packages_distributions, version
from typing import Any

from _common import REPO_ROOT, git, provenance, require_clean_tree, write_artifact

# import name -> distribution name where they differ
DIST_ALIASES = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "stable_baselines3": "stable-baselines3",
}


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    return normalise(re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0])


def main() -> None:
    require_clean_tree()
    files = [
        f
        for f in git("ls-files", "src", "scripts", "tests", "tools", "notebooks").split()
        if f.endswith(".py")
    ]
    local_modules = {p.stem for p in (REPO_ROOT / "scripts").glob("*.py")} | {
        p.stem for p in (REPO_ROOT / "scripts" / "verify").glob("*.py")
    }
    local_modules |= {"twinlink_ml", "conftest", "__future__"}
    imported: dict[str, set[str]] = defaultdict(set)
    for f in files:
        tree = ast.parse((REPO_ROOT / f).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported[alias.name.split(".")[0]].add(f)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported[node.module.split(".")[0]].add(f)
    third_party = {
        m: fs
        for m, fs in imported.items()
        if m not in sys.stdlib_module_names and m not in local_modules
    }

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {requirement_name(s) for s in pyproject["project"]["dependencies"]}
    extras = {
        name: {requirement_name(s) for s in specs}
        for name, specs in pyproject["project"]["optional-dependencies"].items()
    }
    dist_of = packages_distributions()
    rows: list[dict[str, Any]] = []
    for module, fs in sorted(third_party.items()):
        dist = normalise(DIST_ALIASES.get(module, (dist_of.get(module) or [module])[0]))
        where = (["base"] if dist in base else []) + [e for e, s in extras.items() if dist in s]
        try:
            installed = version(dist)
        except Exception:
            installed = None
        rows.append(
            {
                "module": module,
                "distribution": dist,
                "declared_in": where,
                "installed_version": installed,
                "n_files": len(fs),
                "example_files": sorted(fs)[:3],
            }
        )
    declared_but_unused = sorted(
        (base | set().union(*extras.values())) - {r["distribution"] for r in rows}
    )
    write_artifact(
        "verify_te1_declared_dependencies",
        {
            "_provenance": provenance(),
            "n_files_parsed": len(files),
            "third_party_imports": rows,
            "undeclared": [r["module"] for r in rows if not r["declared_in"]],
            "declared_but_no_static_import_found": declared_but_unused,
            "dynamic_imports_note": "importlib loads of repository scripts (rl/env.py, "
            "probe_critical_moisture_sensitivity.py); __import__ of the MLflow logger in "
            "utils/tracking.py (extra 'train')",
        },
    )
    for r in rows:
        print(r["module"], r["declared_in"], r["installed_version"])
    print("undeclared:", [r["module"] for r in rows if not r["declared_in"]])
    print("declared but unused:", declared_but_unused)


if __name__ == "__main__":
    main()
