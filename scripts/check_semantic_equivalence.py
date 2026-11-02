"""Machine evidence that a translation commit leaves behaviour unchanged.

Compares the working tree against a git reference (default ``HEAD``), per suffix:

* ``.py``: both versions are parsed, docstrings (the first ``Expr``-``Constant`` string of a
  module, class or function) are removed, and ``ast.dump`` (without attributes) must be
  identical. With ``--rename-map`` the table is first applied to the OLD tree (``Name``,
  ``arg``, ``FunctionDef``/``AsyncFunctionDef``/``ClassDef`` names, ``keyword.arg``,
  ``alias`` and ``Attribute.attr``, the latter only for names of the table), restricted to the
  files each entry lists. The new file must also pass ``py_compile``.
* ``.yaml``/``.yml``/``.toml``/``.json``: parsed objects equal (only comments may change).
  ``--allow-key FILE:dotted.key`` admits one named key whose value may differ (reported).
* ``.ipynb``: same cells, cell types, metadata; code cells (source) and outputs identical, only
  markdown cells may change. With ``--rename-map`` code cells are compared with the same AST
  check instead of textually.
* ``.sh``/``.ps1``: every line of the diff is a comment line (``#`` after whitespace).
* ``.md``: not checked here -- ``tools/check_translation.py`` covers Markdown.
* Any other changed file is a violation.

In addition, no changed text file may introduce a carriage return the old version did not
have (line endings are kept).

``--all-py`` checks every tracked ``.py`` file (and notebook), not only the changed ones -- for
the renaming pass, where a forgotten call site would sit in an unchanged file.

``--validate-map`` checks a rename table instead: injective, no new name already used as an
identifier in a file the entry applies to, and no old name occurring in any string literal
of any ``.py`` file or as a word in any tracked non-Python file.

Exit code 1 on any violation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import py_compile
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

PY_SUFFIXES = (".py",)
DATA_SUFFIXES = (".yaml", ".yml", ".toml", ".json")
SHELL_SUFFIXES = (".sh", ".ps1")
DELEGATED_SUFFIXES = (".md",)


@dataclass(frozen=True, slots=True)
class RenameEntry:
    """One row of the rename table: ``old`` becomes ``new`` in ``files``."""

    old: str
    new: str
    module: str
    scope: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Result:
    """Outcome for one file: ``ok`` and the messages (violations or notes)."""

    path: str
    ok: bool
    messages: tuple[str, ...] = ()


# --- rename table ---------------------------------------------------------------------------


def load_rename_map(path: Path) -> list[RenameEntry]:
    """Read ``{"renames": [{"old", "new", "module", "scope", "files"}, ...]}``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        RenameEntry(
            old=row["old"],
            new=row["new"],
            module=row["module"],
            scope=row["scope"],
            files=tuple(row["files"]),
        )
        for row in payload["renames"]
    ]


def renames_for(entries: Iterable[RenameEntry], path: str) -> dict[str, str]:
    """Old -> new for the entries that apply to ``path``."""
    return {e.old: e.new for e in entries if path in e.files}


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = mapping

    def _sub(self, name: str | None) -> str | None:
        return self.mapping.get(name, name) if name is not None else None

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.mapping.get(node.id, node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self.mapping.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self.mapping.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node.attr = self.mapping.get(node.attr, node.attr)
        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        node.arg = self._sub(node.arg)
        return self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> ast.AST:
        node.name = ".".join(self.mapping.get(p, p) for p in node.name.split("."))
        node.asname = self._sub(node.asname)
        return self.generic_visit(node)


def apply_renames(tree: ast.AST, mapping: Mapping[str, str]) -> ast.AST:
    """Return a renamed copy of ``tree`` (the input stays untouched)."""
    if not mapping:
        return tree
    return _Renamer(mapping).visit(copy.deepcopy(tree))


# --- .py ------------------------------------------------------------------------------------


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the first ``Expr``-``Constant`` string of every module, class and function."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def normalised_dump(source: str, mapping: Mapping[str, str] | None = None) -> str:
    """``ast.dump`` without docstrings (and with ``mapping`` applied) of ``source``."""
    tree = strip_docstrings(ast.parse(source))
    if mapping:
        tree = apply_renames(tree, mapping)
    return ast.dump(tree, include_attributes=False)


def check_python_source(
    old: str, new: str, mapping: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Violations of the AST equivalence (empty tuple = equivalent)."""
    try:
        new_dump = normalised_dump(new)
    except SyntaxError as exc:
        return (f"new version does not parse: {exc}",)
    if normalised_dump(old, mapping) != new_dump:
        return ("AST without docstrings differs" + (" (rename table applied)" if mapping else ""),)
    return ()


def _py_compiles(text: str) -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "m.py"
        src.write_text(text, encoding="utf-8")
        try:
            py_compile.compile(str(src), cfile=str(Path(tmp) / "m.pyc"), doraise=True)
        except py_compile.PyCompileError as exc:
            return str(exc)
    return None


# --- data files -----------------------------------------------------------------------------


def _parse_data(text: str, suffix: str) -> Any:
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if suffix == ".toml":
        return tomllib.loads(text)
    return json.loads(text)


def _pop_key(data: Any, dotted: str) -> Any:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node.pop(parts[-1], None) if isinstance(node, dict) else None


def check_data_source(
    old: str, new: str, suffix: str, allowed_keys: Iterable[str] = ()
) -> tuple[bool, tuple[str, ...]]:
    """Parsed objects equal except for ``allowed_keys`` (each reported if it changed)."""
    try:
        old_data, new_data = _parse_data(old, suffix), _parse_data(new, suffix)
    except Exception as exc:
        return False, (f"does not parse: {exc}",)
    notes = []
    for key in allowed_keys:
        before, after = _pop_key(old_data, key), _pop_key(new_data, key)
        if before != after:
            notes.append(f"allowed key {key} changed: {before!r} -> {after!r}")
    if old_data != new_data:
        return False, ("parsed content differs", *notes)
    return True, tuple(notes)


# --- notebooks ------------------------------------------------------------------------------


def check_notebook_source(
    old: str, new: str, mapping: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Only markdown cells may change; code cells textually (or AST with ``mapping``) equal."""
    a, b = json.loads(old), json.loads(new)
    problems: list[str] = []
    if {k: v for k, v in a.items() if k != "cells"} != {k: v for k, v in b.items() if k != "cells"}:
        problems.append("notebook metadata differs")
    if len(a["cells"]) != len(b["cells"]):
        return (*problems, "number of cells differs")
    for i, (ca, cb) in enumerate(zip(a["cells"], b["cells"], strict=True)):
        if ca["cell_type"] != cb["cell_type"]:
            problems.append(f"cell {i}: cell type differs")
            continue
        if ca["cell_type"] == "markdown":
            if {k: v for k, v in ca.items() if k != "source"} != {
                k: v for k, v in cb.items() if k != "source"
            }:
                problems.append(f"cell {i}: markdown cell fields other than source differ")
            continue
        if {k: v for k, v in ca.items() if k != "source"} != {
            k: v for k, v in cb.items() if k != "source"
        }:
            problems.append(f"cell {i}: outputs or metadata of a code cell differ")
        src_a, src_b = "".join(ca["source"]), "".join(cb["source"])
        if mapping and ca["cell_type"] == "code":
            if check_python_source(src_a, src_b, mapping):
                problems.append(f"cell {i}: code cell AST differs")
        elif src_a != src_b:
            problems.append(f"cell {i}: code cell source differs")
    return tuple(problems)


# --- shell ----------------------------------------------------------------------------------


def check_shell_source(old: str, new: str) -> tuple[str, ...]:
    """Every removed or added line must be a comment line."""
    bad = [
        line
        for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
        if line[:1] in "+-"
        and not line.startswith(("+++", "---"))
        and not line[1:].lstrip().startswith("#")
    ]
    return tuple(f"non-comment line changed: {line!r}" for line in bad)


# --- git plumbing ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def _git_show(ref: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def changed_paths(ref: str) -> list[str]:
    """Tracked files that differ from ``ref`` plus untracked, non-ignored files."""
    tracked = _git("diff", "--name-only", ref, "--").splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked) | set(untracked))


def tracked_paths() -> list[str]:
    """All tracked files of the working tree."""
    return _git("ls-files").splitlines()


# --- per file -------------------------------------------------------------------------------


def check_file(
    path: str,
    ref: str,
    entries: list[RenameEntry],
    allowed: Mapping[str, tuple[str, ...]],
) -> Result:
    """Check one repository-relative ``path`` against ``ref``."""
    suffix = Path(path).suffix
    old = _git_show(ref, path)
    file = REPO_ROOT / path
    if not file.exists():
        return Result(path, False, ("file deleted",))
    new = file.read_bytes().decode("utf-8")
    if old is None:
        return Result(path, True, ("new file (not compared)",))
    # git show with text=True already normalises nothing on an LF repository; compare raw bytes
    old_raw = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout
    if b"\r" in file.read_bytes() and b"\r" not in old_raw:
        return Result(path, False, ("carriage return introduced",))
    mapping = renames_for(entries, path)
    if suffix in PY_SUFFIXES:
        problems = list(check_python_source(old, new, mapping))
        error = _py_compiles(new)
        if error:
            problems.append(f"py_compile failed: {error}")
        return Result(path, not problems, tuple(problems))
    if suffix in DATA_SUFFIXES:
        ok, msgs = check_data_source(old, new, suffix, allowed.get(path, ()))
        return Result(path, ok, msgs)
    if suffix == ".ipynb":
        problems = check_notebook_source(old, new, mapping or None)
        return Result(path, not problems, problems)
    if suffix in SHELL_SUFFIXES:
        problems = check_shell_source(old, new)
        return Result(path, not problems, problems)
    if suffix in DELEGATED_SUFFIXES:
        return Result(path, True, ("markdown: covered by tools/check_translation.py",))
    if old == new:
        return Result(path, True)
    return Result(path, False, (f"changed file of unchecked type {suffix or '(none)'}",))


# --- rename-table validation ----------------------------------------------------------------


def _identifiers(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.update(node.name.split("."))
            if node.asname:
                names.add(node.asname)
    return names


def _string_constants(tree: ast.AST) -> list[str]:
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def validate_rename_map(entries: list[RenameEntry], sources: Mapping[str, str]) -> list[str]:
    """Problems of the table against the OLD repository content ``sources`` (path -> text).

    ``sources`` holds every tracked text file; ``.py`` files are parsed, everything else is
    searched for the old name as a whole word.
    """
    problems: list[str] = []
    olds = [e.old for e in entries]
    news = [e.new for e in entries]
    for name in {n for n in olds if olds.count(n) > 1}:
        rows = [e for e in entries if e.old == name]
        if len({e.new for e in rows}) > 1:
            problems.append(f"not a function: {name} maps to several new names")
        files = [f for e in rows for f in e.files]
        if len(files) != len(set(files)):
            problems.append(f"duplicate entry for {name} in the same file")
    pairs: dict[str, set[str]] = {}
    for e in entries:
        pairs.setdefault(e.new, set()).add(e.old)
    for new, old_names in pairs.items():
        if len(old_names) > 1:
            problems.append(f"not injective: {sorted(old_names)} all map to {new}")
    for e in entries:
        if e.old == e.new:
            problems.append(f"{e.old}: old and new name are equal")
        if not e.new.isidentifier():
            problems.append(f"{e.new}: not a valid identifier")
    trees: dict[str, ast.AST] = {}
    for path, text in sources.items():
        if path.endswith(".py"):
            trees[path] = ast.parse(text)
    for e in entries:
        for f in e.files:
            if f not in trees:
                problems.append(f"{e.old}: listed file {f} is not a tracked .py file")
                continue
            idents = _identifiers(trees[f])
            if e.old not in idents:
                problems.append(f"{e.old}: does not occur in {f}")
            if e.new in idents:
                problems.append(f"collision: new name {e.new} already used in {f}")
            if re.search(rf"\b{re.escape(e.new)}\b", sources[f]):
                problems.append(f"collision: new name {e.new} already occurs in the text of {f}")
    word = {name: re.compile(rf"\b{re.escape(name)}\b") for name in set(olds) | set(news)}
    for path, text in sources.items():
        if path.endswith(".py"):
            for const in _string_constants(trees[path]):
                for name in set(olds):
                    if word[name].search(const):
                        problems.append(f"string literal: {name} occurs in a string of {path}")
        else:
            for name in set(olds):
                if word[name].search(text):
                    problems.append(f"non-Python file: {name} occurs in {path}")
    return sorted(set(problems))


def _repository_sources(ref: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in _git("ls-tree", "-r", "--name-only", ref).splitlines():
        raw = subprocess.run(
            ["git", "show", f"{ref}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True
        ).stdout
        try:
            sources[path] = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary file (png, npz, gz): cannot contain a Python name as text
    return sources


# --- CLI ------------------------------------------------------------------------------------


def _allowed_keys(specs: Iterable[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for spec in specs:
        path, _, key = spec.partition(":")
        out.setdefault(path, []).append(key)
    return {k: tuple(v) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    """Run the check and print one line per file; exit 1 on any violation."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("paths", nargs="*", help="files to check (default: changed vs. --ref)")
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--rename-map", type=Path)
    parser.add_argument("--all-py", action="store_true", help="check every tracked .py/.ipynb")
    parser.add_argument("--allow-key", action="append", default=[], metavar="FILE:KEY")
    parser.add_argument("--validate-map", action="store_true")
    args = parser.parse_args(argv)

    entries = load_rename_map(args.rename_map) if args.rename_map else []
    if args.validate_map:
        if not entries:
            parser.error("--validate-map needs --rename-map")
        problems = validate_rename_map(entries, _repository_sources(args.ref))
        for p in problems:
            print(f"MAP  {p}")
        print(f"rename table: {len(entries)} entries, {len(problems)} problems")
        return 1 if problems else 0

    paths = list(args.paths) or changed_paths(args.ref)
    if args.all_py:
        paths = sorted(set(paths) | {p for p in tracked_paths() if p.endswith((".py", ".ipynb"))})
    allowed = _allowed_keys(args.allow_key)
    results = [check_file(p, args.ref, entries, allowed) for p in paths]
    n_bad = 0
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        n_bad += not r.ok
        print(f"{status} {r.path}" + ("" if not r.messages else " -- " + "; ".join(r.messages)))
    print(f"{len(results)} files checked, {n_bad} violations")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
