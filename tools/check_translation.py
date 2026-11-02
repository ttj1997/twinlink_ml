#!/usr/bin/env python3
r"""Checks whether a translation changed only comments/docstrings/prose.

Created for the planned translation into English (see ``CLAUDE.md``, section
Documentation) before the first translation happens -- cloud sessions should
find the tool in place, not have to build it first.

Compares the working tree against a Git reference (default ``HEAD``), depending on the suffix:

* ``.py``: syntax tree with docstrings removed equal before/after (``ast.dump`` without
  attributes). HARD check -- this proves that only comments and docstrings were
  changed. Every deviation is localised as deeply as possible (file and function/class,
  module level if need be) and reported, recursively: a parent node is only reported if NO
  named child itself already deviates.
* ``.yaml``/``.yml``: parsed content equal. HARD check -- with ONE exception:
  ``provenance.description``/``provenance.open_todos`` are the same purely documentary
  free-text fields that ``twinlink_ml.simulation.config.without_documentary_provenance`` excludes
  from the config fingerprint (see there and ``docs/ARCHIV.md``) -- deliberately
  duplicated here instead of imported, ``tools/`` stays without a dependency on the package
  source tree. Only these two fields may change (e.g. through translation), all other
  values remain a hard check.
* ``.md``: inline/fenced code sections (`` `...` ``/```` ```...``` ````) equal as a multiset,
  HARD -- translation must not touch any code. Numbers outside of
  code sections: German notation before (decimal comma, thousands point) compared as values
  against English notation after (decimal point, thousands comma);
  deviations are only REPORTED, not counted as errors, because the notation in the
  existing material is already mixed (see the task).

Exit code non-zero on at least one hard violation.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DOCUMENTARY_YAML_FIELDS = ("description", "open_todos")
"""Deckt sich mit ``twinlink_ml.simulation.config._DOCUMENTARY_PROVENANCE_FIELDS``."""

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_NUMBER_RE = re.compile(r"(?<![\w.,])\d[\d.,]*\d(?![\w.,])|(?<![\w.,])\d(?![\w.,])")

CHECKED_SUFFIXES = (".py", ".yaml", ".yml", ".md")


@dataclass(frozen=True, slots=True)
class Finding:
    """A finding: file, kind (hard/info) and message text."""

    path: str
    kind: str  # "hard" or "info"
    message: str


# --- .py -----------------------------------------------------------------------------


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]
    return tree


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _direct_named_children(node: ast.AST) -> dict[str, ast.AST]:
    return {
        child.name: child
        for child in ast.iter_child_nodes(node)
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _localize_diffs(
    old_node: ast.AST, new_node: ast.AST, qualname: str, path: str, findings: list[Finding]
) -> None:
    """Reports the deepest possible location(s) at which ``old_node``/``new_node`` deviate.

    Recursive: a node is only reported if none of its named children
    (function/class) itself already deviates; otherwise only the children are descended into.
    Prevents a single changed line in a deeply nested method from also reporting
    all enclosing classes/modules as "deviating".
    """
    if _dump(old_node) == _dump(new_node):
        return
    old_children = _direct_named_children(old_node)
    new_children = _direct_named_children(new_node)
    any_child_reported = False
    for name in sorted(set(old_children) | set(new_children)):
        child_qualname = f"{qualname}.{name}" if qualname else name
        if name not in old_children:
            findings.append(
                Finding(
                    path, "hard", f"{child_qualname}: neu (nicht durch Uebersetzung erklaerbar)"
                )
            )
            any_child_reported = True
        elif name not in new_children:
            findings.append(
                Finding(
                    path,
                    "hard",
                    f"{child_qualname}: entfernt (nicht durch Uebersetzung erklaerbar)",
                )
            )
            any_child_reported = True
        else:
            before = len(findings)
            _localize_diffs(old_children[name], new_children[name], child_qualname, path, findings)
            any_child_reported = any_child_reported or len(findings) > before
    if not any_child_reported:
        location = qualname or "<Modulebene>"
        findings.append(
            Finding(
                path, "hard", f"{location}: Code unterschiedlich (nicht nur Docstring/Kommentar)"
            )
        )


def check_python(old: str, new: str, path: str) -> list[Finding]:
    """Hard check: syntax tree without docstrings identical (see module docstring)."""
    try:
        old_tree = _strip_docstrings(ast.parse(old))
        new_tree = _strip_docstrings(ast.parse(new))
    except SyntaxError as error:
        return [Finding(path, "hard", f"Syntaxfehler: {error}")]
    findings: list[Finding] = []
    _localize_diffs(old_tree, new_tree, "", path, findings)
    return findings


# --- .yaml/.yml ------------------------------------------------------------------------


def _without_documentary_yaml_fields(data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("provenance"), dict):
        data = dict(data)
        data["provenance"] = {
            key: value
            for key, value in data["provenance"].items()
            if key not in _DOCUMENTARY_YAML_FIELDS
        }
    return data


def check_yaml(old: str, new: str, path: str) -> list[Finding]:
    """Hard check: parsed content identical outside documentary fields."""
    try:
        old_data = _without_documentary_yaml_fields(yaml.safe_load(old))
        new_data = _without_documentary_yaml_fields(yaml.safe_load(new))
    except yaml.YAMLError as error:
        return [Finding(path, "hard", f"YAML-Parsefehler: {error}")]
    if old_data != new_data:
        return [
            Finding(
                path,
                "hard",
                "geparster Inhalt unterschiedlich (ausserhalb von "
                f"provenance.{{{', '.join(_DOCUMENTARY_YAML_FIELDS)}}})",
            )
        ]
    return []


# --- .md ---------------------------------------------------------------------------------


def _code_spans_and_remainder(text: str) -> tuple[list[str], str]:
    spans: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return "\x00"

    remainder = _FENCED_CODE_RE.sub(_collect, text)
    remainder = _INLINE_CODE_RE.sub(_collect, remainder)
    return spans, remainder


def _numbers(text: str, *, decimal: str, thousands: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        if token.count(decimal) > 1:
            continue
        cleaned = token.replace(thousands, "")
        if decimal != ".":
            cleaned = cleaned.replace(decimal, ".")
        try:
            values.append(round(float(cleaned), 9))
        except ValueError:
            continue
    return values


def check_markdown(old: str, new: str, path: str) -> list[Finding]:
    """Code sections hard, number notation outside of code only as a notice."""
    findings: list[Finding] = []
    old_spans, old_remainder = _code_spans_and_remainder(old)
    new_spans, new_remainder = _code_spans_and_remainder(new)
    if Counter(old_spans) != Counter(new_spans):
        findings.append(
            Finding(path, "hard", "Inline-/Fenced-Code-Abschnitte unterscheiden sich (Multimenge)")
        )
    old_numbers = Counter(_numbers(old_remainder, decimal=",", thousands="."))
    new_numbers = Counter(_numbers(new_remainder, decimal=".", thousands=","))
    if old_numbers != new_numbers:
        missing = sorted((old_numbers - new_numbers).elements())
        extra = sorted((new_numbers - old_numbers).elements())
        findings.append(
            Finding(
                path,
                "info",
                f"Zahlen ausserhalb von Code veraendert -- nur vorher: {missing}, "
                f"nur nachher: {extra} (gemischte Schreibweise im Bestand, nur Hinweis)",
            )
        )
    return findings


# --- Git/CLI -----------------------------------------------------------------------------


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _git_show(ref: str, rel_path: str, repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _changed_paths(ref: str, repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", ref, "--"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def check_file(rel_path: str, ref: str, repo_root: Path) -> list[Finding]:
    """Dispatches to the matching ``check_*`` function by file suffix."""
    old = _git_show(ref, rel_path, repo_root)
    new_path = repo_root / rel_path
    if old is None or not new_path.is_file():
        return []
    new = new_path.read_text(encoding="utf-8")
    if old == new:
        return []
    if rel_path.endswith(".py"):
        return check_python(old, new, rel_path)
    if rel_path.endswith((".yaml", ".yml")):
        return check_yaml(old, new, rel_path)
    if rel_path.endswith(".md"):
        return check_markdown(old, new, rel_path)
    return []


def main(argv: list[str] | None = None) -> int:
    """CLI entry point -- see the module docstring for the arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Git-Referenz, gegen die verglichen wird")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Zu pruefende Pfade (Default: alle .py/.yaml/.yml/.md, die sich gegen --ref "
        "unterscheiden)",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    paths = args.paths or _changed_paths(args.ref, repo_root)
    paths = [path for path in paths if path.endswith(CHECKED_SUFFIXES)]

    findings: list[Finding] = []
    for rel_path in paths:
        findings.extend(check_file(rel_path, args.ref, repo_root))

    hard = [finding for finding in findings if finding.kind == "hard"]
    info = [finding for finding in findings if finding.kind == "info"]
    for finding in hard:
        print(f"FEHLER {finding.path}: {finding.message}")
    for finding in info:
        print(f"HINWEIS {finding.path}: {finding.message}")
    if not findings:
        print(f"OK -- {len(paths)} Datei(en) geprueft, keine Abweichung ausserhalb Uebersetzung.")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
