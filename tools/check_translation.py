#!/usr/bin/env python3
r"""Prueft, ob eine Uebersetzung nur Kommentare/Docstrings/Prosa geaendert hat.

Fuer die geplante Uebersetzung ins Englische (siehe ``CLAUDE.md``, Abschnitt
Documentation) angelegt, bevor die erste Uebersetzung passiert -- Cloud-Sitzungen sollen
das Werkzeug vorfinden, nicht es erst noch bauen muessen.

Vergleicht den Arbeitsbaum gegen eine Git-Referenz (Standard ``HEAD``), je nach Endung:

* ``.py``: Syntaxbaum mit entfernten Docstrings vorher/nachher gleich (``ast.dump`` ohne
  Attribute). HARTE Pruefung -- das beweist, dass nur Kommentare und Docstrings geaendert
  wurden. Jede Abweichung wird so tief wie moeglich lokalisiert (Datei und Funktion/Klasse,
  notfalls Modulebene) gemeldet, rekursiv: ein Elternknoten wird nur gemeldet, wenn KEIN
  benanntes Kind selbst schon abweicht.
* ``.yaml``/``.yml``: geparster Inhalt gleich. HARTE Pruefung -- mit EINER Ausnahme:
  ``provenance.description``/``provenance.open_todos`` sind dieselben rein dokumentarischen
  Freitextfelder, die ``twinlink_ml.simulation.config.without_documentary_provenance`` vom
  Config-Fingerprint ausschliesst (siehe dort und ``docs/ARCHIV.md``) -- bewusst hier
  dupliziert statt importiert, ``tools/`` bleibt ohne Abhaengigkeit vom Paket-Quellbaum.
  Nur diese zwei Felder duerfen sich aendern (z. B. durch Uebersetzung), alle anderen
  Werte bleiben eine harte Pruefung.
* ``.md``: Inline-/Fenced-Code-Abschnitte (`` `...` ``/```` ```...``` ````) als Multimenge
  gleich, HART -- Uebersetzung darf keinen Code beruehren. Zahlen ausserhalb von
  Code-Abschnitten: deutsche Schreibweise vorher (Komma dezimal, Punkt Tausender) gegen
  englische Schreibweise nachher (Punkt dezimal, Komma Tausender) als Werte verglichen;
  Abweichungen werden nur GEMELDET, nicht als Fehler gewertet, weil die Schreibweise im
  Bestand bereits gemischt ist (siehe Auftrag).

Exit-Code ungleich 0 bei mindestens einem harten Verstoss.
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
    """Ein Fund: Datei, Art (hart/Hinweis) und Meldungstext."""

    path: str
    kind: str  # "hard" oder "info"
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
    """Meldet die tiefstmoegliche(n) Stelle(n), an denen ``old_node``/``new_node`` abweichen.

    Rekursiv: ein Knoten wird nur gemeldet, wenn keines seiner benannten Kinder
    (Funktion/Klasse) selbst schon abweicht, sonst wird nur in die Kinder hinabgestiegen.
    Verhindert, dass eine einzelne geaenderte Zeile in einer tief verschachtelten Methode
    alle umschliessenden Klassen/Module ebenfalls als "abweichend" meldet.
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
                Finding(path, "hard", f"{child_qualname}: neu (nicht durch Uebersetzung "
                "erklaerbar)")
            )
            any_child_reported = True
        elif name not in new_children:
            findings.append(
                Finding(path, "hard", f"{child_qualname}: entfernt (nicht durch Uebersetzung "
                "erklaerbar)")
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
    """Harte Pruefung: Syntaxbaum ohne Docstrings identisch (siehe Moduldocstring)."""
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
    """Harte Pruefung: geparster Inhalt identisch ausserhalb dokumentarischer Felder."""
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
    """Code-Abschnitte hart, Zahlenschreibweise ausserhalb von Code nur als Hinweis."""
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
    """Dispatcht auf die passende ``check_*``-Funktion nach Dateiendung."""
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
    """CLI-Einstiegspunkt -- siehe Moduldocstring fuer die Argumente."""
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
