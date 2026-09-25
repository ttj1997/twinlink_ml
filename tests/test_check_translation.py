"""Tests fuer ``tools/check_translation.py``.

Testet nur die reinen Vergleichsfunktionen (``check_python``/``check_yaml``/
``check_markdown``, kein Git) -- ``tools/`` ist kein Package (analog zu ``scripts/``,
siehe dortige Konvention), deshalb per Dateipfad importiert.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_translation.py"
_SPEC = importlib.util.spec_from_file_location("check_translation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_translation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_translation
_SPEC.loader.exec_module(check_translation)


# --- .py -----------------------------------------------------------------------------


def test_python_docstring_only_change_is_clean() -> None:
    old = (
        '"""Alter Docstring."""\n\n\n'
        'def f(x: int) -> int:\n    """Alter Funktions-Docstring."""\n    return x + 1\n'
    )
    new = (
        '"""New docstring."""\n\n\n'
        'def f(x: int) -> int:\n    """New function docstring."""\n    return x + 1\n'
    )
    assert check_translation.check_python(old, new, "m.py") == []


def test_python_comment_only_change_is_clean() -> None:
    old = "x = 1  # alter Kommentar\n"
    new = "x = 1  # new comment\n"
    assert check_translation.check_python(old, new, "m.py") == []


def test_python_logic_change_is_reported_hard_with_location() -> None:
    old = "def f(x):\n    return x + 1\n"
    new = "def f(x):\n    return x + 2\n"
    findings = check_translation.check_python(old, new, "m.py")
    assert len(findings) == 1
    assert findings[0].kind == "hard"
    assert findings[0].path == "m.py"
    assert "f" in findings[0].message


def test_python_deeply_nested_change_does_not_report_ancestors() -> None:
    old = "class A:\n    class B:\n        def f(self):\n            return 1\n"
    new = "class A:\n    class B:\n        def f(self):\n            return 2\n"
    findings = check_translation.check_python(old, new, "m.py")
    locations = [f.message for f in findings]
    assert len(findings) == 1
    assert "A.B.f" in locations[0]


def test_python_syntax_error_is_reported_hard() -> None:
    findings = check_translation.check_python("x = 1\n", "x = (\n", "m.py")
    assert findings and findings[0].kind == "hard"


def test_python_new_function_is_reported_hard() -> None:
    old = "def f():\n    return 1\n"
    new = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
    findings = check_translation.check_python(old, new, "m.py")
    assert any("g" in f.message for f in findings)


# --- .yaml -----------------------------------------------------------------------------


def test_yaml_open_todos_translation_is_clean() -> None:
    old = (
        'provenance:\n  status: placeholder\n  open_todos:\n    - "Alte Frage"\n'
        "geometry:\n  x: 1.0\n"
    )
    new = (
        'provenance:\n  status: placeholder\n  open_todos:\n    - "Old question"\n'
        "geometry:\n  x: 1.0\n"
    )
    assert check_translation.check_yaml(old, new, "c.yaml") == []


def test_yaml_description_translation_is_clean() -> None:
    old = "provenance:\n  description: Alter Text\ngeometry:\n  x: 1.0\n"
    new = "provenance:\n  description: New text\ngeometry:\n  x: 1.0\n"
    assert check_translation.check_yaml(old, new, "c.yaml") == []


def test_yaml_numeric_change_is_reported_hard() -> None:
    old = "provenance:\n  status: placeholder\ngeometry:\n  x: 1.0\n"
    new = "provenance:\n  status: placeholder\ngeometry:\n  x: 2.0\n"
    findings = check_translation.check_yaml(old, new, "c.yaml")
    assert findings and findings[0].kind == "hard"


def test_yaml_status_change_is_reported_hard() -> None:
    """``status`` ist kategorial, kein Freitext -- bleibt eine harte Pruefung."""
    old = "provenance:\n  status: placeholder\n"
    new = "provenance:\n  status: calibrated\n"
    findings = check_translation.check_yaml(old, new, "c.yaml")
    assert findings and findings[0].kind == "hard"


def test_yaml_parse_error_is_reported_hard() -> None:
    findings = check_translation.check_yaml("a: 1\n", "a: [1, 2\n", "c.yaml")
    assert findings and findings[0].kind == "hard"


# --- .md -----------------------------------------------------------------------------


def test_markdown_prose_translation_is_clean() -> None:
    old = "Der Wert ist `0.712` (siehe `config.py`)."
    new = "The value is `0.712` (see `config.py`)."
    assert check_translation.check_markdown(old, new, "d.md") == []


def test_markdown_code_span_change_is_reported_hard() -> None:
    old = "Siehe `config.py`."
    new = "Siehe `config_new.py`."
    findings = check_translation.check_markdown(old, new, "d.md")
    assert any(f.kind == "hard" for f in findings)


def test_markdown_number_format_change_is_reported_as_info_only() -> None:
    old = "Der Wert ist 1.234,5 kg."
    new = "The value is 1,234.5 kg."
    findings = check_translation.check_markdown(old, new, "d.md")
    assert findings == [] or all(f.kind == "info" for f in findings)


def test_markdown_dropped_number_is_reported_as_info() -> None:
    old = "Wert 0,712 gemessen."
    new = "Value measured."
    findings = check_translation.check_markdown(old, new, "d.md")
    assert findings and all(f.kind == "info" for f in findings)
