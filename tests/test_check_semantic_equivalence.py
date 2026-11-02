"""Tests for ``scripts/check_semantic_equivalence.py`` (pure comparison functions, no git)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_semantic_equivalence as cse  # noqa: E402

# --- .py ------------------------------------------------------------------------------------


def test_docstring_and_comment_changes_are_equivalent() -> None:
    old = (
        '"""Modul."""\n\n\nclass A:\n    """Klasse."""\n\n    def f(self, x: int) -> int:\n'
        '        """Funktion."""\n        return x + 1  # plus eins\n'
    )
    new = (
        '"""Module."""\n\n\nclass A:\n    """Class."""\n\n    def f(self, x: int) -> int:\n'
        '        """Function."""\n        return x + 1  # plus one\n'
    )
    assert cse.check_python_source(old, new) == ()


def test_changed_string_literal_is_a_violation() -> None:
    assert cse.check_python_source('x = "alt"\n', 'x = "old"\n')


def test_changed_attribute_docstring_is_a_violation() -> None:
    # a bare string after an assignment is not a docstring in the sense of the check
    old = 'X = 1\n"""Wert."""\n'
    new = 'X = 1\n"""Value."""\n'
    assert cse.check_python_source(old, new)


def test_added_docstring_where_none_was_is_equivalent_but_code_change_is_not() -> None:
    assert cse.check_python_source("def f():\n    pass\n", 'def f():\n    """Doc."""\n') == ()
    assert cse.check_python_source("def f():\n    return 1\n", "def f():\n    return 2\n")


def test_rename_map_is_applied_to_the_old_tree() -> None:
    old = (
        "import m as heiss\n\n\ndef kalt(anzahl_x: int, *, warm: int = 0) -> int:\n"
        "    total = anzahl_x + warm\n    return heiss.f(total, anzahl_x=anzahl_x)\n\n\n"
        "kalt(1, warm=2)\nobj.anzahl_x\n"
    )
    new = (
        "import m as hot\n\n\ndef cold(count_x: int, *, warm: int = 0) -> int:\n"
        "    total = count_x + warm\n    return hot.f(total, count_x=count_x)\n\n\n"
        "cold(1, warm=2)\nobj.count_x\n"
    )
    mapping = {"heiss": "hot", "kalt": "cold", "anzahl_x": "count_x"}
    assert cse.check_python_source(old, new, mapping) == ()
    assert cse.check_python_source(old, new)  # without the table the trees differ


def test_forgotten_call_site_is_detected() -> None:
    old = "def kalt() -> None:\n    pass\n\n\nkalt()\nkalt()\n"
    new = "def cold() -> None:\n    pass\n\n\ncold()\nkalt()\n"
    assert cse.check_python_source(old, new, {"kalt": "cold"})


def test_apply_renames_leaves_input_untouched_and_string_constants_alone() -> None:
    tree = ast.parse('kalt = "kalt"\n')
    renamed = cse.apply_renames(tree, {"kalt": "cold"})
    assert ast.unparse(renamed) == "cold = 'kalt'"
    assert ast.unparse(tree) == "kalt = 'kalt'"


def test_strip_docstrings_keeps_a_body() -> None:
    tree = cse.strip_docstrings(ast.parse('def f():\n    """Nur Doc."""\n'))
    assert ast.unparse(tree) == "def f():\n    pass"


# --- data files -----------------------------------------------------------------------------


def test_yaml_comment_change_is_equivalent_value_change_is_not() -> None:
    ok, _ = cse.check_data_source("a: 1  # eins\n", "a: 1  # one\n", ".yaml")
    assert ok
    ok, _ = cse.check_data_source("a: 1\n", "a: 2\n", ".yaml")
    assert not ok


def test_toml_allowed_key_is_reported_not_failed() -> None:
    old = '[project]\nname = "x"\ndescription = "ML fuer X"\n# Kommentar\n'
    new = '[project]\nname = "x"\ndescription = "ML for X"\n# comment\n'
    ok, notes = cse.check_data_source(old, new, ".toml")
    assert not ok
    ok, notes = cse.check_data_source(old, new, ".toml", ["project.description"])
    assert ok
    assert any("project.description" in n for n in notes)


def test_json_equivalence() -> None:
    assert cse.check_data_source('{"a": 1}', '{ "a" : 1 }', ".json")[0]
    assert not cse.check_data_source('{"a": 1}', '{"a": 2}', ".json")[0]


# --- notebooks ------------------------------------------------------------------------------


def _nb(md: str, code: str, outputs: list[object] | None = None) -> str:
    return json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": [md]},
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": 1,
                    "outputs": outputs or [],
                    "source": [code],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def test_notebook_markdown_change_is_equivalent() -> None:
    assert (
        cse.check_notebook_source(_nb("# Lernkurve", "x = 1"), _nb("# Learning curve", "x = 1"))
        == ()
    )


def test_notebook_code_or_output_change_is_a_violation() -> None:
    assert cse.check_notebook_source(_nb("a", "x = 1"), _nb("a", "x = 2"))
    assert cse.check_notebook_source(_nb("a", "x = 1", ["o"]), _nb("a", "x = 1", ["p"]))


def test_notebook_code_cells_with_rename_map() -> None:
    old, new = _nb("a", "kalt = 1"), _nb("a", "cold = 1")
    assert cse.check_notebook_source(old, new)
    assert cse.check_notebook_source(old, new, {"kalt": "cold"}) == ()


# --- shell ----------------------------------------------------------------------------------


def test_shell_comment_only_diff() -> None:
    old = "#!/bin/sh\n# Kommentar\necho a\n"
    assert cse.check_shell_source(old, "#!/bin/sh\n# comment\necho a\n") == ()
    assert cse.check_shell_source(old, "#!/bin/sh\n# Kommentar\necho b\n")


# --- rename-table validation ----------------------------------------------------------------


def _entry(old: str, new: str, files: tuple[str, ...] = ("s.py",)) -> cse.RenameEntry:
    return cse.RenameEntry(old=old, new=new, module="s", scope="local", files=files)


def test_valid_table_has_no_problems() -> None:
    sources = {"s.py": "def f(zahl_y: int) -> int:\n    return zahl_y\n", "d.md": "x\n"}
    assert cse.validate_rename_map([_entry("zahl_y", "number_y")], sources) == []


def test_table_must_be_injective() -> None:
    sources = {"s.py": "a = 1\nb = 2\n"}
    problems = cse.validate_rename_map([_entry("a", "c"), _entry("b", "c")], sources)
    assert any("not injective" in p for p in problems)


def test_collision_with_existing_name_is_detected() -> None:
    sources = {"s.py": "kalt = 1\ncold = 2\n"}
    problems = cse.validate_rename_map([_entry("kalt", "cold")], sources)
    assert any("collision" in p for p in problems)


def test_old_name_in_string_literal_or_non_python_file_is_detected() -> None:
    sources = {"s.py": 'kalt = 1\nkey = "kalt"\n'}
    assert any(
        "string literal" in p for p in cse.validate_rename_map([_entry("kalt", "cold")], sources)
    )
    sources = {"s.py": "kalt = 1\n", "conf.yaml": "x: kalt\n"}
    assert any(
        "non-Python" in p for p in cse.validate_rename_map([_entry("kalt", "cold")], sources)
    )


def test_listed_file_must_contain_the_old_name() -> None:
    problems = cse.validate_rename_map([_entry("kalt", "cold")], {"s.py": "x = 1\n"})
    assert any("does not occur" in p for p in problems)


def test_the_table_itself_is_ignored_in_the_text_search() -> None:
    sources = {"s.py": "kalt = 1\n", "rename_map.json": '{"old": "kalt"}\n'}
    entries = [_entry("kalt", "cold")]
    assert cse.validate_rename_map(entries, sources)
    assert cse.validate_rename_map(entries, sources, ignore=["rename_map.json"]) == []
