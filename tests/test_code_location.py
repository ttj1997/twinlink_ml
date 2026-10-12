"""Provenance of the code location: abort if ``twinlink_ml`` is not the repository's own code.

Background: the editable install of an interpreter points at one directory; a second
checkout or worktree silently runs that other code while the record names the commit of the
directory the script was started in (see ``twinlink_ml.utils.code_location``).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from twinlink_ml.utils.code_location import (
    CodeLocationError,
    code_location_fields,
    find_repo_root,
    loaded_package_dir,
    require_code_location,
)
from twinlink_ml.utils.tracking import environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON = REPO_ROOT / "scripts" / "verify" / "_common.py"


def _foreign_root(tmp_path: Path) -> Path:
    """A directory that looks like another repository/worktree (``.git`` marker)."""
    root = tmp_path / "other_worktree"
    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    return root


def test_loaded_package_is_this_repositorys_src() -> None:
    assert loaded_package_dir().samefile(REPO_ROOT / "src" / "twinlink_ml")


def test_fields_are_relative_and_name_the_interpreter() -> None:
    fields = require_code_location(REPO_ROOT)
    assert fields["code/package_dir"] == "src/twinlink_ml"
    assert fields["code/root_checked"] is True
    for key in ("code/package_dir", "code/interpreter"):
        assert not fields[key].startswith(("//", "\\\\")), "no network share in a record"
        assert ":" not in fields[key].split("/")[0], "no drive letter in a record"
    assert fields["code/interpreter"]


def test_other_spelling_of_the_same_root_is_accepted() -> None:
    assert require_code_location(REPO_ROOT / "src" / "..")["code/root_checked"] is True


def test_foreign_root_aborts_with_a_clear_message(tmp_path: Path) -> None:
    with pytest.raises(CodeLocationError, match="PYTHONPATH="):
        require_code_location(_foreign_root(tmp_path))


def test_no_root_skips_the_check_and_says_so() -> None:
    fields = require_code_location(None)
    assert fields["code/root_checked"] is False
    assert code_location_fields(None)["code/root_checked"] is False


def test_find_repo_root_accepts_directory_and_worktree_style_git_marker(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout" / "deep" / "dir"
    checkout.mkdir(parents=True)
    (tmp_path / "checkout" / ".git").mkdir()
    assert find_repo_root(checkout) == (tmp_path / "checkout").resolve()
    worktree = tmp_path / "wt" / "sub"
    worktree.mkdir(parents=True)
    (tmp_path / "wt" / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    assert find_repo_root(worktree) == (tmp_path / "wt").resolve()


def test_environment_fingerprint_records_the_location() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fingerprint = environment_fingerprint()
    assert fingerprint["code/package_dir"] == "src/twinlink_ml"
    assert fingerprint["code/root_checked"] is True
    assert "code/interpreter" in fingerprint


def test_environment_fingerprint_aborts_when_started_from_another_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_foreign_root(tmp_path))
    with pytest.raises(CodeLocationError):
        environment_fingerprint()
    # explicit opt-out still writes the fields (non-artifact callers only)
    fields = environment_fingerprint(check_code_location=False)
    assert fields["code/root_checked"] is True


def _common() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_common_location_test", COMMON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_provenance_records_the_location() -> None:
    block = _common().provenance()
    assert block["code/package_dir"] == "src/twinlink_ml"
    assert block["code/root_checked"] is True
    assert block["code/interpreter"]


def test_script_provenance_aborts_if_the_script_belongs_to_another_worktree(
    tmp_path: Path,
) -> None:
    common = _common()
    common.REPO_ROOT = _foreign_root(tmp_path)  # script lives in a worktree that is not the code's
    with pytest.raises(CodeLocationError):
        common.provenance()


def test_end_to_end_child_process_started_from_a_foreign_worktree(tmp_path: Path) -> None:
    """The real situation: loaded code is this repo's, the run belongs to another worktree."""
    foreign = _foreign_root(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from twinlink_ml.utils.tracking import environment_fingerprint;"
            "environment_fingerprint()",
        ],
        cwd=foreign,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode != 0
    assert "CodeLocationError" in result.stderr
    assert "different code tree" in result.stderr
