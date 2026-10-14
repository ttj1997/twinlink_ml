"""The run's own output file must not make ``git/dirty`` true (Befund NN, Addendum 3 of study B).

``ignore_paths`` exempts exactly the named UNTRACKED files; a modified tracked file or any other
untracked file still counts; the exemption is recorded next to the flag.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

from twinlink_ml.utils import tracking
from twinlink_ml.utils.tracking import environment_fingerprint, status_is_dirty

COMMON = Path(__file__).resolve().parents[1] / "scripts" / "verify" / "_common.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("a\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(tracking, "_GIT_CWD", root)
    return root


def test_status_parsing() -> None:
    porcelain = "?? artifacts/runs/log.jsonl\n"
    assert status_is_dirty(porcelain)
    assert not status_is_dirty(porcelain, ["artifacts/runs/log.jsonl"])
    assert status_is_dirty(porcelain, ["artifacts/runs/other.jsonl"])
    assert status_is_dirty(" M artifacts/runs/log.jsonl\n", ["artifacts/runs/log.jsonl"])
    assert status_is_dirty("A  artifacts/runs/log.jsonl\n", ["artifacts/runs/log.jsonl"])
    assert status_is_dirty("?? a.txt\n?? b.txt\n", ["a.txt"])
    assert not status_is_dirty("", [])
    assert not status_is_dirty('?? "dir/log file.jsonl"\n', ["dir/log file.jsonl"])


def test_own_log_does_not_count_but_everything_else_does(repo: Path) -> None:
    log = repo / "artifacts" / "runs" / "driver.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("{}\n", encoding="utf-8")
    assert tracking._git_dirty() is True
    assert tracking._git_dirty([log]) is False  # absolute path
    assert tracking._git_dirty(["artifacts/runs/driver.jsonl"]) is False  # relative path
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    assert tracking._git_dirty([log]) is True  # another untracked file still counts
    (repo / "stray.txt").unlink()
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert tracking._git_dirty([log]) is True  # a modified tracked file always counts


def test_a_tracked_file_cannot_be_exempted(repo: Path) -> None:
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert tracking._git_dirty(["tracked.txt"]) is True


def test_environment_fingerprint_records_the_exemption(repo: Path) -> None:
    log = repo / "run.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = environment_fingerprint(check_code_location=False)
        exempt = environment_fingerprint(check_code_location=False, ignore_paths=[log])
    assert plain["git/dirty"] is True and "git/dirty_ignored_paths" not in plain
    assert exempt["git/dirty"] is False
    assert exempt["git/dirty_ignored_paths"] == [str(log)]


def _common() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_common_dirty_test", COMMON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_helpers_follow_the_same_rule(repo: Path) -> None:
    common = _common()
    common.REPO_ROOT = repo
    log = repo / "out.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    assert common.working_tree_dirty() is True
    assert common.working_tree_dirty([log]) is False
    with pytest.raises(SystemExit):
        common.require_clean_tree()
    common.require_clean_tree([log])  # does not raise
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        common.require_clean_tree([log])
