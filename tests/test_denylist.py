"""Checks ``local/denylist.txt`` against all git-tracked files.

Prevents real partner/person names (see ``local.example/README.md``) from being
accidentally committed again. ``local/`` exists only locally with the maintainer
of the repo and is gitignored. Every worktree gets a copy of ``local/denylist.txt`` at
creation; the test FAILS (it does not skip) when the list is missing, because a silent
skip would let a commit count as checked that was not.

Search order for ``denylist.txt``: ``$TWINLINK_LOCAL_DIR`` (as with the partner tests),
``<Repo>/local``, ``local/`` of the main checkout (in a git worktree ``local/`` lies
only there). If none of the three locations has the file, the test fails.

Lines with the prefix ``re:`` are regular expressions (case-sensitive, applied per
line, e.g. ``re:(?<![A-Za-z0-9])XYZ(?![A-Za-z0-9])`` for an abbreviation with
word boundaries). They do not apply to ``artifacts/`` (recorded text of runs).
All other lines are, as before, substrings matched case-insensitively.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REGEX_PREFIX = "re:"


def _repo_root() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out.stdout.strip())


def _main_checkout(repo_root: Path) -> Path | None:
    """First entry of ``git worktree list`` (the main checkout), if determinable."""
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    first = out.stdout.splitlines()[0] if out.stdout else ""
    return Path(first.removeprefix("worktree ")) if first.startswith("worktree ") else None


def _find_denylist(repo_root: Path) -> Path | None:
    candidates = []
    env = os.environ.get("TWINLINK_LOCAL_DIR")
    if env:
        candidates.append(Path(env) / "denylist.txt")
    candidates.append(repo_root / "local" / "denylist.txt")
    main = _main_checkout(repo_root)
    if main is not None:
        candidates.append(main / "local" / "denylist.txt")
    return next((c for c in candidates if c.is_file()), None)


def _require_denylist(repo_root: Path) -> Path:
    """Path of ``denylist.txt``; fails the test (never skips) when it cannot be found."""
    path = _find_denylist(repo_root)
    if path is None:
        pytest.fail(
            "local/denylist.txt fehlt: the denylist check cannot run. Copy local/denylist.txt "
            "into this worktree (or set TWINLINK_LOCAL_DIR); a missing list is a failure, "
            "not a skip."
        )
    return path


def _denylist_terms(denylist_path: Path) -> list[str]:
    terms = []
    for line in denylist_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            terms.append(stripped)
    return terms


def _tracked_files(repo_root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    return [repo_root / name for name in out.stdout.split("\0") if name]


def test_denylist_terms_not_present_in_tracked_files() -> None:
    repo_root = _repo_root()
    if repo_root is None:
        pytest.skip("kein Git-Repository (git rev-parse --show-toplevel schlug fehl)")
    denylist_path = _require_denylist(repo_root)

    terms = _denylist_terms(denylist_path)
    if not terms:
        pytest.skip("local/denylist.txt ist leer")

    plain_terms = [t for t in terms if not t.startswith(REGEX_PREFIX)]
    regex_terms = [
        (t, re.compile(t.removeprefix(REGEX_PREFIX))) for t in terms if t.startswith(REGEX_PREFIX)
    ]
    hits: list[str] = []
    for file_path in _tracked_files(repo_root):
        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary file or not readable -- no text content to check
        rel = file_path.relative_to(repo_root)
        lines = text.splitlines()
        lower_lines = [line.lower() for line in lines]
        for term in plain_terms:
            term_lower = term.lower()
            for line_no, line in enumerate(lower_lines, start=1):
                if term_lower in line:
                    hits.append(f"{rel}:{line_no}: enthaelt Sperrlisten-Begriff {term!r}")
        if rel.parts[0] == "artifacts":
            continue
        for term, pattern in regex_terms:
            for line_no, line in enumerate(lines, start=1):
                if pattern.search(line):
                    hits.append(f"{rel}:{line_no}: enthaelt Sperrlisten-Begriff {term!r}")

    assert not hits, "Sperrlisten-Treffer in getrackten Dateien:\n" + "\n".join(hits)


def test_missing_denylist_fails_instead_of_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository without any ``denylist.txt`` must make the check fail, not skip."""
    monkeypatch.setenv("TWINLINK_LOCAL_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(pytest.fail.Exception, match="denylist.txt"):
        _require_denylist(tmp_path)


def test_found_denylist_is_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "denylist.txt").write_text("term\n", encoding="utf-8")
    monkeypatch.setenv("TWINLINK_LOCAL_DIR", str(tmp_path))
    assert _require_denylist(tmp_path) == tmp_path / "denylist.txt"
