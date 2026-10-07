"""Prueft ``local/denylist.txt`` gegen alle git-getrackten Dateien.

Verhindert, dass reale Partner-/Personennamen (siehe ``local.example/README.md``)
versehentlich wieder committet werden. ``local/`` existiert nur lokal beim Pflegenden
des Repos und ist gitignored -- in Cloud-Sitzungen fehlt es normalerweise, dann wird
dieser Test uebersprungen (er kann dort seine eigene Referenz nicht lesen).

Suchreihenfolge fuer ``denylist.txt``: ``$TWINLINK_LOCAL_DIR`` (wie bei den Partner-Tests),
``<Repo>/local``, ``local/`` des Haupt-Checkouts (in einem Git-Worktree liegt ``local/``
nur dort). Erst wenn keine der drei Stellen die Datei hat, wird uebersprungen.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


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
    """Erster Eintrag von ``git worktree list`` (der Haupt-Checkout), falls ermittelbar."""
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
    denylist_path = _find_denylist(repo_root)
    if denylist_path is None:
        pytest.skip("local/denylist.txt fehlt (Normalfall in Cloud-Sitzungen)")

    terms = _denylist_terms(denylist_path)
    if not terms:
        pytest.skip("local/denylist.txt ist leer")

    hits: list[str] = []
    for file_path in _tracked_files(repo_root):
        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # Binaerdatei oder nicht lesbar -- kein Textinhalt zu pruefen
        lower_lines = [line.lower() for line in text.splitlines()]
        for term in terms:
            term_lower = term.lower()
            for line_no, line in enumerate(lower_lines, start=1):
                if term_lower in line:
                    rel = file_path.relative_to(repo_root)
                    hits.append(f"{rel}:{line_no}: enthaelt Sperrlisten-Begriff {term!r}")

    assert not hits, "Sperrlisten-Treffer in getrackten Dateien:\n" + "\n".join(hits)
