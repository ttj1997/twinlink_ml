r"""Provenance of the code location of ``twinlink_ml`` (which checkout is loaded).

Why this exists
---------------
``pip install -e .`` points the interpreter at ONE directory. A second checkout or a git
worktree runs that other code unless ``PYTHONPATH`` says otherwise, and the record would still
carry the commit of the directory the script was started in. Seen on this machine: the
system interpreter's editable install pointed at an older, unrelated checkout (no
``heatnetwork/site_network.py``), while a script in a worktree named its own commit.
Nothing in a record told the two apart.

What a run records and enforces
-------------------------------
* ``code/package_dir`` -- directory of the loaded package, **relative to the repository or
  worktree root** (``src/twinlink_ml`` when it is the expected one; no absolute paths, no
  user names or network shares in a committed artifact).
* ``code/interpreter`` -- the Python executable, relative to the root if inside it (for
  example ``.venv/Scripts/python.exe``); a venv outside the root as ``<venv dir>/<path in it>``;
  otherwise with the home directory shortened to ``~``.
* ``code/root_checked`` -- whether a root was available to check against.

If the loaded package does not lie under the root of the repository (or worktree) that the run
is started from, :func:`require_code_location` raises :class:`CodeLocationError` and nothing is
written. There is deliberately no switch to downgrade this to a warning for artifact runs; the
way out is to start the run so that it loads the right code (``CLAUDE.md``, section
"Provenance": start from the worktree root with ``PYTHONPATH=<worktree>/src``).

Directories are compared with :meth:`pathlib.Path.samefile`, not as strings: the same checkout is
reachable as a drive letter and as a UNC path on this machine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

__all__ = [
    "CodeLocationError",
    "code_location_fields",
    "find_repo_root",
    "loaded_package_dir",
    "require_code_location",
]


class CodeLocationError(RuntimeError):
    """The loaded ``twinlink_ml`` is not the code of the repository the run belongs to."""


def loaded_package_dir() -> Path:
    """Directory of the ``twinlink_ml`` package that is actually imported."""
    import twinlink_ml

    return Path(twinlink_ml.__file__).resolve().parent


def find_repo_root(start: Path | None = None) -> Path | None:
    """Nearest ancestor of ``start`` (default: the working directory) that holds ``.git``.

    ``.git`` is a directory in a checkout and a file in a worktree; both count. Purely a
    filesystem lookup (no subprocess), so it cannot be disturbed by tests that mock ``git``.
    """
    here = (start if start is not None else Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _same(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def _display(path: Path, root: Path | None) -> str:
    """Path for a record: relative to ``root`` if inside it, else home-shortened."""
    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    home = Path.home().resolve()
    try:
        return "~/" + resolved.relative_to(home).as_posix()
    except ValueError:
        return resolved.as_posix()


def _interpreter(root: Path | None) -> str:
    # not resolve(): a venv's python.exe is often a launcher that resolves to the base
    # interpreter, which would hide that a venv was used
    executable = Path(sys.executable)
    absolute = executable.absolute()
    if root is not None:
        for base in (root, root.resolve()):
            try:
                return absolute.relative_to(base).as_posix()
            except ValueError:
                continue
        # the same directory under another spelling (drive letter vs UNC)
        for parent in absolute.parents:
            if _same(parent, root):
                return absolute.relative_to(parent).as_posix()
    # a venv outside the root (the main checkout's .venv used from a worktree): name the venv
    # directory and the path inside it, not the location of the checkout (network share)
    for parent in absolute.parents:
        if (parent / "pyvenv.cfg").exists():
            return f"{parent.name}/{absolute.relative_to(parent).as_posix()}"
    home = Path.home()
    try:
        return "~/" + absolute.relative_to(home).as_posix()
    except ValueError:
        return absolute.as_posix()


def code_location_fields(root: Path | None) -> dict[str, Any]:
    """Provenance fields; ``root`` is the repository/worktree the run belongs to (or ``None``)."""
    package = loaded_package_dir()
    return {
        "code/package_dir": _display(package, root),
        "code/interpreter": _interpreter(root),
        "code/root_checked": root is not None,
    }


def require_code_location(root: Path | None) -> dict[str, Any]:
    """Return :func:`code_location_fields`; raise if the package is not under ``root``.

    ``root=None`` (no repository found) skips the check and says so in ``code/root_checked``.
    The package must be ``<root>/src/twinlink_ml``.
    """
    package = loaded_package_dir()
    if root is not None and not _same(package.parent.parent, root):
        raise CodeLocationError(
            "twinlink_ml is loaded from a different code tree than the one this run belongs "
            "to, so a logged commit would not describe the code that ran.\n"
            f"  loaded package : {package}\n"
            f"  expected under : {root}  (the repository or worktree of the run)\n"
            f"  interpreter    : {sys.executable}\n"
            "Start the run from the root of that repository/worktree with "
            f"PYTHONPATH={root / 'src'} (or use an environment whose editable install "
            "points there), see CLAUDE.md, section Provenance."
        )
    return code_location_fields(root)
