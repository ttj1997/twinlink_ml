"""Experiment tracking from day 1 (work plan step 5).

Why a separate module and not MLflow directly
-----------------------------------------------
MLflow is not a runtime dependency of the package, and a learning-curve run should also work
without a tracking server. The ABC :class:`ExperimentLogger` allows both: the
JSONL log runs everywhere and can be versioned, the MLflow adapter is loaded on demand when
it is available. The same separation as for the backend and the simulator.

What must go into every record
--------------------------------------
Without these six fingerprints a learning-curve point cannot be reconstructed, and one
only notices months later:

* ``process/fingerprint`` -- process configuration including all placeholders
* ``excitation/fingerprint`` -- excitation design; contains ``n_runs``, so it differs per
  pool size
* ``split/digest`` -- assignment *and* order of the training runs
* ``scaler/digest`` -- normalisation statistics; reveals whether they came from the
  wrong split
* ``seed/{data,noise,model}`` -- the three separable random sources
* ``git/commit`` and the package versions
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .code_location import code_location_fields, find_repo_root, require_code_location

__all__ = [
    "DIFF_DIGEST_UNAVAILABLE",
    "DiffDigestUnavailableWarning",
    "DirtyWorkingTreeWarning",
    "ExperimentLogger",
    "JsonlExperimentLogger",
    "MissingGitCommitError",
    "MlflowExperimentLogger",
    "MultiExperimentLogger",
    "RunRecord",
    "TrainingCurves",
    "environment_fingerprint",
    "read_jsonl",
    "status_is_dirty",
    "training_curves",
]

_GIT_CWD = Path(__file__).resolve().parent
"""``cwd`` für jeden ``git``-Subprozessaufruf in diesem Modul UND Basisverzeichnis für die
Pfade, die ``git ls-files --others`` relativ dazu zurückgibt (s. :func:`_git_diff_digest`)
-- eine einzige Stelle statt fünffach wiederholtem ``Path(__file__).resolve().parent``,
und ein einziger Punkt, den ein Test umleiten kann (``monkeypatch.setattr(tracking,
"_GIT_CWD", ...)``), damit Subprozessaufruf und Dateizugriff auf dasselbe (ggf. auf ein
Test-Repository umgeleitete) Verzeichnis zeigen -- beides driftete sonst leicht
auseinander, wenn nur ``subprocess.run`` gemockt wird."""

DIFF_DIGEST_UNAVAILABLE = "unavailable"
"""Marker für ``git/diff_digest``, wenn der Arbeitsbaum dirty ist, der Digest aber trotzdem
nicht ermittelt werden konnte (s. :class:`DiffDigestUnavailableWarning`).

Ein echter Digest ist ein 16-stelliger Hex-String; dieser Wert liegt garantiert außerhalb
dieses Wertebereichs und ist so von ihm unterscheidbar. NICHT für Gleichheitsvergleiche
zwischen Läufen verwenden, um "gleicher Codestand" zu prüfen: zwei Records mit
``git/diff_digest == DIFF_DIGEST_UNAVAILABLE`` haben nicht denselben (unbekannten) Diff,
sondern beide gar keinen ermittelten Diff -- das darf nicht als Übereinstimmung gewertet
werden. Vor einem Codestand-Vergleich immer zuerst ``git/diff_digest_available`` prüfen.
"""


class DirtyWorkingTreeWarning(UserWarning):
    """``git status --porcelain`` is not empty when a record is created.

    A separate class instead of ``UserWarning`` directly, so that it can be filtered
    specifically -- ``pyproject.toml`` marks it as expected (the same rationale as
    ``PlaceholderConfigWarning``: a dirty working tree is the rule during normal iteration,
    not an error case, and must not block the test suite, see the docstring of
    :func:`environment_fingerprint`).
    """


class MissingGitCommitError(RuntimeError):
    """A record with a provenance gap was about to be written.

    Befund 33 (``docs/04_stand_ap1.md``): ``git/commit`` silently stood at ``"unknown"`` in
    every run recorded so far -- without a warning, without a log entry at the
    call site. For ``arch_composite.jsonl`` (440 records) this was only noticed weeks later,
    when the configuration behind the numbers could no longer be reconstructed. This
    exception turns the absence into a default error instead of a silent string; a
    caller who wants to accept this deliberately (e.g. a Docker image without ``.git``)
    sets ``allow_missing_git=True`` on the logger.

    G4a follow-up fix: the same exception now also covers the related gap that a
    dirty working tree has no determinable ``git/diff_digest`` (see
    :func:`_check_git_provenance`) -- the same silent-gap error class as for the commit,
    checked under the same ``allow_missing_git`` switch. ``allow_missing_git`` and the
    class name are thus named somewhat more narrowly than what they cover in substance (they
    are named after the commit, not after provenance as a whole) -- deliberately not renamed,
    because that affects existing callers; that is a separate decision.
    """


class DiffDigestUnavailableWarning(UserWarning):
    """``git/dirty`` is true, but ``git diff HEAD`` could not be turned into a digest.

    A separate class instead of ``DirtyWorkingTreeWarning`` directly: a dirty working tree alone
    is the rule (see there), but additionally being unable to determine a digest for it is the
    silent failure path that, before ``encoding="utf-8"``/``errors="replace"`` in
    :func:`_git_diff_digest`, made ten tests in ``tests/test_protocol.py`` fail under Windows
    (``stdout`` stayed ``None`` without an exception reaching the caller). This
    warning remains as a safeguard for the remaining, genuine error case (no ``git`` on
    the PATH, no repository, ``git diff`` or ``git ls-files --others`` itself fails
    -- the latter since the extension to untracked files, see :func:`_git_diff_digest`).
    """


def _git_commit() -> str | None:
    """Commit hash of the working tree, ``None`` outside a repository.

    ``encoding="utf-8"``/``errors="replace"``: the same Windows trap as in
    :func:`_git_diff_digest` (``text=True`` without ``encoding`` falls back to the locale code page
    of the executing system) applies in principle to every ``subprocess.run`` call
    here -- even if a commit hash is in practice purely hexadecimal and never contains
    undecodable bytes, the explicit setting makes the function correct independently of that
    instead of by chance.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=_GIT_CWD,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_repo_root() -> Path | None:
    """Root directory of the working tree, ``None`` outside a repository.

    For :func:`_relativize_path_params` -- absolute checkpoint paths are to be logged relative
    to this root, not as an absolute path with user name/
    network drive (confidentiality audit 2026-09-25).
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=_GIT_CWD,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(completed.stdout.strip())


def status_is_dirty(
    porcelain: str, ignore_paths: Iterable[Path | str] = (), repo_root: Path | None = None
) -> bool:
    """Whether ``git status --porcelain`` shows changes other than exempted UNTRACKED files.

    ``ignore_paths`` (absolute, or relative to ``repo_root``) exempt exactly those files **if they
    are untracked** (status ``??``): the output file a run is about to write into (Befund 87,
    Addendum 3: a driver's own log made ``git/dirty`` true). A modified, staged or deleted tracked
    file is never exempt, and neither is any other untracked file. The caller records the exempted
    paths next to the flag (``git/dirty_ignored_paths``) so that the exemption is visible.
    """
    ignored = set()
    for raw in ignore_paths:
        path = Path(raw)
        if path.is_absolute() and repo_root is not None:
            try:
                path = path.resolve().relative_to(repo_root.resolve())
            except ValueError:
                continue
        ignored.add(path.as_posix())
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if line.startswith("??") and line[3:].strip().strip('"') in ignored:
            continue
        return True
    return False


def _git_dirty(ignore_paths: Iterable[Path | str] = ()) -> bool:
    """``True`` if the working tree has uncommitted changes.

    ``ignore_paths``: untracked files that do not count (see :func:`status_is_dirty`); with
    them ``git status`` lists untracked files one by one (``--untracked-files=all``) so that a file
    inside a new directory can be matched.

    ``False`` outside a repository or if ``git`` itself is not on the PATH
    (Befund 33 documents exactly this environment) -- ``git/dirty`` then answers
    no meaningful question, and ``git/commit_available=False`` already carries the actual
    information.

    ``encoding="utf-8"``/``errors="replace"``: see :func:`_git_diff_digest` -- ``git status
    --porcelain`` normally quotes unusual file names, but this does not rely
    on it.
    """
    ignore = list(ignore_paths)
    command = ["git", "status", "--porcelain"] + (["--untracked-files=all"] if ignore else [])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            cwd=_GIT_CWD,
        )
    except OSError:
        return False
    if not ignore:
        return bool(result.stdout.strip())
    return status_is_dirty(result.stdout, ignore, _git_repo_root())


_UNTRACKED_FILE_UNREADABLE = "<unreadable>"
"""Marker im Digest-Eintrag einer untracked Datei (s. :func:`_git_diff_digest`), wenn ihr
Inhalt nicht gelesen werden konnte -- z. B. zwischen ``git ls-files --others`` und dem
Lesen gelöscht, oder keine Leseberechtigung. Ersetzt den sonst dort stehenden
Inhalts-Hash, damit die Digest-Bildung insgesamt nicht abbricht, aber die Lücke im Wert
selbst sichtbar bleibt statt die Datei stillschweigend zu überspringen -- dieselbe
Fehlerklasse wie der Encoding-Fehlpfad, den :class:`DiffDigestUnavailableWarning`
dokumentiert: eine Provenienzfunktion darf bei einem Fehlpfad nicht leise einen harmlos
aussehenden Wert liefern."""


def _git_untracked_files() -> list[str] | None:
    """Paths of all non-ignored, untracked files.

    ``git ls-files --others --exclude-standard``, sorted deterministically by path by
    Git itself.

    ``--exclude-standard`` mirrors the same ``.gitignore`` semantics that ``git status
    --porcelain`` (see :func:`_git_dirty`) already applies anyway -- an ignored file
    does not make the tree dirty and should not appear here either.

    ``None`` on any error (no ``git`` on the PATH, no repository, non-zero
    exit) -- like :func:`_git_commit`/:func:`_git_dirty`, the same ``encoding="utf-8"``/
    ``errors="replace"`` rationale for the path names themselves (not for file contents, those
    are read in binary, see :func:`_git_diff_digest`).
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=_GIT_CWD,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in result.stdout.splitlines() if line]


def _git_diff_digest() -> str | None:
    """Makes a dirty state identifiable.

    sha256 (16 hex characters) over ``git diff HEAD`` AND the content of all untracked files.

    ``git diff HEAD`` alone only sees tracked changes (including staged new
    files) -- a completely new file that was never ``git add``ed does NOT appear in it, not even
    as a "new file" hunk. A dirty state consisting exclusively of such a
    file therefore previously returned the digest of the EMPTY diff
    (``sha256("")[:16] == "e3b0c44298fc1c14"``) -- identifiable in name only, not in
    substance: of all cases, the most frequent one (a newly written, not yet
    committed script) thus stayed silent. This function closes the gap by
    additionally querying :func:`_git_untracked_files` and hashing, for every file listed there,
    its content -- not only its name, otherwise an uncommitted script CHANGED between two runs
    would return the same digest as long as the file name
    stays the same (exactly the case that triggered this extension).

    Combination via a structured, not a simple string concatenation:
    ``json.dumps({"diff": <sha256 von git diff HEAD>, "untracked": [[pfad, sha256(inhalt)],
    ...]}, sort_keys=True)``, then sha256 over that. A simple concatenation
    (path+content joined together) would be ambiguous -- path "a"+content "bc" would give the same
    bytes as path "ab"+content "c". Every file is first reduced to its own 64-character
    hash before it enters the outer structure, so that a single large
    untracked file does not inflate the outer hash input. The ``untracked`` list is
    additionally sorted explicitly (not just relying on Git's own, in itself already deterministic
    sorting of ``ls-files``) -- reproducibility should follow from this function
    itself, not from a property of a call it merely consumes.

    If an untracked file cannot be read (race with ``git ls-files --others``,
    missing permission), :data:`_UNTRACKED_FILE_UNREADABLE` replaces its content hash --
    the digest computation does not abort because of this, but the gap stays visible in the value
    itself (no silent fallback to a harmless-looking value).

    Untracked files are read in BINARY, not as decoded text: for the
    tracked diff ``errors="replace"`` is the right choice (see below), for
    file contents it would be a loss of information -- two binary files with different content
    must not be mapped to the same hash through identical replacement characters.

    **Line endings between Windows and Linux:** this repository sets
    ``.gitattributes: * text=auto eol=lf`` -- tracked files are normalised to LF by Git on
    checkout/diff independently of ``core.autocrlf``, so their contribution to the
    digest is stable across platforms. This does NOT hold for UNTRACKED files:
    Git applies neither ``.gitattributes`` nor ``core.autocrlf`` to files it does not
    manage. The same uncommitted file can carry different bytes (CRLF vs. LF) on a Windows and a
    Linux machine, and thus a different
    digest, even if in substance it is "the same code". This is not a bug of this
    function -- the digest is meant to capture the ACTUAL state, and different
    bytes are an actually different state --, but it belongs here so that
    a later comparison of two digests from different machines is not wrongly read as
    "different code" where it is really only different
    line endings of the same, not yet committed file.

    ``None`` outside a repository or if ``git diff``/``git ls-files`` itself
    fails (not on the PATH, no repository, non-zero exit).
    :func:`environment_fingerprint` translates a dirty-but-``None`` into an explicit
    marker instead of letting it silently look like "no diff" (see there) -- this function
    itself stays at ``None``.

    **Not comparable across this change:** before this extension the
    digest was ``sha256(git diff HEAD)[:16]`` (raw diff text). Now it is
    ``sha256(json.dumps({"diff": sha256(git diff HEAD), "untracked": [...]}, sort_keys=True))[:16]``
    -- a different input structure, not just an extension of the same one. A ``git/diff_digest``
    computed BEFORE this commit and one AFTER it are therefore different even with an
    identical tree state (even with no untracked files at all) -- the same
    diff text now produces a different hash, because it is first hashed itself and then
    embedded in a JSON structure, instead of being hashed directly. A comparison
    ``digest_alt == digest_neu`` "same code state?" is therefore meaningless across this commit
    -- the same class of drift as with the ``ProcessConfig`` fingerprint (whose digest
    changes with every extension by a field, even if the new value equals the old
    default). Within a single digest procedure (before OR after this
    commit, not mixed) the comparison remains valid.

    ``encoding="utf-8"``: a dirty diff here routinely contains UTF-8 special characters
    (German documentation, Greek formula symbols, superscript digits). Without
    explicit ``encoding``, ``subprocess.run(text=True)`` falls back to the locale code page of the
    executing system (on Windows mostly cp1252), which cannot decode such bytes.
    The error happens in an internal reader thread of ``subprocess`` and
    is NOT passed on to the caller as an exception -- ``stdout`` simply stays
    ``None`` without the ``except`` block below ever taking effect: exactly the error class
    against which this function was built undermines it through its own decoding.

    ``errors="replace"``: even after choosing UTF-8, a diff can contain undecodable bytes
    (e.g. binary files in the diff). The digest should still return a value for it
    instead of failing on individual bytes -- it is then no longer byte-identical
    to the raw diff (replacement character U+FFFD instead of the original bytes at unreadable
    positions), but stays stable and reproducible over repeated calls on the same state, which
    suffices for the purpose here (making a dirty state identifiable, not reconstructing the
    diff itself). For untracked files the same question does not arise --
    they are read in binary (see above).
    """
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=_GIT_CWD,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    diff_digest = hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest()

    untracked_paths = _git_untracked_files()
    if untracked_paths is None:
        return None
    untracked_digests: list[list[str]] = []
    for rel_path in untracked_paths:
        try:
            content = (_GIT_CWD / rel_path).read_bytes()
        except OSError:
            untracked_digests.append([rel_path, _UNTRACKED_FILE_UNREADABLE])
        else:
            untracked_digests.append([rel_path, hashlib.sha256(content).hexdigest()])
    untracked_digests.sort()

    payload = json.dumps({"diff": diff_digest, "untracked": untracked_digests}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _torch_num_threads() -> int | None:
    """``torch.get_num_threads()``, ``None`` if torch is not installed.

    A separate function (not inline in :func:`environment_fingerprint`), so that tests can
    simulate the torch-free case without faking the real import -- ``torch``
    thus remains an optional import for this otherwise torch-free module, the same
    condition as for the ``version/torch`` field further below.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch.get_num_threads()


def environment_fingerprint(
    *, check_code_location: bool = True, ignore_paths: Iterable[Path | str] = ()
) -> dict[str, Any]:
    """Environment, as far as it can influence the result.

    ``git/commit`` is ``None`` (no longer the string ``"unknown"``) if no
    repository is found; ``git/commit_available`` makes this explicit instead of silently
    hiding it in the value (Befund 33 -- see :class:`MissingGitCommitError`). A dirty
    working tree (``git/dirty=True``) is not reproducible even with a valid commit hash
    -- the commit does not fully describe the state; the current
    session demonstrates this (``xlstm.py``, ``test_composite.py`` uncommitted).
    ``git/diff_digest`` makes even a dirty state identifiable afterwards. A
    dirty working tree is only a loud warning, not an error -- otherwise iteration
    between runs would be impossible. If the working tree is dirty but the digest NEVERTHELESS
    cannot be determined (no ``git`` on the PATH, no repository, ``git diff`` fails),
    this does NOT silently become ``None`` -- that would no longer be distinguishable from a clean
    state (``git/dirty=False``, ``git/diff_digest=None`` likewise), exactly the
    error class of Befund 33. Instead ``git/diff_digest`` gets the marker
    :data:`DIFF_DIGEST_UNAVAILABLE`, ``git/diff_digest_available=False`` makes it
    machine-evaluable (counterpart to ``git/commit_available``), and a
    :class:`DiffDigestUnavailableWarning` is raised. :func:`_check_git_provenance`
    checks this field under the same ``allow_missing_git`` switch as
    ``git/commit_available`` (see there).

    ``torch/num_threads``, ``env/mkl_num_threads``, ``env/omp_num_threads``: the occasion was the
    factor-5 spread in ``s/Epoche`` in ``befund42_tbed_exhaust_sweep.jsonl``, which without
    these fields could only be diagnosed via the temporal clusters of the ``created_unix_s`` values
    (threading/process oversubscription, not convergence -- see the analysis of it). The two
    env-var fields are ``None`` if the variable is not set -- that itself is the
    information (no pinning active), not to be confused with an empty string.

    ``code/package_dir``, ``code/interpreter``, ``code/root_checked`` (see
    :mod:`twinlink_ml.utils.code_location`): where ``twinlink_ml`` was actually loaded from and
    with which interpreter. If the loaded package does not lie under the root of the repository
    or worktree from which the run was started (working directory), the function aborts
    with :class:`~twinlink_ml.utils.code_location.CodeLocationError` before any
    record is created. ``check_code_location=False`` only for callers that explicitly do not
    log an artifact run (the fields are written anyway).
    """
    location = (
        require_code_location(find_repo_root())
        if check_code_location
        else code_location_fields(find_repo_root())
    )
    from ..evaluation.experiments import NEW_STUDY_PRESET_VERSION

    ignored = [str(p) for p in ignore_paths]
    commit = _git_commit()
    dirty = _git_dirty(ignored)
    if dirty:
        warnings.warn(
            "Arbeitsbaum ist dirty (git status --porcelain nicht leer) -- der geloggte "
            "git/commit beschreibt den Stand nicht vollständig reproduzierbar, siehe "
            "git/diff_digest.",
            DirtyWorkingTreeWarning,
            stacklevel=2,
        )
    diff_digest: str | None = None
    diff_digest_available = True
    if dirty:
        diff_digest = _git_diff_digest()
        diff_digest_available = diff_digest is not None
        if not diff_digest_available:
            warnings.warn(
                "Arbeitsbaum ist dirty, aber git/diff_digest ließ sich nicht ermitteln (kein "
                "git auf dem PATH, kein Repository, oder 'git diff HEAD' schlug fehl) -- der "
                "dirty Stand ist damit NICHT nachträglich identifizierbar. git/diff_digest "
                f"wird auf den Marker {DIFF_DIGEST_UNAVAILABLE!r} gesetzt statt still auf "
                "None (nicht von einem sauberen Stand unterscheidbar).",
                DiffDigestUnavailableWarning,
                stacklevel=2,
            )
            diff_digest = DIFF_DIGEST_UNAVAILABLE
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git/commit": commit,
        "git/commit_available": commit is not None,
        "git/dirty": dirty,
        "git/diff_digest": diff_digest,
        "git/diff_digest_available": diff_digest_available,
        "torch/num_threads": _torch_num_threads(),
        "env/mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "env/omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        **location,
        "preset/new_study_version": NEW_STUDY_PRESET_VERSION,
    }
    if ignored:
        # only when an exemption was used: the record says which untracked files did not count
        versions["git/dirty_ignored_paths"] = sorted(ignored)
    for name in ("numpy", "scipy", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[f"version/{name}"] = str(getattr(module, "__version__", "unknown"))
    return versions


def _check_git_provenance(record: RunRecord, *, allow_missing_git: bool) -> None:
    """Refuses to write a record with a provenance gap.

    Two gaps are checked, both under the same ``allow_missing_git`` switch (see
    :class:`MissingGitCommitError` for the reason why name and switch are still named
    as originally, only after the first case):

    * ``git/commit_available`` comes from :func:`environment_fingerprint`. If the
      key is missing (a caller that does not use ``environment_fingerprint()``), this is
      treated as not available (default ``False``) -- better a caller sets
      ``allow_missing_git`` deliberately than a record like those in ``arch_composite.jsonl``
      (``git/commit="unknown"`` in 440 records) becomes untraceable months later.
    * ``git/diff_digest_available`` also comes from :func:`environment_fingerprint`,
      but is only checked if the record itself carries ``git/dirty=True`` -- a record
      without a ``git/dirty`` entry (older caller that never knew the field) should thereby
      not retroactively count as incomplete where there was no expectation before; if
      the key is missing with ``git/dirty=True``, it counts as available (default ``True``), since
      ``environment_fingerprint()`` now always sets the field and its absence here rather
      indicates a foreign caller than a genuine error case.
    """
    commit_available = bool(record.params.get("git/commit_available", False))
    dirty = bool(record.params.get("git/dirty", False))
    diff_digest_available = bool(record.params.get("git/diff_digest_available", True))
    missing_commit = not commit_available
    missing_diff_digest = dirty and not diff_digest_available
    if not missing_commit and not missing_diff_digest:
        return

    reasons = []
    if missing_commit:
        reasons.append("keinen ermittelbaren git/commit")
    if missing_diff_digest:
        reasons.append("trotz dirty Arbeitsbaum keinen ermittelbaren git/diff_digest")
    reason_text = " und ".join(reasons)

    if not allow_missing_git:
        raise MissingGitCommitError(
            f"Record {record.run_uid!r} (group={record.group!r}) hat {reason_text} -- "
            "geschrieben außerhalb eines Git-Repositories, ohne environment_fingerprint() "
            "oder git selbst schlug fehl. Wenn das bewusst in Kauf genommen wird, den Logger "
            "mit allow_missing_git=True konstruieren (der Record trägt die betroffenen "
            "*_available-Felder dann weiterhin korrekt als False)."
        )
    warn_reasons = []
    if missing_commit:
        warn_reasons.append("ohne Git-Commit")
    if missing_diff_digest:
        warn_reasons.append("trotz dirty Arbeitsbaum ohne git/diff_digest")
    warnings.warn(
        f"Record {record.run_uid!r} (group={record.group!r}) wird "
        f"{' und '.join(warn_reasons)} geschrieben (allow_missing_git=True).",
        stacklevel=3,
    )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """An entry in the results log.

    Attributes
    ----------
    group
        Experiment name, for example ``"learning_curve"`` or ``"dt_study"``.
    params
        Everything that determines the run: fingerprints, seeds, window and
        model configuration. Must be JSON-serialisable.
    metrics
        Scalar results, flatly named (see
        :meth:`~twinlink_ml.evaluation.evaluator.EvaluationResult.as_flat_metrics`).
    tags
        Short labels for filtering, for example model and target name.
    """

    group: str
    params: Mapping[str, Any]
    metrics: Mapping[str, float]
    tags: Mapping[str, str] = field(default_factory=dict)
    run_uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_unix_s: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "run_uid": self.run_uid,
            "group": self.group,
            "created_unix_s": self.created_unix_s,
            "tags": dict(self.tags),
            "params": _jsonable(dict(self.params)),
            "metrics": {key: float(value) for key, value in self.metrics.items()},
        }


@dataclass(frozen=True, slots=True)
class TrainingCurves:
    """Loss history of one head, as stored in records written after VH1 (Befund 67)."""

    train: tuple[float, ...]
    validation: tuple[float, ...]
    best_epoch: int


def training_curves(params: Mapping[str, Any], head: str) -> TrainingCurves | None:
    """Loss history of ``head`` from a record's params; ``None`` for records without one.

    Records written before the history was stored (everything up to Befund 67, SX3) have no
    ``training/loss_history_*`` keys and stay valid; callers must handle ``None``.
    """
    train = params.get(f"training/loss_history_train/{head}")
    if train is None:
        return None
    return TrainingCurves(
        train=tuple(float(v) for v in train),
        validation=tuple(float(v) for v in params.get(f"training/loss_history_val/{head}", ())),
        best_epoch=int(params[f"training/best_epoch/{head}"]),
    )


def _relativize_path_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Makes absolute paths in ``*/path`` parameters relative to the repository root directory.

    Affects only records written in future (:meth:`JsonlExperimentLogger.log`) --
    existing lines in ``artifacts/runs/*.jsonl`` stay untouched. Deliberately calls
    :func:`_git_repo_root` (a ``subprocess.run``) only when an absolute ``*/path`` value
    is present at all -- the frequent case without such parameters thus stays
    without an additional process call. A path outside the repo root (e.g. a
    checkpoint on another drive) or no Git repository stays unchanged --
    better an absolute path than a wrong relative one.
    """
    result = dict(params)
    absolute_path_keys = [
        key
        for key, value in params.items()
        if key.endswith("/path") and isinstance(value, str) and Path(value).is_absolute()
    ]
    if not absolute_path_keys:
        return result
    repo_root = _git_repo_root()
    if repo_root is None:
        return result
    for key in absolute_path_keys:
        with contextlib.suppress(ValueError):
            result[key] = Path(result[key]).relative_to(repo_root).as_posix()
    return result


def _jsonable(value: Any) -> Any:
    """Recursive conversion into JSON-compatible types."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Iterable):
        return [_jsonable(item) for item in value]
    return str(value)


class ExperimentLogger(ABC):
    """Sink for result entries."""

    @abstractmethod
    def log(self, record: RunRecord) -> None:
        """Write one entry."""

    def close(self) -> None:  # noqa: B027 - deliberately not abstract
        """Release resources.

        The default implementation does nothing and is deliberately **not** abstract: a logger
        without resources should not have to write an empty method.
        """


@dataclass(slots=True)
class JsonlExperimentLogger(ExperimentLogger):
    """Appending JSON Lines log -- one line per run.

    Deliberately appending and not overwriting: a learning-curve run that aborts after 40 of 45
    points should keep the 40. The format is diffable line by line and can be evaluated
    without additional tooling.

    Attributes
    ----------
    allow_missing_git
        ``False`` (default): :meth:`log` raises :class:`MissingGitCommitError` if the
        record carries no determinable ``git/commit`` (see Befund 33). ``True``:
        the record is written anyway, with a loud warning -- for environments
        without a Git repository, set deliberately by the caller.
    """

    path: Path
    allow_missing_git: bool = False

    def log(self, record: RunRecord) -> None:
        """Append an entry."""
        _check_git_provenance(record, allow_missing_git=self.allow_missing_git)
        relativized_params = _relativize_path_params(record.params)
        record = replace(record, params=relativized_params)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


@dataclass(slots=True)
class MlflowExperimentLogger(ExperimentLogger):
    """Adapter to MLflow; imported only on the first write.

    If MLflow is missing, the construction fails and not only the 40th run.
    """

    experiment_name: str
    tracking_uri: str | None = None
    allow_missing_git: bool = False

    def __post_init__(self) -> None:
        try:
            import mlflow
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "MLflow ist nicht installiert. Entweder 'pip install twinlink-ml[train]' "
                "oder JsonlExperimentLogger verwenden."
            ) from error
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log(self, record: RunRecord) -> None:
        """Write the entry as an MLflow run."""
        _check_git_provenance(record, allow_missing_git=self.allow_missing_git)
        import mlflow

        with mlflow.start_run(run_name=record.run_uid):
            mlflow.set_tags(dict(record.tags))
            params = _jsonable(dict(record.params))
            # Loss histories are long lists (MLflow limits the length of a parameter value):
            # stored as a JSON artifact of the run instead of as parameters.
            curves = {k: params.pop(k) for k in list(params) if "/loss_history_" in k}
            mlflow.log_params(params)
            if curves:
                mlflow.log_dict(curves, "loss_history.json")
            mlflow.log_metrics({key: float(v) for key, v in record.metrics.items()})


@dataclass(slots=True)
class MultiExperimentLogger(ExperimentLogger):
    """Several sinks at once. The order is preserved."""

    loggers: tuple[ExperimentLogger, ...]

    def log(self, record: RunRecord) -> None:
        """Pass the entry on to all sinks."""
        for logger in self.loggers:
            logger.log(record)

    def close(self) -> None:
        """Close all sinks."""
        for logger in self.loggers:
            logger.close()


def read_jsonl(path: Path | str) -> list[RunRecord]:
    """Read a JSONL log as a :class:`RunRecord` list.

    So that a notebook does not have to unpack a JSON structure itself: evaluation and plots
    work on the same type as the drivers, and a change to the log format takes effect
    in one place instead of in every cell.
    """
    records: list[RunRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            RunRecord(
                group=str(payload.get("group", "")),
                params=dict(payload.get("params", {})),
                metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
                tags={str(k): str(v) for k, v in payload.get("tags", {}).items()},
                run_uid=str(payload.get("run_uid", "")),
                created_unix_s=float(payload.get("created_unix_s", 0.0)),
            )
        )
    return records
