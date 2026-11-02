"""Tests for :mod:`twinlink_ml.utils.tracking` (Befund 33, G4a).

``subprocess.run`` is mocked so that the tests do not depend on the actual git state
of this working tree -- a clean checkout in CI and a dirty
development state must yield the same results.
"""

from __future__ import annotations

import locale
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from twinlink_ml.utils import tracking
from twinlink_ml.utils.tracking import (
    DIFF_DIGEST_UNAVAILABLE,
    DiffDigestUnavailableWarning,
    DirtyWorkingTreeWarning,
    JsonlExperimentLogger,
    MissingGitCommitError,
    RunRecord,
    _git_diff_digest,
    environment_fingerprint,
)

HAS_GIT = shutil.which("git") is not None


def _completed(stdout: str, *, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _run_side_effect(
    clean: bool,
    *,
    diff_text: str = "diff --git a/x b/x\n",
    untracked_text: str = "",
    repo_root: str | None = None,
):
    """Replaces ``subprocess.run`` for ``git rev-parse``/``status``/``diff``/``ls-files``.

    ``untracked_text`` (default empty -- no untracked files) simulates the output of
    ``git ls-files --others --exclude-standard`` (see :func:`_git_untracked_files`); a
    path listed in it will NOT really exist when actually read in :func:`_git_diff_digest`
    (this module mocks only ``subprocess.run``, not the file system), and thus falls
    back to ``_UNTRACKED_FILE_UNREADABLE`` -- irrelevant for the tests that use this
    function, because they do not check the concrete digest value, only its
    existence/length.

    ``repo_root`` (default ``None``) is the answer to ``git rev-parse --show-toplevel``
    (see :func:`_git_repo_root`) -- ``None`` yields a path that is guaranteed not to be a prefix
    of a test path, so that :func:`_relativize_path_params` silently does nothing in tests that
    do not check it.
    """

    def _side_effect(args, **kwargs):  # noqa: ANN003 - mock signature
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return _completed((repo_root or "/nonexistent-repo-root") + "\n")
        if args[:2] == ["git", "rev-parse"]:
            return _completed("abc123deadbeef\n")
        if args[:2] == ["git", "status"]:
            return _completed("" if clean else " M src/x.py\n")
        if args[:2] == ["git", "diff"]:
            return _completed(diff_text)
        if args[:2] == ["git", "ls-files"]:
            return _completed(untracked_text)
        raise AssertionError(f"unerwarteter Aufruf: {args}")

    return _side_effect


class _NoWarnings:
    """Context manager: fails if a warning is raised."""

    def __enter__(self) -> None:
        import warnings as _warnings

        self._catch = _warnings.catch_warnings(record=True)
        self._records = self._catch.__enter__()
        _warnings.simplefilter("always")

    def __exit__(self, *exc: object) -> None:
        self._catch.__exit__(*exc)
        assert not self._records, f"unerwartete Warnung(en): {self._records}"


def test_environment_fingerprint_clean_repo() -> None:
    with patch("subprocess.run", side_effect=_run_side_effect(clean=True)), _NoWarnings():
        fingerprint = environment_fingerprint()
    assert fingerprint["git/commit"] == "abc123deadbeef"
    assert fingerprint["git/commit_available"] is True
    assert fingerprint["git/dirty"] is False
    assert fingerprint["git/diff_digest"] is None
    assert fingerprint["git/diff_digest_available"] is True


def test_environment_fingerprint_dirty_repo_warns_and_sets_diff_digest() -> None:
    with (
        patch("subprocess.run", side_effect=_run_side_effect(clean=False)),
        pytest.warns(DirtyWorkingTreeWarning, match="dirty"),
    ):
        fingerprint = environment_fingerprint()
    assert fingerprint["git/commit"] == "abc123deadbeef"
    assert fingerprint["git/commit_available"] is True
    assert fingerprint["git/dirty"] is True
    assert fingerprint["git/diff_digest_available"] is True
    assert fingerprint["git/diff_digest"] is not None
    assert len(fingerprint["git/diff_digest"]) == 16


def test_environment_fingerprint_no_repository() -> None:
    def _side_effect(args, **kwargs):  # noqa: ANN003
        raise FileNotFoundError("git not found")

    with patch("subprocess.run", side_effect=_side_effect):
        fingerprint = environment_fingerprint()
    assert fingerprint["git/commit"] is None
    assert fingerprint["git/commit_available"] is False
    assert fingerprint["git/dirty"] is False
    assert fingerprint["git/diff_digest"] is None
    assert fingerprint["git/diff_digest_available"] is True


def test_environment_fingerprint_thread_env_vars_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset env vars become ``None``, not an empty string."""
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with patch("subprocess.run", side_effect=_run_side_effect(clean=True)), _NoWarnings():
        fingerprint = environment_fingerprint()
    assert fingerprint["env/mkl_num_threads"] is None
    assert fingerprint["env/omp_num_threads"] is None


def test_environment_fingerprint_thread_env_vars_captured_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with patch("subprocess.run", side_effect=_run_side_effect(clean=True)), _NoWarnings():
        fingerprint = environment_fingerprint()
    assert fingerprint["env/mkl_num_threads"] == "1"
    assert fingerprint["env/omp_num_threads"] == "4"


def test_environment_fingerprint_torch_num_threads_present_when_torch_installed() -> None:
    torch = pytest.importorskip("torch")
    with patch("subprocess.run", side_effect=_run_side_effect(clean=True)), _NoWarnings():
        fingerprint = environment_fingerprint()
    assert fingerprint["torch/num_threads"] == torch.get_num_threads()


def test_environment_fingerprint_torch_num_threads_none_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_torch_num_threads`` mocked instead of faking the import (see its docstring)."""
    monkeypatch.setattr("twinlink_ml.utils.tracking._torch_num_threads", lambda: None)
    with patch("subprocess.run", side_effect=_run_side_effect(clean=True)), _NoWarnings():
        fingerprint = environment_fingerprint()
    assert fingerprint["torch/num_threads"] is None


@pytest.mark.skipif(not HAS_GIT, reason="Test braucht ein echtes git-Binary auf dem PATH")
def test_git_diff_digest_survives_narrow_locale_regardless_of_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the cp1252 trap from G4a -- catches it under Linux/CI too.

    The original error (``subprocess.run(text=True)`` without ``encoding=`` falls back to
    the locale code page of the executing system; on Windows usually cp1252) is
    invisible in practice under Linux/CI, because there the default locale is almost always
    UTF-8 already (and newer Python versions moreover automatically "coerce" a pure C locale to
    C.UTF-8, PEP 538) -- a test that relies on environment variables such as
    ``LC_ALL``/``LANG`` could therefore slip through exactly where it would be
    most important. Hence the mechanism that
    ``subprocess.run(text=True)`` without an explicit ``encoding=`` actually consults
    (``locale.getencoding``/``locale.getpreferredencoding``) is additionally patched directly here
    -- this reproduces the narrow, non-UTF-8-capable code page deterministically on every
    operating system and every Python version, independent of the actual locale of the
    executing machine. The environment variables are set in addition (documents the intent
    and takes effect if ``_git_diff_digest`` is ever moved into its own subprocess), but are
    not the load-bearing element of this test's determinism guarantee.

    ``_git_diff_digest`` binds its ``cwd`` firmly to the directory of ``tracking.py`` (the
    actual project repo), so it cannot point directly at a synthetic test repo.
    The redirect below only redirects the ``cwd`` kwarg to the test repo and then calls
    the REAL ``subprocess.run`` with all other kwargs (in particular
    ``encoding=``/``errors=``) unchanged -- the real git and decoding machinery
    thus actually runs, just against a controlled dirty state instead of against the
    real project checkout.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "module.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    # Make dirty with characters outside cp1252: tau, Delta, superscript digits, ~=.
    tracked.write_text("x = 1  # tau=τ Delta=Δ x² x⁶ ungefaehr≈gleich\n", encoding="utf-8")

    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("LC_CTYPE", "C")
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setattr(locale, "getencoding", lambda: "ascii")
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *_args, **_kwargs: "ascii")

    real_run = subprocess.run

    def _redirect_cwd_to_repo(args, **kwargs):  # noqa: ANN003 - passthrough wrapper
        kwargs = dict(kwargs)
        kwargs["cwd"] = repo
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _redirect_cwd_to_repo)

    digest = _git_diff_digest()
    assert digest is not None, (
        "git/diff_digest wurde None -- Regression der cp1252-Falle aus G4a: "
        "_git_diff_digest decodiert 'git diff HEAD' nicht mehr unabhaengig von der "
        "Locale des ausfuehrenden Systems."
    )
    assert len(digest) == 16

    with pytest.warns(DirtyWorkingTreeWarning, match="dirty"):
        fingerprint = environment_fingerprint()
    assert fingerprint["git/diff_digest_available"] is True
    assert fingerprint["git/diff_digest"] == digest


def _init_repo(repo: Path) -> None:
    """Empty git repo with a committed starting point.

    Common basis for the untracked-file tests below.
    Afterwards redirect via ``monkeypatch.setattr(tracking, "_GIT_CWD", repo)``, do not
    mock ``subprocess.run`` -- see the docstring of ``_GIT_CWD``.
    """
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", ".gitkeep"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)


# ------------------------------------------------------- IA1: untracked files in the digest


@pytest.mark.skipif(not HAS_GIT, reason="Test braucht ein echtes git-Binary auf dem PATH")
def test_git_diff_digest_distinguishes_untracked_file_from_clean_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual IA1 case: a completely new, never committed script.

    Before this extension, this case yielded the same digest as a clean tree
    (``sha256("")`` -- ``git diff HEAD`` does not see untracked files). Of all cases, that was
    the most frequent one in practice (Befund 52-54).
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.setattr(tracking, "_GIT_CWD", repo)

    clean_digest = _git_diff_digest()
    assert clean_digest is not None

    (repo / "new_script.py").write_text("print('a')\n", encoding="utf-8")
    dirty_digest = _git_diff_digest()

    assert dirty_digest is not None
    assert dirty_digest != clean_digest, (
        "git/diff_digest unterscheidet nicht zwischen einem sauberen Baum und einem Baum "
        "mit ausschliesslich einer neuen, nie committeten Datei -- IA1-Regression."
    )


@pytest.mark.skipif(not HAS_GIT, reason="Test braucht ein echtes git-Binary auf dem PATH")
def test_git_diff_digest_distinguishes_different_content_of_same_untracked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual trigger of this addendum: the same untracked script, changed.

    A digest that captures only the FILE NAME of the untracked file (instead of its content)
    would wrongly yield the same value here -- exactly the case from Befund 52-54 (the
    script was developed further between two runs, but stayed uncommitted until the end of
    the investigation).
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.setattr(tracking, "_GIT_CWD", repo)

    script = repo / "new_script.py"
    script.write_text("print('version a')\n", encoding="utf-8")
    digest_a = _git_diff_digest()

    script.write_text("print('version b, komplett anders')\n", encoding="utf-8")
    digest_b = _git_diff_digest()

    assert digest_a is not None
    assert digest_b is not None
    assert digest_a != digest_b, (
        "git/diff_digest unterscheidet nicht zwischen zwei verschiedenen Inhalten derselben "
        "untracked Datei -- der Dateiname allein wäre nicht genug."
    )


@pytest.mark.skipif(not HAS_GIT, reason="Test braucht ein echtes git-Binary auf dem PATH")
def test_git_diff_digest_marks_unreadable_untracked_file_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vanishing untracked file must not abort the digest computation.

    If it vanished between ``ls-files`` and the reading (or is unreadable for another
    reason), it gets the marker :data:`_UNTRACKED_FILE_UNREADABLE` instead of a
    content hash, and the result therefore differs from that of the readable file.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.setattr(tracking, "_GIT_CWD", repo)

    (repo / "vanishing.py").write_text("print('x')\n", encoding="utf-8")
    digest_readable = _git_diff_digest()
    assert digest_readable is not None

    real_read_bytes = Path.read_bytes

    def _fail_for_vanishing(self: Path, *args: object, **kwargs: object) -> bytes:
        if self.name == "vanishing.py":
            raise OSError("simuliert: Datei zwischen ls-files und Lesen verschwunden")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _fail_for_vanishing)

    digest_unreadable = _git_diff_digest()
    assert digest_unreadable is not None, (
        "eine einzelne nicht lesbare untracked Datei darf die gesamte Digest-Bildung nicht "
        "zum Scheitern bringen"
    )
    assert len(digest_unreadable) == 16
    assert digest_unreadable != digest_readable, (
        "der Unreadable-Marker muss sich vom Inhalts-Hash derselben Datei unterscheiden, "
        "sonst waere er von einem echten Treffer nicht zu unterscheiden"
    )


def _record(*, commit_available: bool) -> RunRecord:
    return RunRecord(
        group="test",
        params={"git/commit_available": commit_available},
        metrics={"nrmse": 0.1},
    )


def test_jsonl_logger_raises_without_git_commit_by_default(tmp_path: Path) -> None:
    logger = JsonlExperimentLogger(tmp_path / "log.jsonl")
    with pytest.raises(MissingGitCommitError):
        logger.log(_record(commit_available=False))
    assert not (tmp_path / "log.jsonl").exists()


def test_jsonl_logger_writes_with_allow_missing_git(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    logger = JsonlExperimentLogger(path, allow_missing_git=True)
    with pytest.warns(UserWarning, match="ohne Git-Commit"):
        logger.log(_record(commit_available=False))
    assert path.exists()
    from twinlink_ml.utils.tracking import read_jsonl

    records = read_jsonl(path)
    assert len(records) == 1
    assert records[0].params["git/commit_available"] is False


def test_jsonl_logger_writes_normally_with_git_commit(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    logger = JsonlExperimentLogger(path)
    logger.log(_record(commit_available=True))
    assert path.exists()


def test_jsonl_logger_relativizes_absolute_checkpoint_path(tmp_path: Path) -> None:
    """Absolute checkpoint paths contained user names/network drives (audit 2026-09-25).

    Future records are to log ``*/path`` parameters relative to the repo root, provided
    the path lies below it.
    """
    from twinlink_ml.utils.tracking import read_jsonl

    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "artifacts" / "checkpoints" / "gru_seed1.pt"
    outside = tmp_path / "elsewhere" / "gru_seed2.pt"
    record = RunRecord(
        group="test",
        params={
            "git/commit_available": True,
            "checkpoint/path": str(checkpoint),
            "other/path": str(outside),
            "model/label": "gru",
        },
        metrics={"nrmse": 0.1},
    )
    log_path = tmp_path / "log.jsonl"
    logger = JsonlExperimentLogger(log_path)
    with patch(
        "subprocess.run", side_effect=_run_side_effect(clean=True, repo_root=str(repo_root))
    ):
        logger.log(record)
    records = read_jsonl(log_path)
    assert records[0].params["checkpoint/path"] == "artifacts/checkpoints/gru_seed1.pt"
    assert records[0].params["other/path"] == str(outside)  # outside the repo root


def test_jsonl_logger_leaves_checkpoint_path_absolute_without_git(tmp_path: Path) -> None:
    """No git repository (or ``git`` not on the PATH) -- the path stays unchanged.

    No wrong relative path is produced.
    """
    from twinlink_ml.utils.tracking import read_jsonl

    checkpoint = tmp_path / "somewhere" / "gru_seed1.pt"
    record = RunRecord(
        group="test",
        params={"git/commit_available": True, "checkpoint/path": str(checkpoint)},
        metrics={"nrmse": 0.1},
    )
    log_path = tmp_path / "log.jsonl"
    logger = JsonlExperimentLogger(log_path)

    def _no_repo(args, **kwargs):  # noqa: ANN003 - mock signature
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            raise FileNotFoundError("git nicht gefunden")
        return _run_side_effect(clean=True)(args, **kwargs)

    with patch("subprocess.run", side_effect=_no_repo):
        logger.log(record)
    records = read_jsonl(log_path)
    assert records[0].params["checkpoint/path"] == str(checkpoint)


def test_jsonl_logger_treats_missing_field_as_unavailable(tmp_path: Path) -> None:
    """Records without ``git/commit_available`` (old callers) count as not available."""
    logger = JsonlExperimentLogger(tmp_path / "log.jsonl")
    record = RunRecord(group="test", params={}, metrics={"nrmse": 0.1})
    with pytest.raises(MissingGitCommitError):
        logger.log(record)


# ------------------------------------------------------------ R2: git/diff_digest gap


def test_environment_fingerprint_dirty_but_diff_unobtainable_sets_marker_and_warns() -> None:
    """``git/dirty=True``, but ``git diff`` itself fails.

    The silent failure path from G4a: ``git/diff_digest`` would otherwise be ``None``,
    indistinguishable from a clean state.
    """

    def _side_effect(args, **kwargs):  # noqa: ANN003 - mock signature
        if args[:2] == ["git", "rev-parse"]:
            return _completed("abc123deadbeef\n")
        if args[:2] == ["git", "status"]:
            return _completed(" M src/x.py\n")
        if args[:2] == ["git", "diff"]:
            raise subprocess.CalledProcessError(1, args)
        raise AssertionError(f"unerwarteter Aufruf: {args}")

    with (
        patch("subprocess.run", side_effect=_side_effect),
        pytest.warns(DiffDigestUnavailableWarning, match="diff_digest"),
    ):
        fingerprint = environment_fingerprint()
    assert fingerprint["git/dirty"] is True
    assert fingerprint["git/diff_digest"] == DIFF_DIGEST_UNAVAILABLE
    assert fingerprint["git/diff_digest_available"] is False
    # The marker is not a 16-digit hex digest -- distinguishable from a real digest.
    assert len(fingerprint["git/diff_digest"]) != 16


def _record_with_diff_digest(*, dirty: bool, diff_digest_available: bool) -> RunRecord:
    return RunRecord(
        group="test",
        params={
            "git/commit_available": True,
            "git/dirty": dirty,
            "git/diff_digest_available": diff_digest_available,
        },
        metrics={"nrmse": 0.1},
    )


def test_jsonl_logger_raises_when_dirty_and_diff_digest_unavailable(tmp_path: Path) -> None:
    logger = JsonlExperimentLogger(tmp_path / "log.jsonl")
    with pytest.raises(MissingGitCommitError, match="diff_digest"):
        logger.log(_record_with_diff_digest(dirty=True, diff_digest_available=False))
    assert not (tmp_path / "log.jsonl").exists()


def test_jsonl_logger_writes_with_allow_missing_git_when_diff_digest_unavailable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "log.jsonl"
    logger = JsonlExperimentLogger(path, allow_missing_git=True)
    with pytest.warns(UserWarning, match="diff_digest"):
        logger.log(_record_with_diff_digest(dirty=True, diff_digest_available=False))
    assert path.exists()


def test_jsonl_logger_ignores_diff_digest_available_when_not_dirty(tmp_path: Path) -> None:
    """``git/diff_digest_available=False`` is only relevant if the record itself is dirty.

    A clean state does not expect a digest anyway.
    """
    path = tmp_path / "log.jsonl"
    logger = JsonlExperimentLogger(path)
    logger.log(_record_with_diff_digest(dirty=False, diff_digest_available=False))
    assert path.exists()
