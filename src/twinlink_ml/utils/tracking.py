"""Experiment-Tracking ab Tag 1 (Arbeitsplan Schritt 5).

Warum ein eigenes Modul und nicht direkt MLflow
-----------------------------------------------
MLflow ist keine Laufzeitabhängigkeit des Pakets, und ein Lernkurvenlauf soll auch ohne
Tracking-Server funktionieren. Die ABC :class:`ExperimentLogger` erlaubt beides: das
JSONL-Log läuft überall und ist versionierbar, der MLflow-Adapter wird nachgeladen, wenn
er verfügbar ist. Dieselbe Trennung wie beim Backend und beim Simulator.

Was zwingend in jeden Datensatz gehört
--------------------------------------
Ohne diese sechs Fingerprints ist ein Lernkurvenpunkt nicht rekonstruierbar, und das
merkt man erst Monate später:

* ``process/fingerprint`` -- Prozesskonfiguration inklusive aller Platzhalter
* ``excitation/fingerprint`` -- Anregungsdesign; enthält ``n_runs``, ist also je
  Pool-Größe verschieden
* ``split/digest`` -- Zuordnung *und* Reihenfolge der Trainingsläufe
* ``scaler/digest`` -- Normalisierungsstatistiken; deckt auf, wenn sie aus dem
  falschen Split kamen
* ``seed/{data,noise,model}`` -- die drei trennbaren Zufallsquellen
* ``git/commit`` und die Paketversionen
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
    """``git status --porcelain`` ist nicht leer, wenn ein Record entsteht.

    Eigene Klasse statt ``UserWarning`` direkt, damit sie gezielt gefiltert werden kann --
    ``pyproject.toml`` markiert sie als erwartet (dieselbe Begründung wie
    ``PlaceholderConfigWarning``: ein dirty Arbeitsbaum ist während normaler Iteration die
    Regel, kein Fehlerfall, und darf die Testsuite nicht blockieren, siehe Docstring von
    :func:`environment_fingerprint`).
    """


class MissingGitCommitError(RuntimeError):
    """Ein Record mit einer Provenienz-Lücke sollte geschrieben werden.

    Befund 33 (``docs/04_stand_ap1.md``): ``git/commit`` stand in jedem bisher
    aufgezeichneten Lauf still auf ``"unknown"`` -- ohne Warnung, ohne Log-Eintrag an der
    Aufrufstelle. Bei ``arch_composite.jsonl`` (440 Records) fiel das erst Wochen später
    auf, als die Konfiguration hinter den Zahlen nicht mehr rekonstruierbar war. Diese
    Ausnahme macht das Fehlen zum Standardfehler statt zur stillen Zeichenkette; ein
    Aufrufer, der das bewusst in Kauf nehmen will (z. B. ein Docker-Image ohne ``.git``),
    setzt ``allow_missing_git=True`` am Logger.

    G4a-Nachbesserung: dieselbe Ausnahme deckt jetzt auch die verwandte Lücke ab, dass ein
    dirty Arbeitsbaum keinen ermittelbaren ``git/diff_digest`` hat (s.
    :func:`_check_git_provenance`) -- dieselbe stille-Lücke-Fehlerklasse wie beim Commit,
    unter demselben ``allow_missing_git``-Schalter geprüft. ``allow_missing_git`` und der
    Klassenname sind dadurch etwas enger benannt, als sie inhaltlich abdecken (sie heißen
    nach dem Commit, nicht nach der Provenienz insgesamt) -- absichtlich nicht umbenannt,
    weil das bestehende Aufrufer betrifft; das ist eine eigene Entscheidung.
    """


class DiffDigestUnavailableWarning(UserWarning):
    """``git/dirty`` ist wahr, aber ``git diff HEAD`` ließ sich nicht in einen Digest übersetzen.

    Eigene Klasse statt ``DirtyWorkingTreeWarning`` direkt: ein dirty Arbeitsbaum allein ist
    die Regel (siehe dort), aber zusätzlich keinen Digest dafür ermitteln zu können, ist der
    stille Fehlpfad, der vor ``encoding="utf-8"``/``errors="replace"`` in
    :func:`_git_diff_digest` zehn Tests in ``tests/test_protocol.py`` unter Windows scheitern
    ließ (``stdout`` blieb ``None``, ohne dass eine Exception den Aufrufer erreichte). Diese
    Warnung bleibt als Absicherung für den verbleibenden, echten Fehlerfall (kein ``git`` auf
    dem PATH, kein Repository, ``git diff`` oder ``git ls-files --others`` selbst schlägt
    fehl -- Letzteres seit der Erweiterung um untracked Dateien, s. :func:`_git_diff_digest`).
    """


def _git_commit() -> str | None:
    """Commit-Hash des Arbeitsbaums, ``None`` außerhalb eines Repositories.

    ``encoding="utf-8"``/``errors="replace"``: dieselbe Windows-Falle wie in
    :func:`_git_diff_digest` (``text=True`` ohne ``encoding`` fällt auf die Locale-Codepage
    des ausführenden Systems zurück) trifft grundsätzlich jeden ``subprocess.run``-Aufruf
    hier -- auch wenn ein Commit-Hash in der Praxis rein hexadezimal ist und nie
    undekodierbare Bytes enthält, macht die explizite Angabe die Funktion unabhängig davon
    korrekt statt zufällig.
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
    """Wurzelverzeichnis des Arbeitsbaums, ``None`` außerhalb eines Repositories.

    Für :func:`_relativize_path_params` -- absolute Checkpoint-Pfade sollen relativ
    zu dieser Wurzel geloggt werden, nicht als absoluter Pfad mit Benutzername/
    Netzlaufwerk (Vertraulichkeits-Audit 2026-09-25).
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


def _git_dirty() -> bool:
    """``True``, wenn der Arbeitsbaum uncommittete Änderungen hat.

    ``False`` außerhalb eines Repositories oder wenn ``git`` selbst nicht auf dem PATH
    liegt (Befund 33 dokumentiert genau diese Umgebung) -- ``git/dirty`` beantwortet dann
    keine sinnvolle Frage, und ``git/commit_available=False`` trägt die eigentliche
    Information bereits.

    ``encoding="utf-8"``/``errors="replace"``: s. :func:`_git_diff_digest` -- ``git status
    --porcelain`` quotet untypische Dateinamen zwar normalerweise, verlässt sich aber nicht
    darauf.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            cwd=_GIT_CWD,
        )
    except OSError:
        return False
    return bool(result.stdout.strip())


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
    """Pfade aller nicht-ignorierten, nicht getrackten Dateien.

    ``git ls-files --others --exclude-standard``, von Git selbst deterministisch nach
    Pfad sortiert.

    ``--exclude-standard`` spiegelt dieselbe ``.gitignore``-Semantik, die ``git status
    --porcelain`` (s. :func:`_git_dirty`) ohnehin schon anwendet -- eine ignorierte Datei
    macht den Baum nicht dirty und soll auch hier nicht auftauchen.

    ``None`` bei jedem Fehler (kein ``git`` auf dem PATH, kein Repository, nicht-Null
    Exit) -- wie :func:`_git_commit`/:func:`_git_dirty`, dieselbe ``encoding="utf-8"``/
    ``errors="replace"``-Begründung für die Pfadnamen selbst (nicht für Dateiinhalte, die
    werden binär gelesen, s. :func:`_git_diff_digest`).
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
    """Macht einen dirty Stand identifizierbar.

    sha256 (16 Hex-Zeichen) über ``git diff HEAD`` UND den Inhalt aller untracked Dateien.

    ``git diff HEAD`` allein sieht nur getrackte Änderungen (inklusive gestageter neuer
    Dateien) -- eine komplett neue, nie ``git add``ete Datei taucht darin NICHT auf, auch
    nicht als "neue Datei"-Hunk. Ein dirty Stand, der ausschließlich aus einer solchen
    Datei besteht, lieferte deshalb bislang den Digest des LEEREN Diffs
    (``sha256("")[:16] == "e3b0c44298fc1c14"``) -- identifizierbar nur im Namen, nicht in
    der Sache: ausgerechnet der häufigste Fall (ein neu geschriebenes, noch nicht
    committetes Skript) blieb damit stumm. Diese Funktion schließt die Lücke, indem sie
    zusätzlich :func:`_git_untracked_files` abfragt und für jede darin gelistete Datei
    ihren Inhalt hasht -- nicht nur ihren Namen, sonst würde ein zwischen zwei Läufen
    GEÄNDERTES uncommittetes Skript denselben Digest liefern, solange der Dateiname
    gleich bleibt (genau der Fall, der diese Erweiterung ausgelöst hat).

    Kombination über eine strukturierte, nicht über eine simple String-Konkatenation:
    ``json.dumps({"diff": <sha256 von git diff HEAD>, "untracked": [[pfad, sha256(inhalt)],
    ...]}, sort_keys=True)``, danach sha256 darüber. Eine simple Konkatenation
    (Pfad+Inhalt aneinandergehängt) wäre mehrdeutig -- Pfad "a"+Inhalt "bc" ergäbe dieselben
    Bytes wie Pfad "ab"+Inhalt "c". Jede Datei wird zuerst auf ihren eigenen 64-stelligen
    Hash reduziert, bevor sie in die äußere Struktur eingeht, damit eine einzelne große
    untracked Datei nicht die äußere Hash-Eingabe aufbläht. Die ``untracked``-Liste wird
    zusätzlich explizit sortiert (nicht nur auf Gits eigene, an sich schon deterministische
    Sortierung von ``ls-files`` vertraut) -- Reproduzierbarkeit soll aus dieser Funktion
    selbst folgen, nicht aus einer Eigenschaft eines Aufrufs, den sie nur konsumiert.

    Kann eine untracked Datei nicht gelesen werden (Race mit ``git ls-files --others``,
    fehlende Berechtigung), ersetzt :data:`_UNTRACKED_FILE_UNREADABLE` ihren Inhalts-Hash --
    die Digest-Bildung bricht dadurch nicht ab, aber die Lücke bleibt im Wert selbst
    sichtbar (kein stiller Fallback auf einen harmlos aussehenden Wert).

    Untracked Dateien werden BINÄR gelesen, nicht als dekodierter Text: für den
    getrackten Diff ist ``errors="replace"`` die richtige Wahl (s. unten), für
    Dateiinhalte wäre es Informationsverlust -- zwei inhaltlich verschiedene Binärdateien
    dürften nicht durch identische Ersetzungszeichen auf denselben Hash abgebildet werden.

    **Zeilenenden zwischen Windows und Linux:** dieses Repository setzt
    ``.gitattributes: * text=auto eol=lf`` -- getrackte Dateien werden von Git beim
    Checkout/Diff unabhängig von ``core.autocrlf`` auf LF normalisiert, ihr Beitrag zum
    Digest ist deshalb plattformunabhängig stabil. Für UNTRACKED Dateien gilt das NICHT:
    Git wendet weder ``.gitattributes`` noch ``core.autocrlf`` auf Dateien an, die es nicht
    verwaltet. Dieselbe uncommittete Datei kann auf einer Windows- und einer
    Linux-Maschine unterschiedliche Bytes (CRLF vs. LF) und damit einen unterschiedlichen
    Digest tragen, auch wenn sie inhaltlich "derselbe Code" ist. Das ist kein Fehler dieser
    Funktion -- der Digest soll den TATSÄCHLICHEN Zustand erfassen, und unterschiedliche
    Bytes sind ein tatsächlich unterschiedlicher Zustand --, aber es gehört hierhin, damit
    ein späterer Vergleich zweier Digests von verschiedenen Maschinen nicht fälschlich als
    "unterschiedlicher Code" gelesen wird, wo es eigentlich nur unterschiedliche
    Zeilenenden derselben, noch nicht committeten Datei sind.

    ``None`` außerhalb eines Repositories oder wenn ``git diff``/``git ls-files`` selbst
    fehlschlägt (nicht auf dem PATH, kein Repository, nicht-Null Exit).
    :func:`environment_fingerprint` übersetzt ein dirty-aber-``None`` in einen expliziten
    Marker statt es still wie "kein Diff" aussehen zu lassen (s. dort) -- diese Funktion
    selbst bleibt bei ``None``.

    **Nicht vergleichbar über diese Änderung hinweg:** vor dieser Erweiterung war der
    Digest ``sha256(git diff HEAD)[:16]`` (roher Diff-Text). Jetzt ist er
    ``sha256(json.dumps({"diff": sha256(git diff HEAD), "untracked": [...]}, sort_keys=True))[:16]``
    -- eine andere Eingabestruktur, nicht nur eine Erweiterung derselben. Ein VOR diesem
    Commit berechneter ``git/diff_digest`` und einer DANACH sind deshalb selbst bei
    identischem Baumzustand (auch ganz ohne untracked Dateien) verschieden -- derselbe
    Diff-Text erzeugt jetzt einen anderen Hash, weil er zuerst selbst gehasht und dann in
    eine JSON-Struktur eingebettet wird, statt direkt gehasht zu werden. Ein Vergleich
    ``digest_alt == digest_neu`` "gleicher Codestand?" ist damit über diesen Commit hinweg
    sinnlos -- dieselbe Klasse Drift wie beim ``ProcessConfig``-Fingerprint (dessen Digest
    sich bei jeder Erweiterung um ein Feld ändert, auch wenn der neue Wert dem alten
    Default entspricht). Innerhalb eines einzelnen Digest-Verfahrens (vor ODER nach diesem
    Commit, nicht gemischt) bleibt der Vergleich weiterhin gültig.

    ``encoding="utf-8"``: ein dirty Diff enthält hier routinemäßig UTF-8-Sonderzeichen
    (deutsche Dokumentation, griechische Formelzeichen, hochgestellte Ziffern). Ohne
    explizites ``encoding`` fällt ``subprocess.run(text=True)`` auf die Locale-Codepage des
    ausführenden Systems zurück (auf Windows meist cp1252), die solche Bytes nicht dekodieren
    kann. Der Fehler passiert dabei in einem internen Reader-Thread von ``subprocess`` und
    wird NICHT als Exception an den Aufrufer durchgereicht -- ``stdout`` bleibt einfach
    ``None``, ohne dass der ``except``-Block unten je greift: genau die Fehlerklasse, gegen
    die diese Funktion gebaut wurde, unterläuft sie an ihrer eigenen Decodierung.

    ``errors="replace"``: ein Diff kann auch nach Wahl von UTF-8 nicht-dekodierbare Bytes
    enthalten (z. B. Binärdateien im Diff). Der Digest soll dafür trotzdem einen Wert
    liefern statt an einzelnen Bytes zu scheitern -- er ist dann nicht mehr byte-identisch
    zum Rohdiff (Ersetzungszeichen U+FFFD statt der Originalbytes an unlesbaren Stellen),
    bleibt aber über wiederholte Aufrufe an demselben Stand stabil und reproduzierbar, was
    für den Zweck hier (einen dirty Stand identifizierbar machen, nicht den Diff selbst
    rekonstruieren) ausreicht. Für untracked Dateien stellt sich dieselbe Frage nicht --
    die werden binär gelesen (s. oben).
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
    """``torch.get_num_threads()``, ``None`` wenn Torch nicht installiert ist.

    Eigene Funktion (nicht inline in :func:`environment_fingerprint`), damit Tests den
    torch-freien Fall simulieren koennen, ohne den echten Import zu faelschen -- ``torch``
    bleibt damit ein optionaler Import fuer dieses ansonsten torch-freie Modul, dieselbe
    Bedingung wie beim ``version/torch``-Feld weiter unten.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch.get_num_threads()


def environment_fingerprint(*, check_code_location: bool = True) -> dict[str, Any]:
    """Umgebung, so weit sie das Ergebnis beeinflussen kann.

    ``git/commit`` ist ``None`` (nicht mehr die Zeichenkette ``"unknown"``), wenn kein
    Repository gefunden wird; ``git/commit_available`` macht das explizit statt es still
    im Wert zu verstecken (Befund 33 -- siehe :class:`MissingGitCommitError`). Ein dirty
    Arbeitsbaum (``git/dirty=True``) ist selbst mit gültigem Commit-Hash nicht
    reproduzierbar -- der Commit beschreibt den Stand nicht vollständig; die aktuelle
    Sitzung demonstriert das (``xlstm.py``, ``test_composite.py`` uncommittet).
    ``git/diff_digest`` macht auch einen dirty Stand nachträglich identifizierbar. Ein
    dirty Arbeitsbaum ist nur eine laute Warnung, kein Fehler -- sonst wäre Iteration
    zwischen Läufen unmöglich. Ist der Arbeitsbaum dirty, der Digest aber TROTZDEM nicht
    ermittelbar (kein ``git`` auf dem PATH, kein Repository, ``git diff`` schlägt fehl),
    wird das NICHT still zu ``None`` -- das wäre von einem sauberen Stand (``git/dirty
    =False``, ``git/diff_digest=None`` ebenfalls) nicht mehr unterscheidbar, exakt die
    Fehlerklasse aus Befund 33. Stattdessen bekommt ``git/diff_digest`` den Marker
    :data:`DIFF_DIGEST_UNAVAILABLE`, ``git/diff_digest_available=False`` macht es
    maschinell auswertbar (Pendant zu ``git/commit_available``), und eine
    :class:`DiffDigestUnavailableWarning` wird ausgelöst. :func:`_check_git_provenance`
    prüft dieses Feld unter demselben ``allow_missing_git``-Schalter wie
    ``git/commit_available`` (s. dort).

    ``torch/num_threads``, ``env/mkl_num_threads``, ``env/omp_num_threads``: Anlass war die
    Faktor-5-Streuung in ``s/Epoche`` in ``befund42_tbed_exhaust_sweep.jsonl``, die ohne
    diese Felder nur ueber die zeitlichen Cluster der ``created_unix_s``-Werte diagnostizierbar
    war (Threading/Prozess-Ueberzeichnung, nicht Konvergenz -- siehe Analyse dazu). Die beiden
    Env-Var-Felder sind ``None``, wenn die Variable nicht gesetzt ist -- das ist selbst die
    Information (kein Pinning aktiv), nicht mit einem leeren String zu verwechseln.

    ``code/package_dir``, ``code/interpreter``, ``code/root_checked`` (siehe
    :mod:`twinlink_ml.utils.code_location`): wo ``twinlink_ml`` tatsächlich geladen wurde und
    mit welchem Interpreter. Liegt das geladene Paket nicht unter der Wurzel des Repositorys
    bzw. Worktrees, aus dem der Lauf gestartet wurde (Arbeitsverzeichnis), bricht die Funktion
    mit :class:`~twinlink_ml.utils.code_location.CodeLocationError` ab, bevor irgendein
    Record entsteht. ``check_code_location=False`` nur für Aufrufer, die ausdrücklich keinen
    Artefakt-Lauf protokollieren (die Felder werden trotzdem geschrieben).
    """
    location = (
        require_code_location(find_repo_root())
        if check_code_location
        else code_location_fields(find_repo_root())
    )
    from ..evaluation.experiments import NEW_STUDY_PRESET_VERSION

    commit = _git_commit()
    dirty = _git_dirty()
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
    for name in ("numpy", "scipy", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[f"version/{name}"] = str(getattr(module, "__version__", "unknown"))
    return versions


def _check_git_provenance(record: RunRecord, *, allow_missing_git: bool) -> None:
    """Verweigert das Schreiben eines Records mit einer Provenienz-Lücke.

    Zwei Lücken werden geprüft, beide unter demselben ``allow_missing_git``-Schalter (s.
    :class:`MissingGitCommitError` zur Begründung, warum Name und Schalter trotzdem so
    heißen wie ursprünglich, nur nach dem ersten Fall):

    * ``git/commit_available`` kommt aus :func:`environment_fingerprint`. Fehlt der
      Schlüssel (ein Aufrufer, der ``environment_fingerprint()`` nicht verwendet), wird das
      als nicht verfügbar behandelt (Standard ``False``) -- lieber ein Aufrufer setzt
      ``allow_missing_git`` bewusst, als dass ein Record wie in ``arch_composite.jsonl``
      (``git/commit="unknown"`` in 440 Records) Monate später unauffindbar wird.
    * ``git/diff_digest_available`` kommt ebenfalls aus :func:`environment_fingerprint`,
      wird aber nur geprüft, wenn der Record selbst ``git/dirty=True`` trägt -- ein Record
      ohne ``git/dirty``-Angabe (älterer Aufrufer, der das Feld nie kannte) soll dadurch
      nicht rückwirkend als lückenhaft gelten, wo vorher gar keine Erwartung bestand; fehlt
      der Schlüssel bei ``git/dirty=True``, gilt er als verfügbar (Standard ``True``), da
      ``environment_fingerprint()`` das Feld inzwischen immer setzt und ein Fehlen hier eher
      einen fremden Aufrufer als einen echten Fehlerfall anzeigt.
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
    """Ein Eintrag im Ergebnis-Log.

    Attributes
    ----------
    group
        Experimentname, etwa ``"learning_curve"`` oder ``"dt_study"``.
    params
        Alles, was den Lauf bestimmt: Fingerprints, Seeds, Fenster- und
        Modellkonfiguration. Muss JSON-serialisierbar sein.
    metrics
        Skalare Ergebnisse, flach benannt (siehe
        :meth:`~twinlink_ml.evaluation.evaluator.EvaluationResult.as_flat_metrics`).
    tags
        Kurze Kennzeichnungen für Filterung, etwa Modell- und Zielgrößenname.
    """

    group: str
    params: Mapping[str, Any]
    metrics: Mapping[str, float]
    tags: Mapping[str, str] = field(default_factory=dict)
    run_uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_unix_s: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        """Serialisierbare Form."""
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
    """Macht absolute Pfade in ``*/path``-Parametern relativ zum Repository-Wurzelverzeichnis.

    Betrifft nur künftig geschriebene Records (:meth:`JsonlExperimentLogger.log`) --
    bestehende Zeilen in ``artifacts/runs/*.jsonl`` bleiben unangetastet. Ruft
    :func:`_git_repo_root` (ein ``subprocess.run``) bewusst erst auf, wenn überhaupt ein
    absoluter ``*/path``-Wert vorliegt -- der häufige Fall ohne solche Parameter bleibt so
    ohne zusätzlichen Prozessaufruf. Ein Pfad außerhalb der Repo-Wurzel (z. B. ein
    Checkpoint auf einem anderen Laufwerk) oder kein Git-Repository bleibt unverändert --
    lieber ein absoluter Pfad als ein falscher relativer.
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
    """Rekursive Konversion in JSON-taugliche Typen."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Iterable):
        return [_jsonable(item) for item in value]
    return str(value)


class ExperimentLogger(ABC):
    """Senke für Ergebnis-Einträge."""

    @abstractmethod
    def log(self, record: RunRecord) -> None:
        """Einen Eintrag schreiben."""

    def close(self) -> None:  # noqa: B027 - absichtlich nicht abstrakt
        """Ressourcen freigeben.

        Standardimplementierung tut nichts und ist bewusst **nicht** abstrakt: ein Logger
        ohne Ressourcen soll keine leere Methode schreiben müssen.
        """


@dataclass(slots=True)
class JsonlExperimentLogger(ExperimentLogger):
    """Anfügendes JSON-Lines-Log -- eine Zeile pro Lauf.

    Bewusst anfügend und nicht überschreibend: ein Lernkurvenlauf, der nach 40 von 45
    Punkten abbricht, soll die 40 behalten. Das Format ist zeilenweise diffbar und ohne
    Zusatzwerkzeug auswertbar.

    Attributes
    ----------
    allow_missing_git
        ``False`` (Standard): :meth:`log` wirft :class:`MissingGitCommitError`, wenn der
        Record keinen ermittelbaren ``git/commit`` trägt (siehe Befund 33). ``True``:
        der Record wird trotzdem geschrieben, mit einer lauten Warnung -- für Umgebungen
        ohne Git-Repository, bewusst vom Aufrufer gesetzt.
    """

    path: Path
    allow_missing_git: bool = False

    def log(self, record: RunRecord) -> None:
        """Eintrag anfügen."""
        _check_git_provenance(record, allow_missing_git=self.allow_missing_git)
        relativized_params = _relativize_path_params(record.params)
        record = replace(record, params=relativized_params)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


@dataclass(slots=True)
class MlflowExperimentLogger(ExperimentLogger):
    """Adapter auf MLflow; wird nur beim ersten Schreiben importiert.

    Fehlt MLflow, schlägt die Konstruktion fehl und nicht erst der 40. Lauf.
    """

    experiment_name: str
    tracking_uri: str | None = None
    allow_missing_git: bool = False

    def __post_init__(self) -> None:
        try:
            import mlflow
        except ImportError as error:  # pragma: no cover - abhängig von der Umgebung
            raise ImportError(
                "MLflow ist nicht installiert. Entweder 'pip install twinlink-ml[train]' "
                "oder JsonlExperimentLogger verwenden."
            ) from error
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log(self, record: RunRecord) -> None:
        """Eintrag als MLflow-Run schreiben."""
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
    """Mehrere Senken gleichzeitig. Reihenfolge bleibt erhalten."""

    loggers: tuple[ExperimentLogger, ...]

    def log(self, record: RunRecord) -> None:
        """Den Eintrag an alle Senken weitergeben."""
        for logger in self.loggers:
            logger.log(record)

    def close(self) -> None:
        """Alle Senken schließen."""
        for logger in self.loggers:
            logger.close()


def read_jsonl(path: Path | str) -> list[RunRecord]:
    """JSONL-Log als :class:`RunRecord`-Liste einlesen.

    Damit ein Notebook keine JSON-Struktur selbst auspacken muss: Auswertung und Plots
    arbeiten auf demselben Typ wie die Treiber, und eine Änderung am Log-Format schlägt
    an einer Stelle durch statt in jeder Zelle.
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
