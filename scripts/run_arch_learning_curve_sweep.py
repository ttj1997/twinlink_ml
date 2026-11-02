r"""Orchestrates the learning-curve sweep: 5 seeds, each 4 budget points x N archs, parallel.

Starts ``run_arch_learning_curve_worker.py`` as independent processes -- one job per seed,
five in total. No hyperparameter and no seed is changed by this; only the wall-clock time
drops. Structure deliberately identical to ``run_arch_constant_holdout_sweep.py`` (staggered
start, merge with duplicate-run_uid check) -- only the job unit is the seed instead of
(architecture, seed), because ``run_arch_learning_curve_worker.py`` does the pool generation once
per seed and then runs through all four budget points internally.

``MKL_NUM_THREADS``/``OMP_NUM_THREADS=1`` are passed here as PROCESS environment variables to
every worker -- ``torch.set_num_threads(1)`` alone is not enough (factor-4
slowdown through thread oversubscription with parallel process execution, see the
``befund42_tbed_exhaust_sweep.jsonl`` analysis).

Resumability (OA2b, Transformer/xLSTM/Mamba extension)
-------------------------------------------------------------
The worker stays UNCHANGED -- resuming is pure orchestrator logic, at the level of
whole seed jobs (not individual architecture/budget combinations): one seed job generates
all architectures x all budget points x both test sets together in one process; a
partial rerun of individual cells within a seed is not possible with the worker in its current
form. The task key of a record is therefore
``(master_seed, n_train, model/label, test_set)`` -- read bit for bit from an existing
record in ``arch_learning_curve_projected.jsonl`` (``params["master_seed"]``,
``params["n_train"]``, ``params["model/label"]``, ``params["test_set"]``). There are
NO ``variance/replicate``/``variance/source`` fields in this schema (they come from
another protocol, ``evaluation/resolution.py`` -- not relevant here).

A seed counts as complete if a record exists for EVERY expected combination
(``sizes`` x ``BASELINE_FACTORIES``/``ARCHITECTURE_LABELS`` of the worker x both
test sets) -- then this seed is skipped (default:
``--resume`` on, see ``parse_arguments``). If even one combination is missing, the
ENTIRE seed is recomputed (the worker knows no subset), and its old records
are discarded at the merge, not mixed with the new ones -- otherwise duplicate ``run_uid``
or inconsistent partial states within the same seed.

PROVENANCE GUARD: a resumed dataset whose points come partly from the old, partly from the
new ``git/commit`` is worthless as ONE learning curve. Therefore, when resuming,
the ``git/commit`` of the adopted (complete) seeds is compared against the current HEAD
-- a mismatch aborts hard (both hashes in the message), overridable only explicitly with
``--allow-commit-mismatch``. A dirty working tree also aborts hard when
resuming (no silent continuation -- unlike
``environment_fingerprint()``, which treats dirty only as a warning).

A line truncated at the end of the existing log file (abort in the middle of
writing) is discarded on reading, not treated as an error -- the associated
seed then counts as incomplete and is recomputed.

Open point (SA4, docs/04_stand_ap1.md Befund 58/59): the guard only checks at the
``--resume`` START. A commit that lands WHILE a sweep is already running (no
resumption, all jobs already running) is not detected by it -- exactly how the
commit drift in Befund 58 arose, noticed manually there instead of caught automatically.
Only noted, not fixed. Possible addition for later: a commit check at the
start of each individual fit in the worker, not only in the orchestrator at start-up.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

from twinlink_ml.utils.tracking import DirtyWorkingTreeWarning, environment_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = Path(__file__).resolve().parent / "run_arch_learning_curve_worker.py"
SEEDS = (1, 2, 3, 4, 5)
TEST_SETS = ("excited", "constant_holdout")


def _load_worker_module() -> types.ModuleType:
    """Load ``run_arch_learning_curve_worker.py`` as a module to read the current
    architecture list (do not keep it duplicated here -- otherwise the
    resume logic will eventually drift away from the worker silently). No ``torch`` import at
    load time (in the worker it sits in ``main()``, not at module level), so harmless
    for the orchestrator, which itself needs no torch.
    """
    spec = importlib.util.spec_from_file_location(
        "_run_arch_learning_curve_worker_for_sweep", WORKER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_labels(architectures: list[str] | None, *, skip_baselines: bool) -> tuple[str, ...]:
    """Which labels a seed job with this selection actually produces.

    Must reflect the same selection that ``--architectures``/``--skip-baselines`` pass through
    to the worker (see ``_launch``) -- otherwise a seed considers the resumption
    wrongly incomplete (or wrongly complete) when deliberately
    only a subset is computed (e.g. only the three new architectures, without
    regenerating GRU/LSTM/baselines, which already exist under the same protocol in
    another dataset).
    """
    worker = _load_worker_module()
    baseline_labels = () if skip_baselines else tuple(worker.BASELINE_FACTORIES)
    arch_labels = tuple(architectures) if architectures is not None else worker.ARCHITECTURE_LABELS
    return baseline_labels + arch_labels


def _expected_keys(
    seed: int, sizes: list[int], labels: tuple[str, ...]
) -> set[tuple[int, int, str, str]]:
    return {
        (seed, n_train, label, test_set)
        for n_train in sizes
        for label in labels
        for test_set in TEST_SETS
    }


def _read_existing_records(
    log_path: Path,
) -> tuple[list[dict], set[tuple[int, int, str, str]], str | None]:
    """Read the existing log file, build the task key per record.

    Tolerates a truncated last line (abort in the middle of writing) -- it is
    discarded, not treated as an error (the associated seed then counts as
    incomplete via ``_expected_keys`` and is recomputed). A truncated
    line that is NOT the last one, by contrast, is a real file defect and aborts.

    Returns: (records, set of task keys, first found ``git/commit`` with
    ``git/commit_available=True``, ``None`` if there is none).
    """
    if not log_path.exists():
        return [], set(), None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    records: list[dict] = []
    keys: set[tuple[int, int, str, str]] = set()
    commit: str | None = None
    for index, raw_line in enumerate(lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(
                    f"  WARNUNG: letzte Zeile in {log_path} ist unvollstaendig (Abbruch "
                    "mitten im Schreiben) -- verworfen, der zugehoerige Seed wird als "
                    "unvollstaendig behandelt."
                )
                continue
            raise ValueError(
                f"{log_path}: Zeile {index + 1} ist kein gueltiges JSON und ist NICHT die "
                "letzte Zeile -- echter Datei-Defekt, kein abgebrochener Schreibvorgang. "
                "Abbruch."
            ) from None
        records.append(record)
        params = record["params"]
        keys.add(
            (
                int(params["master_seed"]),
                int(params["n_train"]),
                str(params["model/label"]),
                str(params["test_set"]),
            )
        )
        if commit is None and params.get("git/commit_available"):
            commit = params.get("git/commit")
    return records, keys, commit


def _worker_paths(work_dir: Path, seed: int) -> tuple[Path, Path]:
    output = work_dir / f"_worker_seed{seed}.jsonl"
    log = work_dir / f"_worker_seed{seed}.log"
    return output, log


def _launch(
    python: str,
    seed: int,
    *,
    config: Path,
    checkpoint_dir: Path,
    work_dir: Path,
    num_threads: int,
    sizes: list[int],
    architectures: list[str] | None,
    skip_baselines: bool,
    net_max_epochs: int,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    output, log_path = _worker_paths(work_dir, seed)
    args = [
        python,
        str(WORKER_SCRIPT),
        "--seed",
        str(seed),
        "--config",
        str(config),
        "--output",
        str(output),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--num-threads",
        str(num_threads),
        "--sizes",
        *[str(size) for size in sizes],
        "--net-max-epochs",
        str(net_max_epochs),
    ]
    if architectures is not None:
        args += ["--architectures", *architectures]
    if skip_baselines:
        args.append("--skip-baselines")
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        args, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=env
    )
    return process, output, log_path


def _setup_marker_present(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return "Setup fertig" in log_path.read_text(encoding="utf-8", errors="ignore")


def _run_batch(
    seeds: list[int],
    *,
    python: str,
    config: Path,
    checkpoint_dir: Path,
    work_dir: Path,
    num_threads: int,
    sizes: list[int],
    architectures: list[str] | None,
    skip_baselines: bool,
    net_max_epochs: int,
    stagger_first: int,
    stagger_wait_s: float,
) -> list[tuple[int, int, Path]]:
    """Start one batch -- optionally staggered -- and wait for all to finish.

    Returns per job: ``(seed, returncode, output_path)``.
    """
    handles: list[tuple[subprocess.Popen[bytes], int, Path]] = []

    staggered = seeds[:stagger_first] if stagger_first else []
    rest = seeds[stagger_first:] if stagger_first else seeds

    for seed in staggered:
        process, output, _log_path = _launch(
            python,
            seed,
            config=config,
            checkpoint_dir=checkpoint_dir,
            work_dir=work_dir,
            num_threads=num_threads,
            sizes=sizes,
            architectures=architectures,
            skip_baselines=skip_baselines,
            net_max_epochs=net_max_epochs,
        )
        handles.append((process, seed, output))
        print(f"  gestartet (gestaffelt): seed={seed}, PID {process.pid}")

    if staggered:
        print(f"  warte {stagger_wait_s:.0f}s, pruefe die {len(staggered)} gestaffelten Jobs ...")
        time.sleep(stagger_wait_s)
        for process, seed, _output in handles:
            log_path = _worker_paths(work_dir, seed)[1]
            alive = process.poll() is None
            marker = _setup_marker_present(log_path)
            if alive and marker:
                status = "laeuft, Setup bestaetigt"
            elif alive:
                status = "laeuft, Setup noch nicht bestaetigt"
            else:
                status = f"BEENDET (returncode={process.returncode})"
            print(f"    seed={seed}: {status}")
            if not alive and process.returncode != 0:
                raise RuntimeError(
                    f"Gestaffelter Job seed={seed} ist mit Fehler beendet "
                    f"(returncode={process.returncode}) -- Log: {log_path}. Abbruch vor dem "
                    "Start der restlichen Jobs."
                )
            if alive and not marker:
                print(
                    f"    WARNUNG: seed={seed} laeuft, aber der 'Setup fertig'-Marker fehlt "
                    "noch nach der Wartezeit -- ungewoehnlich langsame Datengenerierung oder "
                    "ein Problem. Nicht automatisch abgebrochen, aber pruefen."
                )

    for seed in rest:
        process, output, _log_path = _launch(
            python,
            seed,
            config=config,
            checkpoint_dir=checkpoint_dir,
            work_dir=work_dir,
            num_threads=num_threads,
            sizes=sizes,
            architectures=architectures,
            skip_baselines=skip_baselines,
            net_max_epochs=net_max_epochs,
        )
        handles.append((process, seed, output))
        print(f"  gestartet: seed={seed}, PID {process.pid}")

    results: list[tuple[int, int, Path]] = []
    for process, seed, output in handles:
        returncode = process.wait()
        results.append((seed, returncode, output))
        status = "OK" if returncode == 0 else f"FEHLER (returncode={returncode})"
        print(f"  fertig: seed={seed}: {status}")
    return results


def _merge(outputs: list[Path], final_path: Path) -> None:
    """Merge all job JSONL files, with duplicate-run_uid check."""
    seen_run_uids: dict[str, Path] = {}
    lines: list[str] = []
    for output in outputs:
        if not output.exists():
            raise FileNotFoundError(f"Erwartete Job-Ausgabe fehlt: {output}")
        with output.open(encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                record = json.loads(raw_line)
                run_uid = record["run_uid"]
                if run_uid in seen_run_uids:
                    raise ValueError(
                        f"Doppelte run_uid {run_uid!r} in {output} und "
                        f"{seen_run_uids[run_uid]} -- Verteilungslogik-Fehler, kein Race. "
                        "Merge abgebrochen."
                    )
                seen_run_uids[run_uid] = output
                lines.append(raw_line)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    print(f"Merge fertig: {len(lines)} Records aus {len(outputs)} Job-Dateien -> {final_path}")


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=_load_worker_module().SELECTABLE_LABELS,
        default=None,
        help="An jeden Worker durchgereicht (Default: alle -- siehe "
        "run_arch_learning_curve_worker.py --architectures). Bewusst NICHT vorbelegt mit "
        "der vollen Liste, damit --resume die Auswahl korrekt in den Auftragsschluessel "
        "einrechnet (siehe _expected_labels).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="An jeden Worker durchgereicht -- persistence/static/arx nicht erneut "
        "rechnen (z. B. wenn diese Records unter demselben Protokoll bereits in einem "
        "anderen Datensatz vorliegen).",
    )
    parser.add_argument(
        "--net-max-epochs",
        type=int,
        default=200,
        help="An jeden Worker durchgereicht (siehe run_arch_learning_curve_worker.py "
        "--net-max-epochs). Default 200 -- unveraendertes Protokoll.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_projected.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "_arch_learning_curve_work",
    )
    parser.add_argument("--stagger-first", type=int, default=2)
    parser.add_argument("--stagger-wait-s", type=float, default=30.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bereits vollstaendige Seeds (alle erwarteten Kombinationen in --log vorhanden) "
        "ueberspringen (Default: an). --no-resume ignoriert eine bestehende --log-Datei "
        "vollstaendig und rechnet alle --seeds neu (bisheriges Verhalten).",
    )
    parser.add_argument(
        "--allow-commit-mismatch",
        action="store_true",
        help="Fortsetzen erlauben, obwohl die uebernommenen Records einen anderen "
        "git/commit tragen als der aktuelle HEAD (sonst harter Abbruch, siehe "
        "Moduldocstring, Provenienz-Wache).",
    )
    return parser.parse_args()


def _check_provenance_for_resume(existing_commit: str | None, *, allow_mismatch: bool) -> None:
    """Catch a dirty working tree and commit drift hard, before anything is computed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirtyWorkingTreeWarning)
        fingerprint = environment_fingerprint()
    if fingerprint["git/dirty"]:
        raise RuntimeError(
            "Arbeitsbaum ist dirty -- Abbruch vor dem Sweep (gilt auch fuer einen frischen "
            "Lauf, nicht nur beim Fortsetzen: alle Skripte muessen committet sein, siehe "
            "Auftragsvorgabe)."
        )
    current_commit = fingerprint["git/commit"]
    if existing_commit is not None and existing_commit != current_commit and not allow_mismatch:
        raise RuntimeError(
            f"Bestehende Records in --log tragen git/commit={existing_commit!r}, aktueller "
            f"HEAD ist {current_commit!r} -- Fortsetzen wuerde eine Lernkurve aus zwei "
            "verschiedenen Codestaenden zusammenmischen, deren Punkte nicht vergleichbar "
            "sind. Mit --allow-commit-mismatch explizit erzwingen, falls das hier "
            "ausdruecklich gewollt ist."
        )


def main() -> None:
    """Run the sweep: skip seeds that are already complete, start the rest, merge."""
    arguments = parse_arguments()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if arguments.resume:
        existing_records, existing_keys, existing_commit = _read_existing_records(arguments.log)
    else:
        existing_records, existing_keys, existing_commit = [], set(), None

    _check_provenance_for_resume(existing_commit, allow_mismatch=arguments.allow_commit_mismatch)

    labels = _expected_labels(arguments.architectures, skip_baselines=arguments.skip_baselines)
    seeds = arguments.seeds
    complete_seeds = [
        seed for seed in seeds if _expected_keys(seed, arguments.sizes, labels) <= existing_keys
    ]
    seeds_to_run = [seed for seed in seeds if seed not in complete_seeds]
    kept_records = [
        record
        for record in existing_records
        if int(record["params"]["master_seed"]) in complete_seeds
    ]

    if complete_seeds:
        print(
            f"Fortsetzung: {len(complete_seeds)} von {len(seeds)} Seeds bereits vollstaendig "
            f"in {arguments.log} vorhanden, uebersprungen: {complete_seeds}"
        )
    if not seeds_to_run:
        print("Alle angeforderten Seeds bereits vollstaendig -- kein neuer Job noetig.")

    kept_path = arguments.work_dir / "_resume_kept.jsonl"
    if kept_records:
        with kept_path.open("w", encoding="utf-8") as handle:
            for record in kept_records:
                handle.write(json.dumps(record) + "\n")

    batches = [
        seeds_to_run[i : i + arguments.parallel]
        for i in range(0, len(seeds_to_run), arguments.parallel)
    ]

    if batches:
        print(
            f"{len(seeds_to_run)} Jobs (Seeds) in {len(batches)} Batches "
            f"(Parallelitaet {arguments.parallel})"
        )
    start = time.perf_counter()
    all_results: list[tuple[int, int, Path]] = []
    for index, batch in enumerate(batches):
        print(f"\n=== Batch {index + 1}/{len(batches)} ({len(batch)} Jobs) ===")
        stagger = arguments.stagger_first if index == 0 else 0
        results = _run_batch(
            batch,
            python=python,
            config=arguments.config,
            checkpoint_dir=arguments.checkpoint_dir,
            work_dir=arguments.work_dir,
            num_threads=arguments.num_threads,
            sizes=arguments.sizes,
            architectures=arguments.architectures,
            skip_baselines=arguments.skip_baselines,
            net_max_epochs=arguments.net_max_epochs,
            stagger_first=stagger,
            stagger_wait_s=arguments.stagger_wait_s,
        )
        all_results.extend(results)

    elapsed = time.perf_counter() - start
    failed = [r for r in all_results if r[1] != 0]
    print(f"\nGesamtzeit: {elapsed:.1f}s ({elapsed / 60.0:.1f} Min)")
    if failed:
        print(f"{len(failed)} Jobs fehlgeschlagen:")
        for seed, returncode, _ in failed:
            print(f"  seed={seed}: returncode={returncode}")
        raise SystemExit(1)

    outputs = ([kept_path] if kept_records else []) + [output for _, _, output in all_results]
    _merge(outputs, arguments.log)


if __name__ == "__main__":
    main()
