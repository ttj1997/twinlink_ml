r"""Orchestriert den Konstantlauf-Holdout-Sweep: GRU/LSTM, 10 Seeds, parallel.

Startet ``run_arch_constant_holdout_worker.py`` als unabhängige Prozesse -- ein Job je
(Architektur, Seed)-Kombination, 20 insgesamt. Kein Hyperparameter, kein Seed wird dadurch
verändert; nur die Wanduhrzeit sinkt (siehe Thread-Skalierungsmessung, 04_stand_ap1.md).

Gestaffelter Start (nur der erste Batch)
-----------------------------------------
Netzlaufwerk-Risiko: acht Prozesse, die gleichzeitig Checkpoints und JSONL-Teil-Dateien auf
eine SMB-Freigabe schreiben, sind eine andere Belastungsklasse als der Ein-Prozess-Thread-
Benchmark, der die Parallelitätsentscheidung begründet hat. Deshalb starten im ersten Batch
zunächst zwei Jobs, kurze Wartezeit, Prüfung (Prozess lebt noch, Log-Datei zeigt den
"Setup fertig"-Marker aus dem Worker -- die Phase, die Konfigurations-/Netzlaufwerkfehler am
ehesten zeigen würde), erst dann die restlichen sechs. Kostet keine zusätzliche Rechenzeit --
es sind ohnehin reale Sweep-Läufe, nur zeitversetzt gestartet statt blind alle acht auf
einmal.

Zusammenführen
---------------
Jeder Job schreibt in eine eigene JSONL-Datei (kein gemeinsames Schreiben mehrerer Prozesse
in dieselbe Datei). Am Ende werden alle Dateien zusammengeführt; dabei wird geprüft, dass
keine ``run_uid`` doppelt vorkommt -- ein Duplikat wäre ein Zeichen für einen
Verteilungslogik-Fehler (zwei Jobs mit derselben Architektur/demselben Seed), kein
harmloses Race, und bricht den Merge ab statt es zu übergehen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = Path(__file__).resolve().parent / "run_arch_constant_holdout_worker.py"
SEEDS = tuple(range(1, 11))
ARCHITECTURES = ("gru", "lstm")


def _jobs() -> list[tuple[str, int, bool]]:
    """(Architektur, Seed, mit_Baselines) je Job -- Baselines nur einmal je Seed."""
    return [
        (architecture, seed, architecture == ARCHITECTURES[0])
        for seed in SEEDS
        for architecture in ARCHITECTURES
    ]


def _worker_paths(work_dir: Path, architecture: str, seed: int) -> tuple[Path, Path]:
    output = work_dir / f"_worker_{architecture}_{seed}.jsonl"
    log = work_dir / f"_worker_{architecture}_{seed}.log"
    return output, log


def _launch(
    python: str,
    architecture: str,
    seed: int,
    with_baselines: bool,
    *,
    config: Path,
    checkpoint_dir: Path,
    work_dir: Path,
    num_threads: int,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    output, log_path = _worker_paths(work_dir, architecture, seed)
    args = [
        python,
        str(WORKER_SCRIPT),
        "--architecture",
        architecture,
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
    ]
    if with_baselines:
        args.append("--with-baselines")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
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
    jobs: list[tuple[str, int, bool]],
    *,
    python: str,
    config: Path,
    checkpoint_dir: Path,
    work_dir: Path,
    num_threads: int,
    stagger_first: int,
    stagger_wait_s: float,
) -> list[tuple[str, int, int, Path]]:
    """Einen Batch starten -- optional gestaffelt -- und auf Abschluss aller warten.

    Rückgabe je Job: ``(architecture, seed, returncode, output_path)``.
    """
    handles: list[tuple[subprocess.Popen[bytes], str, int, Path]] = []

    staggered = jobs[:stagger_first] if stagger_first else []
    rest = jobs[stagger_first:] if stagger_first else jobs

    for architecture, seed, with_baselines in staggered:
        process, output, log_path = _launch(
            python,
            architecture,
            seed,
            with_baselines,
            config=config,
            checkpoint_dir=checkpoint_dir,
            work_dir=work_dir,
            num_threads=num_threads,
        )
        handles.append((process, architecture, seed, output))
        print(f"  gestartet (gestaffelt): {architecture} seed={seed}, PID {process.pid}")

    if staggered:
        print(f"  warte {stagger_wait_s:.0f}s, pruefe die {len(staggered)} gestaffelten Jobs ...")
        time.sleep(stagger_wait_s)
        for process, architecture, seed, _output in handles:
            log_path = _worker_paths(work_dir, architecture, seed)[1]
            alive = process.poll() is None
            marker = _setup_marker_present(log_path)
            if alive and marker:
                status = "laeuft, Setup bestaetigt"
            elif alive:
                status = "laeuft, Setup noch nicht bestaetigt"
            else:
                status = f"BEENDET (returncode={process.returncode})"
            print(f"    {architecture} seed={seed}: {status}")
            if not alive and process.returncode != 0:
                raise RuntimeError(
                    f"Gestaffelter Job {architecture} seed={seed} ist mit Fehler beendet "
                    f"(returncode={process.returncode}) -- Log: {log_path}. Abbruch vor dem "
                    "Start der restlichen Jobs."
                )
            if alive and not marker:
                print(
                    f"    WARNUNG: {architecture} seed={seed} laeuft, aber der "
                    "'Setup fertig'-Marker fehlt noch nach der Wartezeit -- ungewoehnlich "
                    "langsame Datengenerierung oder ein Problem. Nicht automatisch "
                    "abgebrochen, aber pruefen."
                )

    for architecture, seed, with_baselines in rest:
        process, output, log_path = _launch(
            python,
            architecture,
            seed,
            with_baselines,
            config=config,
            checkpoint_dir=checkpoint_dir,
            work_dir=work_dir,
            num_threads=num_threads,
        )
        handles.append((process, architecture, seed, output))
        print(f"  gestartet: {architecture} seed={seed}, PID {process.pid}")

    results: list[tuple[str, int, int, Path]] = []
    for process, architecture, seed, output in handles:
        returncode = process.wait()
        results.append((architecture, seed, returncode, output))
        status = "OK" if returncode == 0 else f"FEHLER (returncode={returncode})"
        print(f"  fertig: {architecture} seed={seed}: {status}")
    return results


def _merge(outputs: list[Path], final_path: Path) -> None:
    """Alle Job-JSONL-Dateien zusammenführen, mit Duplikat-run_uid-Pruefung."""
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
    """Kommandozeile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml",
    )
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "arch_gab_constant_holdout.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "_arch_constant_holdout_work",
    )
    parser.add_argument("--stagger-first", type=int, default=2)
    parser.add_argument("--stagger-wait-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    """Sweep fahren: Batches starten, warten, zusammenführen."""
    arguments = parse_arguments()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    jobs = _jobs()
    batches = [jobs[i : i + arguments.parallel] for i in range(0, len(jobs), arguments.parallel)]

    print(f"{len(jobs)} Jobs in {len(batches)} Batches (Parallelitaet {arguments.parallel})")
    start = time.perf_counter()
    all_results: list[tuple[str, int, int, Path]] = []
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
            stagger_first=stagger,
            stagger_wait_s=arguments.stagger_wait_s,
        )
        all_results.extend(results)

    elapsed = time.perf_counter() - start
    failed = [r for r in all_results if r[2] != 0]
    print(f"\nGesamtzeit: {elapsed:.1f}s ({elapsed / 60.0:.1f} Min)")
    if failed:
        print(f"{len(failed)} Jobs fehlgeschlagen:")
        for architecture, seed, returncode, _ in failed:
            print(f"  {architecture} seed={seed}: returncode={returncode}")
        raise SystemExit(1)

    outputs = [output for _, _, _, output in all_results]
    _merge(outputs, arguments.log)


if __name__ == "__main__":
    main()
