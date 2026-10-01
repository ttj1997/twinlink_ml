r"""Orchestriert den xLSTM-Konstantlauf-Holdout-Sweep: 10 Seeds, parallel.

Startet ``run_xlstm_constant_holdout_worker.py`` als unabhaengige Prozesse -- ein Job je
Seed, 10 insgesamt (Baselines/GRU/LSTM liegen bereits vor, Befund 36). Kein
Hyperparameter, kein Seed wird dadurch veraendert; nur die Wanduhrzeit sinkt. Struktur
bewusst identisch zu ``run_mamba_constant_holdout_sweep.py`` (Befund 40, selbst wiederum
nach ``run_arch_constant_holdout_sweep.py`` gebaut: gestaffelter Start,
Merge-mit-Duplikatpruefung) -- nur auf xLSTM statt Mamba umgestellt.

Gestaffelter Start (nur der erste Batch)
-----------------------------------------
Dieselbe Begruendung wie im GRU/LSTM- und Mamba-Sweep: mehrere Prozesse, die
gleichzeitig Checkpoints und JSONL-Teil-Dateien schreiben, sind eine andere
Belastungsklasse als ein Einzelprozess-Benchmark. Erst zwei Jobs, kurze Wartezeit,
Pruefung (Prozess lebt noch, "Setup fertig"-Marker im Log), dann der Rest.

Zusammenfuehren
-----------------
Jeder Job schreibt in eine eigene JSONL-Datei; am Ende werden alle zusammengefuehrt, mit
Pruefung auf doppelte ``run_uid`` (Abbruch bei Duplikat, kein stillschweigendes
Uebergehen). Ziel-Datei bewusst NICHT ``arch_gab_constant_holdout.jsonl`` (die Befund-36-
Datei bleibt unangetastet) -- eigene Datei
``artifacts/runs/xlstm_gab_constant_holdout.jsonl``, die zusammen mit der Befund-36-Datei
fuer die Vergleichstabelle gelesen wird.
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
WORKER_SCRIPT = Path(__file__).resolve().parent / "run_xlstm_constant_holdout_worker.py"
SEEDS = tuple(range(1, 11))


def _parse_seeds(raw: str) -> list[int]:
    """``--seeds`` als Komma-Liste parsen, z. B. ``"2,3,4"``.

    Fuer Teil-Reruns (z. B. Seed 1 bereits als Einzelsonde gelaufen, nur 2..10 fehlen noch).
    """
    return [int(part) for part in raw.split(",") if part.strip()]


def _worker_paths(work_dir: Path, seed: int) -> tuple[Path, Path]:
    output = work_dir / f"_worker_xlstm_{seed}.jsonl"
    log = work_dir / f"_worker_xlstm_{seed}.log"
    return output, log


def _launch(
    python: str,
    seed: int,
    *,
    config: Path,
    checkpoint_dir: Path,
    work_dir: Path,
    num_threads: int,
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
    ]
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        # Bestaetigt (Diagnose in dieser Sitzung): torch.set_num_threads(1) im Worker
        # begrenzt nur ATen/PyTorchs eigenen Intraop-Pool, NICHT Intel MKL/OpenMP
        # darunter -- ohne diese Variablen liest MKL bei jedem Worker-Prozess
        # unabhaengig mkl_get_max_threads()=8 (unbelastet gemessen), was bei 8-9
        # gleichzeitigen Prozessen zu massiver Thread-Ueberzeichnung fuehrte (~7.2x
        # langsamer als der unbelastete Einzellauf). Muessen als echte
        # Prozessumgebungsvariablen VOR dem Interpreterstart gesetzt sein -- ein
        # `os.environ[...] = ...` im Worker-Skript selbst kaeme zu spaet, MKL liest
        # das beim Bibliotheks-Import.
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
    stagger_first: int,
    stagger_wait_s: float,
) -> list[tuple[int, int, Path]]:
    """Einen Batch starten -- optional gestaffelt -- und auf Abschluss aller warten.

    Rueckgabe je Job: ``(seed, returncode, output_path)``.
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
        )
        handles.append((process, seed, output))
        print(f"  gestartet (gestaffelt): xlstm seed={seed}, PID {process.pid}")

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
            print(f"    xlstm seed={seed}: {status}")
            if not alive and process.returncode != 0:
                raise RuntimeError(
                    f"Gestaffelter Job xlstm seed={seed} ist mit Fehler beendet "
                    f"(returncode={process.returncode}) -- Log: {log_path}. Abbruch vor dem "
                    "Start der restlichen Jobs."
                )
            if alive and not marker:
                print(
                    f"    WARNUNG: xlstm seed={seed} laeuft, aber der 'Setup fertig'-Marker "
                    "fehlt noch nach der Wartezeit -- ungewoehnlich langsame "
                    "Datengenerierung oder ein Problem. Nicht automatisch abgebrochen, "
                    "aber pruefen."
                )

    for seed in rest:
        process, output, _log_path = _launch(
            python,
            seed,
            config=config,
            checkpoint_dir=checkpoint_dir,
            work_dir=work_dir,
            num_threads=num_threads,
        )
        handles.append((process, seed, output))
        print(f"  gestartet: xlstm seed={seed}, PID {process.pid}")

    results: list[tuple[int, int, Path]] = []
    for process, seed, output in handles:
        returncode = process.wait()
        results.append((seed, returncode, output))
        status = "OK" if returncode == 0 else f"FEHLER (returncode={returncode})"
        print(f"  fertig: xlstm seed={seed}: {status}")
    return results


def _merge(outputs: list[Path], final_path: Path) -> None:
    """Alle Job-JSONL-Dateien zusammenfuehren, mit Duplikat-run_uid-Pruefung."""
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
        default=REPO_ROOT / "artifacts" / "runs" / "xlstm_gab_constant_holdout.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "_xlstm_constant_holdout_work",
    )
    parser.add_argument("--stagger-first", type=int, default=2)
    parser.add_argument("--stagger-wait-s", type=float, default=30.0)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Komma-Liste statt aller 10 Seeds, z. B. '2,3,4,5,6,7,8,9,10'",
    )
    return parser.parse_args()


def main() -> None:
    """Sweep fahren: Batches starten, warten, zusammenfuehren."""
    arguments = parse_arguments()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    seeds = _parse_seeds(arguments.seeds) if arguments.seeds is not None else list(SEEDS)
    batches = [seeds[i : i + arguments.parallel] for i in range(0, len(seeds), arguments.parallel)]

    print(f"{len(seeds)} Jobs in {len(batches)} Batches (Parallelitaet {arguments.parallel})")
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
            print(f"  xlstm seed={seed}: returncode={returncode}")
        raise SystemExit(1)

    outputs = [output for _, _, output in all_results]
    _merge(outputs, arguments.log)


if __name__ == "__main__":
    main()
