r"""Orchestriert den KONSTANT trainierten Lernkurven-Sweep: 5 Seeds, je 4 Budgetpunkte x 5
Architekturen, parallel.

Identisch zu ``run_arch_learning_curve_sweep.py``, nur Worker-Skript und Default-``--log``
ausgetauscht -- startet ``run_arch_constant_trained_learning_curve_worker.py`` statt
``run_arch_learning_curve_worker.py`` und schreibt nach
``artifacts/runs/arch_learning_curve_constant_trained.jsonl``, NICHT in den bestehenden
``arch_learning_curve_projected.jsonl``-Datensatz.

``MKL_NUM_THREADS``/``OMP_NUM_THREADS=1`` werden -- wie im Zwillingsskript -- als PROZESS-
Umgebungsvariablen an jeden Worker weitergegeben, unabhaengig von ``--num-threads``.
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
WORKER_SCRIPT = (
    Path(__file__).resolve().parent / "run_arch_constant_trained_learning_curve_worker.py"
)
SEEDS = (1, 2, 3, 4, 5)


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
    ]
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
            sizes=sizes,
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
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "arch_learning_curve_constant_trained.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "artifacts" / "checkpoints"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs" / "_arch_constant_trained_learning_curve_work",
    )
    parser.add_argument("--stagger-first", type=int, default=2)
    parser.add_argument("--stagger-wait-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    """Sweep fahren: Batches starten, warten, zusammenfuehren."""
    arguments = parse_arguments()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    seeds = arguments.seeds
    batches = [seeds[i : i + arguments.parallel] for i in range(0, len(seeds), arguments.parallel)]

    print(
        f"{len(seeds)} Jobs (Seeds) in {len(batches)} Batches (Parallelitaet {arguments.parallel})"
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

    outputs = [output for _, _, output in all_results]
    _merge(outputs, arguments.log)


if __name__ == "__main__":
    main()
