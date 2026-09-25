r"""K1 (BA4): ``probe_l1_clip_sensitivity.py`` UNVERAENDERT aufgerufen, unter der neuen
Projektions-Config (``enforce_gas_velocity_projection=True``, Fingerprint
``bddca37f716f5cfc`` -- vor dem Pseudonymisierungs-Schnitt 2026-09-25: ``2d2679eeed716577``,
siehe ``docs/ARCHIV.md``).

Laedt ``probe_l1_clip_sensitivity.py`` per Dateipfad (``importlib.util``, nicht als
Paket-Import -- ``scripts/`` ist kein Package) und ersetzt ausschliesslich zwei Modul-Objekte
NACH dem Laden, keine Zeile der Originaldatei wird editiert:

* ``_load_config`` -- dieselbe Funktion, zusaetzlich mit
  ``numerics.enforce_gas_velocity_projection=True`` (die Originalfunktion setzt nur
  ``material.max_driving_humidity_kg_per_kg``, Numerik bleibt beim YAML-/Dataclass-Default
  AUS).
* ``REPO_ROOT`` -- auf ein Scratch-Verzeichnis umgebogen, NUR damit ``main()`` (das seinen
  Ausgabepfad intern aus dem Modul-Global ``REPO_ROOT`` neu zusammensetzt, nicht aus der
  ``OUT_PATH``-Konstante) das committete Original-Artefakt
  ``artifacts/runs/probe_l1_clip_sensitivity.json`` NICHT ueberschreibt. Nach dem Lauf wird
  die Ausgabe von dort nach ``artifacts/runs/probe_l1_clip_sensitivity_projected.json``
  kopiert.

``_fir_design``/``_violation_mask``/``_r2_and_variances``/``run_for_max_driving_humidity``/
``main`` laufen bitgleich zum Original -- exakt dieselbe FIR-Merkmalslogik und
Maskierungskriterien, nur die Prozess-Config unterscheidet sich in einem Feld.

Referenz (unprojiziert, Default AUS, committetes Original):
dR2(k=43) fuer ``product_moisture_kg_per_kg`` = +0.431. K1 erwartet unter Projektion grob
+0.2 (etwa Halbierung, siehe Auftrag/``docs/04_stand_ap1.md``).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from twinlink_ml.utils.fingerprint_history import fingerprints_equivalent

if TYPE_CHECKING:
    from twinlink_ml.simulation.config import ProcessConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = REPO_ROOT / "scripts" / "probe_l1_clip_sensitivity.py"
FINAL_OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "probe_l1_clip_sensitivity_projected.json"
EXPECTED_PROJECTED_FINGERPRINT = "bddca37f716f5cfc"


def _load_original_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "probe_l1_clip_sensitivity_original_unmodified", ORIGINAL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_original_module()
    original_load_config = module._load_config

    def _load_config_projected(max_driving_humidity: float) -> ProcessConfig:
        config = original_load_config(max_driving_humidity)
        return dataclasses.replace(
            config,
            numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True),
        )

    module._load_config = _load_config_projected

    # Verifikation der Projektion: `config.fingerprint`, wie im Original-Ergebnis geloggt,
    # ist der Fingerprint VOR generate_data()s sample_time_s-Ueberschreibung (5.0 s, YAML-
    # Default) -- das Original loggt diesen Wert nur informativ, prueft ihn nirgends gegen
    # eine Erwartung. Fuer BA0(c)/K1 zaehlt der TATSAECHLICH ANGEWENDETE Fingerprint (nach
    # sample_time_s=25.0, wie generate_data ihn intern bildet) -- hier separat nachgebildet,
    # nicht aus dem JSON gelesen.
    probe_config = _load_config_projected(0.50)
    applied_config = dataclasses.replace(
        probe_config,
        numerics=dataclasses.replace(probe_config.numerics, sample_time_s=25.0),
    )
    if not fingerprints_equivalent(applied_config.fingerprint, EXPECTED_PROJECTED_FINGERPRINT):
        raise RuntimeError(
            f"angewendeter process/fingerprint={applied_config.fingerprint!r} != erwartet "
            f"{EXPECTED_PROJECTED_FINGERPRINT!r} -- Abbruch, Ergebnis nicht uebernommen."
        )

    with tempfile.TemporaryDirectory() as scratch:
        scratch_root = Path(scratch)
        (scratch_root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
        module.REPO_ROOT = scratch_root
        module.main()
        scratch_out = scratch_root / "artifacts" / "runs" / "probe_l1_clip_sensitivity.json"
        if not scratch_out.exists():
            raise RuntimeError(f"erwartete Ausgabe des Original-main() fehlt: {scratch_out}")

        payload = json.loads(scratch_out.read_text(encoding="utf-8"))
        FINAL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scratch_out, FINAL_OUT_PATH)

    moisture = next(
        entry
        for entry in payload["results"]
        if entry["max_driving_humidity_kg_per_kg"] == 0.50
    )["channels"]["product_moisture_kg_per_kg"]
    print(
        "\n=== K1 (projiziert, max_driving_humidity=0.50, Default) ==="
        f"\ndR2(k=43) masked-unmasked, product_moisture_kg_per_kg: "
        f"{moisture['k43_r2_holdout_diff_masked_minus_unmasked']:+.4f}"
        f"\n(Referenz unprojiziert: +0.431; Erwartung K1: grob +0.2)"
    )
    print(f"kopiert nach: {FINAL_OUT_PATH}")


if __name__ == "__main__":
    main()
