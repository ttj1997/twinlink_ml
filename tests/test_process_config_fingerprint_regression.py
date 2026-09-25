r"""Regressionstest: ``ProcessConfig.fingerprint`` bleibt stabil ueber den Hash-Refactor.

Extraktion von :func:`config_fingerprint` aus der Property, siehe
``twinlink_ml.simulation.config``. Der committete Lernkurvendatensatz
(``artifacts/runs/arch_learning_curve_projected.jsonl``, Befund 51) traegt diese
Fingerprints, und ein Refactor der Hash-Mechanik darf sie nicht bewegen. Dieser Test
fehlte und haette den Refactor abgesichert.

Werte hier sind die AKTUELLEN (nach dem Pseudonymisierungs-Schnitt 2026-09-25, der
``provenance.open_todos`` aus dem Hash ausschloss, siehe ``without_documentary_provenance``
und ``twinlink_ml.utils.fingerprint_history``). Die vor dem Schnitt geloggten Records
tragen die AELTEREN, dort als aequivalent eingetragenen Werte -- dieser Test prueft nur die
aktuelle Config, nicht die alten Records.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"

FINGERPRINT_PROJECTION_OFF = "6ecedc6bd993fb3d"
FINGERPRINT_PROJECTION_ON = "bddca37f716f5cfc"


def _load_config(*, projection_active: bool) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_PATH)
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(
            config.numerics,
            sample_time_s=25.0,
            enforce_gas_velocity_projection=projection_active,
        ),
    )


def test_greybox_gab_placeholder_fingerprint_projection_off_is_unchanged() -> None:
    config = _load_config(projection_active=False)
    assert config.fingerprint == FINGERPRINT_PROJECTION_OFF


def test_greybox_gab_placeholder_fingerprint_projection_on_is_unchanged() -> None:
    config = _load_config(projection_active=True)
    assert config.fingerprint == FINGERPRINT_PROJECTION_ON
