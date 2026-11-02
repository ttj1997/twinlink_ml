r"""Regression test: ``ProcessConfig.fingerprint`` stays stable across the hash refactor.

Extraction of :func:`config_fingerprint` from the property, see
``twinlink_ml.simulation.config``. The committed learning-curve data set
(``artifacts/runs/arch_learning_curve_projected.jsonl``, Befund 51) carries these
fingerprints, and a refactor of the hash mechanics must not move them. This test
was missing and would have secured the refactor.

Values here are the CURRENT ones (after the pseudonymisation cut 2026-09-25, which
excluded ``provenance.open_todos`` from the hash, see ``without_documentary_provenance``
and ``twinlink_ml.utils.fingerprint_history``). The records logged before the cut
carry the OLDER values, entered there as equivalent -- this test checks only the
current config, not the old records.
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
