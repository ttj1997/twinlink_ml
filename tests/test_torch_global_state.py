"""Beweist, dass die autouse-Fixture ``_restore_torch_global_state`` das Leck eindämmt.

Torch-dtype-Leck aus ``test_greybox_numerics.py``, Fixture in conftest.py. Reihenfolge
ist hier Teil des Tests, nicht nur Zufall: pytest sammelt Testfunktionen
innerhalb einer Datei in Definitionsreihenfolge (kein Ordering-Plugin nötig). Die zweite
Funktion unten läuft deshalb garantiert direkt nach der ersten -- genau die Reihenfolge,
unter der das Leck ursprünglich reproduzierbar war (ein Test setzt global, der nächste
sieht den unrückgängig gemachten Zustand).
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_default_dtype_leak_reproduced() -> None:
    """Bildet das Leckverhalten nach: setzt den Default-Dtype, setzt ihn NICHT zurück.

    Absichtlich kein try/finally hier -- genau das Fehlen davon ist der reproduzierte
    Fehler aus den beiden betroffenen Gradiententests. Ob die Fixture ihn trotzdem
    eindämmt, prüft die nächste Funktion in dieser Datei.
    """
    import torch

    torch.set_default_dtype(torch.float64)
    assert torch.get_default_dtype() == torch.float64  # Leck tatsächlich aktiv


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_default_dtype_restored_after_leaking_test() -> None:
    """Muss unmittelbar NACH ``test_torch_default_dtype_leak_reproduced`` stehen (s. o.).

    Läuft garantiert direkt nach der leckenden Funktion oben (Definitionsreihenfolge).
    Schlägt dieser Test fehl, ist entweder die autouse-Fixture in conftest.py kaputt
    oder die beiden Funktionen wurden versehentlich umsortiert.
    """
    import torch

    assert torch.get_default_dtype() == torch.float32
