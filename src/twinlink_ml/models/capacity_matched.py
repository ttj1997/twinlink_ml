"""Capacity-matched variants for the xLSTM / Mamba comparison (Befund 67, SX3).

Question: are xLSTM (8 649 parameters) and Mamba (8 139) behind LSTM (57 027) and GRU (42 819)
because of the architecture or because of their size? Every configuration below is fixed
**a priori** (before any training run) so that the three-head composite model has between
43 000 and 57 000 parameters, the range spanned by GRU and LSTM. Only width, depth and head
count are set; learning rate, weight decay, epochs, patience, batch size and clipping stay at the
:class:`TorchSequenceConfig` defaults used in Befund 51/58. The counts are the exact ones of the
built networks (7 exogenous inputs + 1 target per head, three heads):

===================  ==========================================================  ==========
label                configuration                                               parameters
===================  ==========================================================  ==========
``mlstm_cm``         mLSTM only, hidden 58, 2 layers                             50 301
``slstm_cm``         sLSTM only, hidden 48, 4 heads, 2 layers                    47 955
``slstm_l1_cm``      sLSTM only, hidden 112, 4 heads, 1 layer                   50 739
``mixed_cm``         mLSTM layer + sLSTM layer, hidden 52, 4 heads               48 525
``mamba_cm``         Mamba, hidden 64, state 16, expand 2, 1 layer               53 211
``lstm_small``       LSTM, hidden 22 (control at the size of xLSTM/Mamba)         8 517
===================  ==========================================================  ==========

Why depth 2 for the mLSTM: the mLSTM cell has no recurrent matrix and only 8 input channels, so
a single layer needs a hidden size of several hundred for 50 000 parameters (the earlier
attempt with hidden size about 317, ``scripts/study_xlstm_wide.py``, is reported as having cost
more than 12 h per training point); two layers of hidden 58 reach the target with a matrix
state of 2 x 58^2 instead of 64^2 entries per sample. The
sLSTM-only variant is therefore also given with two layers (``slstm_cm``, depth-matched with
``mlstm_cm`` and ``mixed_cm``); ``slstm_l1_cm`` is the single-layer counterpart.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from .torch_sequence import TorchSequenceConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["CAPACITY_MATCHED_CONFIGS", "CAPACITY_MATCHED_PARAMETERS"]

CAPACITY_MATCHED_CONFIGS: Mapping[str, TorchSequenceConfig] = MappingProxyType(
    {
        "mlstm_cm": TorchSequenceConfig(core="xlstm", hidden_size=58, num_layers=2),
        "slstm_cm": TorchSequenceConfig(core="slstm", hidden_size=48, num_heads=4, num_layers=2),
        "slstm_l1_cm": TorchSequenceConfig(
            core="slstm", hidden_size=112, num_heads=4, num_layers=1
        ),
        "mixed_cm": TorchSequenceConfig(
            core="xlstm_mixed", hidden_size=52, num_heads=4, num_layers=2
        ),
        "mamba_cm": TorchSequenceConfig(core="mamba", hidden_size=64, state_size=16, num_layers=1),
        "lstm_small": TorchSequenceConfig(core="lstm", hidden_size=22, num_layers=1),
    }
)

CAPACITY_MATCHED_PARAMETERS: Mapping[str, int] = MappingProxyType(
    {
        "mlstm_cm": 50301,
        "slstm_cm": 47955,
        "slstm_l1_cm": 50739,
        "mixed_cm": 48525,
        "mamba_cm": 53211,
        "lstm_small": 8517,
    }
)
"""Composite parameter counts (three heads) of the configurations above; checked by a test."""
