"""Static transfer study: surrogate of the dryer balance, pretrained on the partner balance.

Fine-tuned on few plant points and compared with calibration and a residual MLP
(docs/NN_static_transfer.md).

Modules: :mod:`.config` (settings), :mod:`.balance` (the partner balance in the intensive
formulation and the synthetic plants), :mod:`.design` (designs, test sets, input errors, noise),
:mod:`.surrogates` (methods L0-L4, B1, B2), :mod:`.study` (units), :mod:`.analysis` (paired
verdicts with Holm per hypothesis family).
"""
