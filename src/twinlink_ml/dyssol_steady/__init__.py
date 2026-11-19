"""Stationary Dyssol flowsheets as a data source for surrogate models.

Design and reading: ``docs/NN_dyssol_steady_state.md``.

* :mod:`.materials` -- the materials database the units need (one added constant) and the
  properties the rebuild reads from it;
* :mod:`.reference` -- Python rebuild of the three stationary units (dryer, heater, heat recovery)
  and of the flowsheets built from them;
* :mod:`.flowsheet` -- topology of a ``.dflw`` file, script generation and the runner of the
  command-line version of Dyssol;
* :mod:`.design` -- box, Sobol design, nested training subsets;
* :mod:`.models` -- linear and quadratic regression, MLP;
* :mod:`.study` -- learning curve, metrics, pre-registered reading.

Plant values, the box and every generated data set stay outside the repository (``local/``).
"""
