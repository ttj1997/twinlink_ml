"""Reconstruction of the partner's workbook ``Model_v4.1.xlsx`` (heat-transfer-medium circuits).

Own package next to :mod:`twinlink_ml.partner_v4`, which stays unchanged and describes
``Model_v4.xlsx``. The new version replaces the open ``#REF!`` cells of v4 by two circuits of
a heat transfer medium (equations 4.1/4.2 of the partner's equation document) and has
different dryers; the structure (case drawings of the document) is the one assumed as A1 in
v4. Only the stable primitives of ``partner_v4`` (dryer balance, streams, dew point) are
imported from there.

* :mod:`.inputs` -- sheet ``Inputs`` incl. the mass flows of the two circuits,
* :mod:`.loops` -- the equations of the document (2.3/2.4, 3.2, 4.1/4.2) in SI, plus the
  counter-current bound used to document the equal-outlet-temperature assumption,
* :mod:`.cascade` -- the six cases of the sheet ``Scenarios`` from the document's equations,
* :mod:`.layout` -- where each quantity sits in the sheet (cell addresses),
* :mod:`.as_built` -- the sheet's formulas as they are (bugs included),
* :mod:`.deviations` -- three-way comparison document / workbook / rebuild per cell.
"""
