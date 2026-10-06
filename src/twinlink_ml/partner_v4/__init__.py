"""Reconstruction of the partner's workbook ``Model_v4.xlsx`` (two dryers, heat recovery, gas).

Separate from :mod:`twinlink_ml.heatnetwork`, whose structure (one cascade, shared cold
temperature) does not match the partner's v4 layout. No comparison with the lab model.

* :mod:`.inputs` -- sheet ``Inputs`` in sheet units and in SI,
* :mod:`.as_built` -- the workbook's formulas reproduced as they are (bugs included),
* :mod:`.corrected` -- the same model in SI with consistent units,
* :mod:`.cascade` -- the six scenario cases as a corrected cascade (our reconstruction),
* :mod:`.deviations` -- the list of deviations between the workbook and the rebuild.
"""
