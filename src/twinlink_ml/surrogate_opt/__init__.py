"""Surrogate-based steady RTO with few true evaluations (docs/25_surrogate_rto.md).

How many evaluations of the true steady model does a surrogate need so that optimising on it hits
the true energy optimum feasibly? Two true models (:mod:`.line_problem` P1, :mod:`.greybox_problem`
P2) behind the :class:`~.problem.SteadyProblem` ABC; black-box surrogates M1-M3 behind the
:class:`~.surrogates.Surrogate` ABC; the grey-box fit M4 on a :class:`~.problem.ProblemFamily`;
designs, reference optima, arms and scoring in :mod:`.study`; the optimiser in :mod:`.optimize`;
a quadratic test problem with a known optimum in :mod:`.quadratic`.

Optional dependencies: scikit-learn (extra ``surrogate``) for M2/M3, CasADi (extra
``optimal-control``) for P2.
"""

from .problem import Box, ConstraintSpec, Evaluation, ParameterSpec, ProblemFamily, SteadyProblem

__all__ = ["Box", "ConstraintSpec", "Evaluation", "ParameterSpec", "ProblemFamily", "SteadyProblem"]
