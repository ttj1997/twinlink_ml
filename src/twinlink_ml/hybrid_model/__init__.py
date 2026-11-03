"""Hybrid model (prior grey-box knowledge + learned residual) against a pure GRU with few runs.

Study docs/NN_hybrid_model.md; pre-registration in the docstring of
``scripts/analyze_hybrid_model.py``.
"""

from .steady_batch import BatchSteadyResult, solve_steady_states

__all__ = ["BatchSteadyResult", "solve_steady_states"]
