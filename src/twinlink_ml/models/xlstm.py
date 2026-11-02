r"""mLSTM core after Beck et al. (2024), arXiv:2405.04517 -- the ``xlstm`` slot.

**Only the mLSTM block, no sLSTM.** In the paper's notation this is xLSTM[1:0]
(exclusively mLSTM branches, none of the scalar-memory sLSTM branches). The
work plan (``05_arbeitsplan_ap1.md``) specifies: "mLSTM-heavy first, sLSTM only if
needed". Whoever expects an sLSTM part under the label ``"xlstm"`` finds
none here -- this is intentional, not a gap still to be closed.

**Only the cell, no block.** Just as ``nn.GRU``/``nn.LSTM`` in
:mod:`twinlink_ml.models.torch_sequence` are raw recurrent cells (no
encoder-decoder, no residual backbone), :class:`XLSTMCore` (factory
:func:`build_xlstm_core_class`) implements only the mLSTM **recurrence** -- cell state, normaliser,
gates. What additionally exists in the official xLSTM block (up/down projection,
causal short convolution before query/key, several heads, block residual with pre-LayerNorm) is
**not** included here. This is consistent with the existing abstraction level of this
module (``core=`` is a recurrent core, not a block) and not a truncated copy of a
block whose parts are missing.

Where the equations come from and how certain that is
-----------------------------------------------------
The paper itself could not be retrieved with its full set of equations via the tools available
in this session (the HTML version on arXiv breaks off at the mLSTM part). The
equations implemented here are instead checked against the **official reference
implementation** (``NX-AI/xlstm``, files ``xlstm/blocks/mlstm/cell.py`` and
``xlstm/blocks/mlstm/backends.py``, function ``recurrent_step_stabilized_simple``,
retrieved via the source code on GitHub). This is a different kind of source than the
paper itself -- code instead of equation numbers --, but one that actually runs
and is maintained by the original authors. Adopted from this reference:

* The recurrence for cell state ``C_t``, normaliser ``n_t`` and max stabiliser
  ``m_t`` (see :meth:`_XLSTMLayer.step`), including the order in the
  outer product (``k`` first, ``v`` second -- this determines which index is
  contracted with ``q`` on readout).
* The scaling of ``k`` by :math:`1/\sqrt{d}` (not of ``q``).
* The gate pre-activations from ``concat([q, k, v])`` instead of directly from ``x_t`` --
  a detail that is in the cell class itself (``self.igate``/``self.fgate`` with
  ``in_features = 3 * embedding_dim``), not in the block diagrams of the paper.
* The denominator ``max(|n_t^T q_t|, exp(-m_t)) + eps`` instead of the
  ``max(|n_t^T q_t|, 1)`` often simplified in the running text -- the ``exp(-m_t)`` variant is the
  equivalent form stabilised in log space (see derivation below).

**Resolved (genuinely checked against the reference, ``NX-AI/xlstm``, ``mlstm_cell.py``):** the
final normalisation (``outnorm``) is applied **before** the multiplication with the
output gate ``o_t`` -- exactly as implemented here (on :math:`\tilde h_t`,
see below). Originally marked as uncertain (only the source excerpt on the
recurrence itself had been retrieved before, not that of the cell class); the later
check of the actual call confirms this order.

Equations, as implemented here
------------------------------
Per time step :math:`t`, state :math:`(C_{t-1}, n_{t-1}, m_{t-1})`:

.. math::
   q_t &= W_q x_t + b_q \qquad k_t = \frac{W_k x_t + b_k}{\sqrt{d}}
   \qquad v_t = W_v x_t + b_v

   \tilde\imath_t, \tilde f_t &= W_{if}\,[q_t; k_t; v_t] + b_{if}
   \qquad \log f_t = -\operatorname{softplus}(-\tilde f_t)

   m_t &= \max(\log f_t + m_{t-1},\ \tilde\imath_t)

   i'_t &= \exp(\tilde\imath_t - m_t) \qquad f'_t = \exp(\log f_t + m_{t-1} - m_t)

   C_t &= f'_t\,C_{t-1} + i'_t\,(k_t \otimes v_t) \qquad n_t = f'_t\,n_{t-1} + i'_t\,k_t

   \tilde h_t &= \operatorname{LayerNorm}\!\left(\frac{q_t^\top C_t}
   {\max(|q_t^\top n_t|,\ \exp(-m_t)) + \varepsilon}\right)

   o_t &= \sigma(W_o x_t + b_o) \qquad h_t = o_t \odot \tilde h_t

Why this recurrence does not overflow (the required stabilisation)
-------------------------------------------------------------------
Without :math:`m_t`, :math:`i_t = \exp(\tilde\imath_t)` and :math:`f_t = \exp(\log f_t)` would be
unbounded, or rather accumulate multiplicatively over many steps to values that
overflow ``float32`` (this is exactly the case the task statement warns against).
:math:`m_t` is defined as the running maximum of the log-gate sums
(:math:`\max(\log f_t + m_{t-1}, \tilde\imath_t)` -- the same log-sum-exp idea as in
a stable softmax). As a result, both exponents in :math:`i'_t, f'_t` are by
construction :math:`\le 0`, i.e. :math:`i'_t, f'_t \in (0, 1]` -- **bounded for every
input**, independently of sequence length or weight initialisation. This is not an
approximation of the exact value, but a mathematically equivalent reformulation (see the
derivation in the comments of ``backends.py`` of the reference): the denominator
``max(|n_t^T q_t|, exp(-m_t))`` exactly compensates the factor by which ``C_t``/``n_t``
are rescaled relative to their unstabilised counterparts.

Simplifications and assumptions, listed openly
----------------------------------------------
1. **One head** (``NH = 1``), not several. The official block splits ``q``/``k``/``v``
   into several heads with a smaller head dimension. With the ``hidden_size`` usual here
   (8-64) and the small input dimension of this project (~8 channels per submodel),
   an additional ``num_heads`` hyperparameter would have had to be introduced in
   ``TorchSequenceConfig``, which the hard integration requirement (call contract of
   ``nn.GRU``/``nn.LSTM``, no new constructor parameters) does not provide for. One head is
   the limiting case ``NH=1`` of the same equations, not a different architecture.
2. **No up/down projection, no causal short convolution before q/k.** See module docstring
   above -- this is the cell, not the block.
3. **Forget gate bias initialised to +3**, a special case of the widespread LSTM trick
   (Jozefowicz et al. 2015) that makes early forgetting harder. This is **not** a
   requirement taken from the paper -- I could not find an explicit
   initialisation rule of the reference for this gate in the available sources. Neutral would be
   ``bias=0``; +3 is a training aid whose effect is unchecked.
4. **``bias=True`` on all linear projections** (``q``, ``k``, ``v``, ``o``, gates).
   **Deliberate deviation from the reference** (genuinely checked, ``NX-AI/xlstm``,
   ``mlstm_layer.py``): there the configuration default is ``bias=False`` for the
   ``q``/``k``/``v``/``o`` projections (``LinearHeadwiseExpand``/``proj_down``). This
   implementation instead adopts the ``nn.Linear`` default (``bias=True``)
   -- not a correctness error (additional degree of freedom, no structural difference),
   but a deviation from the reference default, not merely an unchecked assumption.
5. **``eps = 1e-6``** in the denominator, a value, not a number taken from the paper (the
   reference names an ``eps``, but the retrieved code excerpt showed no
   concrete numerical value).
6. **Only the sequential (recurrent) form implemented**, not the parallel
   matrix form that the reference additionally offers for faster training. At the
   sequence lengths of this project (history 16-32, horizon up to 24) this is a
   speed compromise, not a correctness compromise -- explicitly accepted in order to have a
   single, unambiguous implementation of the recurrence instead of two that
   would have to be tested against each other.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["build_xlstm_core_class"]


def build_xlstm_core_class(torch_module: Any) -> type:
    """Creates the ``XLSTMCore`` class at run time, with ``torch_module`` as its base.

    The same factory-instead-of-module-class justification as for ``_build_network`` in
    ``torch_sequence.py``: ``torch.nn.Module`` must be available at definition time,
    and this package must remain importable without PyTorch. The caller (``_build_network``)
    therefore imports this module only after ``_require_torch()`` has already run.
    """
    nn_module = torch_module.nn
    functional = nn_module.functional
    base: type = nn_module.Module

    class _XLSTMLayer(base):  # type: ignore[misc]
        """One mLSTM layer: state ``(C, n, m)`` -> state, one time step per call."""

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.w_q = nn_module.Linear(input_size, hidden_size)
            self.w_k = nn_module.Linear(input_size, hidden_size)
            self.w_v = nn_module.Linear(input_size, hidden_size)
            self.w_o = nn_module.Linear(input_size, hidden_size)
            self.w_i = nn_module.Linear(3 * hidden_size, 1)
            self.w_f = nn_module.Linear(3 * hidden_size, 1)
            # Forget gate bias trick (Jozefowicz et al. 2015), see module docstring
            # point 3 -- not a requirement of the paper, a common training aid.
            nn_module.init.constant_(self.w_f.bias, 3.0)
            self.norm = nn_module.LayerNorm(hidden_size)

        def init_state(self, batch_size: int, reference: Any) -> tuple[Any, Any, Any]:
            """Zero state, shape/device/dtype taken from ``reference`` (input tensor)."""
            cell = reference.new_zeros(batch_size, self.hidden_size, self.hidden_size)
            normalizer = reference.new_zeros(batch_size, self.hidden_size)
            stabilizer = reference.new_zeros(batch_size, 1)
            return cell, normalizer, stabilizer

        def step(self, x_t: Any, state: tuple[Any, Any, Any]) -> tuple[Any, tuple[Any, Any, Any]]:
            """One time step. For the equations see the module docstring."""
            cell, normalizer, stabilizer = state
            query = self.w_q(x_t)
            key = self.w_k(x_t) / math.sqrt(self.hidden_size)
            value = self.w_v(x_t)

            gate_input = torch_module.cat([query, key, value], dim=-1)
            i_preact = self.w_i(gate_input)
            f_preact = self.w_f(gate_input)
            log_f = -functional.softplus(-f_preact)

            m_new = torch_module.maximum(log_f + stabilizer, i_preact)
            i_act = torch_module.exp(i_preact - m_new)
            f_act = torch_module.exp(log_f + stabilizer - m_new)

            outer = torch_module.einsum("bi,bj->bij", key, value)
            cell_new = f_act.unsqueeze(-1) * cell + i_act.unsqueeze(-1) * outer
            normalizer_new = f_act * normalizer + i_act * key

            h_num = torch_module.einsum("bi,bij->bj", query, cell_new)
            qn_dot = torch_module.einsum("bi,bi->b", query, normalizer_new).unsqueeze(-1)
            denom = torch_module.maximum(qn_dot.abs(), torch_module.exp(-m_new)) + 1e-6
            h_tilde = self.norm(h_num / denom)

            output_gate = torch_module.sigmoid(self.w_o(x_t))
            hidden = output_gate * h_tilde
            return hidden, (cell_new, normalizer_new, m_new)

    class XLSTMCore(base):  # type: ignore[misc]
        """mLSTM stack with the same call contract as ``nn.GRU``/``nn.LSTM``.

        ``forward(input, hx=None) -> (output, hidden)``, ``batch_first=True`` enforced
        (the only call form that ``torch_sequence.py`` ever uses). ``hidden`` is
        a tuple of ``num_layers`` triples ``(C, n, m)`` -- as with GRU/LSTM it is
        unpacked nowhere in ``torch_sequence.py``, only passed through.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            batch_first: bool = True,
        ) -> None:
            super().__init__()
            if not batch_first:
                raise ValueError("XLSTMCore unterstützt nur batch_first=True")
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.layers = nn_module.ModuleList(
                _XLSTMLayer(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(num_layers)
            )
            self.dropout = nn_module.Dropout(dropout)

        def forward(self, sequence: Any, hx: Any | None = None) -> tuple[Any, Any]:
            """Encode history or a single step -- sequence length is arbitrary."""
            batch_size, seq_len, _ = sequence.shape
            if hx is None:
                states = [layer.init_state(batch_size, sequence) for layer in self.layers]
            else:
                states = list(hx)

            layer_input = sequence
            for layer_index, layer in enumerate(self.layers):
                state = states[layer_index]
                outputs = []
                for step in range(seq_len):
                    hidden, state = layer.step(layer_input[:, step, :], state)
                    outputs.append(hidden)
                states[layer_index] = state
                layer_input = torch_module.stack(outputs, dim=1)
                if layer_index < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            return layer_input, tuple(states)

    # Exposed so that mixed stacks (models/slstm.py) reuse the exact same mLSTM layer.
    XLSTMCore.layer_class = _XLSTMLayer  # type: ignore[attr-defined]
    return XLSTMCore
