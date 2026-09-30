r"""sLSTM core after Beck et al. (2024), arXiv:2405.04517 -- slots ``slstm`` and ``xlstm_mixed``.

Complements :mod:`twinlink_ml.models.xlstm` (mLSTM only). The sLSTM differs from the mLSTM in
exactly the point that matters for tracking a dynamic process state: it has **recurrent memory
mixing**. The gate pre-activations depend on the previous cell output through a recurrent matrix
``R`` (block-diagonal over heads: mixing inside a head, no mixing between heads), while in the
mLSTM every projection and gate is computed from the current input only.

Same abstraction level as the mLSTM core: a recurrent **cell** stack with the
``forward(sequence, hx=None) -> (output, hidden)`` contract of ``nn.GRU``/``nn.LSTM``; no
up/down projection, no causal convolution, no feed-forward part, no block residual. Listed
deviations from the reference layer (``NX-AI/xlstm``, ``xlstm/blocks/slstm``) are at the end.

Equations, one time step, ``H`` hidden units in ``NH`` heads of size ``d = H / NH``
--------------------------------------------------------------------------------------
With ``y_{t-1}`` the previous (un-normalised) cell output:

.. math::
   \begin{pmatrix}\tilde\imath_t\\\tilde f_t\\\tilde z_t\\\tilde o_t\end{pmatrix}
   = W x_t + b + R\,y_{t-1},\qquad R = \operatorname{blockdiag}_{\text{heads}}

   \log f_t = \log\sigma(\tilde f_t),\qquad
   m_t = \max(\log f_t + m_{t-1},\ \tilde\imath_t)

   i'_t = \exp(\tilde\imath_t - m_t),\quad f'_t = \exp(\log f_t + m_{t-1} - m_t)

   c_t = f'_t c_{t-1} + i'_t\tanh(\tilde z_t),\quad n_t = f'_t n_{t-1} + i'_t,\quad
   y_t = \sigma(\tilde o_t)\,c_t / n_t

The output of the layer is ``GroupNorm_{NH}(y_t)``; the recurrence uses ``y_t`` itself.

Stabilisation. Without ``m_t`` the input gate ``exp(\tilde\imath_t)`` overflows for
pre-activations above about 88 (float32). With ``m_t`` both exponents are ``<= 0``, so
``i', f' \in (0, 1]`` and one of them equals 1; therefore ``n_t >= 1`` for all t (no epsilon
needed) and ``|c_t| <= n_t``, hence ``|y_t| <= 1``. The stabilised and the unstabilised
recurrence are the same function in exact arithmetic (``c_t/n_t`` is invariant under the
common rescaling by ``exp(-m_t)``); the flag ``stabilized=False`` exists only so that this
equivalence and the overflow of the plain form can be tested.

Initial state. ``c = n = y = 0``, ``m = -1e30`` (a finite stand-in for ``-inf``): the first
step then gives ``m_1 = \tilde\imath_1`` and ``f'_1 = 0``, exactly the reference's
``if all(n == 0): m = i`` branch, without a data-dependent branch.

Deviations from the reference layer and open choices (not tuned)
------------------------------------------------------------------
1. Cell level only (see above): no conv1d on the gate inputs, no gated feed-forward (post-up
   projection), no residual block. Same simplification as for the mLSTM core.
2. Input projection ``W`` is one ``nn.Linear(input_size, 4H)`` **with** bias; the reference has a
   bias in the cell and none in the projection -- the same function.
3. Recurrent matrix initialised uniform(-1/sqrt(H), 1/sqrt(H)) ("standard" in the reference).
   The reference's default is ``"zeros"`` (memory mixing starts switched off); it is not used here
   because it would handicap exactly the property under test.
4. Forget-gate bias initialised to +3 (as in the mLSTM core; the reference uses a per-unit power-law
   schedule). The effect is untested.
5. Sigmoid forget gate (``logsigmoid``), as in the reference sLSTM; the paper also allows
   exponential forget gates.
6. Sequential Python loop, like the other cores (the CUDA kernel of the reference is not usable
   on the CPU).

Numerical equivalence of this cell with the official CPU implementation
(``xlstm==2.0.6``, ``backend="vanilla"``) is checked by
``scripts/verify/sx2_official_equivalence.py``; the unit tests (``tests/test_slstm.py``) do not
need the package.
"""

from __future__ import annotations

import math
from typing import Any

from .xlstm import build_xlstm_core_class

__all__ = ["build_slstm_core_class", "build_xlstm_mixed_core_class"]

_NEG_INIT = -1.0e30
_FORGET_BIAS_INIT = 3.0


def _build_slstm_layer_class(torch_module: Any, num_heads: int, stabilized: bool) -> type:
    nn_module = torch_module.nn
    functional = nn_module.functional
    base: type = nn_module.Module

    class SLSTMLayer(base):  # type: ignore[misc]
        """One sLSTM layer: state ``(c, n, m, y)`` -> state, one time step per call."""

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            if hidden_size % num_heads != 0:
                raise ValueError(f"hidden_size {hidden_size} must be divisible by {num_heads}")
            self.hidden_size = hidden_size
            self.head_size = hidden_size // num_heads
            self.w_x = nn_module.Linear(input_size, 4 * hidden_size)
            bound = 1.0 / math.sqrt(hidden_size)
            self.recurrent = nn_module.Parameter(
                torch_module.empty(num_heads, 4, self.head_size, self.head_size).uniform_(
                    -bound, bound
                )
            )
            with torch_module.no_grad():
                # gate order i, f, z, o along the 4H axis
                self.w_x.bias[hidden_size : 2 * hidden_size] = _FORGET_BIAS_INIT
            self.norm = nn_module.GroupNorm(num_heads, hidden_size)

        def init_state(self, batch_size: int, reference: Any) -> tuple[Any, Any, Any, Any]:
            """Zero state (``m`` at a large negative constant), dtype/device of ``reference``."""
            shape = (batch_size, self.hidden_size)
            zeros = reference.new_zeros(shape)
            return zeros, reference.new_zeros(shape), reference.new_full(shape, _NEG_INIT), zeros

        def recurrent_contribution(self, y_prev: Any) -> Any:
            """``R y_{t-1}`` in gate-major layout ``(B, 4H)``; block-diagonal over heads."""
            batch = y_prev.shape[0]
            per_head = y_prev.reshape(batch, num_heads, self.head_size)
            mixed = torch_module.einsum("bhd,hged->bghe", per_head, self.recurrent)
            return mixed.reshape(batch, 4 * self.hidden_size)

        def cell(
            self, x_t: Any, state: tuple[Any, Any, Any, Any]
        ) -> tuple[Any, tuple[Any, Any, Any, Any]]:
            """Recurrence of the module docstring; returns ``y_t`` (un-normalised) and state."""
            cell_state, normalizer, stabilizer, y_prev = state
            raw = self.w_x(x_t) + self.recurrent_contribution(y_prev)
            i_raw, f_raw, z_raw, o_raw = raw.chunk(4, dim=-1)
            log_f = functional.logsigmoid(f_raw)
            if stabilized:
                m_new = torch_module.maximum(log_f + stabilizer, i_raw)
                i_act = torch_module.exp(i_raw - m_new)
                f_act = torch_module.exp(log_f + stabilizer - m_new)
            else:
                m_new = stabilizer
                i_act = torch_module.exp(i_raw)
                f_act = torch_module.exp(log_f)
            cell_new = f_act * cell_state + i_act * torch_module.tanh(z_raw)
            normalizer_new = f_act * normalizer + i_act
            y_new = torch_module.sigmoid(o_raw) * cell_new / normalizer_new
            return y_new, (cell_new, normalizer_new, m_new, y_new)

        def step(
            self, x_t: Any, state: tuple[Any, Any, Any, Any]
        ) -> tuple[Any, tuple[Any, Any, Any, Any]]:
            """One time step: normalised output and new state."""
            y_new, state_new = self.cell(x_t, state)
            return self.norm(y_new), state_new

    return SLSTMLayer


def _build_stack_class(torch_module: Any, layer_factory: Any, name: str) -> type:
    """Stack of per-step layers with the ``nn.GRU``/``nn.LSTM`` call contract."""
    nn_module = torch_module.nn
    base: type = nn_module.Module

    class Core(base):  # type: ignore[misc]
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
                raise ValueError(f"{name} supports batch_first=True only")
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.layers = nn_module.ModuleList(
                layer_factory(index, num_layers, input_size, hidden_size)
                for index in range(num_layers)
            )
            self.dropout = nn_module.Dropout(dropout)

        def forward(self, sequence: Any, hx: Any | None = None) -> tuple[Any, Any]:
            """Encode a history or a single step; the sequence length is arbitrary."""
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

    Core.__name__ = name
    return Core


def build_slstm_core_class(torch_module: Any, num_heads: int, *, stabilized: bool = True) -> type:
    """``SLSTMCore``: a stack of sLSTM layers (``xlstm[0:1]``-style, sLSTM only)."""
    layer_class = _build_slstm_layer_class(torch_module, num_heads, stabilized)

    def factory(index: int, num_layers: int, input_size: int, hidden_size: int) -> Any:
        del num_layers
        return layer_class(input_size if index == 0 else hidden_size, hidden_size)

    core = _build_stack_class(torch_module, factory, "SLSTMCore")
    core.layer_class = layer_class  # type: ignore[attr-defined]
    return core


def build_xlstm_mixed_core_class(torch_module: Any, num_heads: int) -> type:
    """``XLSTMMixedCore``: mLSTM layers followed by one final sLSTM layer.

    Needs ``num_layers >= 2``. Layer order (mLSTM first, sLSTM last) is fixed a priori: the
    matrix memory reads the inputs, the recurrent sLSTM layer mixes over the mLSTM outputs and
    feeds the output head.
    """
    mlstm_layer_class = build_xlstm_core_class(torch_module).layer_class  # type: ignore[attr-defined]
    slstm_layer_class = _build_slstm_layer_class(torch_module, num_heads, True)

    def factory(index: int, num_layers: int, input_size: int, hidden_size: int) -> Any:
        if num_layers < 2:
            raise ValueError("xlstm_mixed needs num_layers >= 2")
        width = input_size if index == 0 else hidden_size
        if index == num_layers - 1:
            return slstm_layer_class(width, hidden_size)
        return mlstm_layer_class(width, hidden_size)

    return _build_stack_class(torch_module, factory, "XLSTMMixedCore")
