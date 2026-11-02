r"""Selective state-space core (Mamba, Gu & Dao 2023, arXiv:2312.00752) -- the ``mamba`` slot.

Why pure PyTorch and not ``mamba-ssm``
--------------------------------------
Checked (step 0 of this work stream): ``pip install mamba-ssm`` fails on this
CPU-only Windows environment due to a hard dependency on ``nvcc`` (CUDA compiler) --
the official package builds its own CUDA kernels for the parallel scan and cannot even be
installed without the CUDA toolkit (not merely not run). This file
instead implements the recurrence itself, sequentially, without its own CUDA kernels --
see the module docstring section "Simplifications" below for what is deliberately
dropped in the process.

Only the cell, no block -- no more sources of error than with xLSTM
-------------------------------------------------------------------
Just like ``nn.GRU``/``nn.LSTM``/``XLSTMCore`` (:mod:`twinlink_ml.models.xlstm`),
:class:`MambaCore` (factory :func:`build_mamba_core_class`) is a raw recurrent cell with
the same ``forward(sequence, hx=None) -> (output, hidden)`` call contract, not an
encoder-decoder pair. Unlike the Transformer core (which needed its own branch
``_build_transformer_network``, because ``nn.TransformerEncoder``/``-Decoder`` do not know
the ``(input, hidden)`` call pattern), for Mamba **one additional line**
in the ``cell_type`` dict of ``torch_sequence._build_network`` suffices -- the trainer
(``TorchSequenceModel.fit``/``predict``/``state_dict``/``load_checkpoint``) remains
unchanged, as with xLSTM.

The recurrence, step by step (this is the whole reason for this architecture)
-----------------------------------------------------------------------------
Input at time step :math:`t`: :math:`u_t \in \mathbb{R}^{d_{model}}`, first
``RMSNorm``-normalised (reference: ``Block.forward`` in ``state-spaces/mamba`` strictly
follows ``residual = hidden + residual; hidden = norm(residual); hidden = mixer(hidden)``
-- the mixer never gets to see the unnormalised residual stream). Without this norm,
the autoregressive rollout of this project (``last = last + head(...)`` in
``torch_sequence.Network.forward``) can build up: an output that is once slightly too large
becomes an even larger input in the next step, exponentially over 36 horizon steps
-- diagnosed on a real training run (step 2 comparison, ``n_train=16``,
subnetwork ``energy_target``, epoch 12: the raw layer input grew within 5
steps from ~2 to >1e19, `h` collapsed to NaN). GRU/LSTM, by contrast, are structurally
robust (saturating tanh/sigmoid gates), ``nn.Transformer{Encoder,Decoder}Layer`` come with
LayerNorm built in -- this self-written SSM cell needed the norm
retrofitted; then projected via ``in_proj`` into two halves :math:`x_t, z_t \in
\mathbb{R}^{d_{inner}}` (``d_inner = expand * d_model``, ``expand = 2``).

**Learned, time-independent parameters:**

.. math:: A = -\exp(A_{log}) \in \mathbb{R}^{d_{inner} \times d_{state}}
   \qquad D \in \mathbb{R}^{d_{inner}}

``A_log`` initialised with :math:`\log(n)`, :math:`n = 1 \ldots d_{state}`, repeated
identically per row -- the S4D-real initialisation from the reference implementation
(guarantees :math:`A < 0`, i.e. stability, independently of training progress).

**Selective, input-dependent parameters** (linear from :math:`x_t`, AFTER SiLU activation
-- this makes the selection "selective" instead of time-invariant as in S4):

.. math:: \Delta_t = \operatorname{softplus}(W_\Delta x_t + b_\Delta) \in \mathbb{R}_+^{d_{inner}}
   \qquad B_t = W_B x_t \in \mathbb{R}^{d_{state}}
   \qquad C_t = W_C x_t \in \mathbb{R}^{d_{state}}

(``W_\Delta`` runs via a low-pass projection with rank ``dt_rank = max(1, d_model //
16)`` -- parameter economy from the reference, hardly relevant at this project's small
``d_model``, but adopted for fidelity to the original.)

**Discretisation** (Euler approximation for :math:`\bar B`, as in the reference):

.. math:: \bar A_t[i,n] = \exp(\Delta_t[i] \cdot A[i,n]) \qquad
   \bar B_t[i,n] = \Delta_t[i] \cdot B_t[n]

**State update -- O(1) per step, the actual goal of this architecture:**

.. math:: h_t[i,n] = \bar A_t[i,n] \cdot h_{t-1}[i,n] + \bar B_t[i,n] \cdot x_t[i]

:math:`h_t \in \mathbb{R}^{d_{inner} \times d_{state}}` depends exclusively on
:math:`h_{t-1}` and the quantities of the current step -- no reference to
:math:`t' < t-1`, no growing sequence is traversed anew at every step (unlike
the Transformer decoder, which at step :math:`t` attends anew over all :math:`t` previous
tokens, see the ``torch_sequence._build_transformer_network`` docstring and
Befund 37, ``docs/04_stand_ap1.md``). Exactly the same property that a
GRU/LSTM cell has: a carried-forward state of fixed size, no growing
context set.

**Output:**

.. math:: y_t[i] = \sum_n C_t[n] \cdot h_t[i,n] + D[i] \cdot x_t[i]
   \qquad y_t \leftarrow y_t \odot \operatorname{SiLU}(z_t)
   \qquad \text{out}_t = W_{out}\, y_t \in \mathbb{R}^{d_{model}}

For a full sequence (e.g. the history) the same recurrence is executed sequentially over
time in a Python ``for`` loop -- no parallel (cumsum/scan)
trick, as explicitly instructed (no own CUDA kernel development, no attempt
to rebuild the parallel scan performance of the reference).

Simplifications and assumptions, listed openly (analogous to ``xlstm.py``)
--------------------------------------------------------------------------
1. **No causal depthwise Conv1d before the SSM.** In the official Mamba block, a short
   (kernel 4) causal Conv1d for local context sits before the selection projection.
   Omitted here -- confirmed in the step 0 feedback as no blocker for the
   O(1) core property, can be retrofitted if needed (would need a small FIFO buffer as
   part of ``hidden``, would remain O(1) per step).
2. **``dt_proj`` bias with ``nn.Linear`` default initialisation**, not with the
   special ``dt_init`` softplus-inverse initialisation of the reference (which there controls
   the initial effective time step). A question of training dynamics, not of correctness.
3. **Only the sequential recurrence**, not the parallel associative scan of the
   reference -- at the sequence lengths of this project (history 16-32, horizon up to 36)
   a speed compromise, not a correctness compromise (the same justification as for
   xLSTM, point 6 there).
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_mamba_core_class"]


def build_mamba_core_class(torch_module: Any, state_size: int, expand: int = 2) -> type:
    """Creates the ``MambaCore`` class at run time, with ``torch_module`` as its base.

    ``state_size``/``expand`` are bound here (not part of the constructor call) --
    so ``MambaCore.__init__`` has the same signature as ``nn.GRU``/``nn.LSTM``/
    ``XLSTMCore`` (``input_size, hidden_size, num_layers, dropout, batch_first``) and fits
    unchanged into the ``cell_type`` dict of ``torch_sequence._build_network``. The same
    factory-instead-of-module-class justification as for ``build_xlstm_core_class``:
    ``torch.nn.Module`` must be available at definition time, the package must remain
    importable without PyTorch.
    """
    nn_module = torch_module.nn
    functional = nn_module.functional
    base: type = nn_module.Module

    class _MambaLayer(base):  # type: ignore[misc]
        """One selective SSM layer: state ``h`` -> state, one time step per call."""

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.d_inner = expand * hidden_size
            self.dt_rank = max(1, hidden_size // 16)

            # Reference (state-spaces/mamba, `Block.forward`): "Add -> LN -> Mixer" --
            # the mixer NEVER gets to see the unnormalised residual stream, only the
            # normalised view of it. Here this norm was at first missing entirely;
            # without it the autoregressive rollout (`last = last + head(...)` in
            # `torch_sequence.Network.forward`) can build up: an output that is once slightly
            # too large becomes an even larger input in the next step,
            # exponentially over 36 horizon steps (diagnosed on a
            # real training run, n_train=16, subnetwork energy_target, epoch 12: the
            # raw layer input itself grew within 5 steps from ~2 to >1e19,
            # before the state collapsed to NaN -- a mere delta clamp did not help,
            # because `b_t`/`c_t` from the same unnormalised input exploded as well).
            # GRU/LSTM, by contrast, are structurally robust (saturating tanh/sigmoid gates),
            # `nn.Transformer{Encoder,Decoder}Layer` come with LayerNorm built in --
            # the self-written SSM cell had no counterpart.
            self.norm = nn_module.RMSNorm(input_size)
            self.in_proj = nn_module.Linear(input_size, 2 * self.d_inner, bias=False)
            self.x_proj = nn_module.Linear(self.d_inner, self.dt_rank + 2 * state_size, bias=False)
            self.dt_proj = nn_module.Linear(self.dt_rank, self.d_inner, bias=True)

            # S4D-real initialisation (reference): A = -(1, 2, ..., state_size), identical
            # per d_inner row -- guarantees A < 0 independently of training.
            log_a_init = torch_module.log(
                torch_module.arange(1, state_size + 1, dtype=torch_module.float32)
            )
            self.A_log = nn_module.Parameter(log_a_init.unsqueeze(0).repeat(self.d_inner, 1))
            self.D = nn_module.Parameter(torch_module.ones(self.d_inner))
            # Reference convention (state-spaces/mamba, mamba_simple.py): A_log/D carry
            # the time-scale and skip information of the SSM directly (not learned
            # weights in the usual sense) -- weight decay would homogenise the S4D-real
            # diversity of the decay rates towards A_log=0 over many epochs.
            # `TorchSequenceModel._param_groups` (torch_sequence.py) reads exactly this
            # attribute generically, no Mamba special case there.
            self.A_log._no_weight_decay = True
            self.D._no_weight_decay = True
            self.out_proj = nn_module.Linear(self.d_inner, hidden_size, bias=False)

        def init_state(self, batch_size: int, reference: Any) -> Any:
            """Zero state, shape/device/dtype taken from ``reference`` (input tensor)."""
            return reference.new_zeros(batch_size, self.d_inner, state_size)

        def step(self, x_t: Any, h_prev: Any) -> tuple[Any, Any]:
            """One time step -- O(1): depends only on ``h_prev`` and ``x_t``.

            For the equations see the module docstring. ``self.norm`` first, as in the reference
            (see the ``__init__`` comment) -- everything that follows sees only the normalised
            input, never the unbounded raw value of ``x_t`` itself.
            """
            xz = self.in_proj(self.norm(x_t))
            x, z = xz.chunk(2, dim=-1)
            x = functional.silu(x)

            projected = self.x_proj(x)
            delta_in, b_t, c_t = torch_module.split(
                projected, [self.dt_rank, state_size, state_size], dim=-1
            )
            delta = functional.softplus(self.dt_proj(delta_in))
            a = -torch_module.exp(self.A_log)

            discrete_a = torch_module.exp(delta.unsqueeze(-1) * a.unsqueeze(0))
            discrete_b = delta.unsqueeze(-1) * b_t.unsqueeze(1)
            h = discrete_a * h_prev + discrete_b * x.unsqueeze(-1)

            y = (h * c_t.unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * x
            y = y * functional.silu(z)
            return self.out_proj(y), h

    class MambaCore(base):  # type: ignore[misc]
        """Selective SSM stack with the same call contract as ``nn.GRU``/``nn.LSTM``.

        ``forward(input, hx=None) -> (output, hidden)``, ``batch_first=True`` enforced
        (the only call form that ``torch_sequence.py`` ever uses). ``hidden`` is
        a tuple of ``num_layers`` state tensors -- as with GRU/LSTM/xLSTM it is
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
                raise ValueError("MambaCore unterstuetzt nur batch_first=True")
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.layers = nn_module.ModuleList(
                _MambaLayer(input_size if index == 0 else hidden_size, hidden_size)
                for index in range(num_layers)
            )
            self.dropout = nn_module.Dropout(dropout)

        def forward(self, sequence: Any, hx: Any | None = None) -> tuple[Any, Any]:
            """Encode history or a single step -- sequence length is arbitrary.

            Sequential over time (no parallel scan), see module docstring.
            """
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
                    out_t, state = layer.step(layer_input[:, step, :], state)
                    outputs.append(out_t)
                states[layer_index] = state
                layer_input = torch_module.stack(outputs, dim=1)
                if layer_index < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            return layer_input, tuple(states)

    return MambaCore
