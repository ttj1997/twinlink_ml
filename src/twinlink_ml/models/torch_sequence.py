r"""Recurrent architectures (GRU, LSTM) behind a common trainer.

Work plan step 6. Deliberately **after** the four computation-free models: the interface is
validated by persistence, mean prediction, static regression, linear ARX and the
random-feature NARX before it can shape itself around a neural network.

Why one module and not one module per architecture
--------------------------------------------------
For step 6 the work plan demands that a new architecture requires **no change to
training or evaluation code** (open-closed). This can only be kept if the
training loop, rollout semantics, residual head and early stopping exist *once*
and the architecture is an exchangeable core. ``core="gru"`` and ``core="lstm"``
differ here in four lines; xLSTM, Mamba and Transformer are added as further
cores without the trainer being touched. Were they separate modules, the training loops
would differ by the third model -- and an architecture comparison whose
candidates are trained differently compares the training loops.

Three design decisions with justification
-----------------------------------------
**Residual parametrisation.** The head does not predict :math:`\hat y(t+h)`, but the
change relative to the last value:

.. math:: \hat y(t+h) = \hat y(t+h-1) + \Delta_\theta\bigl(h_{t+h}\bigr)

A network with zero weights is thus **exactly** the persistence baseline. For this
process that is not a cosmetic advantage: at :math:`\Delta t = 25` s persistence reaches
run-level NRMSE 0.17 on product moisture; a network with direct output must first find this
operating point and uses up for it, at the small support points of the learning curve,
exactly the data that matter.

**No teacher forcing (default 0.0).** The primary metric is the
multi-step rollout error. If training uses the true past values and evaluation
the model's own predictions, exposure bias arises -- the model has never seen the
state space it ends up in during the rollout. The price is more expensive training
(the sequence has to be traversed step by step) and a harder
optimisation problem. ``teacher_forcing`` exists as a parameter because a curriculum
from 1.0 to 0.0 helps at long horizons, but the default is the honest case.

**One core, no encoder-decoder pair.** History and horizon differ only
in where the target inputs come from: from the measurement or from the model's own
prediction. Two separate networks would have to learn the same dynamics twice.

**Observability needs no special path.** In the soft sensor case the history channel
of the unmeasured target is replaced by the training mean (see
:mod:`twinlink_ml.data.windows`). The anchor of the residual parametrisation is then zero,
the network starts without level information and carries its own state forward. Here
lies the only structural advantage a rollout-trained network has over the
NARX: it sees its own predictions in training and can actually learn the
feedback. A model fitted on one-step error cannot do that.

Status -- explicitly
--------------------
This module has by now been run on a machine with PyTorch (``04_stand_ap1.md``,
Befund 24). The sanity checks hold: a zero-initialised head reproduces
persistence exactly, GRU and LSTM share the same trainer, the rollout does not look into the
future. What is **not** checked by this is whether GRU or LSTM beat persistence with a realistic
seed budget -- a single training run with one seed at small
``n_train`` does not answer that, see the resolution threshold from step 5. This question
is the task of the learning curve (``scripts/study_architectures.py``, several seeds), not of a
unit test. Until this learning curve is available, the architecture comparison is still also
carried by ``RandomFeatureNARX``, which runs without Torch and is checked over several seeds.
On machines without PyTorch the six tests of this module still skip themselves --
this is the intended fallback, not a sign of missing maturity.

``xLSTM`` (only the mLSTM part, see :mod:`twinlink_ml.models.xlstm`) has been implemented since
step 6, part 2 and hooks in here via exactly one additional line in the
``cell_type`` dict of ``_build_network`` -- the trainer itself (``fit``, rollout,
loss, early stopping) has remained unchanged for it, as promised in the docstring above for
every additional core. ``Mamba`` (selective state-space core, see
:mod:`twinlink_ml.models.mamba`) is implemented the same way -- likewise exactly
one additional ``cell_type`` line, no special branch needed as for the Transformer core,
because the SSM cell, like GRU/LSTM/xLSTM, has a carried-forward state of fixed size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray
from .base import SequenceModel

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import Tensor, nn
else:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None

__all__ = [
    "GruConfig",
    "GruSequenceModel",
    "LstmSequenceModel",
    "RecurrentCore",
    "TorchSequenceConfig",
    "TorchSequenceModel",
    "XLSTMSequenceModel",
]

RecurrentCore = Literal["gru", "lstm", "xlstm", "transformer", "mamba", "slstm", "xlstm_mixed"]
"""Austauschbare Kerne. Erweiterung um weitere Kerne ändert den Trainer nicht.

``"xlstm"`` löst auf :func:`twinlink_ml.models.xlstm.build_xlstm_core_class` auf --
**nur** der mLSTM-Anteil (kein sLSTM), siehe Modul-Docstring dort für die Gleichungen und
die dokumentierten Vereinfachungen.

``"slstm"`` (nur sLSTM-Schichten, rekurrente Speichermischung) und ``"xlstm_mixed"`` (mLSTM-
Schichten, zuletzt eine sLSTM-Schicht; ``num_layers >= 2``) lösen auf
:func:`twinlink_ml.models.slstm.build_slstm_core_class` bzw.
:func:`twinlink_ml.models.slstm.build_xlstm_mixed_core_class` auf; ``num_heads`` ist die
Kopfzahl der sLSTM-Schichten.

``"transformer"`` ist **keine** Rekurrenz im engeren Sinn -- der Name bleibt trotzdem
stehen (kein Alias eingeführt), weil er nur intern als Typ des ``core``-Felds verwendet
wird und eine Umbenennung keinen Vertrag ändert. Er löst auf :func:`_build_transformer_network`
auf, nicht auf den ``cell_type``-Dict-Eintrag in :func:`_build_network`, weil das dortige
``forward()`` fest an das ``(input, hidden)``-Aufrufmuster von ``nn.GRU``/``nn.LSTM``/dem
mLSTM-Kern gebunden ist -- ein Aufrufmuster, das ``nn.TransformerEncoder``/``-Decoder``
nicht kennen. Der Trainer selbst (``TorchSequenceModel.fit``/``predict``/``state_dict``/
``load_checkpoint``) bleibt unverändert, weil er nur den ``forward(history,
future_exogenous, teacher, teacher_forcing) -> (batch, horizon, n_targets)``-Vertrag
kennt, nicht die Interna von ``_build_network``.

``"mamba"`` löst auf :func:`twinlink_ml.models.mamba.build_mamba_core_class` auf -- anders
als ``"transformer"`` **über** den ``cell_type``-Dict-Eintrag in :func:`_build_network`,
weil eine selektive State-Space-Zelle wie GRU/LSTM/xLSTM einen fortgeschriebenen Zustand
fester Größe hat und damit exakt in das ``(input, hidden) -> (output, hidden)``-
Aufrufmuster passt -- kein Sonderzweig wie bei Transformer nötig. Siehe
:mod:`twinlink_ml.models.mamba` für die Rekursionsgleichungen und die O(1)-pro-Schritt-
Eigenschaft, die der Grund für diese Architektur ist (Befund 37, ``docs/04_stand_ap1.md``:
der Transformer-Decoder-Loop ist strukturell O(H²), Mamba soll das genau vermeiden).
"""


def _require_torch() -> Any:
    """Load PyTorch on demand; an understandable error instead of ``ModuleNotFoundError``."""
    try:
        import torch as torch_module
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "TorchSequenceModel braucht PyTorch: pip install 'twinlink-ml[torch]'"
        ) from error
    return torch_module


@dataclass(frozen=True, slots=True)
class TorchSequenceConfig:
    """Architecture and training parameters.

    Attributes
    ----------
    core
        Core: ``"gru"``, ``"lstm"``, ``"xlstm"`` (mLSTM only, see
        :mod:`twinlink_ml.models.xlstm`) or ``"transformer"``. Everything else --
        residual head, rollout, loss weighting, early stopping -- is core-independent.
    hidden_size, num_layers
        Size of the GRU/LSTM/xLSTM, or ``d_model``/number of layers of the Transformer encoder and
        decoder. At the small support points of the learning curve capacity is not the
        limiting factor; start small. **For ``core="transformer"`` the
        dataclass default 64 is not a sensible starting point** -- a single
        encoder layer at ``d_model=64`` alone has around 50000 parameters (four
        attention projections plus FFN block), more than the entire
        three-target GRU composite model from Befund 28 put together (42819). Must be even for
        ``core="transformer"`` (sinusoidal positional encoding).
    num_heads
        Only used for ``core="transformer"`` (attention heads) and ``core="slstm"``/
        ``"xlstm_mixed"`` (heads of the sLSTM layers). Must divide ``hidden_size``.
        Without effect for the other cores.
    state_size
        Only used for ``core="mamba"``: dimension of the SSM state per channel
        (:math:`d_{state}` in :mod:`twinlink_ml.models.mamba`), independent of
        ``hidden_size`` (:math:`d_{model}`). Reference default 16. Without effect for the other
        cores -- exactly following the pattern of ``num_heads``.
    dropout
        Only effective for ``num_layers > 1`` (GRU/LSTM/xLSTM), or always effective for
        ``core="transformer"`` (PyTorch's ``TransformerEncoderLayer``/``-DecoderLayer``
        apply dropout independently of the number of layers).
    learning_rate, weight_decay, max_epochs, batch_size
        Optimisation (AdamW). ``weight_decay`` does not apply without exception to every parameter:
        ``TorchSequenceModel._param_groups`` sets it to 0 for parameters with the
        attribute ``_no_weight_decay=True`` -- currently only ``A_log``/``D`` for
        ``core="mamba"`` (see :mod:`twinlink_ml.models.mamba`, reference convention from
        ``state-spaces/mamba``). Without effect for all other cores, since no
        parameter is marked that way there -- bit-identical to the previous uniform AdamW call.
    patience
        Early stopping on the validation rollout loss. ``0`` switches it off and
        trains the fixed epoch budget -- the variant for the metric of
        training data requirement, which must not use up additional validation runs.
    teacher_forcing
        Fraction of horizon steps in which the true instead of the predicted value
        is fed in. Default 0.0, see module docstring.
    horizon_weight_decay
        Exponential weighting of the loss over the horizon. ``1.0`` weights all
        steps equally; values < 1 emphasise short horizons. At 1.0 the late
        steps dominate the gradient, because their error is larger.
    gradient_clip
        Norm clipping. In rollout training over 24 steps without teacher forcing,
        gradient explosions are the rule, not the exception.
    device
        ``"cpu"``, ``"cuda"`` or ``None`` for automatic choice.
    """

    core: RecurrentCore = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    num_heads: int = 2
    state_size: int = 16
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    teacher_forcing: float = 0.0
    horizon_weight_decay: float = 1.0
    gradient_clip: float = 1.0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.core not in (
            "gru",
            "lstm",
            "xlstm",
            "transformer",
            "mamba",
            "slstm",
            "xlstm_mixed",
        ):
            raise ValueError(
                "unbekannter Kern "
                f"{self.core!r}, erlaubt: gru, lstm, xlstm, transformer, mamba, slstm, "
                "xlstm_mixed"
            )
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size und num_layers müssen >= 1 sein")
        if not 0.0 <= self.teacher_forcing <= 1.0:
            raise ValueError("teacher_forcing muss in [0, 1] liegen")
        if not 0.0 < self.horizon_weight_decay <= 1.0:
            raise ValueError("horizon_weight_decay muss in (0, 1] liegen")
        if self.core == "transformer":
            if self.num_heads < 1:
                raise ValueError("num_heads muss >= 1 sein")
            if self.hidden_size % 2 != 0:
                raise ValueError(
                    "hidden_size muss bei core='transformer' gerade sein "
                    "(sinusoidale Positionskodierung)"
                )
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    "hidden_size muss bei core='transformer' durch num_heads teilbar sein"
                )
        if self.core == "mamba" and self.state_size < 1:
            raise ValueError("state_size muss bei core='mamba' >= 1 sein")
        if self.core in ("slstm", "xlstm_mixed") and (
            self.num_heads < 1 or self.hidden_size % self.num_heads != 0
        ):
            raise ValueError(
                f"hidden_size muss bei core={self.core!r} durch num_heads teilbar sein"
            )
        if self.core == "xlstm_mixed" and self.num_layers < 2:
            raise ValueError("num_layers muss bei core='xlstm_mixed' >= 2 sein")


class TorchSequenceModel(SequenceModel):
    """Recurrent network with autoregressive rollout and residual head.

    Parameters
    ----------
    config
        Core, architecture and optimisation.
    """

    def __init__(self, config: TorchSequenceConfig | None = None) -> None:
        self._config = config if config is not None else TorchSequenceConfig()
        self._torch = _require_torch()
        self._network: Any | None = None
        self._n_exogenous = 0
        self._n_targets = 0
        self._target_names: tuple[str, ...] = ()
        self._autoregressive_index: tuple[int, ...] = ()
        self._history: list[dict[str, float]] = []
        self._best_epoch: int | None = None
        self._n_train_windows = 0

    # ------------------------------------------------------------------- Interface

    @property
    def name(self) -> str:
        """Short name with core and capacity."""
        return f"{self._config.core}_h{self._config.hidden_size}_l{self._config.num_layers}"

    @property
    def config(self) -> TorchSequenceConfig:
        """Architecture and training parameters, for checkpoint persistence."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        if self._network is None:
            return 0
        return sum(p.numel() for p in self._network.parameters() if p.requires_grad)

    @property
    def training_history(self) -> tuple[dict[str, float], ...]:
        """Loss per epoch, for diagnosing the learning-curve points."""
        return tuple(self._history)

    _FLATTEN_WINDOW = 5

    def training_diagnostics(self) -> dict[str, Any]:
        """Epoch, gradient-step and flattening metrics for tracking.

        ``gradient_steps`` uses the same rounding up as
        :meth:`~twinlink_ml.data.windows.SequenceDataset.iter_batches` (the last,
        smaller batch counts too) -- a floor calculation would systematically underestimate
        the real number of steps.

        ``train_loss_ratio_last10`` compares the mean of the last 5 epochs
        against the mean of the 5 epochs before (epochs 6-10 before the end) -- **not**
        single point against single point. A two-point comparison (earlier version) proved
        misleading on the real `n_train=8` LSTM run: `product_moisture_kg_per_kg`
        oscillated without trend between 0.000075 and 0.000105 over the last 20 epochs
        (~±20 % relative amplitude) -- the two-point quotient (1.074) depended there on
        chance, on which point of the fluctuation the compared epochs happened to
        fall, not on the actual course of the curve. The averaged window slice
        cushions exactly this epoch-to-epoch noise, without introducing the complexity of a
        regression slope -- sufficient for a yes/no diagnosis (flattened or
        not). Close to 1.0 means flattened (plateau), clearly below means
        the loss was still falling at the stopping point -- the same distinction as in
        Befund 38(e). With ``patience=20`` at least 21 epochs are guaranteed, so the
        full 10-epoch window (two times 5) is practically always available; for
        shorter histories (only possible with ``patience=0`` or a very small ``max_epochs``)
        the window shrinks accordingly; with fewer than two epochs
        the metric stays neutral at 1.0 (no basis for a statement).
        """
        epochs_run = len(self._history)
        if epochs_run == 0:
            return {}
        batches_per_epoch = -(-self._n_train_windows // self._config.batch_size)
        window = min(self._FLATTEN_WINDOW, epochs_run // 2)
        if window == 0:
            ratio = 1.0
        else:
            recent = self._history[-window:]
            prior = self._history[-2 * window : -window]
            recent_mean = sum(record["train_loss"] for record in recent) / window
            prior_mean = sum(record["train_loss"] for record in prior) / window
            ratio = float(recent_mean / prior_mean) if prior_mean != 0.0 else 1.0
        return {
            "epochs_run": epochs_run,
            "gradient_steps": epochs_run * batches_per_epoch,
            "train_loss_ratio_last10": ratio,
            **self._loss_curves(),
        }

    def _loss_curves(self) -> dict[str, Any]:
        """Per-epoch training/validation loss and the epoch whose weights were kept (VH1).

        Lists of floats rounded to 7 significant digits (compact; relative error 1e-7). The
        validation list is empty when the fit ran without validation or with ``patience=0``
        (no validation loss is computed then). ``best_epoch`` is the epoch whose weights were
        restored at the end of :meth:`fit` (the last epoch that improved the validation loss by
        more than the early-stopping tolerance); without validation the weights of the last
        epoch are kept, so ``best_epoch`` is the last epoch. With early stopping,
        ``epochs_run == best_epoch + 1 + patience``.
        """
        epochs_run = len(self._history)
        best = self._best_epoch if self._best_epoch is not None else epochs_run - 1
        return {
            "loss_history_train": [float(f"{r['train_loss']:.7g}") for r in self._history],
            "loss_history_val": [
                float(f"{r['val_loss']:.7g}") for r in self._history if "val_loss" in r
            ],
            "best_epoch": best,
        }

    def describe(self) -> dict[str, Any]:
        """Configuration for tracking.

        ``net_device`` is the CONFIGURED wish (``TorchSequenceConfig.device``, may be
        ``None``); ``net_actual_device`` is the device ACTUALLY used at fit time
        -- ``device=None`` resolves in :meth:`fit` to ``"cuda"`` if available,
        otherwise to ``"cpu"`` (see there); without this field that would no longer be
        reconstructible with a CUDA-capable Torch build (occasion:
        ``befund42_tbed_exhaust_sweep.jsonl``, Torch build with CUDA support, no device field in
        the log). ``None`` before the first :meth:`fit` call (``self._network`` not yet built) --
        all current call sites call ``describe()`` only afterwards (after the evaluation, which
        presupposes a trained model); this case is guarded defensively, not
        observed, analogous to :attr:`n_parameters`.
        """
        return {
            **super().describe(),
            **{f"net_{key}": value for key, value in asdict(self._config).items()},
            "net_actual_device": (
                None if self._network is None else str(next(self._network.parameters()).device)
            ),
        }

    # --------------------------------------------------------------------- Training

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
        verbose: bool = False,
    ) -> None:
        """Train, optionally with early stopping on the validation rollout.

        ``verbose`` is **not** part of ``TorchSequenceConfig`` (and thus not in the
        checkpoint schema either) -- a pure run-time diagnostic switch for long runs without
        other progress output, off by default, changes no training behaviour.
        """
        torch_module = self._torch
        config = self._config
        self._n_exogenous = len(train.exogenous_names)
        self._n_targets = len(train.target_names)
        self._target_names = train.target_names
        self._autoregressive_index = train.autoregressive_index
        self._n_train_windows = train.n_windows

        torch_module.manual_seed(seed)
        device = torch_module.device(
            config.device
            if config.device is not None
            else ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self._network = _build_network(
            torch_module,
            n_exogenous=self._n_exogenous,
            n_targets=self._n_targets,
            config=config,
        ).to(device)
        optimizer = torch_module.optim.AdamW(
            self._param_groups(config.weight_decay),
            lr=config.learning_rate,
        )
        weights = self._horizon_weights(train.config.horizon, device)
        generator = np.random.default_rng(seed)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0
        self._history = []
        self._best_epoch = None

        for epoch in range(config.max_epochs):
            self._network.train()
            total, count = 0.0, 0
            per_target_total = np.zeros(self._n_targets, dtype=np.float64)
            for batch in train.iter_batches(config.batch_size, rng=generator):
                tensors = self._to_tensors(batch, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self._network(
                    tensors["history"],
                    tensors["future"],
                    teacher=tensors["targets"] if config.teacher_forcing > 0.0 else None,
                    teacher_forcing=config.teacher_forcing,
                )
                loss = self._weighted_mse(prediction, tensors["targets"], weights)
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(
                    self._network.parameters(), config.gradient_clip
                )
                optimizer.step()
                total += float(loss.detach()) * batch.n_windows
                count += batch.n_windows
                # Purely diagnostic (Befund 25, 04_stand_ap1.md): the same calculation as
                # above, only broken down before averaging over targets. Computed on the
                # already detached `prediction` tensor, does not enter
                # `loss.backward()` and therefore changes neither gradients nor optimizer
                # state -- pure observation, no intervention.
                with torch_module.no_grad():
                    per_target = self._weighted_mse_per_target(
                        prediction, tensors["targets"], weights
                    )
                per_target_total += per_target.cpu().numpy().astype(np.float64) * batch.n_windows
            record = {"epoch": float(epoch), "train_loss": total / max(count, 1)}
            for index, name in enumerate(self._target_names):
                record[f"train_loss/{name}"] = float(per_target_total[index] / max(count, 1))

            if validation is not None and config.patience > 0:
                validation_loss = self._evaluate_loss(validation, device, weights)
                record["val_loss"] = validation_loss
                if validation_loss < best_loss - 1e-6:
                    best_loss, stale = validation_loss, 0
                    self._best_epoch = epoch
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self._network.state_dict().items()
                    }
                else:
                    stale += 1
            self._history.append(record)
            if verbose:
                message = f"    epoch {epoch}: train_loss={record['train_loss']:.5f}"
                if "val_loss" in record:
                    message += f" val_loss={record['val_loss']:.5f} stale={stale}"
                print(message, flush=True)
            if config.patience > 0 and stale >= config.patience:
                break

        if best_state is not None:
            self._network.load_state_dict(best_state)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Autoregressive rollout, return value standardised as required by the interface."""
        if self._network is None:
            raise RuntimeError("Modell ist nicht angepasst")
        torch_module = self._torch
        device = next(self._network.parameters()).device
        self._network.eval()
        with torch_module.no_grad():
            tensors = self._to_tensors(batch, device)
            prediction = self._network(tensors["history"], tensors["future"])
        return np.asarray(prediction.cpu().numpy(), dtype=np.float64)

    # ----------------------------------------------------------------- Persistence

    def state_dict(self) -> dict[str, Any]:
        """Network weights, detached from the computation graph -- for checkpoint persistence.

        CPU copies, not the live tensors: a saved dict must no longer change
        if training continued afterwards (not the case here, but the copy
        keeps the method correct even if that changes).
        """
        if self._network is None:
            raise RuntimeError("Modell ist nicht angepasst")
        return {
            key: value.detach().cpu().clone() for key, value in self._network.state_dict().items()
        }

    def load_checkpoint(
        self, *, state_dict: dict[str, Any], target_names: Sequence[str], n_exogenous: int
    ) -> None:
        """Reconstruct the network from a saved state, without training.

        Builds the same architecture as :meth:`fit` (from ``self._config``), then
        the weights are loaded -- no training step, no seed needed, the weights
        are already fixed. ``target_names``/``n_exogenous`` come from the
        training dataset of the original run, not from the checkpoint schema
        itself (see :mod:`twinlink_ml.utils.checkpoints`).
        """
        torch_module = self._torch
        self._target_names = tuple(target_names)
        self._n_targets = len(self._target_names)
        self._n_exogenous = n_exogenous
        self._autoregressive_index = tuple(n_exogenous + i for i in range(self._n_targets))
        device = torch_module.device(
            self._config.device
            if self._config.device is not None
            else ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self._network = _build_network(
            torch_module,
            n_exogenous=self._n_exogenous,
            n_targets=self._n_targets,
            config=self._config,
        ).to(device)
        self._network.load_state_dict(state_dict)
        self._network.eval()

    # --------------------------------------------------------------------- internal

    def _param_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Two parameter groups: ``weight_decay=0`` for parameters marked as sensitive.

        Generic, no core special case here: a parameter counts as sensitive if it
        carries an attribute ``_no_weight_decay=True`` -- exactly the marking that the
        Mamba reference implementation (``state-spaces/mamba``, ``mamba_simple.py``) sets for
        ``A_log``/``D`` (see :mod:`twinlink_ml.models.mamba`). GRU/LSTM/xLSTM/
        Transformer mark nothing, so they fall entirely into the second group --
        bit-identical behaviour to the previous ``AdamW(self._network.parameters(), ...)``
        for every core except Mamba (see tests in ``test_composite.py``, which confirm this
        again after this rework).
        """
        assert self._network is not None  # internal only, called in fit() right after building
        no_decay = [p for p in self._network.parameters() if getattr(p, "_no_weight_decay", False)]
        decay = [p for p in self._network.parameters() if not getattr(p, "_no_weight_decay", False)]
        groups = [{"params": decay, "weight_decay": weight_decay}]
        if no_decay:  # an empty params list would make AdamW abort with ValueError
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def _horizon_weights(self, horizon: int, device: Any) -> Any:
        decay = self._config.horizon_weight_decay
        weights = np.power(decay, np.arange(horizon, dtype=np.float64))
        weights = weights / weights.mean()
        return self._torch.tensor(weights, dtype=self._torch.float32, device=device)

    def _to_tensors(self, batch: WindowBatch, device: Any) -> dict[str, Any]:
        torch_module = self._torch

        def convert(array: FloatArray) -> Any:
            return torch_module.tensor(
                np.ascontiguousarray(array), dtype=torch_module.float32, device=device
            )

        return {
            "history": convert(batch.history_inputs),
            "future": convert(batch.future_exogenous),
            "targets": convert(batch.future_targets),
        }

    def _weighted_mse(self, prediction: Any, target: Any, weights: Any) -> Any:
        squared = (prediction - target) ** 2
        return (squared.mean(dim=-1) * weights[None, :]).mean()

    def _weighted_mse_per_target(self, prediction: Any, target: Any, weights: Any) -> Any:
        """Like ``_weighted_mse``, only broken down before averaging over targets.

        Purely diagnostic (Befund 25, ``04_stand_ap1.md``): the return value has shape
        ``(n_targets,)`` and averages out to ``_weighted_mse``, because the same
        calculation only sums at a different place. Used nowhere in the training's return
        path, exclusively for logging.
        """
        squared = (prediction - target) ** 2
        return (squared * weights[None, :, None]).mean(dim=(0, 1))

    def _evaluate_loss(self, dataset: SequenceDataset, device: Any, weights: Any) -> float:
        torch_module = self._torch
        assert self._network is not None
        self._network.eval()
        total, count = 0.0, 0
        with torch_module.no_grad():
            for batch in dataset.iter_batches(self._config.batch_size):
                tensors = self._to_tensors(batch, device)
                prediction = self._network(tensors["history"], tensors["future"])
                loss = self._weighted_mse(prediction, tensors["targets"], weights)
                total += float(loss) * batch.n_windows
                count += batch.n_windows
        return total / max(count, 1)


def _build_network(
    torch_module: Any, *, n_exogenous: int, n_targets: int, config: TorchSequenceConfig
) -> Any:
    """Creates the ``nn.Module``.

    As a factory and not as a class at module level, because ``nn.Module`` would have to be
    available at definition time -- the package is meant to remain importable without PyTorch.

    ``core="transformer"`` branches off entirely before the ``cell_type`` dict
    (:func:`_build_transformer_network`): the ``Network.forward()`` below is tied to the
    ``(input, hidden)`` call pattern of ``nn.GRU``/``nn.LSTM``/the mLSTM core --
    ``nn.TransformerEncoder``/``-Decoder`` do not have this call pattern, so a single
    additional ``cell_type`` entry as for xLSTM would not work here.
    The trainer (``TorchSequenceModel.fit``/``predict``/persistence) nevertheless remains
    unchanged, because it only knows the ``forward(history, future_exogenous, teacher,
    teacher_forcing) -> (batch, horizon, n_targets)`` contract, not these internals.
    """
    if config.core == "transformer":
        return _build_transformer_network(
            torch_module, n_exogenous=n_exogenous, n_targets=n_targets, config=config
        )

    nn_module = torch_module.nn

    # Local import, not at module level: the same justification as for
    # `_resolve_torch_factory` in `registry.py`, so that this module remains importable
    # without Torch. Only the dict entry is new -- the rest of
    # `_build_network` and the entire trainer remain unchanged (see module
    # docstring, section on xLSTM/Mamba).
    from .mamba import build_mamba_core_class
    from .slstm import build_slstm_core_class, build_xlstm_mixed_core_class
    from .xlstm import build_xlstm_core_class

    cell_type = {
        "gru": nn_module.GRU,
        "lstm": nn_module.LSTM,
        "xlstm": build_xlstm_core_class(torch_module),
        "mamba": build_mamba_core_class(torch_module, config.state_size),
        "slstm": build_slstm_core_class(torch_module, config.num_heads),
        "xlstm_mixed": build_xlstm_mixed_core_class(torch_module, config.num_heads),
    }[config.core]

    # The base class is a run-time attribute access chain (`torch_module.nn.Module`)
    # and not a statically resolvable type reference -- exactly for that reason this whole block
    # is a factory function and not a module-level class (see docstring above). Only via
    # an intermediate variable explicitly annotated as `type` does `class Network(base):`
    # become readable as a class expression for mypy at all; the remaining message (subclass
    # of `Any`) is unavoidable with a base class loaded at run time and is the
    # intended way according to the mypy documentation. At run time `base` is exactly
    # `torch.nn.Module`, run and tested on a machine with PyTorch (see
    # tests/test_experiments.py, HAS_TORCH gate).
    base: type = nn_module.Module

    class Network(base):  # type: ignore[misc]
        """Recurrent core over history and horizon with a shared cell."""

        def __init__(self) -> None:
            super().__init__()
            self.core = cell_type(
                input_size=n_exogenous + n_targets,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn_module.Linear(config.hidden_size, n_targets)
            # Zero initialisation of the head: the untrained network is thus exactly
            # the persistence baseline, not a random mapping.
            nn_module.init.zeros_(self.head.weight)
            nn_module.init.zeros_(self.head.bias)

        def forward(
            self,
            history: Any,
            future_exogenous: Any,
            teacher: Any | None = None,
            teacher_forcing: float = 0.0,
        ) -> Any:
            """Encode the history, then continue step by step.

            ``hidden`` is a tensor for the GRU and a pair ``(h, c)`` for the LSTM. It is
            therefore unpacked nowhere, only passed on -- this makes the core
            really exchangeable and not just nominally.
            """
            _, hidden = self.core(history)
            last = history[:, -1, n_exogenous:]
            horizon = future_exogenous.shape[1]
            outputs = []
            for step in range(horizon):
                step_input = torch_module.cat(
                    [future_exogenous[:, step, :], last], dim=-1
                ).unsqueeze(1)
                output, hidden = self.core(step_input, hidden)
                last = last + self.head(output[:, 0, :])
                outputs.append(last)
                if teacher is not None and teacher_forcing > 0.0 and self.training:
                    mask = torch_module.rand(last.shape[0], 1, device=last.device) < teacher_forcing
                    last = torch_module.where(mask, teacher[:, step, :], last)
            return torch_module.stack(outputs, dim=1)

    return Network()


def _sinusoidal_position_encoding(torch_module: Any, positions: Any, d_model: int) -> Any:
    """Fixed (not learned) positional encoding after Vaswani et al. (2017).

    No learnable parameters -- adapts without recalibration to a later change of
    ``history_length``/``horizon`` (step 0 discussion, point c: reason 3 from the
    ``composite.py`` module docstring -- no new hyperparameters at small
    learning-curve support points). ``positions`` is a 1D tensor of arbitrary length; ``d_model``
    must be even (validated in ``TorchSequenceConfig.__post_init__``).
    """
    device = positions.device
    half = d_model // 2
    index = torch_module.arange(half, device=device, dtype=torch_module.float32)
    inverse_freq = torch_module.pow(10000.0, -2.0 * index / d_model)
    angles = positions.to(torch_module.float32).unsqueeze(-1) * inverse_freq.unsqueeze(0)
    encoding = torch_module.zeros(
        positions.shape[0], d_model, device=device, dtype=torch_module.float32
    )
    encoding[:, 0::2] = torch_module.sin(angles)
    encoding[:, 1::2] = torch_module.cos(angles)
    return encoding


def _build_transformer_network(
    torch_module: Any, *, n_exogenous: int, n_targets: int, config: TorchSequenceConfig
) -> Any:
    r"""Encoder-decoder Transformer as core, with the same return contract as the RNN cores.

    **Sequential generation, no explicit causal mask** (plain-language discussion before the
    implementation): with ``teacher_forcing=0.0`` -- the default, see module docstring
    -- the decoder input at step :math:`t>0` is exclusively the model's own prediction
    from step :math:`t-1`, never the true target value; this holds structurally and not just via
    a run-time check, because ``TorchSequenceModel.fit`` already passes ``teacher=None`` when
    ``config.teacher_forcing == 0.0`` (no tensor with future values exists at all
    then). At step ``t`` the decoder attends only over the sequence actually generated so far
    (length ``t + 1``) -- at this point there are no "future"
    decoder tokens that could be attended over, so causality is given by the
    growing sequence itself, not by a triangular mask. ``H = 36`` sequential
    sub-calls are trivially cheap at this ``d_model`` (confirmed in step 1 by the
    real timing, not by a cost formula).

    The encoder sees the full history **unmasked** (already completely known at the
    prediction time, see the step 0 answer on point b) -- only the decoder part is
    autoregressive.

    Positional encoding: sinusoidal, with segment offset (history: positions
    ``0 .. history_length - 1``; horizon: positions ``history_length ..
    history_length + horizon - 1``), as proposed and confirmed in step 0.
    """
    nn_module = torch_module.nn
    d_model = config.hidden_size
    n_features = n_exogenous + n_targets
    feedforward_size = 4 * d_model

    base: type = nn_module.Module

    class Network(base):  # type: ignore[misc]
        """Encoder over the history, sequential decoder over the horizon."""

        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn_module.Linear(n_features, d_model)
            encoder_layer = nn_module.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=feedforward_size,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn_module.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            decoder_layer = nn_module.TransformerDecoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=feedforward_size,
                dropout=config.dropout,
                batch_first=True,
            )
            self.decoder = nn_module.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
            self.head = nn_module.Linear(d_model, n_targets)
            # Zero initialisation of the head: the same residual equivalence as for
            # GRU/LSTM/xLSTM -- the untrained network is exactly the persistence baseline,
            # independently of the (random) encoder/decoder initialisation.
            nn_module.init.zeros_(self.head.weight)
            nn_module.init.zeros_(self.head.bias)

        def forward(
            self,
            history: Any,
            future_exogenous: Any,
            teacher: Any | None = None,
            teacher_forcing: float = 0.0,
        ) -> Any:
            """Encode the history unmasked, then decode autoregressively step by step."""
            device = history.device
            history_length = history.shape[1]
            horizon = future_exogenous.shape[1]

            history_positions = torch_module.arange(history_length, device=device)
            encoder_embed = self.input_proj(history) + _sinusoidal_position_encoding(
                torch_module, history_positions, d_model
            )
            memory = self.encoder(encoder_embed)

            last = history[:, -1, n_exogenous:]
            decoder_embeds: list[Any] = []
            outputs = []
            for step in range(horizon):
                step_input = torch_module.cat(
                    [future_exogenous[:, step, :], last], dim=-1
                ).unsqueeze(1)
                step_position = torch_module.tensor([history_length + step], device=device)
                step_embed = self.input_proj(step_input) + _sinusoidal_position_encoding(
                    torch_module, step_position, d_model
                ).unsqueeze(0)
                decoder_embeds.append(step_embed)
                # Only the sequence generated so far (length step + 1) -- see docstring:
                # no "future" tokens exist at this point, so no
                # explicit causal mask is needed.
                decoder_sequence = torch_module.cat(decoder_embeds, dim=1)
                decoded = self.decoder(decoder_sequence, memory)
                last = last + self.head(decoded[:, -1, :])
                outputs.append(last)
                if teacher is not None and teacher_forcing > 0.0 and self.training:
                    mask = torch_module.rand(last.shape[0], 1, device=device) < teacher_forcing
                    last = torch_module.where(mask, teacher[:, step, :], last)
            return torch_module.stack(outputs, dim=1)

    return Network()


GruConfig = TorchSequenceConfig
"""Alter Name aus Schritt 3. Bleibt als Alias, damit vorhandene Skripte laufen."""


def GruSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> TorchSequenceModel:
    """GRU variant -- factory for the model registry."""
    return TorchSequenceModel(config if config is not None else TorchSequenceConfig(core="gru"))


def LstmSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> TorchSequenceModel:
    """LSTM variant.

    With few training runs, the comparison GRU against LSTM is the more interesting part
    of the pair: with the cell state the LSTM has a second, additively updated
    memory and thus around a third more parameters at the same width. With eight
    runs, capacity is not a resource but a risk.
    """
    return TorchSequenceModel(config if config is not None else TorchSequenceConfig(core="lstm"))


def XLSTMSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> TorchSequenceModel:
    """The mLSTM variant (no sLSTM) -- factory for the model registry.

    See :mod:`twinlink_ml.models.xlstm` for equations and documented
    simplifications. Unlike GRU/LSTM, the capacity at the same
    ``hidden_size`` is **not** automatically larger here -- the mLSTM cell has no
    state-to-state weight matrix (that is precisely the reason why it is
    parallelisable), and at the small input dimension of this project
    exactly this matrix dominates the parameter budget of GRU/LSTM. See
    ``docs/04_stand_ap1.md`` for the measured numbers.
    """
    return TorchSequenceModel(config if config is not None else TorchSequenceConfig(core="xlstm"))
