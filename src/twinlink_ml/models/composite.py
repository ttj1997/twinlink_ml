r"""One independent network per target, instead of a shared one (Befund 26).

Why, with reference to ``docs/04_stand_ap1.md``
------------------------------------------------
Befund 26 diagnoses negative transfer in soft sensor operation:
``torch_sequence.py`` trains all targets with one shared recurrent state
and a single scalar loss. The two quality variables are practically
unsolvable there (Befund 19); their gradient noise degrades, via the shared state,
the otherwise good solution for ``energy_target`` -- measured as a factor of 14 in NRMSE
(0.3422 trained jointly against 0.0238 individually, ``n_train = 32``, median over three
seeds). A follow-up experiment ruled out the alternative cause -- the global
``gradient_clip``: with a 100-fold higher clip norm the error stayed
at 0.3574, no recovery towards 0.0238. The shared state itself is thus
the remaining, unrefuted cause.

Three reasons for this implementation (option 2 from Befund 26) instead of loss weighting
(option 1) or one of the two-phase proposals:

1. **Loss weighting only dampens the gradient magnitude.** A weighting changes how
   strongly a target counts in the scalar loss, but not the fact that its gradient still
   runs through the same recurrent state as that of ``energy_target``. The
   entanglement in the shared representation -- the actual mechanism from
   Befund 26 -- would remain, only weaker.
2. **The classical baselines are already decoupled per target.** ``LinearARXModel``
   and ``RandomFeatureNARX`` solve ``(X^T X + \lambda I)^{-1} X^T Y`` via
   :func:`~twinlink_ml.models.baselines._ridge_solve` with a multi-column
   target matrix ``Y``. This normal equation decomposes exactly into independent
   regressions per column, because ``gram = X^T X + \lambda I`` is identical for all columns
   and the solution vector per column depends only on the corresponding column of the
   right-hand side -- checked numerically (residual difference to the separate solution at
   machine precision, 1.1e-16). The previous comparison GRU/LSTM against ARX in
   soft sensor operation was therefore not a fair architecture comparison: ARX was structurally
   never exposed to the same transfer risk as the shared network. With this class
   the same decoupling now applies to GRU/LSTM as well.
3. **No new hyperparameters.** With few training runs (the learning curve starts at
   eight) every additional hyperparameter is itself a degree of freedom that would have to
   be calibrated without having data left over for it -- a central project requirement.
   Loss weighting after Kendall et al. or GradNorm would have needed exactly that.

What this costs
---------------
:class:`CompositeSequenceModel` sees per target only its own history, not that
of the others -- a deliberate compromise (see the plain-language discussion that preceded the
implementation, in the session history on Befund 26): an architecture that
additionally feeds in the history of the respective other targets would need either a
new channel type in :mod:`twinlink_ml.data.windows` with a placeholder policy for the
rollout, or a coupled step-by-step training loop that
duplicates ``torch_sequence.py`` outside that file. Both enlarge the intervention
considerably and were not part of this decision. This class instead reproduces
exactly case (b) checked in the diagnostic experiment: ``target_names = (eine Zielgröße,)``
per submodel.

Why ``torch_sequence.py`` remains unchanged
-------------------------------------------
Open-closed, as justified in the module docstring of ``torch_sequence.py`` itself: a
new combination of existing building blocks must not need to touch the trainer.
:class:`CompositeSequenceModel` holds ``n_targets`` independent
:class:`~twinlink_ml.models.torch_sequence.TorchSequenceModel` instances and a
generic :class:`~twinlink_ml.data.windows.SequenceDataset` view
(:class:`_SingleTargetView`) that reduces a multi-column dataset to one target.
The reduction is not an approximation: :meth:`Standardizer.fit
<twinlink_ml.data.windows.Standardizer.fit>` estimates mean and spread per channel
independently of the other channels, so ``scaler.select([i])`` is bit-identical to a
fit that would have been estimated from scratch on this one channel only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch, WindowConfig
from ..simulation.backend import FloatArray
from ..utils.seeding import _derive
from .base import SequenceModel
from .torch_sequence import TorchSequenceConfig, TorchSequenceModel

__all__ = [
    "CompositeGruSequenceModel",
    "CompositeLstmSequenceModel",
    "CompositeMambaSequenceModel",
    "CompositeSequenceModel",
    "CompositeTransformerSequenceModel",
    "CompositeXLSTMSequenceModel",
]


def _submodel_seed(seed: int, target: str) -> int:
    """Submodel seed, SHA-256-derived like :class:`~twinlink_ml.utils.seeding.SeedBundle`.

    **Not** ``seed + index``: ``utils/seeding.py`` explicitly justifies the switch from a
    linear derivation to SHA-256 by the fact that neighbouring master seeds otherwise produce
    correlated sub-seeds -- the same argument applies here between the
    target submodels. The target name (not the index) is the salt, so that the
    seed stays stable if the order of ``target_names`` changes.
    """
    return _derive(seed, "composite_submodel", target)


def _select_target(
    batch: WindowBatch, history_columns: Sequence[int], target_index: int
) -> WindowBatch:
    """Trim a ``WindowBatch`` to the input channels and the one target of a submodel."""
    return WindowBatch(
        history_inputs=batch.history_inputs[..., history_columns],
        future_exogenous=batch.future_exogenous,
        future_targets=batch.future_targets[..., [target_index]],
        future_targets_truth=(
            None
            if batch.future_targets_truth is None
            else batch.future_targets_truth[..., [target_index]]
        ),
        run_index=batch.run_index,
        window_start=batch.window_start,
    )


class _SingleTargetView(SequenceDataset):
    """A multi-column dataset, reduced to exactly one target.

    No recomputation from the trajectory -- pure channel selection on already
    standardised arrays of the underlying dataset, see module docstring.
    Works via the :class:`~twinlink_ml.data.windows.SequenceDataset` abstraction,
    not via ``TrajectoryWindowDataset`` internals, and thus stays valid for every
    future implementation of this ABC.
    """

    def __init__(self, base: SequenceDataset, target_index: int) -> None:
        self._base = base
        self._target_index = target_index
        self._history_columns = [
            *range(len(base.exogenous_names)),
            base.autoregressive_index[target_index],
        ]

    @property
    def config(self) -> WindowConfig:
        name = self.target_names[0]
        observed = name in self._base.observed_target_names
        return dataclasses.replace(
            self._base.config,
            target_names=(name,),
            observed_target_names=(name,) if observed else (),
        )

    @property
    def n_windows(self) -> int:
        return self._base.n_windows

    @property
    def exogenous_names(self) -> tuple[str, ...]:
        return self._base.exogenous_names

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self._base.target_names[self._target_index],)

    @property
    def input_scaler(self) -> Any:
        return self._base.input_scaler.select(self._history_columns)

    @property
    def target_scaler(self) -> Any:
        return self._base.target_scaler.select([self._target_index])

    def batch(self, indices: Sequence[int]) -> WindowBatch:
        return _select_target(self._base.batch(indices), self._history_columns, self._target_index)


class CompositeSequenceModel(SequenceModel):
    r"""One :class:`TorchSequenceModel` per target instead of a shared network.

    See the module docstring for the justification (Befund 26, ``docs/04_stand_ap1.md``).

    Each submodel receives a seed derived via :func:`_submodel_seed` from ``(seed, Zielgröße)``
    -- deterministic, but without the correlation between neighbouring
    sub-seeds that a linear derivation (``seed + i``) would introduce (see
    ``utils/seeding._derive``). Whoever wants to reproduce the same prediction with individually
    called ``TorchSequenceModel`` instances uses the same function.

    Parameters
    ----------
    config
        Architecture and optimisation, passed on unchanged to each submodel. One
        object for all targets -- different capacity per target would be a
        further degree of freedom that this module deliberately does not introduce (reason 3
        above).
    """

    def __init__(self, config: TorchSequenceConfig | None = None) -> None:
        self._config = config if config is not None else TorchSequenceConfig()
        self._submodels: dict[str, TorchSequenceModel] = {}
        self._target_names: tuple[str, ...] = ()
        self._n_exogenous = 0

    # ------------------------------------------------------------------- Interface

    @property
    def name(self) -> str:
        """Short name, with ``composite_`` prefix against confusion with the old single network."""
        return (
            f"composite_{self._config.core}_h{self._config.hidden_size}_l{self._config.num_layers}"
        )

    @property
    def config(self) -> TorchSequenceConfig:
        """Architecture and training parameters, passed on to each submodel."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Sum of the trainable parameters of all submodels."""
        return sum(model.n_parameters for model in self._submodels.values())

    @property
    def training_history(self) -> dict[str, tuple[dict[str, float], ...]]:
        """Training history per target -- separate networks, separate curves."""
        return {name: model.training_history for name, model in self._submodels.items()}

    def training_diagnostics(self) -> dict[str, Any]:
        """Epoch/gradient-step metrics per submodel, with target suffix."""
        return {
            f"{key}/{target}": value
            for target, model in self._submodels.items()
            for key, value in model.training_diagnostics().items()
        }

    def describe(self) -> dict[str, Any]:
        """Configuration for tracking, the same ``net_*`` keys as for the single network.

        ``net_actual_device`` is taken from the first submodel (``dict`` iteration order
        = insertion order, see :meth:`fit`) instead of being recomputed here -- all submodels
        share the same ``TorchSequenceConfig`` and therefore resolve ``device=None`` identically
        (see :meth:`TorchSequenceModel.fit`); a second look at each subnetwork would be
        redundant. ``None`` before the first :meth:`fit` call (``self._submodels`` still empty).
        """
        first_submodel = next(iter(self._submodels.values()), None)
        return {
            **super().describe(),
            **{f"net_{key}": value for key, value in asdict(self._config).items()},
            "composite/n_submodels": len(self._submodels),
            "net_actual_device": (
                None if first_submodel is None else first_submodel.describe()["net_actual_device"]
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
        """Fit one submodel per target, independently of each other.

        No case distinction by observability operating mode: ``_SingleTargetView``
        trims the already windowed dataset; the masking of unobserved
        targets is already contained in it (see
        :mod:`twinlink_ml.data.windows`). The same loop thus runs identically in
        ``INLINE`` and ``SOFT_SENSOR``.

        ``verbose`` is only passed through to :meth:`TorchSequenceModel.fit` (see there) --
        not part of ``TorchSequenceConfig``, changes no training behaviour.
        """
        self._target_names = train.target_names
        self._n_exogenous = len(train.exogenous_names)
        submodels: dict[str, TorchSequenceModel] = {}
        for index, target in enumerate(self._target_names):
            sub_train = _SingleTargetView(train, index)
            sub_validation = None if validation is None else _SingleTargetView(validation, index)
            submodel = TorchSequenceModel(self._config)
            if verbose:
                print(f"-- Teilmodell {target!r} --", flush=True)
            submodel.fit(
                sub_train,
                seed=_submodel_seed(seed, target),
                validation=sub_validation,
                verbose=verbose,
            )
            submodels[target] = submodel
        self._submodels = submodels

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Apply each submodel to its own column, result in target order."""
        if not self._submodels:
            raise RuntimeError("Modell ist nicht angepasst")
        predictions = []
        for index, target in enumerate(self._target_names):
            history_columns = [*range(self._n_exogenous), self._n_exogenous + index]
            sub_batch = _select_target(batch, history_columns, index)
            predictions.append(self._submodels[target].predict(sub_batch))
        return np.concatenate(predictions, axis=-1)

    # ----------------------------------------------------------------- Persistence

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """One state dict per submodel -- Befund 26, one network per target."""
        if not self._submodels:
            raise RuntimeError("Modell ist nicht angepasst")
        return {target: model.state_dict() for target, model in self._submodels.items()}

    def load_checkpoint(
        self,
        *,
        submodel_state_dicts: Mapping[str, dict[str, Any]],
        target_names: Sequence[str],
        n_exogenous: int,
    ) -> None:
        """Reconstruct all submodels from saved states, without training.

        Each submodel receives the same architecture (``self._config``) as in the
        original training and exactly one target -- the same setup as in
        :meth:`fit`, only without the training step.
        """
        self._target_names = tuple(target_names)
        self._n_exogenous = n_exogenous
        submodels: dict[str, TorchSequenceModel] = {}
        for target in self._target_names:
            submodel = TorchSequenceModel(self._config)
            submodel.load_checkpoint(
                state_dict=submodel_state_dicts[target],
                target_names=(target,),
                n_exogenous=n_exogenous,
            )
            submodels[target] = submodel
        self._submodels = submodels


def CompositeGruSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """GRU variant -- factory for the model registry."""
    return CompositeSequenceModel(config if config is not None else TorchSequenceConfig(core="gru"))


def CompositeLstmSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """LSTM variant -- factory for the model registry."""
    return CompositeSequenceModel(
        config if config is not None else TorchSequenceConfig(core="lstm")
    )


def CompositeXLSTMSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """The mLSTM variant (no sLSTM) -- factory for the model registry.

    The same reason for a composite model as for GRU/LSTM (Befund 26): the shared
    recurrent state harms ``energy_target`` in soft sensor operation, independently of the
    core. See :mod:`twinlink_ml.models.xlstm` for the mLSTM equations themselves.
    """
    return CompositeSequenceModel(
        config if config is not None else TorchSequenceConfig(core="xlstm")
    )


def CompositeTransformerSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """Transformer variant -- factory for the model registry.

    Unlike for GRU/LSTM/xLSTM, ``hidden_size=64`` (dataclass default) is **not** an
    appropriate default for this core: a single encoder layer at
    ``d_model=64`` alone has around 50000 parameters (four attention projections plus
    FFN block, ``torch_sequence._build_transformer_network``), more than the entire
    three-target GRU composite model from Befund 28 put together (42819, ``docs/04_stand_ap1.md``).
    Hence a deliberately small default here (``hidden_size=16``, ``num_heads=2``,
    ``num_layers=1``) instead of the GRU/LSTM/xLSTM default -- capacity matching against GRU
    is only an option after a real cost measurement, not by formula (the same
    pitfall that already once cost >12h/training point in the capacity-matched xLSTM
    attempt).
    """
    return CompositeSequenceModel(
        config
        if config is not None
        else TorchSequenceConfig(core="transformer", hidden_size=16, num_heads=2, num_layers=1)
    )


def CompositeMambaSequenceModel(  # noqa: N802 - factory name follows the model label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """Mamba variant (selective state-space core) -- factory for the model registry.

    Unlike the Transformer, Mamba needs no reduced ``hidden_size`` compared with the
    dataclass default: the cell (:mod:`twinlink_ml.models.mamba`) has only a
    handful of linear projections per block plus a ``(d_inner, d_state)`` matrix, no
    attention projection growing quadratically in ``hidden_size``. Nevertheless deliberately
    kept small here (``hidden_size=16``, ``state_size=16``, ``num_layers=1``) --
    comparability with the Transformer default (Befund 37) and the same principle as
    there: capacity at small learning-curve support points is not a head start to be
    taken unchecked.
    """
    return CompositeSequenceModel(
        config
        if config is not None
        else TorchSequenceConfig(core="mamba", hidden_size=16, state_size=16, num_layers=1)
    )
