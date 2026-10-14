"""Fine-tuning of a pretrained sequence model (study A1).

``TorchSequenceModel.fit`` always builds a new network, and ``build_datasets`` always estimates
new scalers. For the transfer question -- pretrain on the simulator, then adapt to a plant with
few runs -- both are wrong, so this module adds the missing pieces next to them (nothing in
``torch_sequence.py`` or ``experiments.py`` is changed):

* :class:`TransferSequenceModel` -- a :class:`TorchSequenceModel` with ``fine_tune`` (start from
  the weights present, learning rate and epochs set per call, optionally frozen parameters) and
  ``initialise``;
* :func:`load_pretrained` -- model plus the scalers of the pretraining from a checkpoint
  (``utils.checkpoints``);
* :func:`build_datasets_with_scalers` -- like ``build_datasets`` but the scalers are given, so
  the plant's windows are standardised with the statistics of the pretraining.

The training loop of :meth:`TransferSequenceModel.fine_tune` is written anew (``fit`` is not
called). A test pins it to ``fit``: a freshly initialised model trained with ``fine_tune`` for
k epochs is bit-identical to ``fit`` for k epochs without early stopping.

Fine-tuning semantics (all fixed by the arguments, nothing is inherited silently):

* optimiser: a NEW AdamW (``weight_decay``, ``batch_size``, gradient clipping, horizon
  weights, teacher forcing from the model's config); its state is not carried over from any
  earlier training;
* learning rate and number of epochs are the arguments, not ``config.learning_rate`` /
  ``config.max_epochs``;
* without a validation set (``validation=None`` or ``patience=0``) the budget is run in full
  and the weights of the last epoch are kept -- the case for small N;
* frozen parameters (:class:`FreezeSpec`) do not enter the optimiser; dropout, if any, stays
  active in frozen layers while training (no ``eval()`` of sub-modules).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.splits import SplitName
from ..data.windows import SequenceDataset, Standardizer, TrajectoryWindowDataset
from ..evaluation.experiments import GeneratedData, ValidationPolicy
from .torch_sequence import TorchSequenceConfig, TorchSequenceModel

__all__ = [
    "FreezeSpec",
    "TransferSequenceModel",
    "build_datasets_with_scalers",
    "load_pretrained",
]

_LAYER_SUFFIX = re.compile(r"_l(\d+)(?:_reverse)?$")


@dataclass(frozen=True, slots=True)
class FreezeSpec:
    """Which parameters are NOT trained during fine-tuning. The default freezes nothing.

    ``core``: every parameter outside the output head (the recurrent/attention core);
    ``head``: the linear output head; ``layers``: single layers of a ``gru``/``lstm`` core by
    index (parameters ``..._l{i}``). The three combine (union).
    """

    core: bool = False
    head: bool = False
    layers: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.core or self.head or self.layers)

    def frozen_names(self, names: Sequence[str], core_kind: str) -> set[str]:
        """The parameter names to freeze out of ``names``; errors if a rule matches nothing."""
        frozen: set[str] = set()
        head_names = {n for n in names if n.split(".")[0] == "head"}
        if self.head:
            frozen |= head_names
        if self.core:
            frozen |= set(names) - head_names
        if self.layers:
            if core_kind not in ("gru", "lstm"):
                raise ValueError(
                    f"freezing single layers is only defined for gru/lstm, not {core_kind!r}"
                )
            for index in self.layers:
                matched = {
                    n
                    for n in names
                    if (m := _LAYER_SUFFIX.search(n)) is not None and int(m.group(1)) == index
                }
                if not matched:
                    raise ValueError(f"no parameters of layer {index} to freeze")
                frozen |= matched
        return frozen


class TransferSequenceModel(TorchSequenceModel):
    """A :class:`TorchSequenceModel` that can be trained further, see the module docstring."""

    def __init__(self, config: TorchSequenceConfig | None = None) -> None:
        super().__init__(config)
        self.pretrained_scalers: tuple[Standardizer, Standardizer] | None = None
        """Scalers of the pretraining, set by :func:`load_pretrained`. If present,
        :meth:`fine_tune` refuses a dataset standardised with different ones."""
        self.frozen_parameters: tuple[str, ...] = ()
        """Names frozen in the last :meth:`fine_tune` (for the record)."""

    # ------------------------------------------------------------------- set-up

    def initialise(self, train: SequenceDataset, *, seed: int) -> None:
        """Build the untrained network exactly as :meth:`fit` does, without training."""
        torch_module = self._torch
        self._n_exogenous = len(train.exogenous_names)
        self._n_targets = len(train.target_names)
        self._target_names = train.target_names
        self._autoregressive_index = train.autoregressive_index
        self._n_train_windows = train.n_windows
        torch_module.manual_seed(seed)
        self._network = self._new_network(self._device())

    def _device(self) -> Any:
        torch_module = self._torch
        return torch_module.device(
            self._config.device
            if self._config.device is not None
            else ("cuda" if torch_module.cuda.is_available() else "cpu")
        )

    def _new_network(self, device: Any) -> Any:
        from .torch_sequence import _build_network

        return _build_network(
            self._torch,
            n_exogenous=self._n_exogenous,
            n_targets=self._n_targets,
            config=self._config,
        ).to(device)

    # ---------------------------------------------------------------- fine-tuning

    def fine_tune(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        learning_rate: float,
        max_epochs: int,
        freeze: FreezeSpec = FreezeSpec(),  # noqa: B008 - frozen dataclass, immutable
        validation: SequenceDataset | None = None,
        patience: int = 0,
        verbose: bool = False,
    ) -> None:
        """Train the present weights further on ``train``.

        :param learning_rate: of the new AdamW optimiser (not ``config.learning_rate``)
        :param max_epochs: the epoch budget (not ``config.max_epochs``)
        :param freeze: parameters that are not trained
        :param validation, patience: early stopping on the validation rollout loss as in
            :meth:`fit`; ``patience=0`` (default) or no validation set runs the full budget
            and keeps the last epoch
        :raises RuntimeError: no network yet (load a checkpoint or :meth:`initialise` first)
        :raises ValueError: the dataset does not match the network's channels or the scalers
            of the pretraining
        """
        if self._network is None:
            raise RuntimeError("no network: load a pretrained model or call initialise() first")
        if learning_rate <= 0.0 or max_epochs < 1:
            raise ValueError("learning_rate must be positive and max_epochs >= 1")
        self._check_compatible(train)

        torch_module = self._torch
        config = self._config
        device = next(self._network.parameters()).device
        self._n_train_windows = train.n_windows

        torch_module.manual_seed(seed)
        names = [name for name, _ in self._network.named_parameters()]
        frozen = freeze.frozen_names(names, config.core)
        for name, parameter in self._network.named_parameters():
            parameter.requires_grad_(name not in frozen)
        self.frozen_parameters = tuple(sorted(frozen))
        trainable = [p for p in self._network.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("every parameter is frozen: nothing to train")

        optimizer = torch_module.optim.AdamW(
            self._trainable_groups(config.weight_decay), lr=learning_rate
        )
        weights = self._horizon_weights(train.config.horizon, device)
        generator = np.random.default_rng(seed)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0
        self._history = []
        self._best_epoch = None

        for epoch in range(max_epochs):
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
                torch_module.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
                optimizer.step()
                total += float(loss.detach()) * batch.n_windows
                count += batch.n_windows
                with torch_module.no_grad():
                    per_target = self._weighted_mse_per_target(
                        prediction, tensors["targets"], weights
                    )
                per_target_total += per_target.cpu().numpy().astype(np.float64) * batch.n_windows
            record = {"epoch": float(epoch), "train_loss": total / max(count, 1)}
            for index, name in enumerate(self._target_names):
                record[f"train_loss/{name}"] = float(per_target_total[index] / max(count, 1))

            if validation is not None and patience > 0:
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
                message = f"    fine-tune epoch {epoch}: train_loss={record['train_loss']:.5f}"
                if "val_loss" in record:
                    message += f" val_loss={record['val_loss']:.5f} stale={stale}"
                print(message, flush=True)
            if patience > 0 and validation is not None and stale >= patience:
                break

        if best_state is not None:
            self._network.load_state_dict(best_state)

    def _trainable_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Like ``_param_groups``, over the parameters that are not frozen."""
        assert self._network is not None
        parameters = [p for p in self._network.parameters() if p.requires_grad]
        no_decay = [p for p in parameters if getattr(p, "_no_weight_decay", False)]
        decay = [p for p in parameters if not getattr(p, "_no_weight_decay", False)]
        groups: list[dict[str, Any]] = []
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def _check_compatible(self, train: SequenceDataset) -> None:
        if len(train.exogenous_names) != self._n_exogenous:
            raise ValueError(
                f"dataset has {len(train.exogenous_names)} exogenous channels, the network "
                f"{self._n_exogenous}"
            )
        if tuple(train.target_names) != tuple(self._target_names):
            raise ValueError(
                f"dataset targets {train.target_names} differ from the network's "
                f"{self._target_names}"
            )
        if self.pretrained_scalers is not None:
            for given, expected, what in zip(
                train.scalers, self.pretrained_scalers, ("input", "target"), strict=True
            ):
                if not (
                    np.array_equal(given.mean, expected.mean)
                    and np.array_equal(given.scale, expected.scale)
                ):
                    raise ValueError(
                        f"the {what} scaler of the dataset is not the pretraining's; build the "
                        "datasets with build_datasets_with_scalers"
                    )


def load_pretrained(path: Path) -> TransferSequenceModel:
    """A :class:`TransferSequenceModel` from a checkpoint, with the pretraining's scalers.

    The checkpoint must hold a single (non-composite) model. ``pretrained_scalers`` is set, so
    :meth:`TransferSequenceModel.fine_tune` checks the datasets against it.
    """
    from ..utils.checkpoints import load_checkpoint

    loaded, metadata = load_checkpoint(path)
    if not isinstance(loaded, TorchSequenceModel):
        raise ValueError("composite checkpoints are not supported for fine-tuning")
    model = TransferSequenceModel(loaded.config)
    model.load_checkpoint(
        state_dict=loaded.state_dict(),
        target_names=metadata["target_names"],
        n_exogenous=metadata["n_exogenous"],
    )
    model.pretrained_scalers = (metadata["input_scaler"], metadata["target_scaler"])
    return model


def build_datasets_with_scalers(
    data: GeneratedData,
    n_train: int,
    scalers: tuple[Standardizer, Standardizer],
    *,
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset | None, TrajectoryWindowDataset]:
    """Like ``experiments.build_datasets``, but ALL splits use the given scalers.

    ``build_datasets`` estimates the scalers from the training split of the learning-curve
    point; for fine-tuning they must stay those of the pretraining. The signature and the
    return value are the same.
    """
    split = data.split.subset(n_train)
    window = data.spec.window
    train = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, split, SplitName.TRAIN, window, scalers=scalers
    )
    test = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, split, SplitName.TEST, window, scalers=scalers
    )
    validation = (
        TrajectoryWindowDataset.from_trajectory(
            data.trajectory, split, SplitName.VAL, window, scalers=scalers
        )
        if policy is ValidationPolicy.FIXED_POOL and split.val
        else None
    )
    return train, validation, test
