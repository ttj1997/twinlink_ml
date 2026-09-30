"""SX0 -- where the parameter counts of xLSTM and Mamba come from (read-only inspection).

Builds one head network of every core exactly as the composite model does (7 exogenous inputs,
1 target, ``history`` unused here) with the configurations of the learning-curve runs
(Befund 51/58) and lists the parameter tensors of the core, so that the totals quoted in
Befund 63 (57027 / 42819 / 8649 / 8139 = three heads each) are traced to their tensors. Also
lists, for the recurrent cores, how many parameters are hidden-to-hidden (recurrent) weights.

    python scripts/verify/sx0_parameter_breakdown.py
"""

from __future__ import annotations

from typing import Any

import torch
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.models.torch_sequence import TorchSequenceConfig, _build_network

N_EXOGENOUS = 7
N_TARGETS = 1
N_HEADS_COMPOSITE = 3
CONFIGS = {
    "lstm": TorchSequenceConfig(core="lstm", hidden_size=64),
    "gru": TorchSequenceConfig(core="gru", hidden_size=64),
    "xlstm": TorchSequenceConfig(core="xlstm", hidden_size=64),
    "mamba": TorchSequenceConfig(core="mamba", hidden_size=16, state_size=16),
}


def main() -> None:
    require_clean_tree()
    out: dict[str, Any] = {"_provenance": provenance(), "cores": {}}
    for name, config in CONFIGS.items():
        network = _build_network(
            torch, n_exogenous=N_EXOGENOUS, n_targets=N_TARGETS, config=config
        )
        tensors = {key: int(value.numel()) for key, value in network.named_parameters()}
        per_head = sum(tensors.values())
        out["cores"][name] = {
            "hidden_size": config.hidden_size,
            "state_size": config.state_size if name == "mamba" else None,
            "input_size": N_EXOGENOUS + N_TARGETS,
            "parameters_per_head": per_head,
            "parameters_composite_three_heads": N_HEADS_COMPOSITE * per_head,
            "tensors": tensors,
        }
        print(name, per_head, N_HEADS_COMPOSITE * per_head)
    write_artifact("sx0_parameter_breakdown", out)


if __name__ == "__main__":
    main()
