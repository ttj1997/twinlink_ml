"""SX2 -- numerical equivalence of the sLSTM cell with the official CPU implementation.

Compares ``twinlink_ml.models.slstm`` (stabilised, ``stabilized=True``) with
``xlstm.blocks.slstm.cell.sLSTMCell(backend="vanilla")`` from the official package (NX-AI,
``xlstm==2.0.6``, pure-Python wheel, not installed into the project environment): same weights
(mapped tensor by tensor), same inputs, output of the recurrence ``y_t`` (before the GroupNorm
that belongs to the layer, not to the cell). Cases: several random seeds, hidden sizes and head
counts, input scales up to 30 (large enough for the gate pre-activations to reach the stabilised
regime), sequences of 40 steps. Reports the maximum absolute and relative deviation per case.

    python scripts/verify/sx2_official_equivalence.py --xlstm-path <dir containing 'xlstm/'>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from _common import provenance, require_clean_tree, write_artifact

from twinlink_ml.models.slstm import build_slstm_core_class

CASES = [
    {"seed": seed, "hidden": hidden, "heads": heads, "input_scale": scale}
    for seed in (0, 1, 2)
    for hidden, heads in ((16, 1), (32, 4), (52, 4))
    for scale in (0.5, 5.0, 30.0)
]
SEQUENCE_LENGTH = 40
BATCH = 6
INPUT_SIZE = 8
TOLERANCE_ABS = 1e-5


def _run_case(case: dict[str, Any], official: Any) -> dict[str, Any]:
    torch.manual_seed(case["seed"])
    hidden, heads = case["hidden"], case["heads"]
    layer = build_slstm_core_class(torch, heads)(INPUT_SIZE, hidden).layers[0]
    with torch.no_grad():
        layer.w_x.bias.add_(0.3 * torch.randn_like(layer.w_x.bias))
        layer.recurrent.mul_(3.0)  # exercise the memory mixing harder than the default init
    cell = official.sLSTMCell(
        official.sLSTMCellConfig(
            hidden_size=hidden,
            num_heads=heads,
            backend="vanilla",
            dtype="float32",
            dtype_b="float32",
            dtype_r="float32",
            dtype_w="float32",
            dtype_s="float32",
            dtype_g="float32",
            dtype_a="float32",
            enable_automatic_mixed_precision=False,
            input_shape="SBGNH",
            output_shape="SBH",
        )
    )
    head_size = hidden // heads
    with torch.no_grad():
        cell._recurrent_kernel_.copy_(layer.recurrent.reshape(heads, 4 * head_size, head_size))
        cell._bias_.copy_(layer.w_x.bias)
    x = case["input_scale"] * torch.randn(BATCH, SEQUENCE_LENGTH, INPUT_SIZE)
    with torch.no_grad():
        wx = torch.nn.functional.linear(x, layer.w_x.weight)  # bias lives in the official cell
        expected, _ = cell(wx.transpose(0, 1).contiguous())
        state = layer.init_state(BATCH, x)
        outputs = []
        for step in range(SEQUENCE_LENGTH):
            y, state = layer.cell(x[:, step], state)
            outputs.append(y)
        ours = torch.stack(outputs, dim=0)
    deviation = (ours - expected).abs()
    return {
        **case,
        "max_abs_deviation": float(deviation.max()),
        "max_abs_value": float(expected.abs().max()),
        "all_finite": bool(torch.isfinite(ours).all() and torch.isfinite(expected).all()),
        "within_tolerance": bool(deviation.max() <= TOLERANCE_ABS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlstm-path", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, default=None, help="wheel file, for its sha256")
    arguments = parser.parse_args()
    require_clean_tree()
    sys.path.insert(0, str(arguments.xlstm_path))
    import xlstm
    from xlstm.blocks.slstm import cell as official

    results = [_run_case(case, official) for case in CASES]
    for result in results:
        print(
            f"seed={result['seed']} H={result['hidden']:3d} NH={result['heads']} "
            f"scale={result['input_scale']:5.1f}  max|dev|={result['max_abs_deviation']:.2e}  "
            f"finite={result['all_finite']}"
        )
    payload: dict[str, Any] = {
        "_provenance": provenance(),
        "official_package": {
            "name": "xlstm",
            "version": xlstm.__version__,
            "backend": "vanilla",
            "wheel_sha256": (
                hashlib.sha256(arguments.wheel.read_bytes()).hexdigest()
                if arguments.wheel
                else None
            ),
        },
        "tolerance_abs": TOLERANCE_ABS,
        "sequence_length": SEQUENCE_LENGTH,
        "cases": results,
        "all_within_tolerance": all(r["within_tolerance"] for r in results),
        "max_abs_deviation_overall": max(r["max_abs_deviation"] for r in results),
    }
    write_artifact("sx2_official_equivalence", payload)
    if not payload["all_within_tolerance"]:
        raise SystemExit("deviation above tolerance")


if __name__ == "__main__":
    main()
