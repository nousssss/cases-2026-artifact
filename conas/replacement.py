"""Model surgery: replacing a convolution layer ``l`` with a learned operator ``l'``.

Given a network ``A``, CONAS selects one or more convolution layers and replaces
them with learned operators, producing an optimised architecture ``A'``
(paper, Sec. III-A).  The optimisation is performed one layer at a time and can
be applied iteratively to multiple layers.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .graph import ComputationGraph, LayerSpec
from .models.registry import get_module, normalize_layer_name, set_module
from .operator import GraphOperator


class Identity(nn.Module):
    """``f(x) = x`` -- the no-op baseline used in the Q1 ablation (Sec. IV-B).

    Only defined when the layer preserves shape; otherwise a shape-preserving
    fallback is impossible and the ablation reports the layer as inapplicable.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def layer_is_shape_preserving(conv: nn.Conv2d) -> bool:
    return (
        conv.in_channels == conv.out_channels
        and tuple(conv.stride) == (1, 1)
        and all(
            p == d * (k - 1) // 2
            for p, d, k in zip(conv.padding, conv.dilation, conv.kernel_size)
        )
    )


# --------------------------------------------------------------------------- #
# Shape / activation probing
# --------------------------------------------------------------------------- #
@dataclass
class LayerProbe:
    """Everything captured about a layer by a single forward pass."""

    path: str
    conv: nn.Conv2d
    spec: LayerSpec
    in_hw: Tuple[int, int]
    out_hw: Tuple[int, int]
    inputs: Optional[torch.Tensor] = None


@torch.no_grad()
def probe_layer(
    model: nn.Module,
    layer: str,
    example_input: torch.Tensor,
    model_name: str = "",
    capture_inputs: bool = False,
    max_capture: int = 64,
) -> LayerProbe:
    """Run one forward pass to record the layer's spatial shapes (and inputs)."""
    path = normalize_layer_name(model_name, layer) if model_name else layer
    conv = get_module(model, path)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"layer '{path}' is a {type(conv).__name__}, not Conv2d")

    captured: Dict[str, torch.Tensor] = {}

    def hook(_mod, inp, out):
        captured["in"] = inp[0].detach()
        captured["out"] = out.detach()

    handle = conv.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        model(example_input)
    finally:
        handle.remove()
        model.train(was_training)

    x, y = captured["in"], captured["out"]
    spec = LayerSpec.from_conv(conv, in_hw=tuple(x.shape[2:]), name=layer)
    return LayerProbe(
        path=path,
        conv=conv,
        spec=spec,
        in_hw=tuple(x.shape[2:]),
        out_hw=tuple(y.shape[2:]),
        inputs=x[:max_capture].clone() if capture_inputs else None,
    )


@torch.no_grad()
def collect_calibration(
    model: nn.Module,
    layer_path: str,
    loader: Iterable,
    n_samples: int = 128,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Capture real activations at the input of ``layer_path``.

    Used by constant optimisation so that the constants are fitted on the
    distribution the layer actually receives, rather than on Gaussian noise.
    """
    conv = get_module(model, layer_path)
    chunks: List[torch.Tensor] = []
    total = 0

    def hook(_mod, inp, _out):
        nonlocal total
        if total < n_samples:
            take = inp[0].detach()[: n_samples - total].cpu()
            chunks.append(take)
            total += take.shape[0]

    handle = conv.register_forward_hook(hook)
    model = model.to(device).eval()
    try:
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(x.to(device))
            if total >= n_samples:
                break
    finally:
        handle.remove()
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


# --------------------------------------------------------------------------- #
# Replacement
# --------------------------------------------------------------------------- #
def replace_layer(model: nn.Module, layer_path: str, new_module: nn.Module) -> nn.Module:
    """Replace ``layer_path`` in-place and return the model."""
    set_module(model, layer_path, new_module)
    return model


def build_operator(
    graph: ComputationGraph,
    state_dict: Optional[dict] = None,
    chunk: Optional[int] = None,
) -> GraphOperator:
    op = GraphOperator(graph, chunk=chunk)
    if state_dict:
        op.load_state_dict(state_dict)
    return op


@contextlib.contextmanager
def temporarily_replaced(model: nn.Module, layer_path: str, new_module: nn.Module):
    """Context manager that swaps a layer and restores it afterwards.

    The search evaluates hundreds of candidates against the same network, so it
    never mutates ``A`` permanently -- only the accepted operator is committed.
    """
    original = get_module(model, layer_path)
    set_module(model, layer_path, new_module)
    try:
        yield model
    finally:
        set_module(model, layer_path, original)


def apply_operators(
    model: nn.Module,
    operators: Dict[str, nn.Module],
) -> nn.Module:
    """Commit a set of ``{layer_path: operator}`` replacements to ``A``."""
    for path, op in operators.items():
        set_module(model, path, op)
    return model


def count_replaced(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, GraphOperator))
