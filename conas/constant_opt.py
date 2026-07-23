"""Constant optimisation -- Algorithm 1 of the paper (Sec. III-B2).

The values of the constant nodes are found by treating the candidate operator as
a small network to be trained: an Adam optimiser (learning rate 0.01) minimises
the mean-squared error between the output of the original convolution layer
``l`` and the output of the graph operator ``l'`` on the same inputs.

This is the "lightweight operator-level adaptation step" of Sec. III-E: it is
what makes candidates worth evaluating without any network-level retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .graph import LayerSpec
from .operator import GraphOperator


@dataclass
class ConstantOptConfig:
    """Hyper-parameters of Algorithm 1."""

    epochs: int = 100
    lr: float = 0.01
    batch_size: int = 8
    #: Spatial size of the synthetic inputs.  Defaults to ``spec.in_hw``.
    in_hw: Optional[tuple] = None
    #: Standard deviation of ``generate_tensor_input``.
    input_std: float = 1.0
    #: Optional gradient clipping; candidate graphs can be badly conditioned.
    grad_clip: Optional[float] = 5.0
    #: Stop early once the relative MSE stops improving by this much.
    tolerance: float = 1e-6
    patience: int = 15


def make_input_sampler(
    spec: LayerSpec,
    cfg: ConstantOptConfig,
    device: torch.device,
    calibration: Optional[torch.Tensor] = None,
) -> Callable[[], torch.Tensor]:
    """Return ``generate_tensor_input()``.

    With ``calibration`` provided (a tensor of real activations captured at the
    layer input) minibatches are drawn from it, which fits the constants on the
    distribution the layer actually sees.  Otherwise inputs are sampled from a
    Gaussian, matching the pseudo-code of Algorithm 1.
    """
    if calibration is not None and len(calibration) > 0:
        pool = calibration.to(device)

        def sample_real() -> torch.Tensor:
            idx = torch.randint(0, pool.shape[0], (min(cfg.batch_size, pool.shape[0]),), device=device)
            return pool.index_select(0, idx)

        return sample_real

    hw = cfg.in_hw or spec.in_hw or (8, 8)

    def sample_synth() -> torch.Tensor:
        return torch.randn(cfg.batch_size, spec.in_channels, hw[0], hw[1], device=device) * cfg.input_std

    return sample_synth


@torch.enable_grad()
def optimize_constants(
    conv: nn.Module,
    operator: GraphOperator,
    cfg: Optional[ConstantOptConfig] = None,
    calibration: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> List[float]:
    """Algorithm 1.  Returns the loss trace.

    Parameters
    ----------
    conv:
        The convolution layer ``l`` being approximated (kept frozen).
    operator:
        The graph operator ``l'`` whose constants are optimised in place.
    optimizer:
        Reuse an existing optimiser to continue a previous run -- this is how
        the ``constant_optimization`` mutation performs *additional* iterations
        on a selected graph without resetting Adam's moments.
    """
    cfg = cfg or ConstantOptConfig()
    device = device or next(operator.parameters(), torch.zeros(1)).device
    conv = conv.to(device).eval()
    operator = operator.to(device).train()

    params = [p for p in operator.parameters() if p.requires_grad]
    if not params:
        return []
    opt = optimizer or torch.optim.Adam(params, lr=cfg.lr)
    sampler = make_input_sampler(operator.spec, cfg, device, calibration)

    losses: List[float] = []
    best = float("inf")
    stale = 0
    for _ in range(cfg.epochs):
        inputs = sampler()
        with torch.no_grad():
            conv_output = conv(inputs)
        graph_output = operator(inputs)
        if graph_output.shape != conv_output.shape:
            raise RuntimeError(
                f"operator output {tuple(graph_output.shape)} != conv output {tuple(conv_output.shape)}"
            )
        loss = torch.mean((graph_output - conv_output) ** 2)
        if not torch.isfinite(loss):
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        value = float(loss.detach())
        losses.append(value)
        if value < best - cfg.tolerance:
            best, stale = value, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    operator.eval()
    return losses


def approximation_error(
    conv: nn.Module,
    operator: GraphOperator,
    n_batches: int = 4,
    cfg: Optional[ConstantOptConfig] = None,
    calibration: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Mean MSE between ``l`` and ``l'`` -- a cheap proxy fitness for smoke tests."""
    cfg = cfg or ConstantOptConfig()
    device = device or next(operator.parameters(), torch.zeros(1)).device
    sampler = make_input_sampler(operator.spec, cfg, device, calibration)
    conv, operator = conv.to(device).eval(), operator.to(device).eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x = sampler()
            total += float(torch.mean((operator(x) - conv(x)) ** 2))
    return total / max(n_batches, 1)
