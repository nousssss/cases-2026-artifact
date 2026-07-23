"""Evaluation method (paper, Sec. III-E).

Each candidate operator ``l'`` is evaluated by replacing the original layer ``l``
in model ``A``, giving ``A'``.  The validation accuracy of ``A'`` is computed
**using inference only, without any training during search**, and serves as the
fitness score to maximise.  When several candidates reach similar accuracy they
are compared on execution time; "similar" is defined by an
accuracy-equivalence threshold.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: Iterable,
    device: torch.device = torch.device("cpu"),
    max_batches: Optional[int] = None,
    topk: int = 1,
) -> float:
    """Top-``k`` accuracy in percent.

    ``max_batches`` caps the number of validation batches used as the fitness
    signal, which is what makes a 200-iteration x 300-candidate search tractable.
    """
    model = model.to(device).eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            return 0.0
        pred = logits.topk(min(topk, logits.shape[1]), dim=1).indices
        correct += int((pred == y.unsqueeze(1)).any(dim=1).sum())
        total += y.numel()
    return 100.0 * correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
@dataclass
class LatencyConfig:
    """How inference latency is measured on a target device."""

    batch_size: int = 1
    warmup: int = 10
    repeats: int = 50
    threads: Optional[int] = None
    #: ``median`` is robust to scheduler noise on shared machines; ``mean`` and
    #: ``min`` are also available.
    reduction: str = "median"


def _reduce(samples: Sequence[float], how: str) -> float:
    if how == "mean":
        return statistics.fmean(samples)
    if how == "min":
        return min(samples)
    return statistics.median(samples)


@torch.no_grad()
def measure_latency(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    cfg: Optional[LatencyConfig] = None,
) -> float:
    """Wall-clock inference latency in milliseconds."""
    cfg = cfg or LatencyConfig()
    prev_threads = torch.get_num_threads()
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    try:
        module = module.to(device).eval()
        x = torch.randn(cfg.batch_size, *input_shape, device=device)
        for _ in range(cfg.warmup):
            module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(cfg.repeats):
            start = time.perf_counter()
            module(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
        return _reduce(samples, cfg.reduction)
    finally:
        torch.set_num_threads(prev_threads)


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    """``Latency(A) / Latency(A')``; > 1 means the replacement is faster."""
    return baseline_ms / max(candidate_ms, 1e-9)


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One evaluated operator: fitness plus measured cost."""

    graph: object
    accuracy: float
    latency_ms: Optional[float] = None
    state_dict: Optional[dict] = None
    mse: Optional[float] = None
    meta: dict = field(default_factory=dict)


def select_best(
    candidates: Sequence[Candidate],
    accuracy_equivalence: float = 1.0,
) -> Candidate:
    """Pick the operator to keep.

    Accuracy is the primary criterion.  Among candidates whose accuracy is
    within ``accuracy_equivalence`` percentage points of the best, the
    lower-latency one is selected (paper, Sec. IV-A: "if two candidates differ by
    less than 1% absolute accuracy, the lower-latency operator is selected").
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    best_acc = max(c.accuracy for c in candidates)
    tied = [c for c in candidates if best_acc - c.accuracy <= accuracy_equivalence]
    measured = [c for c in tied if c.latency_ms is not None]
    if measured:
        return min(measured, key=lambda c: c.latency_ms)
    return max(tied, key=lambda c: c.accuracy)


def accuracy_change(baseline: float, candidate: float) -> float:
    """Signed accuracy delta in percentage points, as reported in Figs. 4/5/8."""
    return candidate - baseline
