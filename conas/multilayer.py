"""Multi-layer optimisation (paper, Sec. IV-F).

CONAS follows a **sequential greedy** replacement strategy rather than a fully
joint search.  Starting from the original model ``A(0) = A``, at step ``k`` a
target convolution layer ``l_ik`` is selected in the current model ``A(k-1)``, a
replacement operator is searched for while the rest of the model is kept fixed,
and a new model ``A(k)`` is obtained after accepting the selected operator.

Each candidate is still evaluated through the validation accuracy of the *whole*
modified model, so when several replacements are applied iteratively the effect
of each new replacement is assessed in the context of the model in which it will
be deployed.

A fully joint search over ``m`` layers would require evolving a tuple of ``m``
DAGs and evaluating their combined effect, which greatly increases the search
dimensionality, compiler-generation cost and number of candidate combinations.
The sequential formulation instead lets CONAS target latency-critical layers
progressively, stop once an accuracy-latency budget is reached, and reuse
discovered operators across shape-compatible layers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .constant_opt import ConstantOptConfig, optimize_constants
from .evaluation import evaluate_accuracy, measure_latency
from .graph import ComputationGraph, LayerSpec
from .operator import GraphOperator
from .replacement import collect_calibration, probe_layer, replace_layer, temporarily_replaced
from .search import CONASSearch, SearchConfig, SearchResult
from .utils import get_logger


@dataclass
class MultiLayerConfig:
    """Budget-driven stopping rule for the greedy loop."""

    #: Maximum acceptable accuracy drop, in percentage points, relative to ``A``.
    #: This is the ``gamma`` of the formal statement in Sec. III-A.
    max_accuracy_drop: float = 2.0
    #: Stop once this end-to-end speedup is reached.
    target_speedup: Optional[float] = None
    #: Maximum number of layers to replace.
    max_layers: int = 5
    #: Try reusing already-discovered operators on shape-compatible layers
    #: before launching a fresh search.
    enable_reuse: bool = True
    #: Constant-optimisation epochs used when adapting a reused operator.
    reuse_const_opt_epochs: int = 100
    #: Accept a reused operator only if it costs less than this many points.
    reuse_tolerance: float = 1.0


@dataclass
class MultiLayerResult:
    baseline_accuracy: float
    baseline_latency_ms: float
    steps: List[dict] = field(default_factory=list)
    operators: Dict[str, ComputationGraph] = field(default_factory=dict)
    final_accuracy: Optional[float] = None
    final_latency_ms: Optional[float] = None

    @property
    def speedup(self) -> Optional[float]:
        if self.final_latency_ms:
            return self.baseline_latency_ms / self.final_latency_ms
        return None

    def summary(self) -> dict:
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "final_accuracy": self.final_accuracy,
            "accuracy_change": (self.final_accuracy or 0) - self.baseline_accuracy,
            "baseline_latency_ms": self.baseline_latency_ms,
            "final_latency_ms": self.final_latency_ms,
            "speedup": self.speedup,
            "replaced_layers": list(self.operators),
            "steps": self.steps,
        }


def shape_compatible(a: LayerSpec, b: LayerSpec) -> bool:
    """Two layers accept the same operator when their patch interfaces agree."""
    return (
        a.K == b.K
        and a.in_channels == b.in_channels
        and a.out_channels == b.out_channels
        and a.kernel_size == b.kernel_size
    )


def adapt_operator(
    graph: ComputationGraph,
    state_dict: dict,
    target_spec: LayerSpec,
    conv: nn.Module,
    epochs: int = 100,
    calibration: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[GraphOperator, ComputationGraph]:
    """Re-fit a discovered operator to a new, shape-compatible layer.

    "In practice, this only requires a small number of constant optimization
    iterations to adapt the operator to the new layer."
    """
    retargeted = ComputationGraph.from_dict({**graph.to_dict(), "spec": target_spec.to_dict()})
    op = GraphOperator(retargeted).to(device)
    op.load_state_dict(state_dict)
    cfg = ConstantOptConfig(epochs=epochs, in_hw=target_spec.in_hw)
    optimize_constants(conv, op, cfg, calibration=calibration, device=device)
    return op, retargeted


def sequential_greedy_search(
    model: nn.Module,
    layers: Sequence[str],
    train_loader,
    val_loader,
    search_cfg: SearchConfig,
    multi_cfg: Optional[MultiLayerConfig] = None,
    model_name: str = "",
    device: torch.device = torch.device("cpu"),
    logger=None,
) -> Tuple[nn.Module, MultiLayerResult]:
    """Replace ``layers`` one at a time, each time on top of the current model."""
    multi_cfg = multi_cfg or MultiLayerConfig()
    logger = logger or get_logger()
    model = copy.deepcopy(model).to(device)

    example = next(iter(val_loader))[0].to(device)
    input_shape = tuple(example.shape[1:])
    baseline_acc = evaluate_accuracy(model, val_loader, device)
    baseline_lat = measure_latency(model, input_shape, device, search_cfg.latency)
    logger.info(f"A(0): accuracy {baseline_acc:.2f}%, latency {baseline_lat:.3f} ms")

    result = MultiLayerResult(baseline_accuracy=baseline_acc, baseline_latency_ms=baseline_lat)
    discovered: List[Tuple[LayerSpec, ComputationGraph, dict]] = []

    for k, layer in enumerate(layers[: multi_cfg.max_layers]):
        probe = probe_layer(model, layer, example, model_name, capture_inputs=True)
        calibration = collect_calibration(model, probe.path, val_loader, 128, device)
        conv = copy.deepcopy(probe.conv).to(device).eval()

        chosen_op: Optional[GraphOperator] = None
        chosen_graph: Optional[ComputationGraph] = None
        source = "search"

        # 1) Try to reuse an operator discovered for a shape-compatible layer.
        if multi_cfg.enable_reuse:
            for spec, graph, state in discovered:
                if not shape_compatible(spec, probe.spec):
                    continue
                op, retargeted = adapt_operator(
                    graph, state, probe.spec, conv,
                    multi_cfg.reuse_const_opt_epochs, calibration, device,
                )
                with temporarily_replaced(model, probe.path, op):
                    acc = evaluate_accuracy(model, val_loader, device, search_cfg.fitness_batches)
                if baseline_acc - acc <= multi_cfg.reuse_tolerance:
                    chosen_op, chosen_graph, source = op, retargeted, "reuse"
                    logger.info(f"step {k}: reused an operator on {layer} (acc {acc:.2f}%)")
                    break

        # 2) Otherwise run a fresh search on the *current* model A(k-1).
        if chosen_op is None:
            search = CONASSearch(model, probe, val_loader, search_cfg, device, calibration, logger)
            res: SearchResult = search.run()
            chosen_graph = res.graph
            chosen_op = GraphOperator(res.graph, chunk=search_cfg.operator_chunk).to(device)
            chosen_op.load_state_dict(res.state_dict)

        # 3) Accept the operator, then re-measure the whole model.
        replace_layer(model, probe.path, chosen_op)
        acc = evaluate_accuracy(model, val_loader, device)
        lat = measure_latency(model, input_shape, device, search_cfg.latency)
        step = {
            "step": k,
            "layer": layer,
            "source": source,
            "accuracy": acc,
            "accuracy_change": acc - baseline_acc,
            "latency_ms": lat,
            "speedup": baseline_lat / max(lat, 1e-9),
            "primitives": chosen_graph.n_primitives if chosen_graph else None,
        }
        result.steps.append(step)
        result.operators[probe.path] = chosen_graph
        discovered.append((probe.spec, chosen_graph, chosen_op.state_dict()))
        logger.info(
            f"A({k + 1}): {layer} replaced -> accuracy {acc:.2f}% "
            f"({acc - baseline_acc:+.2f} pts), speedup {step['speedup']:.2f}x"
        )

        # 4) Stop once the accuracy-latency budget is reached.
        if baseline_acc - acc > multi_cfg.max_accuracy_drop:
            logger.info("accuracy budget exhausted; stopping")
            break
        if multi_cfg.target_speedup and step["speedup"] >= multi_cfg.target_speedup:
            logger.info("target speedup reached; stopping")
            break

    result.final_accuracy = evaluate_accuracy(model, val_loader, device)
    result.final_latency_ms = measure_latency(model, input_shape, device, search_cfg.latency)
    return model, result
