"""CONAS -- Compiler Optimization with Neural Architecture Search.

Reference implementation of *"From Scratch or from Structure? Investigating the
Trade-off in Replacing Convolution"*.

CONAS searches for computation graphs built from low-level mathematical
primitives that replace an entire convolution layer, and folds compiler-level
optimisation (MLIR) into the evaluation loop so that discovered operators are
selected for deployment efficiency as well as accuracy.

Quick start
-----------
>>> from conas import CONASSearch, SearchConfig, build_model, probe_layer
>>> model = build_model("resnet20")
>>> probe = probe_layer(model, "1.1.conv1", torch.randn(8, 3, 32, 32), "resnet20")
>>> result = CONASSearch(model, probe, val_loader, SearchConfig(init="struct")).run()
>>> result.summary()
"""

from .constant_opt import ConstantOptConfig, optimize_constants
from .data import DataConfig, build_loaders
from .evaluation import (
    Candidate,
    LatencyConfig,
    evaluate_accuracy,
    measure_latency,
    select_best,
    speedup,
)
from .evolution import apply_mutation, crossover, mutate, tournament_select
from .finetune import FineTuneConfig, fine_tune
from .generation import GenerationConfig, GraphGenerator
from .graph import ComputationGraph, LayerSpec, Node
from .init_struct import (
    conv_equivalent_graph,
    gated_lowrank_patch_graph,
    lowrank_relu_patch_graph,
    structured_population,
)
from .models import build_model, list_conv_layers, normalize_layer_name
from .multilayer import MultiLayerConfig, sequential_greedy_search
from .operator import GraphOperator, is_valid, numeric_probe
from .replacement import collect_calibration, probe_layer, replace_layer
from .search import CONASSearch, SearchConfig, SearchResult

__version__ = "0.1.0"

__all__ = [
    "CONASSearch",
    "SearchConfig",
    "SearchResult",
    "ComputationGraph",
    "LayerSpec",
    "Node",
    "GraphOperator",
    "GraphGenerator",
    "GenerationConfig",
    "ConstantOptConfig",
    "optimize_constants",
    "conv_equivalent_graph",
    "gated_lowrank_patch_graph",
    "lowrank_relu_patch_graph",
    "structured_population",
    "crossover",
    "mutate",
    "apply_mutation",
    "tournament_select",
    "evaluate_accuracy",
    "measure_latency",
    "select_best",
    "speedup",
    "Candidate",
    "LatencyConfig",
    "probe_layer",
    "replace_layer",
    "collect_calibration",
    "build_model",
    "list_conv_layers",
    "normalize_layer_name",
    "build_loaders",
    "DataConfig",
    "fine_tune",
    "FineTuneConfig",
    "sequential_greedy_search",
    "MultiLayerConfig",
    "is_valid",
    "numeric_probe",
]
