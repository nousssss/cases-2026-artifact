"""Test suite for CONAS.

Run with ``pytest tests/ -v``.  Everything here is CPU-only and takes well under
a minute; no dataset or checkpoint is required.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

from conas.compiler import to_mlir
from conas.constant_opt import ConstantOptConfig, approximation_error, optimize_constants
from conas.evaluation import Candidate, select_best
from conas.evolution import MUTATION_TYPES, apply_mutation, crossover, select_subgraph
from conas.generation import GenerationConfig, GraphGenerator
from conas.graph import ComputationGraph, GraphError, LayerSpec, ShapeError
from conas.init_struct import (
    conv_equivalent_graph,
    gated_lowrank_patch_graph,
    lowrank_relu_patch_graph,
    structured_population,
)
from conas.models import build_model, list_conv_layers, normalize_layer_name
from conas.multilayer import shape_compatible
from conas.operator import GraphOperator, is_valid, numeric_probe
from conas.replacement import probe_layer, temporarily_replaced
from conas.utils import structural_hash


@pytest.fixture
def spec() -> LayerSpec:
    return LayerSpec(16, 16, (3, 3), padding=(1, 1), in_hw=(32, 32), name="1.1.conv1")


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


@pytest.fixture
def generator(spec, rng) -> GraphGenerator:
    return GraphGenerator(spec, GenerationConfig(), rng)


# --------------------------------------------------------------------------- #
# Layer specification
# --------------------------------------------------------------------------- #
def test_layer_spec_geometry(spec):
    assert spec.K == 16 * 3 * 3
    assert spec.out_hw == (32, 32)
    assert len(spec.center_index) == spec.in_channels
    # Centre of the first channel's 3x3 patch is index 4.
    assert spec.center_index[0] == 4


def test_layer_spec_from_conv():
    conv = nn.Conv2d(8, 12, 5, stride=2, padding=2, dilation=1, bias=True)
    s = LayerSpec.from_conv(conv, in_hw=(16, 16))
    assert (s.in_channels, s.out_channels, s.K) == (8, 12, 8 * 25)
    assert s.out_hw == (8, 8)


def test_layer_spec_roundtrip(spec):
    assert LayerSpec.from_dict(spec.to_dict()) == spec


# --------------------------------------------------------------------------- #
# Graph structure and shapes
# --------------------------------------------------------------------------- #
def test_generated_graphs_are_valid(generator):
    for _ in range(30):
        g = generator.generate()
        assert g.is_structurally_valid()
        assert g.n_primitives > 0
        # Shape correctness is enforced at the output only.
        assert g.infer_shapes()[g.output][1] == g.spec.out_channels


def test_output_shape_matches_convolution(spec, generator):
    x = torch.randn(2, spec.in_channels, *spec.in_hw)
    conv = nn.Conv2d(spec.in_channels, spec.out_channels, 3, padding=1, bias=False)
    for _ in range(10):
        op = GraphOperator(generator.generate())
        assert op(x).shape == conv(x).shape


def test_cycle_is_rejected(spec):
    g = ComputationGraph(spec)
    a = g.add_node("relu", inputs=[g.patch_id])
    b = g.add_node("relu", inputs=[a])
    g.nodes[a].inputs = [b]  # close the loop
    g.output = b
    with pytest.raises(GraphError):
        g.topo_order()
    assert not g.is_structurally_valid()


def test_shape_mismatch_is_rejected(spec):
    g = ComputationGraph(spec)
    # patch is (M, K), center is (M, Cin): elementwise add cannot type-check.
    bad = g.add_node("matadd", inputs=[g.patch_id, g.center_id])
    g.output = bad
    with pytest.raises(ShapeError):
        g.infer_shapes()


def test_wrong_output_width_is_rejected(spec):
    g = ComputationGraph(spec)
    g.output = g.add_node("relu", inputs=[g.patch_id])  # width K, not Cout
    with pytest.raises(ShapeError):
        g.check_structure()


def test_pruning_removes_unused_nodes(spec):
    g = ComputationGraph(spec)
    theta = g.add_node("const", const_dims=(spec.K, spec.out_channels))
    used = g.add_node("matmul", inputs=[g.patch_id, theta])
    g.add_node("tanh", inputs=[g.patch_id])  # dead branch
    g.add_node("sin", inputs=[g.center_id])  # dead branch
    g.output = used
    assert g.unused_fraction() > 0
    assert g.prune() == 2
    assert g.is_structurally_valid()
    assert g.unused_fraction() == 0


def test_numeric_probe_catches_invalid_domain(spec):
    """log of a signed activation is rejected; log of a positive one is not."""
    bad = ComputationGraph(spec)
    theta = bad.add_node("const", const_dims=(spec.K, spec.out_channels))
    mm = bad.add_node("matmul", inputs=[bad.patch_id, theta])
    bad.output = bad.add_node("log", inputs=[mm])
    assert not numeric_probe(bad, n_samples=64, seed=0)

    good = ComputationGraph(spec)
    theta = good.add_node("const", const_dims=(spec.K, spec.out_channels))
    mm = good.add_node("matmul", inputs=[good.patch_id, theta])
    positive = good.add_node("abs", inputs=[mm])
    good.output = good.add_node("sqrt", inputs=[positive])
    assert is_valid(good)


def test_serialization_roundtrip(generator):
    g = generator.generate()
    restored = ComputationGraph.from_dict(g.to_dict())
    assert structural_hash(restored) == structural_hash(g)
    assert restored.is_structurally_valid()

    x = torch.randn(2, g.spec.in_channels, *g.spec.in_hw)
    op = GraphOperator(g)
    clone = GraphOperator(restored)
    clone.load_state_dict(op.state_dict())
    torch.testing.assert_close(op(x), clone(x))


def test_structural_hash_ignores_constant_values(spec):
    a = conv_equivalent_graph(spec)
    b = conv_equivalent_graph(spec)
    assert structural_hash(a) == structural_hash(b)


# --------------------------------------------------------------------------- #
# Structure-guided initialisation
# --------------------------------------------------------------------------- #
def test_conv_equivalent_graph_reproduces_the_convolution(spec):
    """The manual seed graph is numerically identical to the layer it mimics."""
    conv = nn.Conv2d(spec.in_channels, spec.out_channels, 3, padding=1, bias=False)
    op = GraphOperator(conv_equivalent_graph(spec, conv))
    x = torch.randn(4, spec.in_channels, *spec.in_hw)
    torch.testing.assert_close(op(x), conv(x), atol=1e-4, rtol=1e-4)


def test_conv_equivalent_graph_handles_bias_and_stride():
    conv = nn.Conv2d(8, 12, 3, stride=2, padding=1, bias=True)
    spec = LayerSpec.from_conv(conv, in_hw=(16, 16))
    op = GraphOperator(conv_equivalent_graph(spec, conv))
    x = torch.randn(2, 8, 16, 16)
    torch.testing.assert_close(op(x), conv(x), atol=1e-4, rtol=1e-4)


def test_conv_equivalent_graph_handles_depthwise():
    """Grouped convolutions expand to a block-sparse dense matrix."""
    conv = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
    spec = LayerSpec.from_conv(conv, in_hw=(8, 8))
    op = GraphOperator(conv_equivalent_graph(spec, conv))
    x = torch.randn(2, 16, 8, 8)
    torch.testing.assert_close(op(x), conv(x), atol=1e-4, rtol=1e-4)


def test_paper_example_dags_match_their_written_form(spec):
    """Eq. (1) and Eq. (2) have the primitive counts stated in Sec. III-E."""
    eq1 = gated_lowrank_patch_graph(spec)
    hist = eq1.primitive_histogram()
    assert hist["matmul"] == 3
    assert hist["matadd"] == 3
    assert hist["elemmul"] == 1
    assert hist["tanh"] == 1 and hist["sigmoid"] == 1
    assert eq1.n_constants == 5

    eq2 = lowrank_relu_patch_graph(spec)
    hist = eq2.primitive_histogram()
    assert hist["matmul"] == 2
    assert hist["matadd"] == 2
    assert hist["relu"] == 1
    assert eq2.n_constants == 4
    # The from-scratch example is the more compact of the two.
    assert eq2.n_primitives < eq1.n_primitives


def test_paper_example_dags_execute(spec):
    x = torch.randn(2, spec.in_channels, *spec.in_hw)
    for factory in (gated_lowrank_patch_graph, lowrank_relu_patch_graph):
        g = factory(spec)
        assert is_valid(g)
        assert GraphOperator(g)(x).shape == (2, spec.out_channels, *spec.in_hw)


def test_structured_population_is_diverse_but_valid(spec, rng):
    conv = nn.Conv2d(spec.in_channels, spec.out_channels, 3, padding=1, bias=False)
    pop = structured_population(spec, 16, conv=conv, rng=rng, n_elites=4)
    assert len(pop) == 16
    assert all(g.is_structurally_valid() for g in pop)
    # Elites are untouched; the rest carry mutations, so hashes must vary.
    assert len({structural_hash(g) for g in pop}) > 1


# --------------------------------------------------------------------------- #
# Evolution
# --------------------------------------------------------------------------- #
def test_subgraph_selection_is_bounded_and_connected(generator, rng):
    for _ in range(20):
        g = generator.generate()
        sub = select_subgraph(g, rng)
        assert sub and sub.issubset(set(g.nodes))
        # A single crossover must not swallow the whole parent.
        n_ops = g.n_primitives
        assert len(sub) <= max(1, int(0.4 * n_ops))


def test_crossover_children_are_valid_and_keep_the_interface(generator, rng, spec):
    pop = [generator.generate() for _ in range(12)]
    x = torch.randn(2, spec.in_channels, *spec.in_hw)
    produced = 0
    for _ in range(40):
        a, b = rng.sample(pop, 2)
        for child in crossover(a, b, generator, rng):
            produced += 1
            assert child.is_structurally_valid()
            # External interface stays fixed: same input, same output shape.
            assert GraphOperator(child)(x).shape == (2, spec.out_channels, *spec.in_hw)
    assert produced > 0, "crossover never produced a valid child"


def test_every_mutation_type_is_applicable(generator, rng):
    pop = [generator.generate() for _ in range(12)]
    for kind in MUTATION_TYPES:
        successes = sum(apply_mutation(g, kind, generator, rng) is not None for g in pop)
        assert successes > 0, f"mutation '{kind}' never succeeded"


def test_constant_optimization_mutation_is_structure_preserving(generator, rng):
    g = generator.generate()
    out = apply_mutation(g, "constant_optimization", generator, rng)
    assert structural_hash(out) == structural_hash(g)


def test_mutation_output_stays_valid(generator, rng):
    g = generator.generate()
    for _ in range(25):
        kind = rng.choice(MUTATION_TYPES[:-1])
        mutated = apply_mutation(g, kind, generator, rng)
        if mutated is not None:
            assert mutated.is_structurally_valid()
            g = mutated


# --------------------------------------------------------------------------- #
# Constant optimisation (Algorithm 1)
# --------------------------------------------------------------------------- #
def test_constant_optimization_reduces_the_mse(spec):
    torch.manual_seed(0)
    conv = nn.Conv2d(spec.in_channels, spec.out_channels, 3, padding=1, bias=False)
    op = GraphOperator(lowrank_relu_patch_graph(spec, rank=32))
    cfg = ConstantOptConfig(epochs=150, batch_size=4, in_hw=(8, 8))

    before = approximation_error(conv, op, cfg=cfg)
    losses = optimize_constants(conv, op, cfg)
    after = approximation_error(conv, op, cfg=cfg)

    assert losses, "Algorithm 1 produced no loss trace"
    assert after < before, f"MSE did not improve: {before:.5f} -> {after:.5f}"


def test_constant_optimization_leaves_the_conv_frozen(spec):
    conv = nn.Conv2d(spec.in_channels, spec.out_channels, 3, padding=1, bias=False)
    reference = conv.weight.detach().clone()
    op = GraphOperator(lowrank_relu_patch_graph(spec))
    optimize_constants(conv, op, ConstantOptConfig(epochs=20, batch_size=2, in_hw=(8, 8)))
    torch.testing.assert_close(conv.weight.detach(), reference)


# --------------------------------------------------------------------------- #
# Selection rule
# --------------------------------------------------------------------------- #
def test_accuracy_equivalence_prefers_the_faster_operator():
    """Within the equivalence window, latency decides."""
    fast = Candidate(graph=None, accuracy=90.5, latency_ms=1.0)
    accurate = Candidate(graph=None, accuracy=91.0, latency_ms=5.0)
    assert select_best([accurate, fast], accuracy_equivalence=1.0) is fast
    # Outside the window, accuracy wins.
    assert select_best([accurate, fast], accuracy_equivalence=0.1) is accurate


def test_select_best_ignores_unmeasured_candidates():
    a = Candidate(graph=None, accuracy=90.0, latency_ms=None)
    b = Candidate(graph=None, accuracy=89.8, latency_ms=2.0)
    assert select_best([a, b], accuracy_equivalence=1.0) is b


# --------------------------------------------------------------------------- #
# Model surgery
# --------------------------------------------------------------------------- #
def test_layer_name_normalization():
    assert normalize_layer_name("resnet20", "1.1.conv1") == "layer1.1.conv1"
    assert normalize_layer_name("resnet20", "layer2.0.conv2") == "layer2.0.conv2"
    # ConvNeXt: features = [stem, stage0, down, stage1, ...]
    assert normalize_layer_name("convnext_tiny", "2.3.dwconv") == "features.5.3.block.0"


def test_probe_and_replace_preserves_forward_shape():
    model = build_model("resnet20")
    x = torch.randn(2, 3, 32, 32)
    reference = model(x)

    probe = probe_layer(model, "1.1.conv1", x, "resnet20", capture_inputs=True)
    assert probe.spec.in_channels == 16 and probe.in_hw == (32, 32)
    assert probe.inputs is not None

    op = GraphOperator(conv_equivalent_graph(probe.spec, probe.conv))
    with temporarily_replaced(model, probe.path, op):
        assert model(x).shape == reference.shape
    # The layer must be restored on exit.
    assert isinstance(model.layer1[1].conv1, nn.Conv2d)


def test_conv_equivalent_replacement_preserves_predictions():
    """Swapping in the exact seed graph must not change the network's output."""
    torch.manual_seed(0)
    model = build_model("resnet20").eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        reference = model(x)
    probe = probe_layer(model, "2.0.conv2", x, "resnet20")
    op = GraphOperator(conv_equivalent_graph(probe.spec, probe.conv))
    with torch.no_grad(), temporarily_replaced(model, probe.path, op):
        replaced = model(x)
    torch.testing.assert_close(replaced, reference, atol=1e-3, rtol=1e-3)


def test_list_conv_layers_skips_projections():
    model = build_model("resnet20")
    layers = list_conv_layers(model)
    assert "layer1.0.conv1" in layers
    assert all("shortcut" not in name for name in layers)


def test_operator_chunking_matches_unchunked(spec):
    g = lowrank_relu_patch_graph(spec)
    x = torch.randn(2, spec.in_channels, *spec.in_hw)
    full = GraphOperator(g)
    chunked = GraphOperator(g, chunk=256)
    chunked.load_state_dict(full.state_dict())
    torch.testing.assert_close(full(x), chunked(x))


def test_shape_compatibility_for_operator_reuse():
    a = LayerSpec(16, 16, (3, 3), padding=(1, 1), in_hw=(32, 32))
    b = LayerSpec(16, 16, (3, 3), padding=(1, 1), in_hw=(16, 16))  # different spatial size
    c = LayerSpec(32, 32, (3, 3), padding=(1, 1), in_hw=(16, 16))
    assert shape_compatible(a, b)
    assert not shape_compatible(a, c)


# --------------------------------------------------------------------------- #
# MLIR emission
# --------------------------------------------------------------------------- #
def test_mlir_emission_is_well_formed(spec):
    text = to_mlir(gated_lowrank_patch_graph(spec), rows=1024)
    assert text.startswith("#map")  # aliases precede the module
    assert "func.func @conas_operator" in text
    assert "linalg.matmul" in text
    assert "linalg.generic" in text
    assert "math.tanh" in text
    assert text.count("{") == text.count("}")
    # Static shapes: M x K in, M x Cout out.
    assert f"tensor<1024x{spec.K}xf32>" in text
    assert f"-> tensor<1024x{spec.out_channels}xf32>" in text


def test_mlir_declares_every_constant_as_an_argument(spec, generator):
    g = generator.generate()
    text = to_mlir(g, rows=64)
    for nid, node in g.nodes.items():
        if node.op == "const":
            assert f"%theta{nid}:" in text


def test_mlir_covers_all_generated_primitives(generator):
    """Every primitive the generator can emit must have a lowering."""
    for _ in range(25):
        g = generator.generate()
        to_mlir(g, rows=64)  # raises NotImplementedError on a missing lowering
