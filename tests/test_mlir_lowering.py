"""Correctness of the torch-mlir lowering path (paper, Sec. III-F).
"""

from __future__ import annotations

import pytest
import torch

from conas.compiler.backend import MLIRBackend, MLIRToolchain
from conas.compiler.torch_mlir_export import example_conv_input
from conas.graph import LayerSpec
from conas.init_struct import gated_lowrank_patch_graph, lowrank_relu_patch_graph
from conas.operator import GraphOperator

_TOOLCHAIN = MLIRToolchain()

pytestmark = pytest.mark.skipif(
    not _TOOLCHAIN.available(),
    reason=(
        "torch-mlir toolchain not configured: set CONAS_MLIR_SOLUTION_BUILD_DIR "
        "and CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR, and put torch-mlir's Python "
        "bindings on PYTHONPATH (see Convert-PyTorch-models-to-MLIR/README.md)"
    ),
)


def _small_spec() -> LayerSpec:
    # Kept deliberately small: wrap_for_execution embeds the example input as
    # a literal MLIR dense constant, so the text size grows with N*Cin*H*W.
    return LayerSpec(4, 6, (3, 3), padding=(1, 1), in_hw=(6, 6), name="probe")


@pytest.fixture(params=[gated_lowrank_patch_graph, lowrank_relu_patch_graph])
def graph_and_op(request):
    spec = _small_spec()
    graph = request.param(spec, rank=8)
    torch.manual_seed(0)
    op = GraphOperator(graph).eval()
    return graph, op


def test_lowered_execution_matches_eager(graph_and_op, tmp_path):
    graph, op = graph_and_op
    x = example_conv_input(graph.spec, batch_size=1, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        expected = op(x)

    backend = MLIRBackend(toolchain=_TOOLCHAIN)
    actual = backend.execute(graph, example_input=x, op=op, out_dir=str(tmp_path))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
