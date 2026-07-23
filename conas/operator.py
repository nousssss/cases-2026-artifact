"""Turning a :class:`~conas.graph.ComputationGraph` into an executable module.

``GraphOperator`` is a drop-in replacement for a ``nn.Conv2d``: it takes the
same input tensor and produces an output tensor of the same shape
(paper, Sec. III-A / III-C1).  Internally it materialises the patch view of the
input and evaluates the DAG node by node.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import CONST, DATA, ComputationGraph, GraphError, LayerSpec, ShapeError
from .primitives import PRIMITIVES


class GraphOperator(nn.Module):
    """Executable form of a candidate operator ``l'``.

    Parameters
    ----------
    graph:
        A structurally valid computation graph.
    chunk:
        Optional row-chunk size for the ``M = N.Ho.Wo`` dimension.  Large
        ImageNet layers produce a very wide patch matrix; chunking trades a
        little speed for a much smaller peak memory footprint.  ``None``
        evaluates the whole batch at once.
    strict:
        Evaluate primitives with their exact mathematical definition instead of
        the numerically guarded variants.  Used by the validity probe.
    """

    def __init__(self, graph: ComputationGraph, chunk: Optional[int] = None, strict: bool = False):
        super().__init__()
        graph.check_structure()
        self.graph = graph
        self.spec: LayerSpec = graph.spec
        self.chunk = chunk
        self.strict = strict

        self._order: List[int] = graph.topo_order()
        self._shapes = graph.infer_shapes()

        self.consts = nn.ParameterDict()
        for nid in self._order:
            node = graph.nodes[nid]
            if node.op != "const":
                continue
            dims = tuple(node.const_dims or ())
            if node.init_values is not None:
                t = torch.tensor(node.init_values, dtype=torch.float32).reshape(dims)
            else:
                lo, hi = node.init_range
                t = torch.empty(*dims).uniform_(lo, hi)
            self.consts[str(nid)] = nn.Parameter(t)

        self.register_buffer(
            "center_idx", torch.tensor(self.spec.center_index, dtype=torch.long), persistent=False
        )

    # -- helpers ------------------------------------------------------------- #
    def _patch_view(self, x: torch.Tensor):
        """im2col: ``(N, Cin, H, W) -> (M, K)`` with ``M = N.Ho.Wo``."""
        n = x.shape[0]
        cols = F.unfold(
            x,
            kernel_size=self.spec.kernel_size,
            dilation=self.spec.dilation,
            padding=self.spec.padding,
            stride=self.spec.stride,
        )  # (N, K, L)
        length = cols.shape[-1]
        p = cols.transpose(1, 2).reshape(n * length, -1)  # (M, K)
        return p, n, length

    def _out_hw(self, x: torch.Tensor):
        hw = []
        for i, size in enumerate(x.shape[2:]):
            eff_k = self.spec.dilation[i] * (self.spec.kernel_size[i] - 1) + 1
            hw.append((size + 2 * self.spec.padding[i] - eff_k) // self.spec.stride[i] + 1)
        return hw[0], hw[1]

    def _eval_dag(self, p: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        env: Dict[int, torch.Tensor] = {}
        g = self.graph
        for nid in self._order:
            node = g.nodes[nid]
            if node.op == "patch":
                env[nid] = p
            elif node.op == "center":
                env[nid] = xc
            elif node.op == "const":
                env[nid] = self.consts[str(nid)]
            else:
                prim = PRIMITIVES[node.op]
                args = [env[i] for i in node.inputs]
                env[nid] = prim(*args, strict=self.strict)
        assert g.output is not None
        return env[g.output]

    # -- forward -------------------------------------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, n, length = self._patch_view(x)
        ho, wo = self._out_hw(x)

        if self.chunk is None or p.shape[0] <= self.chunk:
            out = self._eval_dag(p, p.index_select(1, self.center_idx))
        else:
            parts = []
            for start in range(0, p.shape[0], self.chunk):
                block = p[start : start + self.chunk]
                parts.append(self._eval_dag(block, block.index_select(1, self.center_idx)))
            out = torch.cat(parts, dim=0)

        # Final reshape: the layer-level output interface (not a search
        # primitive -- paper, Sec. III-E).
        return out.view(n, ho, wo, self.spec.out_channels).permute(0, 3, 1, 2).contiguous()

    # -- introspection ------------------------------------------------------- #
    def extra_repr(self) -> str:  
        return (
            f"primitives={self.graph.n_primitives}, constants={self.graph.n_constants}, "
            f"params={self.graph.n_parameters()}"
        )


# --------------------------------------------------------------------------- #
# Validity probe (Sec. III-B1: computational correctness)
# --------------------------------------------------------------------------- #
def numeric_probe(
    graph: ComputationGraph,
    n_samples: int = 64,
    scale: float = 1.0,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> bool:
    """Return ``True`` if the graph produces finite values under strict semantics.

    This implements the computational-correctness half of the validity check:
    "the subgraph does not produce invalid numerical behavior such as division
    by zero or log or sqrt of negative values".  Rather than reasoning
    symbolically about each primitive's domain, we evaluate the DAG on random
    inputs with exact (unguarded) primitives and reject anything that yields
    NaN or Inf.
    """
    try:
        graph.check_structure()
    except (GraphError, ShapeError):
        return False

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    try:
        op = GraphOperator(graph, strict=True).to(device)
    except Exception:
        return False

    k = graph.spec.K
    p = torch.randn(n_samples, k, generator=gen).to(device) * scale
    xc = p.index_select(1, op.center_idx)
    with torch.no_grad():
        try:
            out = op._eval_dag(p, xc)
        except Exception:
            return False
    return bool(torch.isfinite(out).all().item())


def is_valid(graph: ComputationGraph, **probe_kwargs) -> bool:
    """Full validity check: structure + shapes + numerical behaviour."""
    return graph.is_structurally_valid() and numeric_probe(graph, **probe_kwargs)
