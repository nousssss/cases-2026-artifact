"""Random computation-graph generation (paper, Sec. III-B1).

Graphs are built iteratively.  For each new node we randomly choose between a
constant, a unary operation or a binary operation.  Inputs are selected from the
existing nodes using *proximity-weighted* probabilities: each existing node is
assigned a sampling weight proportional to ``(i + 1)^2`` where ``i`` is its
creation index, so recently added nodes are more likely to be selected while
long-range connections remain possible.

After each insertion the resulting structure is checked for validity; nodes that
violate it are discarded and construction continues from the last valid state.
Shape correctness is enforced only at the graph's output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .graph import CONST, DATA, ComputationGraph, GraphError, LayerSpec, ShapeError
from .operator import numeric_probe
from .primitives import BROADCASTING_BINARY, UNARY


@dataclass
class GenerationConfig:
    """Knobs of the random graph generator."""

    min_nodes: int = 6
    max_nodes: int = 14
    #: Relative probability of drawing a constant / unary / binary node.
    p_const: float = 0.15
    p_unary: float = 0.40
    p_binary: float = 0.45
    #: Probability that a broadcasting binary takes a constant as its second
    #: operand instead of another activation node.
    p_const_operand: float = 0.35
    #: Exponent of the proximity weighting; ``0`` recovers uniform sampling.
    proximity_power: float = 2.0
    #: How the final node is chosen: ``"proximity"`` samples among all
    #: activation nodes with the same weighting, ``"last"`` always takes the most
    #: recently created one.  ``"last"`` leaves fewer dangling branches (see
    #: ``scripts/proximity_ablation.py``).
    output_selection: str = "proximity"
    #: Candidate trailing dimensions for ``matmul`` constants ("small rank r").
    rank_pool: Optional[Sequence[int]] = None
    init_range: Tuple[float, float] = (-0.1, 0.1)
    #: Attempts per node insertion before giving up on that node.
    node_attempts: int = 8
    #: Attempts to build a whole valid graph before raising.
    graph_attempts: int = 50
    numeric_check: bool = True
    probe_samples: int = 32


def default_rank_pool(spec: LayerSpec) -> List[int]:
    """Small ranks plus the natural channel counts of the layer."""
    cands = {4, 8, 16, 32, spec.out_channels, spec.in_channels}
    upper = max(spec.K, spec.out_channels)
    return sorted({c for c in cands if 1 <= c <= upper})


class GraphGenerator:
    """Builds random candidate DAGs for a given convolution layer."""

    def __init__(self, spec: LayerSpec, cfg: Optional[GenerationConfig] = None, rng=None):
        self.spec = spec
        self.cfg = cfg or GenerationConfig()
        self.rng = rng or random.Random()
        self.ranks = list(self.cfg.rank_pool or default_rank_pool(spec))
        #: Diagnostics filled in by :meth:`generate` (used by the pruning ablation).
        self.last_unused_fraction: float = 0.0

    # -- input selection ----------------------------------------------------- #
    def _proximity_weights(self, n: int) -> List[float]:
        p = self.cfg.proximity_power
        return [float((i + 1) ** p) for i in range(n)]

    def _sample_data_node(
        self, order: List[int], shapes, dim: Optional[int] = None, exclude=()
    ) -> Optional[int]:
        """Pick an activation node, weighted by creation index."""
        idx = [
            (i, nid)
            for i, nid in enumerate(order)
            if nid not in exclude
            and shapes[nid][0] == DATA
            and (dim is None or shapes[nid][1] == dim)
        ]
        if not idx:
            return None
        weights = self._proximity_weights(len(order))
        w = [weights[i] for i, _ in idx]
        return self.rng.choices([nid for _, nid in idx], weights=w, k=1)[0]

    # -- node insertion ------------------------------------------------------ #
    def _try_add_node(self, g: ComputationGraph, order: List[int]) -> Optional[int]:
        """Attempt one insertion.  Returns the new node id, or ``None``."""
        cfg = self.cfg
        shapes = g.infer_shapes()
        kind = self.rng.choices(
            ["const", "unary", "binary"],
            weights=[cfg.p_const, cfg.p_unary, cfg.p_binary],
            k=1,
        )[0]

        if kind == "const":
            src = self._sample_data_node(order, shapes)
            if src is None:
                return None
            d = shapes[src][1]
            return g.add_node("const", const_dims=(d,), init_range=cfg.init_range)

        if kind == "unary":
            op = self.rng.choice(UNARY)
            src = self._sample_data_node(order, shapes)
            if src is None:
                return None
            return g.add_node(op, inputs=[src])

        # Binary.
        if self.rng.random() < 0.5:
            op = "matmul"
        else:
            op = self.rng.choice(BROADCASTING_BINARY)

        left = self._sample_data_node(order, shapes)
        if left is None:
            return None
        d = shapes[left][1]

        if op == "matmul":
            r = self.rng.choice(self.ranks)
            c = g.add_node("const", const_dims=(d, r), init_range=self._matmul_init(d))
            return g.add_node(op, inputs=[left, c])

        # Broadcasting binary: another activation of the same width, or a constant.
        use_const = self.rng.random() < cfg.p_const_operand
        right = None
        if not use_const:
            right = self._sample_data_node(order, shapes, dim=d, exclude=(left,))
        if right is None:
            dims = (d,) if self.rng.random() < 0.7 else (1,)
            right = g.add_node("const", const_dims=dims, init_range=cfg.init_range)
        return g.add_node(op, inputs=[left, right])

    def _matmul_init(self, fan_in: int) -> Tuple[float, float]:
        """U(-a, a) with a = 1/sqrt(fan_in): keeps activations in a sane range."""
        a = 1.0 / max(fan_in, 1) ** 0.5
        return (-a, a)

    # -- output interface ---------------------------------------------------- #
    def enforce_output(self, g: ComputationGraph, node: int) -> int:
        """Make ``node`` the graph output, projecting to ``Cout`` if needed.

        Shape correctness is enforced only here (paper, Sec. III-B1): if the
        final node's width already matches the number of output channels it is
        used directly, otherwise a single ``matmul`` with a learned constant
        maps it onto the required width.  This is what keeps the *external*
        interface of every candidate identical to the convolution it replaces.
        """
        shapes = g.infer_shapes()
        d = shapes[node][1]
        if d == self.spec.out_channels:
            g.output = node
            return node
        c = g.add_node("const", const_dims=(d, self.spec.out_channels), init_range=self._matmul_init(d))
        out = g.add_node("matmul", inputs=[node, c])
        g.output = out
        return out

    # -- public API ---------------------------------------------------------- #
    def generate(self) -> ComputationGraph:
        """Build one valid random candidate operator."""
        for _ in range(self.cfg.graph_attempts):
            g = self._build_once()
            if g is not None:
                return g
        raise RuntimeError(
            "failed to generate a valid graph; loosen GenerationConfig or check the layer spec"
        )

    def _build_once(self) -> Optional[ComputationGraph]:
        cfg = self.cfg
        g = ComputationGraph(self.spec)
        order: List[int] = [g.patch_id, g.center_id]
        target = self.rng.randint(cfg.min_nodes, cfg.max_nodes)

        for _ in range(target):
            for _ in range(cfg.node_attempts):
                snapshot = g.clone()
                try:
                    nid = self._try_add_node(g, order)
                except (ShapeError, GraphError):
                    nid = None
                if nid is None:
                    # Roll back to the last valid state and try a different node.
                    g = snapshot
                    continue
                try:
                    g.infer_shapes()
                except (ShapeError, GraphError):
                    g = snapshot
                    continue
                # Register newly created nodes in creation order.
                for new in sorted(set(g.nodes) - set(snapshot.nodes)):
                    order.append(new)
                break

        # Choose the graph output among the activation nodes, then fix its width.
        shapes = g.infer_shapes()
        if cfg.output_selection == "last":
            tail = next(
                (
                    n
                    for n in reversed(order)
                    if shapes[n][0] == DATA and n not in (g.patch_id, g.center_id)
                ),
                None,
            )
        else:
            tail = self._sample_data_node(order, shapes, exclude=(g.patch_id, g.center_id))
        if tail is None:
            return None
        self.enforce_output(g, tail)

        self.last_unused_fraction = g.unused_fraction()
        g.prune()

        if not g.is_structurally_valid():
            return None
        if g.n_primitives == 0:
            return None
        if cfg.numeric_check and not numeric_probe(g, n_samples=cfg.probe_samples):
            return None
        return g

    def population(self, size: int) -> List[ComputationGraph]:
        """Random initialisation: a completely random population (Sec. III-C)."""
        return [self.generate() for _ in range(size)]
