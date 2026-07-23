"""Computation-graph representation for CONAS candidate operators.

A candidate operator ``l'`` is a directed acyclic graph whose nodes are Table I
primitives and constant tensors (paper, Sec. III-B).  The graph replaces an
*entire* convolution layer, so its external interface is fixed: it consumes the
output ``o`` of the preceding layer and must produce a tensor with the same
shape as the convolution it replaces.

Tensor layout
-------------
The layer-level interface is expressed in the patch (im2col) view used in
Sec. III-E of the paper::

    X  in R^{N x Cin x H x W}          input tensor
    P  = P_{kh x kw}(X) in R^{M x K}   patch view,  M = N.Ho.Wo,  K = Cin.kh.kw
    Xc in R^{M x Cin}                  centre vector of each patch

Every intermediate node therefore carries a matrix of shape ``(M, d)`` where
only the trailing dimension ``d`` varies; ``M`` is shared by construction.  

Shape correctness is enforced **only at the output** (paper, Sec. III-B1):
intermediate nodes may take arbitrary trailing dimensions as long as the final
node produces ``d == Cout``, which the reshape then maps back to
``(N, Cout, Ho, Wo)``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .primitives import BROADCASTING_BINARY, INTERFACE_OPS, PRIMITIVES, primitive_count

# A shape is either ("data", d) for an (M, d) activation matrix, or
# ("const", (d1, ...)) for a learned constant tensor.
Shape = Tuple[str, object]

DATA = "data"
CONST = "const"


class ShapeError(ValueError):
    """Raised when shape inference fails for a graph."""


class GraphError(ValueError):
    """Raised when a graph is structurally invalid (cycle, bad arity, ...)."""


# --------------------------------------------------------------------------- #
# Layer specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayerSpec:

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    #: Spatial size of the layer input, needed to make MLIR shapes static.
    in_hw: Optional[Tuple[int, int]] = None
    name: str = "conv"

    @property
    def K(self) -> int:
        kh, kw = self.kernel_size
        return self.in_channels * kh * kw

    @property
    def out_hw(self) -> Optional[Tuple[int, int]]:
        if self.in_hw is None:
            return None
        out = []
        for i in range(2):
            eff_k = self.dilation[i] * (self.kernel_size[i] - 1) + 1
            out.append((self.in_hw[i] + 2 * self.padding[i] - eff_k) // self.stride[i] + 1)
        return (out[0], out[1])

    @property
    def center_index(self) -> List[int]:
        """Column indices of ``P`` holding the centre pixel of each channel."""
        kh, kw = self.kernel_size
        off = (kh // 2) * kw + (kw // 2)
        return [c * kh * kw + off for c in range(self.in_channels)]

    def to_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "in_hw": list(self.in_hw) if self.in_hw else None,
            "name": self.name,
        }

    @staticmethod
    def from_dict(d: dict) -> "LayerSpec":
        return LayerSpec(
            in_channels=d["in_channels"],
            out_channels=d["out_channels"],
            kernel_size=tuple(d["kernel_size"]),
            stride=tuple(d["stride"]),
            padding=tuple(d["padding"]),
            dilation=tuple(d["dilation"]),
            groups=d.get("groups", 1),
            in_hw=tuple(d["in_hw"]) if d.get("in_hw") else None,
            name=d.get("name", "conv"),
        )

    @staticmethod
    def from_conv(conv, in_hw=None, name="conv") -> "LayerSpec":
        """Build a spec from a ``torch.nn.Conv2d`` module."""
        return LayerSpec(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=tuple(conv.kernel_size),
            stride=tuple(conv.stride),
            padding=tuple(conv.padding) if isinstance(conv.padding, (tuple, list)) else (conv.padding,) * 2,
            dilation=tuple(conv.dilation),
            groups=conv.groups,
            in_hw=tuple(in_hw) if in_hw else None,
            name=name,
        )


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """A single node of the DAG.

    ``op`` is either a Table I primitive name, ``"const"``, or one of the
    interface pseudo-ops ``"patch"`` / ``"center"``.
    """

    nid: int
    op: str
    inputs: List[int] = field(default_factory=list)
    #: Only for ``op == "const"``: the concrete tensor dimensions.
    const_dims: Optional[Tuple[int, ...]] = None
    #: Only for ``op == "const"``: the U(a, b) initialisation range.
    init_range: Tuple[float, float] = (-0.1, 0.1)
    #: Optional explicit initial values (used by structure-guided init to seed
    #: a constant with the original convolution weights).
    init_values: Optional[list] = None

    def clone(self) -> "Node":
        return Node(
            nid=self.nid,
            op=self.op,
            inputs=list(self.inputs),
            const_dims=self.const_dims,
            init_range=self.init_range,
            init_values=copy.deepcopy(self.init_values),
        )


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class ComputationGraph:
    """A candidate replacement operator ``l'``."""

    def __init__(self, spec: LayerSpec):
        self.spec = spec
        self.nodes: Dict[int, Node] = {}
        self.output: Optional[int] = None
        self._next_id = 0
        # Interface nodes are always present and always id 0 / 1.
        self.patch_id = self.add_node("patch")
        self.center_id = self.add_node("center")

    # -- construction ------------------------------------------------------- #
    def add_node(
        self,
        op: str,
        inputs: Sequence[int] = (),
        const_dims: Optional[Tuple[int, ...]] = None,
        init_range: Tuple[float, float] = (-0.1, 0.1),
        init_values: Optional[list] = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(nid, op, list(inputs), const_dims, init_range, init_values)
        return nid

    def remove_nodes(self, ids: Iterable[int]) -> None:
        for nid in ids:
            self.nodes.pop(nid, None)

    def clone(self) -> "ComputationGraph":
        g = ComputationGraph.__new__(ComputationGraph)
        g.spec = self.spec
        g.nodes = {nid: n.clone() for nid, n in self.nodes.items()}
        g.output = self.output
        g._next_id = self._next_id
        g.patch_id = self.patch_id
        g.center_id = self.center_id
        return g

    # -- topology ----------------------------------------------------------- #
    def successors(self, nid: int) -> List[int]:
        return [m for m, n in self.nodes.items() if nid in n.inputs]

    def topo_order(self) -> List[int]:
        """Kahn's algorithm; raises :class:`GraphError` on a cycle."""
        indeg = {nid: 0 for nid in self.nodes}
        for n in self.nodes.values():
            for i in n.inputs:
                if i not in self.nodes:
                    raise GraphError(f"node {n.nid} references missing input {i}")
                indeg[n.nid] += 1
        ready = sorted([nid for nid, d in indeg.items() if d == 0])
        order: List[int] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for s in sorted(self.successors(nid)):
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
        if len(order) != len(self.nodes):
            raise GraphError("graph contains a cycle")
        return order

    def ancestors(self, nid: int) -> Set[int]:
        seen: Set[int] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in self.nodes:
                continue
            seen.add(cur)
            stack.extend(self.nodes[cur].inputs)
        return seen

    # -- shape inference ---------------------------------------------------- #
    def infer_shapes(self) -> Dict[int, Shape]:
        """Propagate trailing dimensions through the DAG.

        Rules (see module docstring):

        * ``patch``  -> data(K),  ``center`` -> data(Cin)
        * unary(x)   -> same shape as ``x``
        * ``matmul(a, b)``: ``a`` is data(d1), ``b`` is a 2-D constant
          ``(d1, d2)`` -> data(d2)
        * other binaries: data(d) with data(d), or data(d) with a constant
          broadcastable to ``(d,)`` or ``(1,)`` -> data(d)
        """
        shapes: Dict[int, Shape] = {}
        for nid in self.topo_order():
            node = self.nodes[nid]
            op = node.op
            if op == "patch":
                shapes[nid] = (DATA, self.spec.K)
            elif op == "center":
                shapes[nid] = (DATA, self.spec.in_channels)
            elif op == "const":
                if node.const_dims is None:
                    raise ShapeError(f"constant node {nid} has no dimensions")
                shapes[nid] = (CONST, tuple(node.const_dims))
            else:
                prim = PRIMITIVES.get(op)
                if prim is None:
                    raise ShapeError(f"unknown primitive '{op}' at node {nid}")
                if len(node.inputs) != prim.arity:
                    raise ShapeError(
                        f"node {nid} ({op}) has {len(node.inputs)} inputs, expected {prim.arity}"
                    )
                ins = [shapes[i] for i in node.inputs]
                shapes[nid] = self._infer_op_shape(nid, op, ins)
        return shapes

    def _infer_op_shape(self, nid: int, op: str, ins: List[Shape]) -> Shape:
        if len(ins) == 1:
            kind, d = ins[0]
            if kind != DATA:
                raise ShapeError(f"node {nid} ({op}): unary applied to a constant")
            return (DATA, d)

        (ka, da), (kb, db) = ins
        if op == "matmul":
            if ka != DATA:
                raise ShapeError(f"node {nid} (matmul): left operand must be an activation")
            if kb != CONST or len(db) != 2:  # type: ignore[arg-type]
                raise ShapeError(f"node {nid} (matmul): right operand must be a 2-D constant")
            if db[0] != da:  # type: ignore[index]
                raise ShapeError(f"node {nid} (matmul): inner dims {da} vs {db[0]}")  # type: ignore[index]
            return (DATA, db[1])  # type: ignore[index]

        # Broadcasting binaries.
        if ka == DATA and kb == DATA:
            if da != db:
                raise ShapeError(f"node {nid} ({op}): dim mismatch {da} vs {db}")
            return (DATA, da)
        if ka == DATA and kb == CONST:
            if tuple(db) not in ((da,), (1,)):  # type: ignore[arg-type]
                raise ShapeError(f"node {nid} ({op}): constant {db} not broadcastable to ({da},)")
            return (DATA, da)
        if ka == CONST and kb == DATA:
            if tuple(da) not in ((db,), (1,)):  # type: ignore[arg-type]
                raise ShapeError(f"node {nid} ({op}): constant {da} not broadcastable to ({db},)")
            return (DATA, db)
        raise ShapeError(f"node {nid} ({op}): both operands are constants")

    # -- pruning ------------------------------------------------------------ #
    def prune(self) -> int:
        """Drop nodes that do not contribute to the output.

        Returns the number of removed nodes.  Sec. III-B1: unused nodes are
        pruned "to avoid occupying memory unnecessarily and improve search
        efficiency".  Interface nodes are always retained so that the external
        interface of the operator stays fixed.
        """
        if self.output is None:
            return 0
        keep = self.ancestors(self.output) | {self.patch_id, self.center_id}
        dead = [nid for nid in self.nodes if nid not in keep]
        self.remove_nodes(dead)
        return len(dead)

    def unused_fraction(self) -> float:
        """Fraction of non-interface nodes that would be pruned."""
        if self.output is None:
            return 0.0
        body = [n for n in self.nodes if n not in (self.patch_id, self.center_id)]
        if not body:
            return 0.0
        keep = self.ancestors(self.output)
        return sum(1 for n in body if n not in keep) / len(body)

    # -- validity ----------------------------------------------------------- #
    def check_structure(self) -> None:
        """Structural + shape correctness (numeric checks live in ``validity``)."""
        if self.output is None:
            raise GraphError("graph has no output node")
        if self.output not in self.nodes:
            raise GraphError("output node is not part of the graph")
        self.topo_order()  # raises on cycles / dangling inputs
        shapes = self.infer_shapes()
        kind, d = shapes[self.output]
        if kind != DATA:
            raise ShapeError("output node is a constant")
        if d != self.spec.out_channels:
            raise ShapeError(
                f"output dim {d} does not match layer out_channels {self.spec.out_channels}"
            )

    def is_structurally_valid(self) -> bool:
        try:
            self.check_structure()
            return True
        except (GraphError, ShapeError):
            return False

    # -- statistics --------------------------------------------------------- #
    @property
    def n_primitives(self) -> int:
        return sum(1 for n in self.nodes.values() if n.op not in INTERFACE_OPS and n.op != "const")

    @property
    def n_constants(self) -> int:
        return sum(1 for n in self.nodes.values() if n.op == "const")

    def primitive_histogram(self) -> Dict[str, int]:
        return primitive_count(n.op for n in self.nodes.values() if n.op != "const")

    def data_nodes(self, shapes: Optional[Dict[int, Shape]] = None) -> List[int]:
        shapes = shapes or self.infer_shapes()
        return [nid for nid, s in shapes.items() if s[0] == DATA]

    def n_parameters(self) -> int:
        total = 0
        for n in self.nodes.values():
            if n.op == "const" and n.const_dims:
                p = 1
                for d in n.const_dims:
                    p *= d
                total += p
        return total

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "output": self.output,
            "patch_id": self.patch_id,
            "center_id": self.center_id,
            "next_id": self._next_id,
            "nodes": [
                {
                    "nid": n.nid,
                    "op": n.op,
                    "inputs": n.inputs,
                    "const_dims": list(n.const_dims) if n.const_dims else None,
                    "init_range": list(n.init_range),
                    "init_values": n.init_values,
                }
                for n in self.nodes.values()
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "ComputationGraph":
        g = ComputationGraph.__new__(ComputationGraph)
        g.spec = LayerSpec.from_dict(d["spec"])
        g.nodes = {}
        for nd in d["nodes"]:
            g.nodes[nd["nid"]] = Node(
                nid=nd["nid"],
                op=nd["op"],
                inputs=list(nd["inputs"]),
                const_dims=tuple(nd["const_dims"]) if nd.get("const_dims") else None,
                init_range=tuple(nd.get("init_range", (-0.1, 0.1))),
                init_values=nd.get("init_values"),
            )
        g.output = d["output"]
        g.patch_id = d["patch_id"]
        g.center_id = d["center_id"]
        g._next_id = d["next_id"]
        return g

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def load(path: str) -> "ComputationGraph":
        with open(path) as fh:
            return ComputationGraph.from_dict(json.load(fh))

    def pretty(self) -> str:
        """Node-by-node breakdown in the style of Eq. (1) / Eq. (2)."""
        shapes = self.infer_shapes()
        lines = []
        for nid in self.topo_order():
            n = self.nodes[nid]
            if n.op == "const":
                lines.append(f"  theta_{nid}: constant {tuple(n.const_dims or ())}")
            elif n.op in INTERFACE_OPS:
                lines.append(f"  {n.op}_{nid}: interface (M, {shapes[nid][1]})  [not counted]")
            else:
                args = ", ".join(f"n{i}" for i in n.inputs)
                mark = "  <- output" if nid == self.output else ""
                lines.append(f"  n{nid} = {n.op}({args})   (M, {shapes[nid][1]}){mark}")
        head = (
            f"ComputationGraph[{self.spec.name}] "
            f"{self.n_primitives} primitives, {self.n_constants} constants, "
            f"{self.n_parameters()} learned parameters"
        )
        return head + "\n" + "\n".join(lines)

    def __repr__(self) -> str:  
        return (
            f"<ComputationGraph nodes={len(self.nodes)} prims={self.n_primitives} "
            f"consts={self.n_constants} out={self.output}>"
        )
