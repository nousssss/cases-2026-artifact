"""UNVERIFIED fallback: hand-written ``linalg`` emission (paper, Sec. III-F).

.. warning::
    This module was written without a working MLIR toolchain to check it
    against, and has never been run through ``mlir-opt``/``mlir-cpu-runner``.
    Treat every construct it emits as unverified. Prefer
    :mod:`conas.compiler.torch_mlir_export`, which goes through torch-mlir the
    way ``Convert-PyTorch-models-to-MLIR/example.ipynb`` (this project's actual
    working pipeline) does; :func:`~conas.compiler.backend.default_backend`
    already prefers it. This module is kept only as a fallback for
    environments without torch-mlir, and should not be trusted for anything
    that requires numerically correct output.

Each graph ``l'`` is lowered to MLIR so that compiler-level optimisation sits
*inside* the evaluation loop rather than being applied as a post-hoc step.  The
emitted module targets ``linalg``-on-tensors, which is the level the MLIR
autoscheduler and the transform dialect operate on.

Shapes are emitted statically: the search fixes a batch size and a layer, so
``M = N.Ho.Wo`` and ``K = Cin.kh.kw`` are known constants.  Static shapes are
what make tiling and vectorisation decisions profitable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..graph import CONST, DATA, ComputationGraph
from ..primitives import PRIMITIVES

F32 = "f32"


def _tensor(dims) -> str:
    return "tensor<" + "x".join(str(d) for d in dims) + f"x{F32}>"


class MLIREmitter:
    """Emit a ``func.func`` implementing the DAG in the linalg dialect."""

    def __init__(self, graph: ComputationGraph, rows: int, func_name: str = "conas_operator"):
        graph.check_structure()
        self.g = graph
        self.M = rows
        self.name = func_name
        self.shapes = graph.infer_shapes()
        self.lines: List[str] = []
        self.ssa: Dict[int, str] = {}
        self._tmp = 0
        self._maps: Dict[str, str] = {}

    # -- small helpers ------------------------------------------------------- #
    def _new(self, hint: str = "v") -> str:
        self._tmp += 1
        return f"%{hint}{self._tmp}"

    def _emit(self, line: str, indent: int = 4) -> None:
        self.lines.append(" " * indent + line)

    def _map(self, expr: str) -> str:
        """Register an ``affine_map`` and return its symbol."""
        if expr not in self._maps:
            self._maps[expr] = f"#map{len(self._maps)}"
        return self._maps[expr]

    def _empty(self, cols: int) -> str:
        t = self._new("init")
        self._emit(f"{t} = tensor.empty() : {_tensor((self.M, cols))}")
        return t

    def _zeros(self, cols: int) -> str:
        init = self._empty(cols)
        z = self._new("zero")
        self._emit(f"{z} = arith.constant 0.0 : {F32}")
        out = self._new("fill")
        self._emit(
            f"{out} = linalg.fill ins({z} : {F32}) outs({init} : {_tensor((self.M, cols))}) "
            f"-> {_tensor((self.M, cols))}"
        )
        return out

    # -- scalar bodies ------------------------------------------------------- #
    @staticmethod
    def _unary_body(op: str) -> List[str]:
        table = {
            "relu": ["%c0 = arith.constant 0.0 : f32", "%r = arith.maximumf %a, %c0 : f32"],
            "tanh": ["%r = math.tanh %a : f32"],
            "exp": ["%r = math.exp %a : f32"],
            "log": ["%r = math.log %a : f32"],
            "sqrt": ["%r = math.sqrt %a : f32"],
            "abs": ["%r = math.absf %a : f32"],
            "sin": ["%r = math.sin %a : f32"],
            "cos": ["%r = math.cos %a : f32"],
            "sigmoid": [
                "%c1 = arith.constant 1.0 : f32",
                "%n = arith.negf %a : f32",
                "%e = math.exp %n : f32",
                "%d = arith.addf %c1, %e : f32",
                "%r = arith.divf %c1, %d : f32",
            ],
        }
        if op not in table:
            raise NotImplementedError(f"no MLIR lowering for unary '{op}'")
        return table[op]

    @staticmethod
    def _binary_body(op: str) -> List[str]:
        table = {
            "matadd": ["%r = arith.addf %a, %b : f32"],
            "matsub": ["%r = arith.subf %a, %b : f32"],
            "elemmul": ["%r = arith.mulf %a, %b : f32"],
            "elemdiv": ["%r = arith.divf %a, %b : f32"],
            "elempow": ["%r = math.powf %a, %b : f32"],
        }
        if op not in table:
            raise NotImplementedError(f"no MLIR lowering for binary '{op}'")
        return table[op]

    # -- op emission --------------------------------------------------------- #
    def _emit_matmul(self, lhs: str, rhs: str, k: int, n: int) -> str:
        acc = self._zeros(n)
        out = self._new("mm")
        self._emit(
            f"{out} = linalg.matmul ins({lhs}, {rhs} : {_tensor((self.M, k))}, {_tensor((k, n))}) "
            f"outs({acc} : {_tensor((self.M, n))}) -> {_tensor((self.M, n))}"
        )
        return out

    def _emit_generic(self, ins: List[tuple], cols: int, body: List[str]) -> str:
        """``ins`` is a list of ``(ssa, map_expr, tensor_type)``."""
        init = self._empty(cols)
        out_map = self._map("affine_map<(d0, d1) -> (d0, d1)>")
        maps = ", ".join(self._map(m) for _, m, _ in ins) + f", {out_map}"
        arg_ssa = ", ".join(s for s, _, _ in ins)
        arg_ty = ", ".join(t for _, _, t in ins)
        res = self._new("g")
        self._emit(
            f"{res} = linalg.generic {{indexing_maps = [{maps}], "
            f'iterator_types = ["parallel", "parallel"]}} '
            f"ins({arg_ssa} : {arg_ty}) outs({init} : {_tensor((self.M, cols))}) {{"
        )
        names = ["%a", "%b"][: len(ins)]
        sig = ", ".join(f"{n}: f32" for n in names) + ", %out: f32"
        self._emit(f"^bb0({sig}):", indent=6)
        for stmt in body:
            self._emit(stmt, indent=8)
        self._emit("linalg.yield %r : f32", indent=8)
        self._emit(f"}} -> {_tensor((self.M, cols))}")
        return res

    def _operand(self, nid: int) -> tuple:
        """Return ``(ssa, map_expr, tensor_type)`` for a linalg.generic operand."""
        kind, dim = self.shapes[nid]
        if kind == DATA:
            return (self.ssa[nid], "affine_map<(d0, d1) -> (d0, d1)>", _tensor((self.M, dim)))
        dims = tuple(dim)  # type: ignore[arg-type]
        if dims == (1,):
            return (self.ssa[nid], "affine_map<(d0, d1) -> (0)>", _tensor(dims))
        return (self.ssa[nid], "affine_map<(d0, d1) -> (d1)>", _tensor(dims))

    # -- module -------------------------------------------------------------- #
    def emit(self) -> str:
        g = self.g
        order = g.topo_order()

        # Function signature: patch view, centre view, then every constant.
        args = [
            (f"%P", _tensor((self.M, g.spec.K))),
            (f"%Xc", _tensor((self.M, g.spec.in_channels))),
        ]
        self.ssa[g.patch_id] = "%P"
        self.ssa[g.center_id] = "%Xc"
        for nid in order:
            node = g.nodes[nid]
            if node.op == "const":
                name = f"%theta{nid}"
                self.ssa[nid] = name
                args.append((name, _tensor(tuple(node.const_dims or ()))))

        for nid in order:
            node = g.nodes[nid]
            if node.op in ("patch", "center", "const"):
                continue
            cols = self.shapes[nid][1]
            prim = PRIMITIVES[node.op]
            if node.op == "matmul":
                lhs, rhs = node.inputs
                k = self.shapes[lhs][1]
                self.ssa[nid] = self._emit_matmul(self.ssa[lhs], self.ssa[rhs], k, cols)
            elif prim.arity == 1:
                operands = [self._operand(node.inputs[0])]
                self.ssa[nid] = self._emit_generic(operands, cols, self._unary_body(node.op))
            else:
                operands = [self._operand(i) for i in node.inputs]
                self.ssa[nid] = self._emit_generic(operands, cols, self._binary_body(node.op))

        assert g.output is not None
        ret_ty = _tensor((self.M, g.spec.out_channels))
        self._emit(f"return {self.ssa[g.output]} : {ret_ty}")

        sig = ", ".join(f"{n}: {t}" for n, t in args)
        # Attribute aliases are declared at file scope, before the module.
        map_defs = [f"{sym} = {expr}" for expr, sym in self._maps.items()]
        body = "\n".join(self.lines)
        return "\n".join(
            map_defs
            + ["module {", f"  func.func @{self.name}({sig}) -> {ret_ty} {{", body, "  }", "}"]
        )


def to_mlir(graph: ComputationGraph, rows: int, func_name: str = "conas_operator") -> str:
    """Convenience wrapper: emit the MLIR text for ``graph``."""
    return MLIREmitter(graph, rows, func_name).emit()


def rows_for(graph: ComputationGraph, batch_size: int = 1) -> int:
    """``M = N . Ho . Wo`` for a given batch size."""
    hw = graph.spec.out_hw
    if hw is None:
        raise ValueError("LayerSpec.in_hw must be set to emit static MLIR shapes")
    return batch_size * hw[0] * hw[1]
