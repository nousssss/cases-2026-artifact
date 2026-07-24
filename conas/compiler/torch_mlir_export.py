"""Lower a :class:`~conas.operator.GraphOperator` to executable MLIR via torch-mlir.

This follows ``Convert-PyTorch-models-to-MLIR/example.ipynb`` -- the ground-truth
record of this project's working torch-mlir pipeline -- rather than hand-writing
``linalg`` (see :mod:`conas.compiler.mlir_export` for that unverified approach).
The three post-processing steps of that pipeline are:

1. :func:`compile_to_linalg_mlir` -- ``torch_mlir.compile(module, example_input,
   output_type="linalg-on-tensors")``, exactly as the notebook's cells 9 and 11.
2. :func:`touchup` -- a Python port of ``touchup.py``: drops the
   ``training is not supported`` assertion and rewrites the ``ml_program.global``
   seed op into a ``memref.global`` (torch-mlir emits the former; the LLVM 17
   toolchain used here expects the latter).
3. :func:`wrap_for_benchmark` / :func:`wrap_for_execution` -- ports of
   ``wrap.py``, which wraps the traced ``@forward`` in an executable ``@main``.
   ``wrap.py`` itself only ever builds a *benchmark* ``main`` (a hard-coded
   input filled with ``2.0``, timed with ``nanoTime``/``printFlops``); it has no
   correctness-checking counterpart, so :func:`wrap_for_execution` is new here:
   it embeds a caller-supplied input as a dense constant and prints the result
   with ``printMemrefF32`` (a real symbol from ``libmlir_c_runner_utils``) so
   that :func:`parse_printed_memref` can recover it for comparison against the
   eager PyTorch output.

Note on timing: ``wrap.py``'s ``nanoTime``/``printFlops`` are declared
``private`` (i.e. expected to be provided externally) but are not defined
anywhere in ``Convert-PyTorch-models-to-MLIR`` and are not standard symbols in
``libmlir_runner_utils`` / ``libmlir_c_runner_utils`` / ``libomp`` -- the three
libraries ``execute.sh`` links against. Timing is therefore left out of the
correctness path entirely; see ``MLIRBackend.benchmark_compiled`` in
``backend.py`` for where this gap is flagged.

Note on what actually gets traced: torch-mlir has no ``linalg`` lowering for
``aten::im2col`` (verified against both this repo's LLVM-17-era checkout and
today's upstream HEAD -- the op is declared but no conversion pattern exists
anywhere in ``lib/Conversion/TorchToLinalg``), so ``GraphOperator.forward``
*as a whole* cannot be lowered: it calls ``F.unfold`` internally to build the
patch view.  :func:`dag_only_module` compiles only the DAG-evaluation core
(``GraphOperator._eval_dag``, taking the already-extracted patch matrix ``P``
and centre vector ``Xc``), leaving patch extraction and the final reshape in
eager PyTorch on both sides of the compiled kernel.  This is not a scope
reduction invented for convenience -- it is exactly the boundary Sec. III-E
and ``graph.py`` already draw ("patch extraction, centre selection and the
final reshape are the layer-level interface... and are *not* counted as
search primitives"), and the same boundary the unverified
:mod:`conas.compiler.mlir_export` emitter already assumed.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..graph import LayerSpec
from ..operator import GraphOperator

__all__ = [
    "compile_to_linalg_mlir",
    "touchup",
    "wrap_for_benchmark",
    "wrap_for_execution",
    "parse_printed_memref",
    "example_conv_input",
    "example_patch_inputs",
    "dag_only_module",
]

_TRAINING_ASSERT = 'cf.assert %3, "training is not supported for now"'
_GLOBAL_SEED_RE = re.compile(
    r"ml_program\.global\s+private\s+mutable\s+@global_seed\s*"
    r"\(\s*dense<0>\s*:\s*tensor<i64>\s*\)\s*:\s*tensor<i64>"
)
_TENSOR_TYPE_RE = re.compile(r"tensor<([\dx]+)xf32>")
_MEMREF_RE = re.compile(r"sizes\s*=\s*\[([^\]]*)\].*?data\s*=\s*\n?(\[.*\])", re.DOTALL)


# --------------------------------------------------------------------------- #
# 1. torch-mlir import
# --------------------------------------------------------------------------- #
def compile_to_linalg_mlir(module: nn.Module, example_input) -> str:
    """``torch_mlir.compile(module, example_input, output_type="linalg-on-tensors")``.

    Mirrors ``example.ipynb`` cells 9 and 11: the same call (with
    ``use_tracing=True`` -- see :func:`dag_only_module` for why scripting
    doesn't work here), then ``str(compiled)`` for the textual IR.
    ``example_input`` may be a single tensor or a tuple of tensors, for
    modules whose ``forward`` takes more than one argument.
    """
    try:
        import torch_mlir
    except ImportError as exc:
        raise RuntimeError(
            "torch-mlir is not importable. Build it from source against this "
            "project's LLVM 17 tree (see mlir_pipeline/BUILD.md) "
            "and put its Python bindings on PYTHONPATH. Do not `pip install "
            "torch-mlir` -- a wheel pins a different LLVM and torch build."
        ) from exc
    compiled = torch_mlir.compile(
        module, example_input, output_type="linalg-on-tensors", use_tracing=True
    )
    return str(compiled)


def example_conv_input(
    spec: LayerSpec, batch_size: int = 1, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """A concrete ``(N, Cin, H, W)`` tensor -- the layer-level input ``x``."""
    if spec.in_hw is None:
        raise ValueError("LayerSpec.in_hw must be set to build a concrete example input")
    return torch.randn(batch_size, spec.in_channels, *spec.in_hw, generator=generator)


def example_patch_inputs(
    spec: LayerSpec, batch_size: int = 1, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concrete ``(P, Xc)`` tensors -- the DAG's own inputs (paper, Sec. III-E)."""
    if spec.out_hw is None:
        raise ValueError("LayerSpec.in_hw must be set to build concrete patch inputs")
    ho, wo = spec.out_hw
    rows = batch_size * ho * wo
    p = torch.randn(rows, spec.K, generator=generator)
    xc = torch.randn(rows, spec.in_channels, generator=generator)
    return p, xc


def dag_only_module(op: GraphOperator) -> nn.Module:
    """A traceable wrapper around ``op._eval_dag`` -- see the module docstring.

    Returns a frozen (``requires_grad_(False)``) deep copy of ``op`` wrapped in
    a fresh ``nn.Module`` that only exposes ``forward(p, xc)``; the original
    ``op`` is left untouched. Freezing is necessary because the wrapper
    captures the copy through a Python closure rather than registering it as a
    submodule (registering it re-triggers torch-mlir's issue converting
    ``GraphOperator``'s own non-tensor attributes, e.g. ``self.spec: LayerSpec``,
    to a TorchScript type) -- and an unregistered module's parameters would
    otherwise be traced as external captured tensors, which the tracer refuses
    to embed as constants while they still require grad.
    """
    frozen = copy.deepcopy(op)
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()

    class _DagOnly(nn.Module):
        def forward(self, p: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
            return frozen._eval_dag(p, xc)

    return _DagOnly()


# --------------------------------------------------------------------------- #
# 2. touchup.py, ported
# --------------------------------------------------------------------------- #
def touchup(mlir_text: str) -> str:
    """Port of ``touchup.py``: drop the training assert, fix the seed global."""
    lines = [ln for ln in mlir_text.splitlines(keepends=True) if _TRAINING_ASSERT not in ln]
    text = "".join(lines)

    # touchup.py only ever looks at the first line (it searches
    # text[:text.find('\n', 0, 20)]); preserved verbatim, including the
    # find()==-1 edge case where the slice silently drops the last character.
    head_end = text.find("\n", 0, 20)
    head = text[:head_end]
    match = _GLOBAL_SEED_RE.search(head)
    if match:
        text = (
            text[: match.start()]
            + 'memref.global "private" @global_seed : memref<i64> = dense<0>'
            + text[match.end() :]
        )
    return text


# --------------------------------------------------------------------------- #
# 3. wrap.py, ported
# --------------------------------------------------------------------------- #
def _find_io_shapes(mlir_text: str, n_inputs: int = 1) -> Tuple[List[str], str]:
    """Return (``n_inputs`` argument shapes, result shape) of ``@forward``.

    ``wrap.py`` hard-codes ``matches[0]``/``matches[1]`` for a single-input
    model; generalised here to ``n_inputs`` arguments followed by one result,
    which is what a tensor<...> type textually appears as in the function
    signature (arguments in order, then the ``->`` result) for the simple,
    single-``func.func`` modules this pipeline produces.
    """
    matches = _TENSOR_TYPE_RE.findall(mlir_text)
    if len(matches) < n_inputs + 1:
        raise ValueError(
            f"expected at least {n_inputs + 1} tensor<...xf32> types in the module "
            f"({n_inputs} argument(s) and the result of @forward); found {len(matches)}"
        )
    return matches[:n_inputs], matches[n_inputs]


def wrap_for_benchmark(mlir_text: str, n_inputs: int = 1, entry: str = "forward") -> str:
    """Port of ``wrap.py``, generalised from one input to ``n_inputs``.

    Builds a ``@main`` that fills each fixed-shape input with ``2.0``, calls
    ``@entry``, times it with ``nanoTime``/``printFlops`` and prints the
    result. ``wrap.py`` itself only ever handles a single input (it was
    written for whole-model exports like resnet18); kept otherwise as close to
    verbatim as the generalisation allows. See the module docstring for why
    ``nanoTime``/``printFlops`` cannot currently be linked.
    """
    end_module_idx = mlir_text.rfind("}")
    if end_module_idx == -1:
        raise ValueError("no closing '}' found in MLIR module")
    input_shapes, output_shape = _find_io_shapes(mlir_text, n_inputs)

    setup_lines = []
    arg_ssa = []
    arg_types = []
    for i, shape in enumerate(input_shapes):
        setup_lines.append(f"        %val{i} = arith.constant 2.00000e+00 : f32")
        setup_lines.append(f"        %out{i} = bufferization.alloc_tensor() : tensor<{shape}xf32>")
        setup_lines.append(
            f"        %exin{i} = linalg.fill ins(%val{i} : f32) outs(%out{i} : tensor<{shape}xf32>) "
            f"-> tensor<{shape}xf32>"
        )
        arg_ssa.append(f"%exin{i}")
        arg_types.append(f"tensor<{shape}xf32>")
    setup = "\n".join(setup_lines)
    call_args = ", ".join(arg_ssa)
    call_types = ", ".join(arg_types)

    new_text = f"""
    func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
    func.func private @printFlops(f64)
    func.func private @printMemrefF32(tensor<*xf32>)

    func.func @main() {{
        %d1 = arith.constant 1: index
        %d0 = arith.constant 0 : index
        %n = arith.constant 2: index

{setup}

      //  scf.for %i = %d0 to %n step %d1 {{
        %0 = func.call @nanoTime() : () -> i64
        %1 = func.call @{entry}({call_args}) : ({call_types}) -> tensor<{output_shape}xf32>
        %2 = func.call @nanoTime() : () -> i64

        %unranked = tensor.cast %1 : tensor<{output_shape}xf32> to tensor<*xf32>
        func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

        %3 = arith.subi %2, %0 : i64
        %4 = arith.uitofp %3 : i64 to f64
        func.call @printFlops(%4) : (f64) -> ()
      //  }}

        return
    }}"""
    return mlir_text[:end_module_idx] + new_text + mlir_text[end_module_idx:]


def _format_dense(tensor: torch.Tensor) -> str:
    """Render a tensor as an MLIR ``dense<[...]>`` nested-list literal."""
    if tensor.dim() == 0:
        return f"{tensor.item():.8e}"
    return "[" + ", ".join(_format_dense(tensor[i]) for i in range(tensor.shape[0])) + "]"


def wrap_for_execution(
    mlir_text: str, example_inputs, entry: str = "forward"
) -> str:
    """Correctness-testing counterpart of :func:`wrap_for_benchmark`.

    ``wrap.py`` has no such mode -- it always hard-codes inputs to ``2.0``,
    which is fine for timing but useless for checking that the lowered kernel
    is numerically right. This embeds each of ``example_inputs`` (a tensor, or
    a tuple of tensors matching ``@entry``'s arguments) as a dense constant
    instead, calls ``@entry`` once and prints the result via
    ``printMemrefF32`` (a real, standard symbol -- unlike ``nanoTime``/
    ``printFlops`` this does not require any extra shared library).
    """
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    end_module_idx = mlir_text.rfind("}")
    if end_module_idx == -1:
        raise ValueError("no closing '}' found in MLIR module")
    input_shapes, output_shape = _find_io_shapes(mlir_text, len(example_inputs))

    arg_ssa = []
    arg_types = []
    const_lines = []
    for i, (shape, tensor) in enumerate(zip(input_shapes, example_inputs)):
        traced_dims = "x".join(str(d) for d in tensor.shape)
        if traced_dims != shape:
            raise ValueError(
                f"example_inputs[{i}] shape ({traced_dims}) does not match the "
                f"traced @{entry} argument shape ({shape})"
            )
        dense = _format_dense(tensor.to(torch.float32))
        const_lines.append(f"        %in{i} = arith.constant dense<{dense}> : tensor<{shape}xf32>")
        arg_ssa.append(f"%in{i}")
        arg_types.append(f"tensor<{shape}xf32>")
    consts = "\n".join(const_lines)
    call_args = ", ".join(arg_ssa)
    call_types = ", ".join(arg_types)

    new_text = f"""
    func.func private @printMemrefF32(tensor<*xf32>)

    func.func @main() {{
{consts}
        %out = func.call @{entry}({call_args}) : ({call_types}) -> tensor<{output_shape}xf32>
        %unranked = tensor.cast %out : tensor<{output_shape}xf32> to tensor<*xf32>
        func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
        return
    }}"""
    return mlir_text[:end_module_idx] + new_text + mlir_text[end_module_idx:]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def parse_printed_memref(stdout: str) -> torch.Tensor:
    """Parse the text an unranked ``printMemrefF32`` call writes to stdout.

    Based on the standard MLIR ``impl::printMemRef`` format (``... sizes = [..]
    strides = [..] data = \\n[[...]]``). Not exercised against a real toolchain
    in this environment (none is installed here) -- if the exact spacing
    differs on your build, adjust ``_MEMREF_RE`` accordingly.
    """
    match = _MEMREF_RE.search(stdout)
    if not match:
        raise ValueError(
            "could not find a printed unranked memref (printMemrefF32 output) in "
            f"mlir-cpu-runner stdout:\n{stdout}"
        )
    sizes = [int(s.strip()) for s in match.group(1).split(",") if s.strip()]
    data_text = match.group(2).strip()
    try:
        data = ast.literal_eval(data_text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"could not parse memref data as a nested list:\n{data_text}") from exc
    return torch.tensor(data, dtype=torch.float32).reshape(sizes)
