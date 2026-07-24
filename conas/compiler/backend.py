"""Compiler backends used to benchmark candidates post-optimisation.

Sec. III-F of the paper lowers every candidate to MLIR and optimises it either
with hand-written schedules (transform dialect) or with the MLIR autoscheduler,
so that "improvements in model structure translate to measurable speedups".

Figures 6 and 7 report each variant **with** and **without** the code
optimisation part of the framework, which isolates the contribution of the
operator replacement itself from the contribution of the compiler.  This module
provides that switch:

============================  ==========================================
``EagerBackend``              without code optimisation
``InductorBackend``           with code optimisation (portable default)
``MLIRBackend``               with code optimisation (paper configuration)
============================  ==========================================

``MLIRBackend`` lowers through torch-mlir using the exact pipeline recorded in
https://github.com/nousssss/Convert-PyTorch-models-to-MLIR (``convert.sh`` / ``execute.sh``), this
project's own working torch-mlir setup -- see :mod:`conas.compiler.torch_mlir_export`.
It requires two LLVM 17 builds (see ``MLIRToolchain``) and, for now, torch-mlir's
Python bindings on ``PYTHONPATH``.  Neither ships with pip, so
:func:`default_backend` falls back to TorchInductor when they are absent.

The "MLIR autoscheduler" mentioned above is a separate tool
(https://github.com/Modern-Compilers-Lab/MLAutoScheduler, Aouadj & Baghdadi);
:meth:`MLIRBackend.run_autoscheduler` runs it on an already-lowered module.
Building and running it is entirely optional and out of scope for this repo --
see that project's own README.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from ..graph import ComputationGraph
from ..operator import GraphOperator
from .torch_mlir_export import (
    compile_to_linalg_mlir,
    dag_only_module,
    example_conv_input,
    example_patch_inputs,
    parse_printed_memref,
    touchup,
    wrap_for_benchmark,
    wrap_for_execution,
)


class CompilerBackend(ABC):
    """Turns a module into a callable that has gone through code optimisation."""

    name: str = "abstract"
    optimizes: bool = False

    @abstractmethod
    def prepare(self, module: nn.Module, example_input: torch.Tensor) -> Callable:
        ...

    def available(self) -> bool:
        return True


class EagerBackend(CompilerBackend):
    """No code optimisation: the reference point of Figs. 6 and 7."""

    name = "eager"
    optimizes = False

    def prepare(self, module: nn.Module, example_input: torch.Tensor) -> Callable:
        return module.eval()


class InductorBackend(CompilerBackend):
    """``torch.compile`` -- fusion, tiling and vectorisation via TorchInductor.

    A portable stand-in for the MLIR pipeline: it performs the same *kind* of
    low-level transformations, so the with/without comparison stays meaningful
    on machines without an LLVM build.
    """

    name = "inductor"
    optimizes = True

    def __init__(self, mode: str = "max-autotune-no-cudagraphs"):
        self.mode = mode

    def available(self) -> bool:
        return hasattr(torch, "compile")

    def prepare(self, module: nn.Module, example_input: torch.Tensor) -> Callable:
        module = module.eval()
        compiled = torch.compile(module, mode=self.mode, dynamic=False)
        with torch.no_grad():  # trigger compilation now, not inside the timer
            compiled(example_input)
        return compiled


@dataclass
class MLIRToolchain:
    """Paths to the two LLVM 17 builds ``convert.sh`` / ``execute.sh`` use.

    ``Convert-PyTorch-models-to-MLIR/convert.sh`` deliberately mixes two
    separate LLVM 17 builds -- this is not a bug and the two are not
    interchangeable:

    * ``solution_build_dir``      -- the "Solution" build: the lowering passes
      (``mlir-opt``) and ``mlir-cpu-runner``.
    * ``autoscheduler_build_dir`` -- the "Autoscheduler" build: the
      ``func-bufferize``/``linalg-bufferize``/... step, and ``libomp.so``.

    Each is configured by its own environment variable, pointing at the
    ``.../llvm-project/build`` directory of that build (the same root the
    hard-coded paths in ``convert.sh``/``execute.sh`` used, e.g.
    ``.../Solution/llvm-project/build``).
    """

    ENV_SOLUTION_BUILD = "CONAS_MLIR_SOLUTION_BUILD_DIR"
    ENV_AUTOSCHEDULER_BUILD = "CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR"
    ENV_MLAUTOSCHEDULER_BIN = "CONAS_MLAUTOSCHEDULER_BIN"

    solution_build_dir: Optional[str] = None
    autoscheduler_build_dir: Optional[str] = None
    #: Path to the built `AutoSchedulerML` binary from
    #: https://github.com/Modern-Compilers-Lab/MLAutoScheduler (optional --
    #: only needed for MLIRBackend.run_autoscheduler()).
    mlautoscheduler_bin: Optional[str] = None

    def __post_init__(self) -> None:
        if self.solution_build_dir is None:
            self.solution_build_dir = os.environ.get(self.ENV_SOLUTION_BUILD)
        if self.autoscheduler_build_dir is None:
            self.autoscheduler_build_dir = os.environ.get(self.ENV_AUTOSCHEDULER_BUILD)
        if self.mlautoscheduler_bin is None:
            self.mlautoscheduler_bin = os.environ.get(self.ENV_MLAUTOSCHEDULER_BIN)

    def _bin(self, build_dir: Optional[str], var_name: str, exe: str) -> str:
        if not build_dir:
            raise RuntimeError(
                f"{var_name} is not set. Point it at the LLVM 17 'build' directory "
                "described in mlir_pipeline/BUILD.md (step VII)."
            )
        return os.path.join(build_dir, "bin", exe)

    def _lib(self, build_dir: Optional[str], var_name: str, lib: str) -> str:
        if not build_dir:
            raise RuntimeError(
                f"{var_name} is not set. Point it at the LLVM 17 'build' directory "
                "described in mlir_pipeline/BUILD.md (step VII)."
            )
        return os.path.join(build_dir, "lib", lib)

    # -- Solution build (lowering + mlir-cpu-runner) ------------------------- #
    def solution_mlir_opt(self) -> str:
        return self._bin(self.solution_build_dir, self.ENV_SOLUTION_BUILD, "mlir-opt")

    def solution_mlir_cpu_runner(self) -> str:
        return self._bin(self.solution_build_dir, self.ENV_SOLUTION_BUILD, "mlir-cpu-runner")

    def solution_runner_utils_lib(self) -> str:
        return self._lib(self.solution_build_dir, self.ENV_SOLUTION_BUILD, "libmlir_runner_utils.so")

    def solution_c_runner_utils_lib(self) -> str:
        return self._lib(self.solution_build_dir, self.ENV_SOLUTION_BUILD, "libmlir_c_runner_utils.so")

    # -- Autoscheduler build (bufferize step + libomp) ------------------------#
    def autoscheduler_mlir_opt(self) -> str:
        return self._bin(self.autoscheduler_build_dir, self.ENV_AUTOSCHEDULER_BUILD, "mlir-opt")

    def autoscheduler_libomp_lib(self) -> str:
        return self._lib(self.autoscheduler_build_dir, self.ENV_AUTOSCHEDULER_BUILD, "libomp.so")

    def autoscheduler_runner_utils_lib(self) -> str:
        return self._lib(self.autoscheduler_build_dir, self.ENV_AUTOSCHEDULER_BUILD, "libmlir_runner_utils.so")

    def autoscheduler_c_runner_utils_lib(self) -> str:
        return self._lib(self.autoscheduler_build_dir, self.ENV_AUTOSCHEDULER_BUILD, "libmlir_c_runner_utils.so")

    def require_mlautoscheduler_bin(self) -> str:
        if not self.mlautoscheduler_bin:
            raise RuntimeError(
                f"{self.ENV_MLAUTOSCHEDULER_BIN} is not set. Point it at the built "
                "AutoSchedulerML binary from "
                "https://github.com/Modern-Compilers-Lab/MLAutoScheduler."
            )
        return self.mlautoscheduler_bin

    # -- availability ---------------------------------------------------------#
    def require_available(self) -> None:
        """Raise a clear, per-variable error if any required path is missing."""
        for fn in (
            self.solution_mlir_opt,
            self.solution_mlir_cpu_runner,
            self.solution_runner_utils_lib,
            self.solution_c_runner_utils_lib,
            self.autoscheduler_mlir_opt,
            self.autoscheduler_libomp_lib,
        ):
            path = fn()  # raises RuntimeError naming the missing env var
            if not os.path.exists(path):
                raise RuntimeError(f"MLIR toolchain path does not exist: {path}")

    def available(self) -> bool:
        try:
            self.require_available()
        except RuntimeError:
            return False
        return True


class MLIRBackend(CompilerBackend):
    """Lower a :class:`GraphOperator`'s DAG core through torch-mlir and execute it.

    Pipeline (see :mod:`conas.compiler.torch_mlir_export` and ``MLIRToolchain``):

    1. ``dag_only_module(op)`` + ``torch_mlir.compile(..., use_tracing=True,
       output_type="linalg-on-tensors")`` -- only ``_eval_dag(P, Xc)`` is
       traced, not the whole operator (torch-mlir cannot lower ``F.unfold``);
       patch extraction and the final reshape stay in eager PyTorch, done by
       :meth:`execute` on either side of the compiled kernel.
    2. ``touchup`` + ``wrap_for_benchmark``/``wrap_for_execution``
    3. Autoscheduler build: ``mlir-opt -func-bufferize -linalg-bufferize
       -arith-bufferize --empty-tensor-to-alloc-tensor -tensor-bufferize``
       (kept even though its output isn't consumed further -- see convert.sh)
    4. Solution build: the full lowering pipeline of ``convert.sh``
    5. Solution build's ``mlir-cpu-runner``, exactly as ``execute.sh`` invokes
       it (``-e main``, all three ``-shared-libs``).
    """

    name = "mlir"
    optimizes = True

    def __init__(self, batch_size: int = 1, toolchain: Optional[MLIRToolchain] = None):
        self.batch_size = batch_size
        self.toolchain = toolchain or MLIRToolchain()

    def available(self) -> bool:
        return self.toolchain.available()

    # -- lowering ------------------------------------------------------------ #
    def compile_module(
        self,
        graph: ComputationGraph,
        example_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        op: Optional[GraphOperator] = None,
        mode: str = "benchmark",
        out_dir: Optional[str] = None,
    ) -> str:
        """Run the full torch-mlir -> convert.sh pipeline; return the ``_llvm.mlir`` path.

        ``example_input``, if given, is a ``(P, Xc)`` tuple -- the DAG's own
        inputs, *not* the layer-level ``(N, Cin, H, W)`` tensor (see
        :meth:`execute` for the version that takes a real image tensor and
        handles patch extraction/reshape around this).

        ``mode="benchmark"`` reproduces ``wrap.py`` verbatim (fixed inputs
        filled with ``2.0``, ``nanoTime``/``printFlops`` timing calls).
        ``mode="execute"`` embeds ``example_input`` as constants instead, for
        numerical correctness checks.
        """
        self.toolchain.require_available()
        out_dir = out_dir or tempfile.mkdtemp(prefix="conas_mlir_")

        module = op if op is not None else GraphOperator(graph, chunk=None).eval()
        p_xc = example_input if example_input is not None else example_patch_inputs(graph.spec, self.batch_size)

        raw = compile_to_linalg_mlir(dag_only_module(module), p_xc)
        raw = touchup(raw)
        if mode == "benchmark":
            wrapped = wrap_for_benchmark(raw, n_inputs=2)
        elif mode == "execute":
            wrapped = wrap_for_execution(raw, p_xc)
        else:
            raise ValueError(f"unknown mode {mode!r}, expected 'benchmark' or 'execute'")

        src = os.path.join(out_dir, "operator.mlir")
        with open(src, "w") as fh:
            fh.write(wrapped)

        # Bufferizing func, arith, tensor, linalg (Autoscheduler build).
        buff = os.path.join(out_dir, "operator_buff.mlir")
        subprocess.run(
            [
                self.toolchain.autoscheduler_mlir_opt(),
                src,
                "-func-bufferize",
                "-linalg-bufferize",
                "-arith-bufferize",
                "--empty-tensor-to-alloc-tensor",
                "-tensor-bufferize",
                "-o",
                buff,
            ],
            check=True,
            capture_output=True,
        )

        # Lowerings (Solution build).
        lowered = os.path.join(out_dir, "operator_llvm.mlir")
        subprocess.run(
            [
                self.toolchain.solution_mlir_opt(),
                src,
                "-loop-invariant-code-motion",
                "-cse",
                "-canonicalize",
                "-cse",
                "-eliminate-empty-tensors",
                "-empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map",
                "-convert-linalg-to-loops",
                "-convert-vector-to-scf",
                "-convert-scf-to-openmp",
                "-canonicalize",
                "-lower-affine",
                "-expand-strided-metadata",
                "-finalize-memref-to-llvm",
                "-convert-scf-to-cf",
                "-lower-affine",
                "-convert-arith-to-llvm",
                "-convert-math-to-llvm",
                "-convert-openmp-to-llvm",
                "-convert-math-to-llvm",
                "-convert-vector-to-llvm",
                "-convert-cf-to-llvm",
                "-convert-func-to-llvm",
                "-reconcile-unrealized-casts",
                "-o",
                lowered,
            ],
            check=True,
            capture_output=True,
        )
        return lowered

    def run_compiled(self, lowered_path: str) -> str:
        """``mlir-cpu-runner``, exactly as ``execute.sh`` invokes it. Returns stdout."""
        proc = subprocess.run(
            [
                self.toolchain.solution_mlir_cpu_runner(),
                "-e",
                "main",
                "-entry-point-result=void",
                f"-shared-libs={self.toolchain.solution_runner_utils_lib()}",
                f"-shared-libs={self.toolchain.solution_c_runner_utils_lib()}",
                f"-shared-libs={self.toolchain.autoscheduler_libomp_lib()}",
                lowered_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout

    def run_autoscheduler(
        self, mlir_path: str, cwd: Optional[str] = None, verbose: bool = True
    ) -> subprocess.CompletedProcess:
        """Run the external MLIR autoscheduler on an already-lowered module.

        https://github.com/Modern-Compilers-Lab/MLAutoScheduler --
        ``AutoSchedulerML <file.mlir>``. Unlike the rest of this pipeline, this
        is *not* a filter that hands back a rewritten module for further
        lowering: verified against its ``main.cpp``, it runs its own beam
        search over tiling/interchange/parallelization/vectorization
        transforms, evaluates each candidate by actually executing it, and
        writes ``benchmark_exhustiveEval_<name>.json`` plus a debug log into
        ``cwd`` (the function name is taken from ``mlir_path``'s basename).
        Read those files for the result -- this method only runs the tool and
        returns its completed process; ``check`` is intentionally not set
        since its exit-code convention hasn't been verified here.

        Needs ``CONAS_MLAUTOSCHEDULER_BIN`` and
        ``CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR``. Per that repo's own README,
        ``LLVM_PATH`` and all three ``SHARED_LIBS`` come from the *same*
        build -- which is exactly the "Autoscheduler build" convert.sh /
        execute.sh already draw ``libomp.so`` from, so no third LLVM build is
        needed to use this.
        """
        binary = self.toolchain.require_mlautoscheduler_bin()
        llvm_path = self.toolchain.autoscheduler_build_dir
        if not llvm_path:
            raise RuntimeError(
                f"{self.toolchain.ENV_AUTOSCHEDULER_BUILD} is not set "
                "(needed as MLAutoScheduler's LLVM_PATH)."
            )
        shared_libs = ",".join(
            [
                self.toolchain.autoscheduler_runner_utils_lib(),
                self.toolchain.autoscheduler_c_runner_utils_lib(),
                self.toolchain.autoscheduler_libomp_lib(),
            ]
        )
        env = dict(os.environ)
        env["LLVM_PATH"] = llvm_path
        env["SHARED_LIBS"] = shared_libs
        if verbose:
            env["AS_VERBOSE"] = "1"
        return subprocess.run(
            [binary, mlir_path],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

    def execute(
        self,
        graph: ComputationGraph,
        example_input: Optional[torch.Tensor] = None,
        op: Optional[GraphOperator] = None,
        out_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """Lower, run and parse the result for a layer-level ``example_input`` (correctness path).

        ``example_input`` is the real ``(N, Cin, H, W)`` image tensor, matching
        what eager ``op(example_input)`` takes. Patch extraction and the final
        reshape/permute happen here in eager PyTorch, mirroring
        ``GraphOperator.forward`` exactly, around the compiled DAG kernel --
        see ``compile_module`` and the module docstring of
        :mod:`conas.compiler.torch_mlir_export` for why.
        """
        module = op if op is not None else GraphOperator(graph, chunk=None).eval()
        x = example_input if example_input is not None else example_conv_input(graph.spec, self.batch_size)
        with torch.no_grad():
            p, n, _ = module._patch_view(x)
            xc = p.index_select(1, module.center_idx)
        lowered = self.compile_module(graph, example_input=(p, xc), op=module, mode="execute", out_dir=out_dir)
        stdout = self.run_compiled(lowered)
        out_patched = parse_printed_memref(stdout)
        ho, wo = graph.spec.out_hw
        return out_patched.view(n, ho, wo, graph.spec.out_channels).permute(0, 3, 1, 2).contiguous()

    def prepare(self, module: nn.Module, example_input: torch.Tensor) -> Callable:
        """Compile the operator, then return a callable for timing.

        The compiled artefact is benchmarked through ``mlir-cpu-runner``; the
        returned Python callable still evaluates the graph in torch so that the
        surrounding network can be run end to end.  Use
        :meth:`benchmark_compiled` for the pure operator timing.
        """
        if isinstance(module, GraphOperator):
            self._last_artifact = self.compile_module(module.graph, op=module, mode="benchmark")
        return module.eval()

    def benchmark_compiled(self, graph: ComputationGraph, op: Optional[GraphOperator] = None) -> float:
        """Time the compiled operator with ``mlir-cpu-runner`` (ms). Not yet implemented.

        ``wrap.py`` (the ground-truth wrapper this project actually uses) times
        the call with hand-declared ``nanoTime``/``printFlops`` externs, *not*
        ``rtclock``/``printF64`` from ``libmlir_runner_utils`` -- the usual MLIR
        idiom. Neither ``nanoTime`` nor ``printFlops`` is defined anywhere in
        ``Convert-PyTorch-models-to-MLIR``, nor are they standard symbols in the
        three libraries ``execute.sh`` links (``libmlir_runner_utils.so``,
        ``libmlir_c_runner_utils.so``, ``libomp.so``). Running the wrapped
        module as-is would fail at symbol resolution.

        Rather than invent a replacement (the previous version of this method
        parsed stdout with a made-up heuristic and passed a ``--repetitions``
        flag that ``mlir-cpu-runner`` doesn't have -- exactly the kind of
        unverified guess this rewrite is meant to remove), this raises until
        someone with the real toolchain confirms how timing should work here:
        either provide a library that defines ``nanoTime``/``printFlops``, or
        switch ``wrap_for_benchmark`` to ``rtclock``/``printF64``.
        """
        raise NotImplementedError(
            "benchmark_compiled() is intentionally not implemented yet -- "
            "wrap_for_benchmark()'s nanoTime/printFlops timing calls are not "
            "linkable with the shared libs execute.sh uses. See the docstring "
            "of this method."
        )


BACKENDS = {
    "eager": EagerBackend,
    "none": EagerBackend,
    "inductor": InductorBackend,
    "mlir": MLIRBackend,
}


def get_backend(name: str, **kwargs) -> CompilerBackend:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend '{name}', choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)


def default_backend(verbose: bool = True) -> CompilerBackend:
    """MLIR (via torch-mlir) when the toolchain is present, TorchInductor otherwise."""
    mlir = MLIRBackend()
    if mlir.available():
        return mlir
    if verbose:
        print("[conas] MLIR toolchain not found; using TorchInductor as the code-optimisation backend.")
    return InductorBackend()
