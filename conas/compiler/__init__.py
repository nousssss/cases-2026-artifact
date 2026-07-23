"""Compiler-aware optimization via MLIR (paper, Sec. III-F)."""

from .backend import (
    BACKENDS,
    CompilerBackend,
    EagerBackend,
    InductorBackend,
    MLIRBackend,
    MLIRToolchain,
    default_backend,
    get_backend,
)
from .mlir_export import MLIREmitter, rows_for, to_mlir
from .torch_mlir_export import (
    compile_to_linalg_mlir,
    example_conv_input,
    parse_printed_memref,
    touchup,
    wrap_for_benchmark,
    wrap_for_execution,
)

__all__ = [
    "CompilerBackend",
    "EagerBackend",
    "InductorBackend",
    "MLIRBackend",
    "MLIRToolchain",
    "BACKENDS",
    "get_backend",
    "default_backend",
    "MLIREmitter",
    "to_mlir",
    "rows_for",
    "compile_to_linalg_mlir",
    "touchup",
    "wrap_for_benchmark",
    "wrap_for_execution",
    "parse_printed_memref",
    "example_conv_input",
]
