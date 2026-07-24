#!/usr/bin/env python3
"""Dump the MLIR of a discovered operator (paper, Sec. III-F).

Example::

    python scripts/export_mlir.py runs/resnet20_cifar10/operators/1_1_conv1_struct.graph.json \
        --batch-size 1 --out operator.mlir
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conas.compiler import rows_for, to_mlir  
from conas.compiler.backend import MLIRBackend  
from conas.graph import ComputationGraph  


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", help="path to a *.graph.json file")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", default=None, help="write to a file instead of stdout")
    p.add_argument(
        "--compile",
        action="store_true",
        help="also run the torch-mlir + convert.sh pipeline (needs both LLVM 17 builds)",
    )
    args = p.parse_args()

    graph = ComputationGraph.load(args.graph)
    # NOTE: this uses the UNVERIFIED hand-written emitter (conas.compiler.mlir_export),
    # kept only for quick inspection of a graph's structure. --compile below goes
    # through the real, torch-mlir-based pipeline instead.
    text = to_mlir(graph, rows_for(graph, args.batch_size))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)

    if args.compile:
        backend = MLIRBackend(batch_size=args.batch_size)
        if not backend.available():
            raise SystemExit(
                "MLIR toolchain not found. Set CONAS_MLIR_SOLUTION_BUILD_DIR and "
                "CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR to the two LLVM 17 'build' "
                "directories described in Convert-PyTorch-models-to-MLIR/README.md, "
                "and make sure torch-mlir's Python bindings are on PYTHONPATH."
            )
        print("lowered module:", backend.compile_module(graph))


if __name__ == "__main__":
    main()
