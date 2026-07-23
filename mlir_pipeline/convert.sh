#!/bin/bash
# Convert a torch-mlir-compiled model (mlir_files/${input_string}.mlir) into an
# executable, lowered MLIR module (mlir_files/${input_string}_llvm.mlir).
#
# Requires two from-source LLVM 17 builds, each identified by an environment
# variable pointing at that build's "build" directory -- see this directory's
# README.md and the top-level REQUIREMENTS.md / INSTALL.md for how to build
# them and why two separate builds are used:
#   CONAS_MLIR_SOLUTION_BUILD_DIR       -- lowering passes (mlir-opt)
#   CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR  -- bufferize step (mlir-opt) + libomp.so
#
# All pass flags below are unchanged from the original script -- verbatim.

input_string="$1"

: "${CONAS_MLIR_SOLUTION_BUILD_DIR:?CONAS_MLIR_SOLUTION_BUILD_DIR is not set -- see mlir_pipeline/README.md}"
: "${CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR:?CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR is not set -- see mlir_pipeline/README.md}"

# bufferizing ml_program
#${CONAS_TORCH_MLIR_BUILD_DIR}/bin/torch-mlir-opt -refback-mlprogram-bufferize "./mlir_files/${input_string}.mlir" -o "./mlir_files/${input_string}.mlir"

# wrapping call @forward in a main function
python ./touchup.py ${input_string}
python ./wrap.py ${input_string}


# bufferizing func, arith, tensor, linalg
"${CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR}/bin/mlir-opt" "./mlir_files/${input_string}.mlir" -func-bufferize -linalg-bufferize -arith-bufferize --empty-tensor-to-alloc-tensor  -tensor-bufferize -o "./mlir_files/${input_string}_buff.mlir"


# lowerings
"${CONAS_MLIR_SOLUTION_BUILD_DIR}/bin/mlir-opt" "./mlir_files/${input_string}.mlir"  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-math-to-llvm -convert-openmp-to-llvm -convert-math-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  -o "./mlir_files/${input_string}_llvm.mlir"
