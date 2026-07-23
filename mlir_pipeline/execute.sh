#!/bin/bash
# Execute an MLIR module lowered by convert.sh (mlir_files/${input_string}_llvm.mlir).
#
# Requires the same two environment variables as convert.sh:
#   CONAS_MLIR_SOLUTION_BUILD_DIR       -- mlir-cpu-runner, libmlir_runner_utils.so, libmlir_c_runner_utils.so
#   CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR  -- libomp.so
#
# All mlir-cpu-runner flags below are unchanged from the original script -- verbatim.

input_string="$1"

: "${CONAS_MLIR_SOLUTION_BUILD_DIR:?CONAS_MLIR_SOLUTION_BUILD_DIR is not set -- see mlir_pipeline/README.md}"
: "${CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR:?CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR is not set -- see mlir_pipeline/README.md}"

# run
"${CONAS_MLIR_SOLUTION_BUILD_DIR}/bin/mlir-cpu-runner" -e main -entry-point-result=void -shared-libs="${CONAS_MLIR_SOLUTION_BUILD_DIR}/lib/libmlir_runner_utils.so" -shared-libs="${CONAS_MLIR_SOLUTION_BUILD_DIR}/lib/libmlir_c_runner_utils.so" -shared-libs="${CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR}/lib/libomp.so" "./mlir_files/${input_string}_llvm.mlir"
