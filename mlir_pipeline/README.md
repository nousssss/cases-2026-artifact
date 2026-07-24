# mlir_pipeline

These are the scripts used for MLIR lowering and benchmarking in the paper
(Sec. III-F). They are archived here (rather than only linked) because they
are original code written for this project, not a third-party dependency.
The live version of this pipeline is developed at
<https://github.com/nousssss/Convert-PyTorch-models-to-MLIR>.

The two external toolchains these scripts drive -- torch-mlir/LLVM and
MLAutoScheduler -- **are** third-party and are not vendored. See `BUILD.md`
in this directory for the torch-mlir/LLVM build steps (author's own
instructions, kept here so they're not lost if the live repo above changes),
and the top-level `REQUIREMENTS.md` for MLAutoScheduler.

## Files

| File | Purpose |
|---|---|
| `example.ipynb` | Step-by-step walkthrough: compile a model with torch-mlir, then run it through `convert.sh` + `execute.sh`. |
| `touchup.py` | Fixes up torch-mlir's raw output (drops an unsupported training-mode assertion, rewrites the RNG seed global to a form the LLVM 17 toolchain accepts). |
| `wrap.py` | Wraps the compiled `@forward` function in an executable `@main` (fixed dummy input, timing calls) so `mlir-cpu-runner` can run it. |
| `convert.sh` | Runs `touchup.py` + `wrap.py`, then the bufferization and lowering `mlir-opt` passes, on `mlir_files/<name>.mlir`. |
| `execute.sh` | Runs the lowered module through `mlir-cpu-runner`. |
| `mlir_files/` | Working directory for `convert.sh`/`execute.sh`. Empty here -- see note below. |
| `BUILD.md` | How to build the torch-mlir/LLVM 17 toolchain these scripts need. |

`conas/compiler/backend.py` and `conas/compiler/torch_mlir_export.py` reproduce
this exact pipeline programmatically (same pass flags, same `touchup`/`wrap`
logic ported to Python) so the search loop can invoke it directly; these
shell scripts are the original, standalone form and the ones actually used to
produce the paper's numbers.

## Usage

```bash
export CONAS_MLIR_SOLUTION_BUILD_DIR=/path/to/solution/llvm-project/build
export CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR=/path/to/autoscheduler/llvm-project/build

# Produces mlir_files/<name>.mlir yourself first, e.g. via example.ipynb's
# torch_mlir.compile(...) call, or via conas/compiler/torch_mlir_export.py.
./convert.sh <name>   # -> mlir_files/<name>_llvm.mlir
./execute.sh  <name>   # runs it
```

