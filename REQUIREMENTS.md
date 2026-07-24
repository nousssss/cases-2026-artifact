# Requirements

## Hardware used in the paper

* **Search** (evolutionary search, constant optimisation, fine-tuning): AMD EPYC server, 128 cores, 512 GB RAM.
* **Latency measurements**: Intel i7-8550U desktop CPU, and a Raspberry Pi 3 Model B+.

None of this is required to *run* the code. The search part runs on any
commodity CPU (or GPU, via standard PyTorch device placement) with just the
pip dependencies below.

## Software: search part (no MLIR toolchain needed)

```
torch>=2.0
torchvision>=0.15
numpy>=1.23
matplotlib>=3.6
pytest>=7.0
```
(see `requirements.txt`; `pyproject.toml` requires Python >=3.9). This is
everything needed for `pytest tests/ -q`, the synthetic-data smoke test in
`INSTALL.md`, and running the search itself.


## Software: MLIR / compiler-integrated part

The code-optimisation backend (`--backend mlir`, `conas/compiler/`) built from source:

1. **Two LLVM 17 builds** (`mlir-opt`, `mlir-cpu-runner`; see
   `mlir_pipeline/README.md` and `conas/compiler/backend.py`'s
   `MLIRToolchain` for exactly which build provides what). Point
   `CONAS_MLIR_SOLUTION_BUILD_DIR` and `CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR` at
   their `build/` directories.
2. **torch-mlir**, built from source against one of the LLVM builds above,
   with its Python bindings on `PYTHONPATH`. Do not `pip install torch-mlir`,
   a wheel pins a different LLVM/torch combination. See
   `mlir_pipeline/BUILD.md` for build steps (also developed live at
   <https://github.com/nousssss/Convert-PyTorch-models-to-MLIR>, which is
   linked rather than vendored here).
3. **[MLAutoScheduler](https://github.com/Modern-Compilers-Lab/MLAutoScheduler)**
   (Aouadj & Baghdadi), the `dev` branch. Needs
   `git submodule update --init --recursive` (pulls in the
   `coreAutoScheduler` submodule), CMake >= 3.20, Ninja, GCC/G++ 13.2, and an
   LLVM build with the `openmp` project enabled.

## Disk and build time


* A **trimmed** from-source LLVM 17 + MLIR + torch-mlir build (no `clang`, no
  `stablehlo`, no LLVM/MLIR's own test suite, shallow single-commit checkout)
  took **roughly 20–30 minutes** on a 16-core/31 GB machine and used about
  **5 GB** of disk.
* That is **not representative of the full build MLAutoScheduler's own README
  asks for**, which additionally builds `clang`, comparable in size to
  building LLVM+MLIR again. Expect this to add on the order of an hour or
  more depending on hardware; not independently measured.
* Budget **several hours and tens of GB of free disk** for the complete MLIR
  part (two LLVM checkouts/builds + torch-mlir + MLAutoScheduler) on modest
  hardware, and check available disk space first.

## Links

* torch-mlir setup used by this project: `mlir_pipeline/` in this repo
  (archived lowering/benchmarking scripts + `BUILD.md`), developed live at
  <https://github.com/nousssss/Convert-PyTorch-models-to-MLIR> (linked, not
  vendored -- see `mlir_pipeline/README.md`).
* MLAutoScheduler: <https://github.com/Modern-Compilers-Lab/MLAutoScheduler>
  (`dev` branch; no LICENSE file at the time of writing, so it is linked, not
  vendored).
