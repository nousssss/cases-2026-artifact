# Building the torch-mlir / LLVM 17 toolchain

These are the build instructions used to set up torch-mlir and the two LLVM
17 builds `convert.sh`/`execute.sh` need. They are the author's own
instructions, kept here so the artifact is self-contained; the live version
(which may have since been updated) is at
<https://github.com/nousssss/Convert-PyTorch-models-to-MLIR>. LLVM and
torch-mlir themselves are third-party and are not vendored — this is build
instructions only, per `REQUIREMENTS.md`.

You need **two** builds of the steps below (see `README.md` in this
directory for why): one for `CONAS_MLIR_SOLUTION_BUILD_DIR`, one for
`CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR`.

## I. Clone the torch-mlir repository

```bash
git clone https://github.com/llvm/torch-mlir
cd torch-mlir
git submodule update --init --progress
```

## II. Set up a Python virtual environment and dependencies

While the submodules are updating, set up a Python virtual environment and
install the necessary dependencies.

1. Create and activate a Python virtual environment:

```bash
python3 -m venv mlir_venv
source mlir_venv/bin/activate
```

2. Upgrade `pip` (older versions may not handle recent PyTorch dependencies):

```bash
python -m pip install --upgrade pip
```

3. Install PyTorch nightlies and the build requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -r torchvision-requirements.txt
```

4. Install Python development headers:

```bash
sudo apt install python3-dev
```

## III. Build with CMake

Depending on whether you already have LLVM installed, follow one of the two
cases below.

### Case 1: no existing LLVM install

Use `cmake` to build the project along with LLVM and MLIR:

```bash
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm
```

Then build:

```bash
cmake --build build
```

### Case 2: you already have LLVM installed

Set `$LLVM_INSTALL_DIR` to point to your LLVM installation directory:

```bash
export LLVM_INSTALL_DIR=/path/to/your/llvm-project
```

Then configure:

```bash
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir/" \
  -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm/" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  .
```

And build:

```bash
cmake --build build --target tools/torch-mlir/all
```

## IV. Export the built Python packages

```bash
export PYTHONPATH=$PWD/build/tools/torch-mlir/python_packages/torch_mlir:$PWD/test/python/fx_importer
```

## V. Optional: Jupyter kernel (for `example.ipynb`)

```bash
python -m ipykernel install --user --name=torch-mlir --env PYTHONPATH="$PYTHONPATH"
```
Then select the "torch-mlir" kernel when opening `example.ipynb`.

## VI. Make the scripts executable

```bash
chmod +x ./convert.sh ./execute.sh
```
(run from wherever you placed `convert.sh`/`execute.sh` -- e.g. this
repository's `mlir_pipeline/` directory.)

## VII. Point the scripts at your LLVM builds

`convert.sh` and `execute.sh` read their LLVM build paths from two
environment variables instead of a hard-coded path:

```bash
export CONAS_MLIR_SOLUTION_BUILD_DIR=/path/to/your/solution/llvm-project/build
export CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR=/path/to/your/autoscheduler/llvm-project/build
```

## VIII. Example notebook

For a step-by-step walkthrough, see `example.ipynb` in this directory.
