# CONAS

Reference implementation of **"From Scratch or from Structure? Investigating the Trade-off in Replacing Convolution"**.

CONAS searches for computation graphs, built from low-level mathematical primitives, that replace an *entire* convolution layer, and folds compiler-level optimisation (MLIR) into the evaluation loop so candidates are selected for deployment efficiency as well as accuracy.

```
Neural Network A ──▶ [ Operator Search (NAS) + Compiler Optimization (MLIR) ] ──▶ A'
                     Latency(A') < Latency(A),  Accuracy(A') ≥ Accuracy(A) − γ
```

**What runs standalone vs. what doesn't.** The search framework itself — graph
generation, mutation/crossover, structure-guided initialisation, constant
optimisation, fitness evaluation, the whole evolutionary loop — is pure
PyTorch and runs with just `pip install -r requirements.txt`; see `INSTALL.md`
for a from-scratch install and a synthetic-data smoke test that needs no
dataset and no external toolchain. The **compiler-integrated speedup numbers**
reported in Figs. 6–7 (the `mlir` backend, Sec. III-F) additionally require
two from-source LLVM 17 builds, torch-mlir built from source, and (for the
autoscheduled variant) MLAutoScheduler built from its own repository — see
`REQUIREMENTS.md`, `mlir_pipeline/README.md`, and
[Compiler backends](#compiler-backends) below. Building that toolchain takes
real time (hours) and disk (tens of GB); it is not required to use or evaluate
the search framework on its own. See `STATUS.md` for exactly what this
artifact's badge claim does and doesn't cover.

---

## Install

```bash
git clone <this-repo> && cd conas
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q          # 38 tests, ~2 s, CPU only (2 skip without the MLIR toolchain)
```

See `INSTALL.md` for a verified synthetic-data smoke test. The MLIR toolchain
is **optional** — see [Compiler backends](#compiler-backends).

---

## Quick start

```bash
# 1. Train the baseline A (no canonical CIFAR ResNet20 checkpoint exists)
python scripts/train_baseline.py --model resnet20 --epochs 200 \
    --out runs/resnet20_baseline.pt

# 2. Search for a replacement operator for one layer
python experiments/run_search.py --model resnet20 --layer 1.1.conv1 \
    --init struct --checkpoint runs/resnet20_baseline.pt

# 3. Inspect what was discovered
python scripts/visualize_dag.py runs/resnet20_cifar10/operators/1_1_conv1_struct.graph.json \
    --breakdown --out fig.dot
python scripts/export_mlir.py runs/resnet20_cifar10/operators/1_1_conv1_struct.graph.json
```

To exercise the pipeline without any dataset, pass `--dataset synthetic`.

---

## Paper → code

| Paper | Code |
|---|---|
| Table I — primitive operations | `conas/primitives.py` |
| Sec. III-B — search space, DAG, validity, pruning | `conas/graph.py`, `conas/operator.py` |
| Sec. III-B1 — random generation, proximity weighting | `conas/generation.py` |
| Sec. III-B2 / Algorithm 1 — constant optimisation | `conas/constant_opt.py` |
| Sec. III-C — tournament selection, crossover, mutation | `conas/evolution.py`, `conas/search.py` |
| Sec. III-D — structure-guided initialisation | `conas/init_struct.py` |
| Sec. III-E — evaluation method, Eq. (1) and Eq. (2) | `conas/evaluation.py`, `conas/init_struct.py` |
| Sec. III-F — MLIR lowering and scheduling | `conas/compiler/` |
| Sec. IV-A — fine-tuning protocol | `conas/finetune.py` |
| Sec. IV-F — sequential greedy multi-layer, operator reuse | `conas/multilayer.py` |

---

## How the search space is represented

Candidate operators are expressed in the patch (im2col) view of Sec. III-E:

```
X  ∈ ℝ^{N×Cin×H×W}          input tensor
P  = P_{kh×kw}(X) ∈ ℝ^{M×K}  patch view,  M = N·Ho·Wo,  K = Cin·kh·kw
Xc ∈ ℝ^{M×Cin}               centre vector of each patch
```

Every intermediate node carries an `(M, d)` matrix, so shape inference reduces to tracking the trailing dimension `d`. Shape correctness is enforced **only at the output** (`d == Cout`), exactly as the paper specifies; intermediate widths are unconstrained.

Patch extraction, centre selection and the final reshape are the layer-level interface and are **not** counted as search primitives — matching Sec. III-E. `graph.n_primitives` and `graph.primitive_histogram()` count only Table I operations.

Computational correctness (no division by zero, no `log`/`sqrt` of a negative) is checked by evaluating the DAG on random inputs with *strict*, unguarded primitives and rejecting any NaN/Inf. During search the guarded variants run instead, so a graph that is valid on the probe cannot blow up on a rare batch.

---

## Compiler backends

Figures 6 and 7 report each variant **with** and **without** code optimisation. That switch is `conas/compiler/backend.py`:

| Backend | Meaning | Requires |
|---|---|---|
| `eager` | *without* code optimisation | — |
| `inductor` | *with* code optimisation (portable default) | `torch.compile` |
| `mlir` | *with* code optimisation (paper configuration) | two LLVM 17 builds + torch-mlir |

The MLIR path is CONAS's integration point for **two separate, external toolchains** — build and use each per its own repo, not this one:

* **torch-mlir / LLVM 17** — this project's own working torch-mlir setup. The
  scripts (`convert.sh`/`execute.sh`/`wrap.py`/`touchup.py`/`example.ipynb`)
  are archived in [`mlir_pipeline/`](mlir_pipeline); `conas/compiler/backend.py`
  reproduces the `convert.sh`/`execute.sh` pipeline exactly (same flags, same
  two-build split), and `conas/compiler/torch_mlir_export.py` ports
  `touchup.py`/`wrap.py` into Python. Build steps are in
  [`mlir_pipeline/BUILD.md`](mlir_pipeline/BUILD.md) (also developed live at
  <https://github.com/nousssss/Convert-PyTorch-models-to-MLIR>, linked rather
  than vendored here since LLVM/torch-mlir version pinning is fragile enough —
  nightly wheels get pruned, pass names change across versions — that keeping
  two copies of the build instructions in sync isn't worth it).
* **[MLAutoScheduler](https://github.com/Modern-Compilers-Lab/MLAutoScheduler)** (Aouadj & Baghdadi) — the actual autoscheduler behind Sec. III-F's "MLIR autoscheduler" mention. It's a standalone beam-search-plus-execution benchmarking tool (`AutoSchedulerML <file.mlir>`), not a filter step CONAS calls automatically: `MLIRBackend.run_autoscheduler()` runs it on an already-lowered module and lets you inspect its own JSON/log output. Build and use it per that repo's README (third-party, not vendored; see `REQUIREMENTS.md`).

Once both are built, point CONAS at them, this is the part that's actually CONAS's contract, so it's documented here:

```bash
# From mlir_pipeline/BUILD.md's build:
export CONAS_MLIR_SOLUTION_BUILD_DIR=/path/to/Solution/llvm-project/build        # mlir-opt lowering + mlir-cpu-runner
export CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR=/path/to/Autoscheduler/llvm-project/build  # bufferize step + libomp.so

# Optional -- only needed for MLIRBackend.run_autoscheduler():
export CONAS_MLAUTOSCHEDULER_BIN=/path/to/MLAutoScheduler/build/bin/AutoSchedulerML

python experiments/q2_init_comparison.py --backend mlir
```

Each `*_BUILD_DIR` must contain `bin/` and `lib/` with the binaries/libraries `MLIRToolchain` in `backend.py` expects (`mlir-opt`, `mlir-cpu-runner`, `libmlir_runner_utils.so`, `libmlir_c_runner_utils.so`, `libomp.so` — see that class's docstring for exactly which build provides which). A missing variable raises naming itself rather than failing silently.

`default_backend()` falls back to TorchInductor with a warning when no toolchain is found. **Numbers from different backends are not interchangeable — always report which one produced them.**

---

## Repository layout

```
conas/
  primitives.py      Table I
  graph.py           DAG, shape inference, validity, pruning, serialisation
  operator.py        executable module + numeric validity probe
  generation.py      random generation, proximity weighting
  init_struct.py     structure-guided init, Eq. (1) and Eq. (2)
  evolution.py       subgraph crossover, four mutations, tournament selection
  constant_opt.py    Algorithm 1
  search.py          the evolutionary search loop
  evaluation.py      accuracy, latency, accuracy-equivalence selection
  replacement.py     model surgery, activation capture
  multilayer.py      sequential greedy, operator reuse
  finetune.py        post-search fine-tuning
  compiler/          MLIR emitter + backends
  models/            ResNet20/32, ConvNeXt, layer-name resolution
experiments/         Q1, Q2, Q3, multi-layer, Table II, plots
scripts/             baseline training, ablations, MLIR export, DAG rendering
tests/               38 tests (2 need the MLIR toolchain, see Compiler backends)
configs/             reference hyper-parameters
mlir_pipeline/       archived convert.sh/execute.sh/wrap.py/touchup.py/example.ipynb + BUILD.md
```

