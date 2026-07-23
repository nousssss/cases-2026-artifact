# Install

These steps cover the **search framework only** — no dataset download, no
MLIR toolchain. For the compiler-integrated half, see `REQUIREMENTS.md` and
`mlir_pipeline/README.md`.

## 1. Environment

```bash
git clone <this-repo> && cd conas
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Test suite

```bash
pytest tests/ -q
```

Expected: **`36 passed, 2 skipped`** (finished in ~2s on the machine this was
verified on). The 2 skipped tests are `tests/test_mlir_lowering.py`'s
correctness checks, which need the MLIR toolchain from `REQUIREMENTS.md` and
skip cleanly without it.

## 3. Smoke test (synthetic data, no checkpoint, no MLIR)

```bash
cd experiments
python run_search.py --model resnet20 --layer 1.1.conv1 --init struct \
    --dataset synthetic --iterations 2 --population 8 --fitness-batches 1 \
    --calibration-samples 0 --out /tmp/conas_smoke
```

This runs an actual (tiny) evolutionary search against randomly-generated
synthetic data, with an untrained (randomly initialised) baseline model — so
the accuracy numbers themselves are meaningless, but a successful run
confirms the whole search pipeline (graph generation, mutation, crossover,
constant optimisation, fitness evaluation, operator selection) executes
correctly. Verified on this machine: it completes in well under a minute and
prints, among the log lines, a JSON summary and the discovered operator's
structure, e.g.:

```
[..] INFO device=cpu dataset=synthetic out=/tmp/conas_smoke/resnet20_synthetic
[..] WARNING no baseline checkpoint supplied: A is randomly initialised, so accuracy numbers are meaningless. Train one with scripts/train_baseline.py.
[..] INFO layer layer1.1.conv1: Cin=16 Cout=16 k=(3, 3) in_hw=(32, 32) K=144
[..] INFO baseline accuracy on the fitness split: 12.11%
[..] INFO initialising 8 graphs from the convolution structure
[..] INFO iter   0  best  12.11%  mean  12.11%  evals 12
[..] INFO iter   1  best  12.11%  mean  12.11%  evals 15
[..] INFO selected operator: acc 12.11% (+0.00 pts), 1 primitives, latency 1.264 ms
[..] INFO saved operator to /tmp/conas_smoke/resnet20_synthetic/operators/1_1_conv1_struct.*
{
  "layer": "1.1.conv1",
  "init": "struct",
  ...
}

Discovered operator:
ComputationGraph[1.1.conv1] 1 primitives, 1 constants, 2304 learned parameters
  patch_0: interface (M, 144)  [not counted]
  center_1: interface (M, 16)  [not counted]
  theta_6: constant (144, 16)
  n7 = matmul(n0, n6)   (M, 16)  <- output
```

Success criterion: the process exits without a traceback and prints a
`Discovered operator:` section like the one above (exact accuracy/latency
numbers will vary run to run — synthetic data and an untrained baseline are
not seeded for reproducible accuracy, only for reproducible *execution*).

## 4. Real experiments (optional, needs real data / a checkpoint)

See the top-level `README.md`'s "Quick start" and "Reproducing the figures"
sections.

## 5. MLIR / compiler-integrated half (optional)

See `REQUIREMENTS.md` for what's needed, and `mlir_pipeline/README.md` /
`Convert-PyTorch-models-to-MLIR/README.md` for build steps. Once built:

```bash
export CONAS_MLIR_SOLUTION_BUILD_DIR=/path/to/solution/llvm-project/build
export CONAS_MLIR_AUTOSCHEDULER_BUILD_DIR=/path/to/autoscheduler/llvm-project/build
pytest tests/test_mlir_lowering.py -v   # should now run instead of skip
```
