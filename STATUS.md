# Artifact Status

We apply for the **Artifacts Available** badge only — not Functional, not
Reusable, not Results Reproduced.

## Why Available, and not more

Roughly half of this artifact (the compiler-integrated speedup numbers in
Figs. 6–7) depends on two from-source LLVM 17 builds, torch-mlir built from
source against one of them, and MLAutoScheduler (a separate third-party
research tool) built from its `dev` branch with submodules. Building all of
that from scratch is realistically several hours plus tens of GB of disk (see
`REQUIREMENTS.md`) — not something a reviewer can do inside a typical
artifact-evaluation window, and we are not providing a Docker image or other
pre-built environment that would let a reviewer skip that build.

## Against the ACM v1.1 criteria for Artifacts Available

* **Permanently archived with a DOI, in a repository built for that purpose
  (e.g. Zenodo, Software Heritage).**
  > **NOT YET SATISFIED — author action required.** At the time of writing,
  > this directory is not even a git repository, so it has not been pushed
  > anywhere, let alone archived with a DOI. This must be done (git init,
  > push to GitHub, archive via Zenodo or Software Heritage, get a DOI)
  > before this badge can actually be granted — see the note at the end of
  > `STATUS.md`'s companion `INSTALL.md`/README changes for what's left.
* **Openly licensed.**
  > MIT, see `LICENSE` — but the copyright holder name in that file is still
  > a placeholder pending author input.
* **Relevant to, and sufficient to (in principle) validate, the claims of the
  paper.**
  > Satisfied for the search-framework claims: the evolutionary search,
  > structure-guided initialisation, primitive DAG representation, mutation/
  > crossover operators, and evaluation methodology described in Sec. III are
  > all present and runnable (see below). Not independently satisfied for the
  > compiler-integrated latency claims without the external toolchain build.

## What a reviewer can run, in this evaluation window

* Full install and the automated test suite (`pytest tests/ -q`) — CPU only,
  no dataset, ~2 seconds, see `INSTALL.md`.
* The synthetic-data smoke test in `INSTALL.md` — exercises the full search
  loop (generation, mutation, crossover, constant optimisation, fitness
  evaluation) without any dataset download or checkpoint.
* The evolutionary search and structure-guided initialisation against real
  data, given a dataset and (optionally) a baseline checkpoint — see the
  top-level README's "Quick start".
* The **`eager`** and **`inductor`** code-optimisation backends
  (`--backend eager` / `--backend inductor`) — `inductor` uses
  `torch.compile`, which ships with PyTorch, so the "with vs. without code
  optimisation" comparison in Figs. 6–7 can be reproduced in *kind*, using a
  portable stand-in, without building anything from source. The reported
  *numbers* will differ from the paper's, which used the `mlir` backend.
