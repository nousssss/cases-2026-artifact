# Artifact Status

We apply for the **Artifacts Available** badge.

## Why Available, and not more

Roughly half of this artifact (the compiler-integrated speedup numbers in
Figs. 6–7) depends on a from-source LLVM 17 build, torch-mlir built from
source against it, and MLAutoScheduler (a separate third-party
research tool) built from its `dev` branch with submodules. Building all of
that from scratch is realistically several hours plus tens of GB of disk (see
`REQUIREMENTS.md`) which is not something a reviewer can do inside a typical
artifact-evaluation window, and a Docker image could not be provided, or other
pre-built environment that would let a reviewer skip that build.

## Against the ACM v1.1 criteria for Artifacts Available

* **Permanently archived with a DOI, in a repository built for that purpose.**
  > This repository is also on
  > GitHub (<https://github.com/nousssss/cases-2026-artifact>).
* **Openly licensed.**
* **Relevant to, and sufficient to validate, the claims of the
  paper.**
  > Satisfied for the search-framework claims: the evolutionary search,
  > structure-guided initialisation, primitive DAG representation, mutation/
  > crossover operators, and evaluation methodology described in Sec. III are
  > all present and runnable (see below). The
  > compiler-integrated latency claims need the external toolchain build.

