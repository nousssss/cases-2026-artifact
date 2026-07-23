#!/usr/bin/env python3
"""Regenerate the paper's figures from the JSON files produced by the experiments.

======  ======================================  ==============================
Figure  Produced from                           Command
======  ======================================  ==============================
4       ``q1_scratch_ablation.json``            ``--fig 4``
5       ``q2_init_comparison_*.json``           ``--fig 5``
6, 7    ``q2_init_comparison_<label>.json``     ``--fig 6 --device-label ...``
8       ``q3_generalization_<model>.json``      ``--fig 8``
======  ======================================  ==============================

Example::

    python experiments/plots.py --fig 5 --results runs/resnet20_cifar10 --out figures/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRATCH_C = "#f4a3a3"
STRUCT_C = "#f7dfa5"
SCRATCH_OPT_C = "#a8c8f0"
STRUCT_OPT_C = "#f5c97a"
NOOP_C = "#c8c8c8"


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _annotate(ax, bars, values, fmt="{:+.2f}%", dy=0.3, size=7):
    for bar, v in zip(bars, values):
        if v is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=size,
        )


def _finish(ax, title, ylabel, xlabel="Layer $\\ell$ to Replace"):
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Fig. 4 -- no-op removal vs from-scratch replacement
# --------------------------------------------------------------------------- #
def figure4(data: dict, out: str) -> str:
    rows = data["rows"]
    labels = [r["layer"] for r in rows]
    baseline = data["baseline_accuracy"]
    noop = [r["noop_accuracy"] if r["noop_accuracy"] is not None else 0 for r in rows]
    scratch = [r["scratch_accuracy"] for r in rows]

    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], noop, w, color=NOOP_C, edgecolor="grey",
                label="Acc(no-op replacement)")
    b2 = ax.bar([i + w / 2 for i in x], scratch, w, color=SCRATCH_C, edgecolor="firebrick",
                label="Acc($A'_{\\mathrm{scratch}}$)")
    ax.axhline(baseline, color="tab:blue", linestyle="--", linewidth=1,
               label=f"Baseline (Acc(A) = {baseline:.2f}%)")
    _annotate(ax, b1, [r["noop_drop"] for r in rows])
    _annotate(ax, b2, [r["scratch_drop"] for r in rows])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, max(100, baseline + 12))
    _finish(ax, "Impact of Computation Graph Replacement on Model Accuracy", "Validation Accuracy (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "fig4_scratch_ablation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Fig. 5 -- initialisation vs accuracy
# --------------------------------------------------------------------------- #
def figure5(data: dict, out: str) -> str:
    rows = data["rows"]
    labels = [r["layer"] for r in rows]
    baseline = rows[0]["baseline_accuracy"]
    scratch = [r["scratch"]["accuracy"] for r in rows]
    struct = [r["struct"]["accuracy"] for r in rows]

    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], scratch, w, color=SCRATCH_C, edgecolor="firebrick",
                label="Acc($A'_{\\mathrm{scratch}}$)")
    b2 = ax.bar([i + w / 2 for i in x], struct, w, color=STRUCT_C, edgecolor="darkgoldenrod",
                label="Acc($A'_{\\mathrm{struct}}$)")
    ax.axhline(baseline, color="tab:blue", linestyle="--", linewidth=1,
               label=f"Baseline (Acc(A) = {baseline:.2f}%)")
    _annotate(ax, b1, [r["scratch"]["accuracy_change"] for r in rows])
    _annotate(ax, b2, [r["struct"]["accuracy_change"] for r in rows])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    lo = min(scratch + struct + [baseline]) - 4
    ax.set_ylim(max(lo, 0), max(scratch + struct + [baseline]) + 5)
    _finish(ax, "Impact of Search Initialization on Validation Accuracy", "Validation Accuracy (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "fig5_init_accuracy.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figs. 6 / 7 -- speedups with and without code optimisation
# --------------------------------------------------------------------------- #
def figure_speedup(data: dict, out: str, filename: str) -> str:
    rows = data["rows"]
    labels = [r["layer"] for r in rows]
    series = [
        ("$A'_{\\mathrm{scratch}}$ without code optimization",
         [r["scratch"]["speedup_without_code_opt"] for r in rows], SCRATCH_C, "firebrick"),
        ("$A'_{\\mathrm{scratch}}$ with code optimization",
         [r["scratch"]["speedup_with_code_opt"] for r in rows], SCRATCH_OPT_C, "steelblue"),
        ("$A'_{\\mathrm{struct}}$ without code optimization",
         [r["struct"]["speedup_without_code_opt"] for r in rows], "#c9e4c5", "seagreen"),
        ("$A'_{\\mathrm{struct}}$ with code optimization",
         [r["struct"]["speedup_with_code_opt"] for r in rows], STRUCT_OPT_C, "darkgoldenrod"),
    ]

    x = range(len(rows))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (label, values, color, edge) in enumerate(series):
        offs = (k - 1.5) * w
        vals = [v if v is not None else 0 for v in values]
        bars = ax.bar([i + offs for i in x], vals, w, color=color, edgecolor=edge, label=label)
        _annotate(ax, bars, values, fmt="{:.2f}x", dy=0.05, size=6)
    ax.axhline(1.0, color="grey", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    _finish(ax, f"Inference speedups on {data.get('device_label', 'target')} "
                f"(backend: {data.get('backend', '?')})", "Speedup")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(out, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Fig. 8 -- generalisation across architectures
# --------------------------------------------------------------------------- #
def figure8(datasets: List[dict], out: str) -> str:
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.5 * len(datasets)), squeeze=False)
    for r, data in enumerate(datasets):
        rows = data["rows"]
        labels = [row["layer"] for row in rows]
        baseline = rows[0]["baseline_accuracy"]
        x = range(len(rows))
        w = 0.38

        ax = axes[r][0]
        b1 = ax.bar([i - w / 2 for i in x], [row["scratch"]["accuracy"] for row in rows], w,
                    color=SCRATCH_C, edgecolor="firebrick", label="$A'_{\\mathrm{scratch}}$")
        b2 = ax.bar([i + w / 2 for i in x], [row["struct"]["accuracy"] for row in rows], w,
                    color=STRUCT_C, edgecolor="darkgoldenrod", label="$A'_{\\mathrm{struct}}$")
        ax.axhline(baseline, color="tab:blue", linestyle="--", linewidth=1,
                   label=f"Baseline = {baseline:.2f}%")
        _annotate(ax, b1, [row["scratch"]["accuracy_change"] for row in rows])
        _annotate(ax, b2, [row["struct"]["accuracy_change"] for row in rows])
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        _finish(ax, f"Accuracy - {data['model']}", "Validation Accuracy (%)")
        ax.legend(fontsize=7)

        ax = axes[r][1]
        b1 = ax.bar([i - w / 2 for i in x], [row["scratch"]["speedup"] for row in rows], w,
                    color=SCRATCH_C, edgecolor="firebrick", label="$A'_{\\mathrm{scratch}}$")
        b2 = ax.bar([i + w / 2 for i in x], [row["struct"]["speedup"] for row in rows], w,
                    color=STRUCT_C, edgecolor="darkgoldenrod", label="$A'_{\\mathrm{struct}}$")
        _annotate(ax, b1, [row["scratch"]["speedup"] for row in rows], fmt="{:.2f}x", dy=0.05, size=6)
        _annotate(ax, b2, [row["struct"]["speedup"] for row in rows], fmt="{:.2f}x", dy=0.05, size=6)
        ax.axhline(1.0, color="grey", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        _finish(ax, f"Speedup - {data['model']}", "Speedup")
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(out, "fig8_generalization.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def search_curve(history_files: List[str], out: str) -> str:
    """Extra diagnostic: best-so-far fitness over generations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for f in history_files:
        data = _load(f)
        hist = data.get("history", data)
        ax.plot([h["iteration"] for h in hist], [h["best_accuracy"] for h in hist],
                label=os.path.basename(f).replace(".result.json", ""))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out, "search_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fig", nargs="+", default=["4", "5", "6", "8"],
                        help="which figures to draw: 4 5 6 7 8 curves")
    parser.add_argument("--results", default="./runs", help="directory holding the JSON results")
    parser.add_argument("--out", default="./figures")
    parser.add_argument("--device-label", default="desktop-cpu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    made = []

    def find(pattern: str) -> List[str]:
        return sorted(glob.glob(os.path.join(args.results, "**", pattern), recursive=True))

    if "4" in args.fig:
        for f in find("q1_scratch_ablation.json"):
            made.append(figure4(_load(f), args.out))
    if "5" in args.fig:
        for f in find("q2_init_comparison_*.json"):
            made.append(figure5(_load(f), args.out))
    for fig_id, label_default, fname in (("6", "desktop-cpu", "fig6_speedup_cpu.png"),
                                         ("7", "rpi3", "fig7_speedup_rpi3.png")):
        if fig_id in args.fig:
            label = args.device_label if fig_id == "6" else label_default
            for f in find(f"q2_init_comparison_{label}.json"):
                made.append(figure_speedup(_load(f), args.out, fname))
    if "8" in args.fig:
        files = find("q3_generalization_*.json")
        if files:
            made.append(figure8([_load(f) for f in files], args.out))
    if "curves" in args.fig:
        files = find("*.result.json")
        if files:
            made.append(search_curve(files, args.out))

    if not made:
        print(f"no result files found under {args.results}; run the experiments first")
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
