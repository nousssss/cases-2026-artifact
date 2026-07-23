#!/usr/bin/env python3
"""Q1 -- Can we replace convolutions from scratch?  (paper, Sec. IV-B; Fig. 4)

For each convolution layer ``l`` of the model we

1. replace ``l`` with an identity-like no-op ``f(x) = x`` to measure the
   accuracy lost by simply removing the layer, and
2. run CONAS with random initialisation to discover a replacement ``l'``.

The comparison shows whether learned operators can recover the functional role
of a convolution without any structural prior.

Usage::

    python experiments/q1_scratch_ablation.py --model resnet20 \
        --checkpoint runs/resnet20_baseline.pt --iterations 200 --population 300
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import torch

from common import add_common_args, build_search_config, setup

from conas.evaluation import evaluate_accuracy, measure_latency
from conas.models import list_conv_layers, normalize_layer_name
from conas.models.resnet_cifar import paper_resnet20_layers
from conas.operator import GraphOperator
from conas.replacement import Identity, collect_calibration, layer_is_shape_preserving, probe_layer, temporarily_replaced
from conas.search import CONASSearch
from conas.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--layers", nargs="*", default=None,
                        help="layers to ablate (default: the seven ResNet20 layers of Fig. 4)")
    parser.add_argument("--all-layers", action="store_true", help="ablate every conv layer")
    args = parser.parse_args()

    ctx = setup(args)
    model, device, logger = ctx.model, ctx.device, ctx.logger

    if args.all_layers:
        layers = list_conv_layers(model)
    else:
        layers = args.layers or paper_resnet20_layers()

    baseline_acc = evaluate_accuracy(model, ctx.val_loader, device)
    input_shape = tuple(ctx.example.shape[1:])
    baseline_lat = measure_latency(model, input_shape, device, ctx.search_cfg.latency)
    logger.info(f"baseline A: {baseline_acc:.2f}%  {baseline_lat:.3f} ms")

    rows = []
    for layer in layers:
        path = normalize_layer_name(args.model, layer)
        probe = probe_layer(model, layer, ctx.example, args.model, capture_inputs=True)

        # 1) No-op removal.
        if layer_is_shape_preserving(probe.conv):
            with temporarily_replaced(model, path, Identity()):
                noop_acc = evaluate_accuracy(model, ctx.val_loader, device)
        else:
            noop_acc = None
            logger.info(f"{layer}: not shape-preserving, no-op baseline is undefined")

        # 2) CONAS from scratch.
        cfg = build_search_config(args, init="scratch")
        calibration = collect_calibration(model, path, ctx.val_loader, 256, device)
        result = CONASSearch(model, probe, ctx.val_loader, cfg, device, calibration, logger).run(
            baseline_accuracy=None
        )

        # Full-validation accuracy of the selected operator (the search itself
        # uses a subset of batches as its fitness signal).
        op = GraphOperator(result.graph, chunk=cfg.operator_chunk).to(device)
        op.load_state_dict(result.state_dict)
        with temporarily_replaced(model, path, op):
            scratch_acc = evaluate_accuracy(model, ctx.val_loader, device)
            scratch_lat = measure_latency(model, input_shape, device, cfg.latency)

        row = {
            "layer": layer,
            "baseline_accuracy": baseline_acc,
            "noop_accuracy": noop_acc,
            "noop_drop": None if noop_acc is None else noop_acc - baseline_acc,
            "scratch_accuracy": scratch_acc,
            "scratch_drop": scratch_acc - baseline_acc,
            "scratch_speedup": baseline_lat / max(scratch_lat, 1e-9),
            "primitives": result.graph.n_primitives,
            "mse": result.mse,
        }
        rows.append(row)
        noop_str = "n/a" if noop_acc is None else format(row["noop_drop"], "+.2f")
        logger.info(
            f"{layer}: no-op {noop_str} pts, "
            f"CONAS-scratch {row['scratch_drop']:+.2f} pts, {row['scratch_speedup']:.2f}x"
        )
        result.save(os.path.join(ctx.out_dir, "operators"))

    drops = [r["scratch_drop"] for r in rows]
    summary = {
        "baseline_accuracy": baseline_acc,
        "baseline_latency_ms": baseline_lat,
        "mean_scratch_drop": sum(drops) / len(drops),
        "rows": rows,
    }
    save_json(summary, os.path.join(ctx.out_dir, "q1_scratch_ablation.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
