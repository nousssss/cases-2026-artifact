#!/usr/bin/env python3
"""Q3 -- Does CONAS generalise across architectures and devices?  (Sec. IV-D; Fig. 8)

Runs both initialisation strategies on five representative layers of ResNet32
(CIFAR-10) and ConvNeXt-T (ImageNet), recording validation accuracy and CPU
latency for each.

Hardware generalisation is covered by running
``experiments/q2_init_comparison.py`` a second time on the other target
(``--device-label rpi3``); the two result files are then plotted together by
``experiments/plots.py``.

Note on ConvNeXt: the target layers are 7x7 depthwise convolutions.  Their patch
matrix is wide (``K = Cin * 49``), so pass ``--operator-chunk`` to keep the peak
memory of the patch view bounded.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from common import add_common_args, build_search_config, setup

from conas.evaluation import evaluate_accuracy, measure_latency
from conas.models import normalize_layer_name
from conas.models.resnet_cifar import paper_resnet32_layers
from conas.operator import GraphOperator
from conas.replacement import collect_calibration, probe_layer, temporarily_replaced
from conas.search import CONASSearch
from conas.utils import save_json

#: The five ConvNeXt depthwise layers reported in Fig. 8(c)-(d).
PAPER_CONVNEXT_LAYERS = ["0.0.dwconv", "0.2.dwconv", "2.0.dwconv", "2.3.dwconv", "3.1.dwconv"]


def default_layers(model_name: str):
    if model_name.startswith("convnext"):
        return PAPER_CONVNEXT_LAYERS
    return paper_resnet32_layers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--layers", nargs="*", default=None)
    args = parser.parse_args()

    ctx = setup(args)
    model, device, logger = ctx.model, ctx.device, ctx.logger
    layers = args.layers or default_layers(args.model)

    input_shape = tuple(ctx.example.shape[1:])
    baseline_acc = evaluate_accuracy(model, ctx.val_loader, device)
    baseline_lat = measure_latency(model, input_shape, device, ctx.search_cfg.latency)
    logger.info(f"baseline {args.model}: {baseline_acc:.2f}%  {baseline_lat:.3f} ms")

    rows = []
    for layer in layers:
        path = normalize_layer_name(args.model, layer)
        probe = probe_layer(model, layer, ctx.example, args.model, capture_inputs=True)
        logger.info(f"{layer}: K={probe.spec.K} groups={probe.spec.groups} in_hw={probe.in_hw}")
        calibration = collect_calibration(model, path, ctx.val_loader, 128, device)
        row = {"layer": layer, "baseline_accuracy": baseline_acc, "baseline_latency_ms": baseline_lat}

        for init in ("scratch", "struct"):
            cfg = build_search_config(args, init=init)
            result = CONASSearch(model, probe, ctx.val_loader, cfg, device, calibration, logger).run()
            result.save(os.path.join(ctx.out_dir, "operators"))

            op = GraphOperator(result.graph, chunk=cfg.operator_chunk).to(device)
            op.load_state_dict(result.state_dict)
            with temporarily_replaced(model, path, op):
                acc = evaluate_accuracy(model, ctx.val_loader, device)
                lat = measure_latency(model, input_shape, device, cfg.latency)
            row[init] = {
                "accuracy": acc,
                "accuracy_change": acc - baseline_acc,
                "latency_ms": lat,
                "speedup": baseline_lat / max(lat, 1e-9),
                "primitives": result.graph.n_primitives,
            }
            logger.info(
                f"{layer} [{init}]: {acc - baseline_acc:+.2f} pts, {row[init]['speedup']:.2f}x"
            )
        rows.append(row)

    out = {"model": args.model, "dataset": args.dataset, "rows": rows}
    path = os.path.join(ctx.out_dir, f"q3_generalization_{args.model}.json")
    save_json(out, path)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
