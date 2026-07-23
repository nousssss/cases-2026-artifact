#!/usr/bin/env python3
"""Search for a replacement operator for one convolution layer.

Examples
--------
Structure-guided search on a ResNet20 layer::

    python experiments/run_search.py --model resnet20 --layer 1.1.conv1 \
        --init struct --checkpoint runs/resnet20_baseline.pt

From scratch, with a smaller budget::

    python experiments/run_search.py --layer 3.2.conv2 --init scratch \
        --iterations 50 --population 100
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from common import add_common_args, build_search_config, setup

from conas.finetune import FineTuneConfig, fine_tune
from conas.operator import GraphOperator
from conas.replacement import collect_calibration, probe_layer, replace_layer
from conas.search import CONASSearch
from conas.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--layer", required=True, help="e.g. 1.1.conv1, 3.2.conv2, 0.0.dwconv")
    parser.add_argument("--init", default="struct", choices=["scratch", "struct"])
    parser.add_argument("--calibration-samples", type=int, default=256,
                        help="real activations used by constant optimisation (0 = Gaussian inputs)")
    parser.add_argument("--fine-tune", action="store_true", help="fine-tune A' after the search")
    parser.add_argument("--fine-tune-epochs", type=int, default=50)
    args = parser.parse_args()

    ctx = setup(args)
    cfg = build_search_config(args, init=args.init)
    cfg.checkpoint_dir = os.path.join(ctx.out_dir, "checkpoints")

    probe = probe_layer(ctx.model, args.layer, ctx.example, args.model, capture_inputs=True)
    ctx.logger.info(
        f"layer {probe.path}: Cin={probe.spec.in_channels} Cout={probe.spec.out_channels} "
        f"k={probe.spec.kernel_size} in_hw={probe.in_hw} K={probe.spec.K}"
    )

    calibration = None
    if args.calibration_samples > 0:
        calibration = collect_calibration(
            ctx.model, probe.path, ctx.val_loader, args.calibration_samples, ctx.device
        )
        ctx.logger.info(f"captured {tuple(calibration.shape)} calibration activations")

    search = CONASSearch(ctx.model, probe, ctx.val_loader, cfg, ctx.device, calibration, ctx.logger)
    result = search.run()
    stem = result.save(os.path.join(ctx.out_dir, "operators"))
    ctx.logger.info(f"saved operator to {stem}.*")
    print(json.dumps(result.summary(), indent=2))
    print("\nDiscovered operator:\n" + result.graph.pretty())

    if args.fine_tune:
        operator = GraphOperator(result.graph, chunk=cfg.operator_chunk).to(ctx.device)
        operator.load_state_dict(result.state_dict)
        replace_layer(ctx.model, probe.path, operator)
        ft_cfg = FineTuneConfig.for_model(args.model)
        ft_cfg.epochs = args.fine_tune_epochs
        record = fine_tune(
            ctx.model, ctx.train_loader, ctx.val_loader, ft_cfg, ctx.device, ctx.logger
        )
        save_json(record, stem + ".finetune.json")
        torch.save(ctx.model.state_dict(), stem + ".model.pt")
        print(json.dumps({k: v for k, v in record.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
