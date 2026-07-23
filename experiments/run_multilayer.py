#!/usr/bin/env python3
"""Multi-layer replacement with the sequential greedy strategy (Sec. IV-F).

Replaces several layers one after another, each search running on the model that
already contains the previously accepted operators, and stopping as soon as the
accuracy-latency budget is exhausted.  Operators are reused across
shape-compatible layers when that costs little accuracy, which avoids a full
search per layer.

Example::

    python experiments/run_multilayer.py --model resnet20 \
        --checkpoint runs/resnet20_baseline.pt \
        --layers 3.2.conv2 3.2.conv1 3.0.conv2 --max-accuracy-drop 2.0
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from common import add_common_args, build_search_config, setup

from conas.finetune import FineTuneConfig, fine_tune
from conas.multilayer import MultiLayerConfig, sequential_greedy_search
from conas.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--layers", nargs="+", required=True, help="layers in replacement order")
    parser.add_argument("--init", default="struct", choices=["scratch", "struct"])
    parser.add_argument("--max-accuracy-drop", type=float, default=2.0,
                        help="gamma: the acceptable accuracy drop relative to A")
    parser.add_argument("--target-speedup", type=float, default=None)
    parser.add_argument("--no-reuse", action="store_true", help="disable operator reuse")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=50)
    args = parser.parse_args()

    ctx = setup(args)
    search_cfg = build_search_config(args, init=args.init)
    multi_cfg = MultiLayerConfig(
        max_accuracy_drop=args.max_accuracy_drop,
        target_speedup=args.target_speedup,
        max_layers=len(args.layers),
        enable_reuse=not args.no_reuse,
    )

    model, result = sequential_greedy_search(
        ctx.model,
        args.layers,
        ctx.train_loader,
        ctx.val_loader,
        search_cfg,
        multi_cfg,
        model_name=args.model,
        device=ctx.device,
        logger=ctx.logger,
    )

    out_path = os.path.join(ctx.out_dir, "multilayer.json")
    save_json(result.summary(), out_path)
    for path, graph in result.operators.items():
        graph.save(os.path.join(ctx.out_dir, "operators", f"{path.replace('.', '_')}.graph.json"))
    print(json.dumps(result.summary(), indent=2))

    if args.fine_tune:
        ft_cfg = FineTuneConfig.for_model(args.model)
        ft_cfg.epochs = args.fine_tune_epochs
        record = fine_tune(model, ctx.train_loader, ctx.val_loader, ft_cfg, ctx.device, ctx.logger)
        save_json(record, os.path.join(ctx.out_dir, "multilayer.finetune.json"))
        torch.save(model.state_dict(), os.path.join(ctx.out_dir, "multilayer.model.pt"))
        print(json.dumps({k: v for k, v in record.items() if k != "history"}, indent=2))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
