#!/usr/bin/env python3
"""Q2 -- Does structure improve accuracy?  (paper, Sec. IV-C; Figs. 5, 6, 7)

For each target layer the script runs **both** CONAS initialisation strategies
and records, for each of them:

* the validation accuracy of ``A'``  (Fig. 5)
* the inference speedup **without** the code-optimization part of the framework
* the inference speedup **with** it   (Figs. 6 and 7)

Reporting both isolates the effect of the operator replacement itself and shows
that the acceleration does not stem solely from code optimisation.

Run it once per hardware target; ``--device-label`` only tags the output file
(use e.g. ``desktop-cpu`` on the Intel machine and ``rpi3`` on the Raspberry Pi).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch

from common import add_common_args, build_search_config, resolve_backend, setup

from conas.evaluation import evaluate_accuracy, measure_latency
from conas.models import normalize_layer_name
from conas.models.resnet_cifar import paper_resnet20_layers
from conas.operator import GraphOperator
from conas.replacement import collect_calibration, probe_layer, temporarily_replaced
from conas.search import CONASSearch
from conas.utils import save_json


def timed_variants(model, path, op, input_shape, device, lat_cfg, backend, example):
    """Latency of ``A'`` without and with code optimisation."""
    with temporarily_replaced(model, path, op):
        raw = measure_latency(model, input_shape, device, lat_cfg)
        try:
            compiled = backend.prepare(model, example)
            opt = measure_latency(compiled, input_shape, device, lat_cfg)
        except Exception as exc:  # a candidate the backend cannot compile
            print(f"  [warn] {backend.name} backend failed: {exc}")
            opt = None
    return raw, opt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--device-label", default="desktop-cpu",
                        help="tag recorded with the timings, e.g. desktop-cpu or rpi3")
    args = parser.parse_args()

    ctx = setup(args)
    model, device, logger = ctx.model, ctx.device, ctx.logger
    layers = args.layers or paper_resnet20_layers()
    backend = resolve_backend(args.backend)
    logger.info(f"code-optimization backend: {backend.name}")

    input_shape = tuple(ctx.example.shape[1:])
    baseline_acc = evaluate_accuracy(model, ctx.val_loader, device)
    baseline_raw = measure_latency(model, input_shape, device, ctx.search_cfg.latency)
    baseline_opt = None
    try:
        baseline_opt = measure_latency(
            backend.prepare(model, ctx.example), input_shape, device, ctx.search_cfg.latency
        )
    except Exception as exc:
        logger.warning(f"could not compile the baseline model: {exc}")
    logger.info(
        f"baseline A: {baseline_acc:.2f}%  {baseline_raw:.3f} ms (eager), "
        f"{baseline_opt if baseline_opt else float('nan'):.3f} ms (optimised)"
    )

    rows = []
    for layer in layers:
        path = normalize_layer_name(args.model, layer)
        probe = probe_layer(model, layer, ctx.example, args.model, capture_inputs=True)
        calibration = collect_calibration(model, path, ctx.val_loader, 256, device)
        row = {"layer": layer, "baseline_accuracy": baseline_acc,
               "baseline_latency_raw_ms": baseline_raw, "baseline_latency_opt_ms": baseline_opt}

        for init in ("scratch", "struct"):
            cfg = build_search_config(args, init=init)
            result = CONASSearch(model, probe, ctx.val_loader, cfg, device, calibration, logger).run()
            result.save(os.path.join(ctx.out_dir, "operators"))

            op = GraphOperator(result.graph, chunk=cfg.operator_chunk).to(device)
            op.load_state_dict(result.state_dict)
            with temporarily_replaced(model, path, op):
                acc = evaluate_accuracy(model, ctx.val_loader, device)
            raw, opt = timed_variants(
                model, path, op, input_shape, device, cfg.latency, backend, ctx.example
            )
            row[init] = {
                "accuracy": acc,
                "accuracy_change": acc - baseline_acc,
                "latency_raw_ms": raw,
                "latency_opt_ms": opt,
                "speedup_without_code_opt": baseline_raw / max(raw, 1e-9),
                "speedup_with_code_opt": (baseline_raw / opt) if opt else None,
                "primitives": result.graph.n_primitives,
                "parameters": result.graph.n_parameters(),
                "mse": result.mse,
            }
            logger.info(
                f"{layer} [{init}]: {acc - baseline_acc:+.2f} pts, "
                f"{row[init]['speedup_without_code_opt']:.2f}x without code opt, "
                f"{row[init]['speedup_with_code_opt'] or float('nan'):.2f}x with"
            )
        rows.append(row)

    out = {
        "device_label": args.device_label,
        "backend": backend.name,
        "model": args.model,
        "rows": rows,
    }
    path = os.path.join(ctx.out_dir, f"q2_init_comparison_{args.device_label}.json")
    save_json(out, path)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
