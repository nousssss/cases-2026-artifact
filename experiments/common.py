"""Shared argument parsing and setup for the experiment scripts."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conas.constant_opt import ConstantOptConfig 
from conas.data import DataConfig, build_loaders, num_classes  
from conas.evaluation import LatencyConfig  
from conas.generation import GenerationConfig  
from conas.models import build_model 
from conas.search import SearchConfig  
from conas.utils import ensure_dir, get_logger, resolve_device, set_seed  

DEFAULT_DATASET = {"resnet20": "cifar10", "resnet32": "cifar10", "convnext_tiny": "imagenet"}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("model & data")
    g.add_argument("--model", default="resnet20", choices=["resnet20", "resnet32", "convnext_tiny"])
    g.add_argument("--dataset", default=None,
                   help="cifar10 | imagenet | synthetic | synthetic_imagenet (default: per model)")
    g.add_argument("--data-root", default="./data")
    g.add_argument("--checkpoint", default=None, help="baseline weights for A")
    g.add_argument("--pretrained", action="store_true", help="use torchvision weights (ConvNeXt)")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--eval-batch-size", type=int, default=256)
    g.add_argument("--workers", type=int, default=4)

    s = parser.add_argument_group("search")
    s.add_argument("--iterations", type=int, default=200, help="evolutionary generations")
    s.add_argument("--population", type=int, default=300)
    s.add_argument("--tournament-k", type=int, default=5)
    s.add_argument("--elitism", type=int, default=4)
    s.add_argument("--crossover-prob", type=float, default=0.8)
    s.add_argument("--mutation-prob", type=float, default=0.9)
    s.add_argument("--acc-equivalence", type=float, default=1.0,
                   help="candidates within this many points are compared on latency")
    s.add_argument("--fitness-batches", type=int, default=4,
                   help="validation batches used for the fitness score (0 = all)")
    s.add_argument("--const-opt-epochs", type=int, default=100)
    s.add_argument("--const-opt-lr", type=float, default=0.01)
    s.add_argument("--max-evaluations", type=int, default=None)
    s.add_argument("--min-nodes", type=int, default=6)
    s.add_argument("--max-nodes", type=int, default=14)
    s.add_argument("--operator-chunk", type=int, default=None,
                   help="row-chunk size for the patch matrix (lowers peak memory)")

    b = parser.add_argument_group("benchmarking")
    b.add_argument("--backend", default="auto", choices=["auto", "eager", "inductor", "mlir"],
                   help="code-optimization backend used for latency numbers")
    b.add_argument("--latency-batch-size", type=int, default=1)
    b.add_argument("--latency-repeats", type=int, default=50)
    b.add_argument("--latency-warmup", type=int, default=10)
    b.add_argument("--threads", type=int, default=None, help="pin torch to N CPU threads")

    m = parser.add_argument_group("misc")
    m.add_argument("--device", default="auto")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default="./runs")
    m.add_argument("--tag", default="", help="suffix appended to the output directory")
    return parser


@dataclass
class Context:
    args: argparse.Namespace
    model: torch.nn.Module
    train_loader: object
    val_loader: object
    example: torch.Tensor
    device: torch.device
    logger: object
    out_dir: str
    search_cfg: SearchConfig


def setup(args: argparse.Namespace, load_data: bool = True) -> Context:
    """Build the model, loaders, config and output directory."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)

    dataset = args.dataset or DEFAULT_DATASET.get(args.model, "cifar10")
    out_dir = ensure_dir(
        os.path.join(args.out, f"{args.model}_{dataset}" + (f"_{args.tag}" if args.tag else ""))
    )
    logger = get_logger("conas", log_file=os.path.join(out_dir, "run.log"))
    logger.info(f"device={device} dataset={dataset} out={out_dir}")

    model = build_model(
        args.model,
        num_classes=num_classes(dataset),
        pretrained=args.pretrained,
        checkpoint=args.checkpoint,
    ).to(device)
    if args.checkpoint is None and not args.pretrained:
        logger.warning(
            "no baseline checkpoint supplied: A is randomly initialised, so accuracy numbers "
            "are meaningless. Train one with scripts/train_baseline.py."
        )

    train_loader = val_loader = example = None
    if load_data:
        data_cfg = DataConfig(
            dataset=dataset,
            root=args.data_root,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            workers=args.workers,
        )
        train_loader, val_loader = build_loaders(data_cfg)
        example = next(iter(val_loader))[0].to(device)

    return Context(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        example=example,
        device=device,
        logger=logger,
        out_dir=out_dir,
        search_cfg=build_search_config(args),
    )


def build_search_config(args: argparse.Namespace, init: str = "scratch") -> SearchConfig:
    return SearchConfig(
        iterations=args.iterations,
        population=args.population,
        tournament_k=args.tournament_k,
        elitism=args.elitism,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        init=init,
        accuracy_equivalence=args.acc_equivalence,
        fitness_batches=args.fitness_batches or None,
        max_evaluations=args.max_evaluations,
        operator_chunk=args.operator_chunk,
        seed=args.seed,
        generation=GenerationConfig(min_nodes=args.min_nodes, max_nodes=args.max_nodes),
        const_opt=ConstantOptConfig(epochs=args.const_opt_epochs, lr=args.const_opt_lr),
        latency=LatencyConfig(
            batch_size=args.latency_batch_size,
            warmup=args.latency_warmup,
            repeats=args.latency_repeats,
            threads=args.threads,
        ),
    )


def resolve_backend(name: str):
    from conas.compiler import default_backend, get_backend

    if name == "auto":
        return default_backend()
    return get_backend(name)
