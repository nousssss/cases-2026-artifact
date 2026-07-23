#!/usr/bin/env python3
"""Train a baseline model ``A``.

There is no canonical pretrained checkpoint for CIFAR-style ResNet20/32, so the
baseline has to be trained once before any search: every accuracy number CONAS
reports is relative to ``Acc(A)``.

The recipe is the standard one for CIFAR ResNets (He et al.): SGD with momentum
0.9, weight decay 5e-4, initial learning rate 0.1, 200 epochs, cosine schedule.


Example::

    python scripts/train_baseline.py --model resnet20 --epochs 200 \
        --out runs/resnet20_baseline.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conas.data import DataConfig, build_loaders, num_classes  # noqa: E402
from conas.evaluation import evaluate_accuracy  # noqa: E402
from conas.models import build_model  # noqa: E402
from conas.utils import ensure_dir, get_logger, resolve_device, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="resnet20", choices=["resnet20", "resnet32"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/baseline.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    logger = get_logger("conas.train")
    ensure_dir(os.path.dirname(args.out) or ".")

    train_loader, val_loader = build_loaders(
        DataConfig(dataset=args.dataset, root=args.data_root, batch_size=args.batch_size,
                   workers=args.workers)
    )
    model = build_model(args.model, num_classes=num_classes(args.dataset)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss) * y.numel()
            seen += y.numel()
        sched.step()
        acc = evaluate_accuracy(model, val_loader, device)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "accuracy": acc, "epoch": epoch}, args.out)
        logger.info(f"epoch {epoch:3d}  loss {running / seen:.4f}  acc {acc:.2f}%  best {best:.2f}%")

    logger.info(f"best accuracy {best:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
