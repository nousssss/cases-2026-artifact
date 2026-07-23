"""Fine-tuning protocol (paper, Sec. IV-A).

Once the search is done the modified model ``A'`` is fine-tuned for 50 epochs
using the baseline training configuration:

* ResNet / CIFAR-10 -- SGD, learning rate 0.1, momentum 0.9
* ConvNeXt / ImageNet -- AdamW, learning rate 5e-5, weight decay 1e-8

Fine-tuning happens **after** the search, never during it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .evaluation import evaluate_accuracy
from .utils import get_logger


@dataclass
class FineTuneConfig:
    epochs: int = 50
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: Optional[str] = "cosine"
    label_smoothing: float = 0.0
    grad_clip: Optional[float] = None
    #: Freeze everything except the learned operators.  Off by default: the
    #: paper fine-tunes the whole model with the baseline configuration.
    operators_only: bool = False
    eval_every: int = 1

    @staticmethod
    def for_model(model_name: str) -> "FineTuneConfig":
        if model_name.startswith("convnext"):
            return FineTuneConfig(
                epochs=50, optimizer="adamw", lr=5e-5, weight_decay=1e-8, momentum=0.0
            )
        return FineTuneConfig(epochs=50, optimizer="sgd", lr=0.1, momentum=0.9, weight_decay=0.0)


def build_optimizer(model: nn.Module, cfg: FineTuneConfig) -> torch.optim.Optimizer:
    from .operator import GraphOperator

    if cfg.operators_only:
        params = [p for m in model.modules() if isinstance(m, GraphOperator) for p in m.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )


def fine_tune(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    cfg: Optional[FineTuneConfig] = None,
    device: torch.device = torch.device("cpu"),
    logger=None,
) -> dict:
    """Fine-tune ``A'`` and return the training record.
    """
    cfg = cfg or FineTuneConfig()
    logger = logger or get_logger()
    model = model.to(device)
    opt = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else None
    )

    before = evaluate_accuracy(model, val_loader, device)
    logger.info(f"accuracy before fine-tuning: {before:.2f}%")

    history = []
    best = before
    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(loss) * y.numel()
            seen += y.numel()
        if sched:
            sched.step()

        record = {"epoch": epoch, "loss": running / max(seen, 1)}
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            acc = evaluate_accuracy(model, val_loader, device)
            record["accuracy"] = acc
            best = max(best, acc)
            logger.info(f"epoch {epoch:3d}  loss {record['loss']:.4f}  acc {acc:.2f}%")
        history.append(record)

    after = evaluate_accuracy(model, val_loader, device)
    logger.info(f"accuracy after fine-tuning: {after:.2f}% (recovered {after - before:+.2f} pts)")
    return {
        "accuracy_before": before,
        "accuracy_after": after,
        "best_accuracy": best,
        "recovered": after - before,
        "history": history,
    }
