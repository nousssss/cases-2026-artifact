"""Datasets used in the evaluation (paper, Sec. IV-A).

ResNet20/ResNet32 are trained on CIFAR-10 and ConvNeXt on ImageNet.  
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    root: str = "./data"
    batch_size: int = 128
    eval_batch_size: int = 256
    workers: int = 4
    download: bool = True
    #: Number of images in the synthetic fallback dataset.
    synthetic_size: int = 512


class SyntheticDataset(Dataset):
    """Deterministic random images -- placeholder for offline runs."""

    def __init__(self, size: int, shape: Tuple[int, int, int], num_classes: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(size, *shape, generator=g)
        self.y = torch.randint(0, num_classes, (size,), generator=g)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _cifar10(cfg: DataConfig, train: bool):
    from torchvision import datasets, transforms

    if train:
        tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    else:
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        )
    return datasets.CIFAR10(cfg.root, train=train, download=cfg.download, transform=tf)


def _imagenet(cfg: DataConfig, train: bool):
    import os

    from torchvision import datasets, transforms

    if train:
        tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    split = "train" if train else "val"
    return datasets.ImageFolder(os.path.join(cfg.root, split), transform=tf)


def build_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)``."""
    name = cfg.dataset.lower()
    if name == "synthetic":
        shape, classes = (3, 32, 32), 10
        train = SyntheticDataset(cfg.synthetic_size, shape, classes, seed=0)
        val = SyntheticDataset(cfg.synthetic_size // 2, shape, classes, seed=1)
    elif name == "synthetic_imagenet":
        shape, classes = (3, 224, 224), 1000
        train = SyntheticDataset(cfg.synthetic_size, shape, classes, seed=0)
        val = SyntheticDataset(cfg.synthetic_size // 2, shape, classes, seed=1)
    elif name == "cifar10":
        train, val = _cifar10(cfg, True), _cifar10(cfg, False)
    elif name == "imagenet":
        train, val = _imagenet(cfg, True), _imagenet(cfg, False)
    else:
        raise ValueError(f"unknown dataset '{cfg.dataset}'")

    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True
    )
    return train_loader, val_loader


def num_classes(dataset: str) -> int:
    return {"cifar10": 10, "synthetic": 10, "imagenet": 1000, "synthetic_imagenet": 1000}[
        dataset.lower()
    ]


def example_batch(loader, device: Optional[torch.device] = None) -> torch.Tensor:
    """One input batch, used to probe layer shapes."""
    x = next(iter(loader))[0]
    return x.to(device) if device else x
