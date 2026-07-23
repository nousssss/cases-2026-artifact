"""CIFAR-style ResNets (He et al., 2016) -- the ResNet20/ResNet32 baselines.

Layer naming follows the convention used in the paper's figures: ``1.1.conv1``
means ``layer1[1].conv1``.  :func:`normalize_layer_name` accepts either form.
"""

from __future__ import annotations

import re
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCifar(nn.Module):
    def __init__(self, blocks_per_stage: int, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_stage(16, blocks_per_stage, 1)
        self.layer2 = self._make_stage(32, blocks_per_stage, 2)
        self.layer3 = self._make_stage(64, blocks_per_stage, 2)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_stage(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet20(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(3, num_classes)


def resnet32(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(5, num_classes)


_SHORT_NAME = re.compile(r"^(\d+)\.(\d+)\.(conv\d+)$")


def normalize_layer_name(name: str) -> str:
    """``"1.1.conv1"`` -> ``"layer1.1.conv1"``; other names pass through."""
    m = _SHORT_NAME.match(name)
    if m:
        return f"layer{m.group(1)}.{m.group(2)}.{m.group(3)}"
    return name


def paper_resnet20_layers() -> List[str]:
    """The seven ResNet20 layers reported in Figs. 4-7."""
    return [
        "1.1.conv1",
        "1.2.conv1",
        "2.0.conv2",
        "2.2.conv2",
        "3.0.conv2",
        "3.2.conv1",
        "3.2.conv2",
    ]


def paper_resnet32_layers() -> List[str]:
    """The five ResNet32 layers reported in Fig. 8(a)-(b)."""
    return ["1.0.conv1", "1.0.conv2", "1.1.conv1", "1.2.conv1", "1.4.conv1"]
