"""Model registry and layer-name resolution.

Supported baselines
-------------------
``resnet20`` / ``resnet32``   CIFAR-10 (Sec. IV-A: main results, generalisation)
``convnext_tiny``             ImageNet (Sec. IV-D)

Layer names may be given either as full module paths
(``layer1.1.conv1``, ``features.1.0.block.0``) or in the short form used in the
paper's figures (``1.1.conv1`` for ResNets, ``0.0.dwconv`` for ConvNeXt).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .resnet_cifar import normalize_layer_name as _resnet_name
from .resnet_cifar import resnet20, resnet32

_CONVNEXT_SHORT = re.compile(r"^(\d+)\.(\d+)\.dwconv$")

MODEL_INPUT_SIZE: Dict[str, Tuple[int, int, int]] = {
    "resnet20": (3, 32, 32),
    "resnet32": (3, 32, 32),
    "convnext_tiny": (3, 224, 224),
}


def build_model(
    name: str,
    num_classes: Optional[int] = None,
    pretrained: bool = False,
    checkpoint: Optional[str] = None,
) -> nn.Module:
    """Instantiate a baseline model ``A``."""
    name = name.lower()
    if name == "resnet20":
        model = resnet20(num_classes or 10)
    elif name == "resnet32":
        model = resnet32(num_classes or 10)
    elif name in ("convnext_tiny", "convnext"):
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        if num_classes and num_classes != 1000:
            in_f = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"unknown model '{name}'")

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
    return model


def normalize_layer_name(model_name: str, layer: str) -> str:
    """Map a short figure-style layer name onto a real module path."""
    if model_name.startswith("convnext"):
        m = _CONVNEXT_SHORT.match(layer)
        if m:
            stage, block = int(m.group(1)), int(m.group(2))
            # torchvision layout: features = [stem, stage0, down, stage1, ...]
            return f"features.{2 * stage + 1}.{block}.block.0"
        return layer
    return _resnet_name(layer)


def get_module(model: nn.Module, path: str) -> nn.Module:
    mod: nn.Module = model
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)) else getattr(mod, part)
    return mod


def set_module(model: nn.Module, path: str, new: nn.Module) -> None:
    parts = path.split(".")
    parent = get_module(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)


def list_conv_layers(model: nn.Module, min_kernel: int = 2) -> List[str]:
    """All ``Conv2d`` module paths, skipping 1x1 projections by default."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and max(mod.kernel_size) >= min_kernel:
            out.append(name)
    return out


def input_size(model_name: str) -> Tuple[int, int, int]:
    return MODEL_INPUT_SIZE.get(model_name.lower(), (3, 32, 32))
