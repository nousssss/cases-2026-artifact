"""Small shared utilities: seeding, hashing, logging and checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> random.Random:
    """Seed every RNG and return a dedicated generator for the search."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return random.Random(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Graph identity
# --------------------------------------------------------------------------- #
def structural_hash(graph) -> str:
    """Canonical fingerprint of a DAG's *structure* (constants' values excluded).

    Used to skip re-evaluating candidates that crossover and mutation have
    already produced, which is a large saving late in the search when the
    population starts to converge.
    """
    order = graph.topo_order()
    index = {nid: i for i, nid in enumerate(order)}
    parts = []
    for nid in order:
        n = graph.nodes[nid]
        ins = ",".join(str(index[i]) for i in n.inputs)
        dims = "x".join(str(d) for d in (n.const_dims or ()))
        parts.append(f"{index[nid]}:{n.op}({ins}){dims}")
    parts.append(f"out={index[graph.output]}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "conas", level: int = logging.INFO, log_file: Optional[str] = None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextmanager
def timed(label: str, logger=None):
    start = time.perf_counter()
    yield
    msg = f"{label} took {time.perf_counter() - start:.1f}s"
    (logger.info if logger else print)(msg)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def sanitize(name: str) -> str:
    """Make a layer name safe for use in a file path."""
    return name.replace(".", "_").replace("/", "_")
