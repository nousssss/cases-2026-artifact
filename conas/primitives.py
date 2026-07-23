"""Primitive operations used to construct candidate computation graphs.

This is a direct implementation of Table I of the paper: the vocabulary from
which candidate DAGs are composed.  Primitives are *not* used as standalone
replacements for convolution layers -- a replacement operator typically
contains several instances of a subset of them (paper, Sec. III-B).

Two evaluation modes are provided per primitive:

``strict``
    The mathematical definition.  Used by the validity checker: if a subgraph
    produces NaN/Inf under strict evaluation (division by zero, ``log``/``sqrt``
    of a negative value) the candidate is rejected, as described in Sec. III-B1.

``guarded``
    A numerically clamped variant used during search and training so that a
    graph that is valid on the probe inputs cannot blow up on a rare batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

EPS = 1e-6

# --------------------------------------------------------------------------- #
# Categories mirror the row groups of Table I.
LINALG = "linear_algebra"
ELEMENTWISE = "elementwise"
NONLINEAR = "nonlinear"
CONSTANT = "constant"


@dataclass(frozen=True)
class Primitive:
    """A single row of Table I."""

    name: str
    arity: int
    category: str
    expression: str
    strict: Optional[Callable[..., torch.Tensor]] = None
    guarded: Optional[Callable[..., torch.Tensor]] = None
    #: ``True`` for primitives with a restricted domain (div, log, sqrt, pow).
    domain_restricted: bool = False
    #: ``True`` when the operand order matters (used by mutation/crossover).
    ordered: bool = True

    def __call__(self, *args: torch.Tensor, strict: bool = False) -> torch.Tensor:
        fn = self.strict if strict else self.guarded
        assert fn is not None, f"primitive {self.name} is not callable"
        return fn(*args)


def _guard_denominator(b: torch.Tensor) -> torch.Tensor:
    """Push a denominator away from zero without changing its sign."""
    sign = torch.where(b >= 0, 1.0, -1.0)
    return torch.where(b.abs() < EPS, sign * EPS, b)


_TABLE: List[Primitive] = [
    # --- Linear algebra operations ----------------------------------------- #
    Primitive(
        "matmul", 2, LINALG, "C = A x B",
        strict=torch.matmul, guarded=torch.matmul,
    ),
    Primitive(
        "matadd", 2, LINALG, "C = A + B",
        strict=torch.add, guarded=torch.add, ordered=False,
    ),
    Primitive(
        "matsub", 2, LINALG, "C = A - B",
        strict=torch.sub, guarded=torch.sub,
    ),
    # --- Elementwise operations -------------------------------------------- #
    Primitive(
        "elemmul", 2, ELEMENTWISE, "C = A o B",
        strict=torch.mul, guarded=torch.mul, ordered=False,
    ),
    Primitive(
        "elemdiv", 2, ELEMENTWISE, "C = A / B",
        strict=torch.div,
        guarded=lambda a, b: a / _guard_denominator(b),
        domain_restricted=True,
    ),
    Primitive(
        "elempow", 2, ELEMENTWISE, "C = A ^ B",
        strict=torch.pow,
        # Fractional exponents are undefined for negative bases, so the guarded
        # form raises the magnitude and restores the sign of the base.
        guarded=lambda a, b: torch.sign(a) * torch.pow(a.abs() + EPS, torch.clamp(b, -4.0, 4.0)),
        domain_restricted=True,
    ),
    Primitive(
        "abs", 1, ELEMENTWISE, "C = |A|",
        strict=torch.abs, guarded=torch.abs,
    ),
    Primitive(
        "sqrt", 1, ELEMENTWISE, "C = sqrt(A)",
        strict=torch.sqrt,
        guarded=lambda a: torch.sqrt(torch.clamp(a, min=EPS)),
        domain_restricted=True,
    ),
    Primitive(
        "log", 1, ELEMENTWISE, "C = log(A)",
        strict=torch.log,
        guarded=lambda a: torch.log(torch.clamp(a, min=EPS)),
        domain_restricted=True,
    ),
    # --- Nonlinear functions ----------------------------------------------- #
    Primitive("relu", 1, NONLINEAR, "C = max(0, A)", strict=torch.relu, guarded=torch.relu),
    Primitive("sigmoid", 1, NONLINEAR, "C = 1/(1+e^-A)", strict=torch.sigmoid, guarded=torch.sigmoid),
    Primitive("tanh", 1, NONLINEAR, "C = tanh(A)", strict=torch.tanh, guarded=torch.tanh),
    Primitive(
        "exp", 1, NONLINEAR, "C = e^A",
        strict=torch.exp,
        guarded=lambda a: torch.exp(torch.clamp(a, max=20.0)),
    ),
    Primitive("sin", 1, NONLINEAR, "C = sin(A)", strict=torch.sin, guarded=torch.sin),
    Primitive("cos", 1, NONLINEAR, "C = cos(A)", strict=torch.cos, guarded=torch.cos),
    # --- Constants ---------------------------------------------------------- #
    Primitive("const", 0, CONSTANT, "C ~ U(a, b)"),
]

PRIMITIVES: Dict[str, Primitive] = {p.name: p for p in _TABLE}

UNARY = [p.name for p in _TABLE if p.arity == 1]
BINARY = [p.name for p in _TABLE if p.arity == 2]
#: Binary primitives whose semantics are elementwise (as opposed to ``matmul``).
BROADCASTING_BINARY = [n for n in BINARY if n != "matmul"]

#: Interface pseudo-nodes.  Sec. III-E of the paper is explicit that patch
#: extraction, center selection and the final reshape express the layer-level
#: input/output interface and are *not* counted as search primitives.
INTERFACE_OPS = ("patch", "center")


def primitive_count(op_names) -> Dict[str, int]:
    """Count only real Table I primitives (interface ops excluded)."""
    counts: Dict[str, int] = {}
    for name in op_names:
        if name in INTERFACE_OPS:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts
