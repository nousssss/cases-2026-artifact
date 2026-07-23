"""Structure-guided initialisation (paper, Sec. III-D).

Instead of starting from a random population, the search is seeded with a
manually defined computation graph that mimics the functional structure of a
standard convolution layer, built from the *same* primitives available in the
search space so that it stays compatible with the rest of CONAS.

Following the PyTorch formulation

.. math::
    y(i,j) = \\sum_{c=1}^{C}\\sum_{u=1}^{K}\\sum_{v=1}^{K}
             x(c, i+u, j+v)\\cdot w(c,u,v) + b

the patch view turns the triple sum into a single multiply-accumulate over the
flattened neighbourhood, i.e. ``Y = P @ W^T + b``.  In DAG terms: a
multiplication node implementing the elementwise products between input patches
and kernel weights, an addition node performing the reduction, and constant
nodes holding the learnable weights and bias.

"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
import torch.nn as nn

from .generation import GenerationConfig, GraphGenerator
from .graph import ComputationGraph, LayerSpec


# --------------------------------------------------------------------------- #
# Weight extraction
# --------------------------------------------------------------------------- #
def dense_im2col_weight(conv: nn.Conv2d) -> torch.Tensor:
    """Return the ``(Cout, K)`` dense matrix equivalent to ``conv``'s kernel.

    For a grouped or depthwise convolution the dense form is block-sparse: the
    entries connecting an output channel to input channels outside its group are
    zero.  Materialising them keeps the learned operator free to *become* dense
    during evolution, which is the point of the structure-guided seed.
    """
    cout, cin_per_group, kh, kw = conv.weight.shape
    cin = conv.in_channels
    k = cin * kh * kw
    dense = conv.weight.new_zeros(cout, k)
    per_group_out = cout // conv.groups
    for o in range(cout):
        g = o // per_group_out
        base = g * cin_per_group
        for c in range(cin_per_group):
            start = (base + c) * kh * kw
            dense[o, start : start + kh * kw] = conv.weight[o, c].reshape(-1)
    return dense


# --------------------------------------------------------------------------- #
# The convolution-mimicking seed graph
# --------------------------------------------------------------------------- #
def conv_equivalent_graph(
    spec: LayerSpec,
    conv: Optional[nn.Conv2d] = None,
    init_range=(-0.05, 0.05),
) -> ComputationGraph:
    """``l'(X) = reshape(P @ Theta_W + Theta_b)`` -- the manual conv graph.

    When ``conv`` is supplied the constants are seeded with the layer's own
    parameters, so the seed graph is numerically identical to the convolution it
    replaces (up to floating-point reassociation).  This mirrors GOS, which
    evolves operators from the original implementation.  With ``conv=None`` the
    same *structure* is used but the constants are drawn from ``U(a, b)``.
    """
    g = ComputationGraph(spec)
    if conv is not None:
        w = dense_im2col_weight(conv).t().contiguous()  # (K, Cout)
        theta_w = g.add_node("const", const_dims=(spec.K, spec.out_channels),
                             init_values=w.detach().cpu().tolist())
        bias = conv.bias if conv.bias is not None else torch.zeros(spec.out_channels)
        theta_b = g.add_node("const", const_dims=(spec.out_channels,),
                             init_values=bias.detach().cpu().tolist())
    else:
        a = 1.0 / spec.K**0.5
        theta_w = g.add_node("const", const_dims=(spec.K, spec.out_channels), init_range=(-a, a))
        theta_b = g.add_node("const", const_dims=(spec.out_channels,), init_range=init_range)

    mm = g.add_node("matmul", inputs=[g.patch_id, theta_w])
    out = g.add_node("matadd", inputs=[mm, theta_b])
    g.output = out
    g.check_structure()
    return g


def lowrank_conv_graph(spec: LayerSpec, rank: int = 16) -> ComputationGraph:
    """A cheaper structured seed: ``reshape(P @ Theta_1 @ Theta_2 + Theta_b)``."""
    g = ComputationGraph(spec)
    a1 = 1.0 / spec.K**0.5
    a2 = 1.0 / rank**0.5
    t1 = g.add_node("const", const_dims=(spec.K, rank), init_range=(-a1, a1))
    t2 = g.add_node("const", const_dims=(rank, spec.out_channels), init_range=(-a2, a2))
    tb = g.add_node("const", const_dims=(spec.out_channels,), init_range=(-0.05, 0.05))
    m1 = g.add_node("matmul", inputs=[g.patch_id, t1])
    m2 = g.add_node("matmul", inputs=[m1, t2])
    g.output = g.add_node("matadd", inputs=[m2, tb])
    g.check_structure()
    return g


# --------------------------------------------------------------------------- #
# Worked examples from Sec. III-E
# --------------------------------------------------------------------------- #
def gated_lowrank_patch_graph(spec: LayerSpec, rank: int = 16) -> ComputationGraph:
    """Eq. (1): the from-structure example -- low-rank gated patch operator.

    ``reshape([tanh(P T1 + T2) o sigmoid(P T1 + T2)] T3 + Xc T4 + T5)``

    Three matrix multiplications, three matrix additions, one elementwise
    multiplication, one tanh, one sigmoid and five learned constant tensors.
    """
    g = ComputationGraph(spec)
    a1 = 1.0 / spec.K**0.5
    a3 = 1.0 / rank**0.5
    ac = 1.0 / spec.in_channels**0.5
    t1 = g.add_node("const", const_dims=(spec.K, rank), init_range=(-a1, a1))
    t2 = g.add_node("const", const_dims=(rank,), init_range=(-0.05, 0.05))
    t3 = g.add_node("const", const_dims=(rank, spec.out_channels), init_range=(-a3, a3))
    t4 = g.add_node("const", const_dims=(spec.in_channels, spec.out_channels), init_range=(-ac, ac))
    t5 = g.add_node("const", const_dims=(spec.out_channels,), init_range=(-0.05, 0.05))

    u1 = g.add_node("matmul", inputs=[g.patch_id, t1])       # u1 = P T1
    u2 = g.add_node("matadd", inputs=[u1, t2])               # u2 = u1 + T2
    u3 = g.add_node("tanh", inputs=[u2])                     # u3 = tanh(u2)
    u4 = g.add_node("sigmoid", inputs=[u2])                  # u4 = sigma(u2)
    u5 = g.add_node("elemmul", inputs=[u3, u4])              # u5 = u3 o u4
    u6 = g.add_node("matmul", inputs=[u5, t3])               # u6 = u5 T3
    u7 = g.add_node("matmul", inputs=[g.center_id, t4])      # u7 = Xc T4
    u8 = g.add_node("matadd", inputs=[u6, u7])               # u8 = u6 + u7
    u9 = g.add_node("matadd", inputs=[u8, t5])               # u9 = u8 + T5
    g.output = u9
    g.check_structure()
    return g


def lowrank_relu_patch_graph(spec: LayerSpec, rank: int = 16) -> ComputationGraph:
    """Eq. (2): the from-scratch example -- compact low-rank ReLU projection.

    ``reshape(ReLU(P T1 + T2) T3 + T4)``
    """
    g = ComputationGraph(spec)
    a1 = 1.0 / spec.K**0.5
    a3 = 1.0 / rank**0.5
    t1 = g.add_node("const", const_dims=(spec.K, rank), init_range=(-a1, a1))
    t2 = g.add_node("const", const_dims=(rank,), init_range=(-0.05, 0.05))
    t3 = g.add_node("const", const_dims=(rank, spec.out_channels), init_range=(-a3, a3))
    t4 = g.add_node("const", const_dims=(spec.out_channels,), init_range=(-0.05, 0.05))

    v1 = g.add_node("matmul", inputs=[g.patch_id, t1])
    v2 = g.add_node("matadd", inputs=[v1, t2])
    v3 = g.add_node("relu", inputs=[v2])
    v4 = g.add_node("matmul", inputs=[v3, t3])
    g.output = g.add_node("matadd", inputs=[v4, t4])
    g.check_structure()
    return g


# --------------------------------------------------------------------------- #
# Seed populations
# --------------------------------------------------------------------------- #
def structured_population(
    spec: LayerSpec,
    size: int,
    conv: Optional[nn.Conv2d] = None,
    rng: Optional[random.Random] = None,
    seed_mutations: int = 2,
    n_elites: int = 4,
    gen_cfg: Optional[GenerationConfig] = None,
) -> List[ComputationGraph]:
    """Initial population for the from-structure variant.

    Every individual descends from the manually defined convolution graph.  A
    few elites are kept untouched; the rest receive a small number of random
    mutations so that the very first generation already carries diversity, while
    the inductive bias aligned with convolutional behaviour is preserved.

    The initialisation does not restrict the search space: all nodes and
    connections remain subject to modification during evolution.
    """
    from .evolution import mutate  # local import: avoids a circular dependency

    rng = rng or random.Random()
    base = conv_equivalent_graph(spec, conv)
    generator = GraphGenerator(spec, gen_cfg, rng)

    pop: List[ComputationGraph] = []
    for i in range(size):
        if i < n_elites:
            pop.append(base.clone())
            continue
        cand = base.clone()
        for _ in range(seed_mutations):
            mutated = mutate(cand, generator, rng, allow_constant_opt=False)
            if mutated is not None:
                cand = mutated
        pop.append(cand)
    return pop
