"""Evolutionary operators over computation graphs (paper, Sec. III-C).

Crossover
---------
Unlike strings or trees, graphs cannot always be split at a single point.  We
therefore randomly select a connected subgraph from each parent and swap them:

* sample an internal node uniformly at random and use it as an **anchor**;
* grow a connected subgraph by randomly expanding to neighbouring *successor*
  nodes, with a bound on the subgraph size so a single crossover step cannot
  replace most of the parent;
* the inputs and outputs of the subgraph are **induced by its boundary** -- an
  input is any edge entering the subgraph from outside, an output is any edge
  leaving it toward the rest of the DAG.  A selected subgraph is therefore not
  restricted to a single-input/single-output block.

The resulting children are accepted only if they pass the validity checks of
Sec. III-B; otherwise they are discarded and another attempt is made.  Crossover
modifies internal computations only: the child still consumes the same input
tensor as the original convolution layer and produces an output of the same
shape.

Mutation
--------
Four types, sampled with equal probability (1/4 each): adding a node, replacing
a node, altering connections, and additional constant optimisation.  The last
one has no structural effect and is carried out by the search loop, which owns
the optimiser state.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .generation import GraphGenerator
from .graph import CONST, DATA, ComputationGraph, GraphError, ShapeError
from .operator import numeric_probe
from .primitives import BROADCASTING_BINARY, INTERFACE_OPS, PRIMITIVES, UNARY

MUTATION_TYPES = ("add_node", "replace_node", "alter_connections", "constant_optimization")

#: A crossover subgraph may cover at most this fraction of the parent's
#: operation nodes, so that the swap does not amount to replacing the parent.
MAX_SUBGRAPH_FRACTION = 0.4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _operation_nodes(g: ComputationGraph) -> List[int]:
    return [
        nid
        for nid, n in g.nodes.items()
        if n.op not in INTERFACE_OPS and n.op != "const"
    ]


def _finalise(
    child: ComputationGraph,
    generator: GraphGenerator,
    repair_output: bool = True,
    numeric_check: bool = True,
) -> Optional[ComputationGraph]:
    """Prune, fix the output interface, and run the validity checks."""
    try:
        if child.output is None or child.output not in child.nodes:
            return None
        shapes = child.infer_shapes()
        if shapes[child.output][0] != DATA:
            return None
        if shapes[child.output][1] != child.spec.out_channels:
            if not repair_output:
                return None
            # Shape correctness is enforced only at the output, so a single
            # learned projection restores the fixed external interface.
            generator.enforce_output(child, child.output)
        child.prune()
        child.check_structure()
    except (GraphError, ShapeError):
        return None
    if child.n_primitives == 0:
        return None
    if numeric_check and not numeric_probe(child, n_samples=32):
        return None
    return child


# --------------------------------------------------------------------------- #
# Crossover
# --------------------------------------------------------------------------- #
def select_subgraph(
    g: ComputationGraph,
    rng: random.Random,
    max_fraction: float = MAX_SUBGRAPH_FRACTION,
) -> Optional[Set[int]]:
    """Anchor-and-expand selection of a connected subgraph of operation nodes."""
    ops = _operation_nodes(g)
    if not ops:
        return None
    anchor = rng.choice(ops)
    max_size = max(1, int(max_fraction * len(ops)))
    sub = {anchor}
    frontier = [anchor]
    while len(sub) < max_size and frontier:
        cur = rng.choice(frontier)
        succ = [s for s in g.successors(cur) if s not in sub and g.nodes[s].op not in INTERFACE_OPS
                and g.nodes[s].op != "const"]
        if not succ:
            frontier.remove(cur)
            continue
        nxt = rng.choice(succ)
        sub.add(nxt)
        frontier.append(nxt)
    return sub


def _subgraph_outputs(g: ComputationGraph, sub: Set[int]) -> List[int]:
    """Nodes of ``sub`` with at least one edge leaving the subgraph (or the graph output)."""
    outs = [n for n in sorted(sub) if any(s not in sub for s in g.successors(n))]
    if g.output in sub and g.output not in outs:
        outs.append(g.output)
    if not outs:
        outs = [max(sub)]
    return outs


def _splice(
    host: ComputationGraph,
    sub_host: Set[int],
    donor: ComputationGraph,
    sub_donor: Set[int],
    rng: random.Random,
) -> Optional[ComputationGraph]:
    """Replace ``sub_host`` inside a copy of ``host`` with a copy of ``sub_donor``."""
    child = host.clone()
    try:
        host_shapes = child.infer_shapes()
        donor_shapes = donor.infer_shapes()
    except (GraphError, ShapeError):
        return None

    # Boundary of the host subgraph.  The donor's dangling inputs are attached
    # to these sources in preference to the interface nodes, so that the part of
    # the host feeding the removed subgraph stays alive instead of being pruned.
    boundary_sources = [
        src
        for nid in sorted(sub_host)
        for src in child.nodes[nid].inputs
        if src not in sub_host and host_shapes[src][0] == DATA
    ]
    fallback_sources = [child.patch_id, child.center_id]
    host_in_sources = boundary_sources + fallback_sources
    host_consumers = [
        (nid, k)
        for nid in sorted(child.nodes)
        if nid not in sub_host
        for k, src in enumerate(child.nodes[nid].inputs)
        if src in sub_host
    ]
    host_output_in_sub = child.output in sub_host

    # Copy the donor subgraph (with any constants it consumes).
    donor_consts = {
        src
        for nid in sub_donor
        for src in donor.nodes[nid].inputs
        if src not in sub_donor and donor.nodes[src].op == "const"
    }
    try:
        donor_order = [n for n in donor.topo_order() if n in sub_donor or n in donor_consts]
    except GraphError:
        return None

    mapping: Dict[int, int] = {}
    for nid in donor_order:
        node = donor.nodes[nid]
        mapping[nid] = child.add_node(
            node.op,
            inputs=[],
            const_dims=node.const_dims,
            init_range=node.init_range,
            init_values=node.init_values,
        )

    # Wire the donor copy: internal edges are remapped, boundary inputs are
    # attached to the host's own boundary sources (dimension-matched when
    # possible).
    for nid in donor_order:
        new_inputs: List[int] = []
        for src in donor.nodes[nid].inputs:
            if src in mapping:
                new_inputs.append(mapping[src])
                continue
            want = donor_shapes[src][1] if donor_shapes[src][0] == DATA else None
            pool = (
                [s for s in boundary_sources if host_shapes[s][1] == want]
                or [s for s in fallback_sources if host_shapes[s][1] == want]
                or host_in_sources
            )
            if not pool:
                return None
            new_inputs.append(rng.choice(pool))
        child.nodes[mapping[nid]].inputs = new_inputs

    donor_outs = [mapping[n] for n in _subgraph_outputs(donor, sub_donor)]

    # Reattach the host's consumers to the transplanted subgraph.
    for i, (nid, k) in enumerate(host_consumers):
        child.nodes[nid].inputs[k] = donor_outs[i % len(donor_outs)]
    if host_output_in_sub:
        child.output = donor_outs[0]

    child.remove_nodes(sub_host)
    return child


def crossover(
    parent_a: ComputationGraph,
    parent_b: ComputationGraph,
    generator: GraphGenerator,
    rng: random.Random,
    attempts: int = 6,
    numeric_check: bool = True,
) -> List[ComputationGraph]:
    """Swap a subgraph between two parents, producing up to two children.

    Returns the children that pass the validity checks (0, 1 or 2 graphs).  The
    search loop evaluates both and keeps the better-performing one.
    """
    for _ in range(attempts):
        sub_a = select_subgraph(parent_a, rng)
        sub_b = select_subgraph(parent_b, rng)
        if not sub_a or not sub_b:
            return []
        children = []
        for host, sh, donor, sd in (
            (parent_a, sub_a, parent_b, sub_b),
            (parent_b, sub_b, parent_a, sub_a),
        ):
            spliced = _splice(host, sh, donor, sd, rng)
            if spliced is None:
                continue
            ok = _finalise(spliced, generator, numeric_check=numeric_check)
            if ok is not None:
                children.append(ok)
        if children:
            return children
    return []


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #
def _mutate_add_node(g: ComputationGraph, generator: GraphGenerator, rng: random.Random) -> bool:
    """Add a node to increase the operator's complexity and functionality."""
    shapes = g.infer_shapes()
    data = [n for n in g.data_nodes(shapes)]
    if not data:
        return False
    src = rng.choice(data)
    d = shapes[src][1]

    if rng.random() < 0.5:
        new = g.add_node(rng.choice(UNARY), inputs=[src])
    elif rng.random() < 0.5:
        r = rng.choice(generator.ranks)
        c = g.add_node("const", const_dims=(d, r), init_range=generator._matmul_init(d))
        new = g.add_node("matmul", inputs=[src, c])
    else:
        op = rng.choice(BROADCASTING_BINARY)
        partner = [n for n in data if n != src and shapes[n][1] == d]
        if partner and rng.random() < 0.5:
            new = g.add_node(op, inputs=[src, rng.choice(partner)])
        else:
            c = g.add_node("const", const_dims=(d,), init_range=generator.cfg.init_range)
            new = g.add_node(op, inputs=[src, c])

    # Splice the new node onto an existing edge so that it is actually used.
    consumers = [
        (nid, k)
        for nid, node in g.nodes.items()
        if nid != new and node.op != "const"
        for k, s in enumerate(node.inputs)
        if s == src
    ]
    new_dim = g.infer_shapes()[new][1]
    consumers = [(nid, k) for nid, k in consumers if _slot_accepts(g, nid, k, new_dim)]
    if consumers and rng.random() < 0.7:
        nid, k = rng.choice(consumers)
        g.nodes[nid].inputs[k] = new
    if src == g.output or not consumers:
        g.output = new
    return True


def _slot_accepts(g: ComputationGraph, nid: int, slot: int, dim: int) -> bool:
    """Cheap pre-check that feeding width ``dim`` into ``(nid, slot)`` can type."""
    node = g.nodes[nid]
    if node.op == "matmul":
        if slot == 1:
            return False
        other = g.nodes[node.inputs[1]]
        return bool(other.const_dims) and other.const_dims[0] == dim
    return True


def _mutate_replace_node(g: ComputationGraph, rng: random.Random) -> bool:
    """Replace an operation node with another of compatible arity and shape."""
    ops = _operation_nodes(g)
    if not ops:
        return False
    nid = rng.choice(ops)
    cur = g.nodes[nid].op
    prim = PRIMITIVES[cur]
    if prim.arity == 1:
        choices = [o for o in UNARY if o != cur]
    elif cur == "matmul":
        return False  # nothing else consumes a 2-D constant in the same way
    else:
        choices = [o for o in BROADCASTING_BINARY if o != cur]
    if not choices:
        return False
    g.nodes[nid].op = rng.choice(choices)
    return True


def _mutate_alter_connections(g: ComputationGraph, rng: random.Random, max_edges: int = 3) -> bool:
    """Regenerate a random subset of the graph's edges."""
    try:
        order = g.topo_order()
        shapes = g.infer_shapes()
    except (GraphError, ShapeError):
        return False
    rank = {nid: i for i, nid in enumerate(order)}

    edges = [
        (nid, k)
        for nid, node in g.nodes.items()
        if node.op not in INTERFACE_OPS and node.op != "const"
        for k, src in enumerate(node.inputs)
        if shapes[src][0] == DATA
    ]
    if not edges:
        return False
    n_edges = rng.randint(1, min(max_edges, len(edges)))
    changed = False
    for nid, k in rng.sample(edges, n_edges):
        want = shapes[g.nodes[nid].inputs[k]][1]
        # Only earlier nodes are eligible, which keeps the graph acyclic.
        cands = [
            m
            for m in order
            if rank[m] < rank[nid] and shapes[m][0] == DATA and shapes[m][1] == want
            and m != g.nodes[nid].inputs[k]
        ]
        if not cands:
            continue
        g.nodes[nid].inputs[k] = rng.choice(cands)
        changed = True
    return changed


def apply_mutation(
    graph: ComputationGraph,
    kind: str,
    generator: GraphGenerator,
    rng: random.Random,
    attempts: int = 6,
    numeric_check: bool = True,
) -> Optional[ComputationGraph]:
    """Apply one structural mutation, returning ``None`` if it cannot be made valid.

    ``kind == "constant_optimization"`` is a no-op here: it is handled by the
    search loop, which runs additional constant-optimisation iterations on the
    selected graph.
    """
    if kind == "constant_optimization":
        return graph.clone()

    for _ in range(attempts):
        cand = graph.clone()
        try:
            if kind == "add_node":
                ok = _mutate_add_node(cand, generator, rng)
            elif kind == "replace_node":
                ok = _mutate_replace_node(cand, rng)
            elif kind == "alter_connections":
                ok = _mutate_alter_connections(cand, rng)
            else:
                raise ValueError(f"unknown mutation type '{kind}'")
        except (GraphError, ShapeError):
            ok = False
        if not ok:
            continue
        out = _finalise(cand, generator, numeric_check=numeric_check)
        if out is not None:
            return out
    return None


def mutate(
    graph: ComputationGraph,
    generator: GraphGenerator,
    rng: random.Random,
    allow_constant_opt: bool = True,
    numeric_check: bool = True,
) -> Optional[ComputationGraph]:
    """Sample one of the four mutation types uniformly and apply it."""
    types = MUTATION_TYPES if allow_constant_opt else MUTATION_TYPES[:-1]
    return apply_mutation(graph, rng.choice(types), generator, rng, numeric_check=numeric_check)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def tournament_select(
    population: Sequence,
    fitness: Sequence[float],
    k: int,
    rng: random.Random,
) -> int:
    """Tournament selection: index of the fittest among ``k`` random entrants."""
    k = max(1, min(k, len(population)))
    entrants = rng.sample(range(len(population)), k)
    return max(entrants, key=lambda i: fitness[i])
