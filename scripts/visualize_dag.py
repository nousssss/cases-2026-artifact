#!/usr/bin/env python3
"""Render a discovered operator as a Graphviz DAG (paper, Figs. 9 and 10).

Rectangular nodes are primitive operations, circles are learned constants, and
dashed nodes are the interface pseudo-ops (patch extraction, centre selection
and the output reshape), which Sec. III-E excludes from the primitive count.

Example::

    python scripts/visualize_dag.py operators/1_1_conv1_struct.graph.json --out fig9.dot
    dot -Tpng fig9.dot -o fig9.png     # requires graphviz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conas.graph import ComputationGraph  
from conas.primitives import INTERFACE_OPS  

STYLE = {
    "op": 'shape=box, style="rounded,filled", fillcolor="#d9e8f7", color="#3b6ea5"',
    "const": 'shape=circle, style=filled, fillcolor="#f7d3d3", color="#b05252"',
    "interface": 'shape=box, style="rounded,dashed", color="#7a7a7a"',
    "output": 'shape=ellipse, style=filled, fillcolor="#ffffff", color="#000000"',
}


def to_dot(graph: ComputationGraph, name: str = "operator") -> str:
    lines = [f'digraph "{name}" {{', "  rankdir=TB;", '  node [fontname="Helvetica", fontsize=11];']
    lines.append(f'  X [label="X", {STYLE["output"]}];')

    for nid in graph.topo_order():
        node = graph.nodes[nid]
        if node.op == "patch":
            lines.append(f'  n{nid} [label="Patch", {STYLE["interface"]}];')
            lines.append(f"  X -> n{nid};")
        elif node.op == "center":
            lines.append(f'  n{nid} [label="Center", {STYLE["interface"]}];')
            lines.append(f"  X -> n{nid};")
        elif node.op == "const":
            dims = "x".join(str(d) for d in (node.const_dims or ()))
            lines.append(f'  n{nid} [label="&Theta;{nid}\\n({dims})", {STYLE["const"]}];')
        else:
            lines.append(f'  n{nid} [label="{node.op}", {STYLE["op"]}];')
        for src in node.inputs:
            lines.append(f"  n{src} -> n{nid};")

    lines.append(f'  reshape [label="reshape", {STYLE["interface"]}];')
    lines.append(f'  Output [label="Output", {STYLE["output"]}];')
    lines.append(f"  n{graph.output} -> reshape;")
    lines.append("  reshape -> Output;")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", help="path to a *.graph.json file")
    p.add_argument("--out", default=None, help="write a .dot file instead of stdout")
    p.add_argument("--breakdown", action="store_true", help="also print the node-by-node listing")
    args = p.parse_args()

    graph = ComputationGraph.load(args.graph)
    dot = to_dot(graph, os.path.basename(args.graph))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(dot)
        print(f"wrote {args.out}  (render with: dot -Tpng {args.out} -o out.png)")
    else:
        print(dot)

    if args.breakdown:
        print()
        print(graph.pretty())


if __name__ == "__main__":
    main()
