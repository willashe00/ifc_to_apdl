"""Mesh-level compatibility screen on a live MAPDL session.

Runs after the deck has been read (before any solve) and inspects the actual
finite-element mesh:

  - connected components of the element/node graph (a globally compatible
    structure is exactly one);
  - hanging nodes: nodes lying on the interior of another element's edge
    without belonging to that element (non-conformal interface);
  - coincident unmerged nodes (distinct ids within the merge tolerance).

Only corner / end nodes count: BEAM188 and PIPE288 orientation nodes,
SOLID187 mid-side nodes and ELBOW290 mid-nodes are not edge endpoints.
Separate supported structures (three equipment racks, three piping loops)
are legitimate: the caller compares the component count with the IR.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshReport:
    nodes: int = 0
    elements: int = 0
    components: int = 0
    component_sizes: list[int] = field(default_factory=list)
    hanging_nodes: int = 0
    unmerged_pairs: int = 0
    unmeshed_areas: int = 0
    samples: list[str] = field(default_factory=list)


def _structural_nodes(nodes: list[int], etype_name: str) -> list[int]:
    if etype_name in ("BEAM188", "PIPE288", "COMBIN14"):
        return nodes[:2]
    if etype_name in ("SHELL181", "SOLID187"):
        return nodes[:4]                     # SOLID187: corner nodes only
    if etype_name == "ELBOW290":
        return nodes[:2]                     # I, J (K is the mid-node)
    return nodes


def mesh_report(mapdl, deck_text: str, merge_tol: float = 1e-3) -> MeshReport:
    from scipy.spatial import cKDTree

    rep = MeshReport()
    mapdl.prep7()
    mapdl.allsel()
    nnum = np.asarray(mapdl.mesh.nnum)
    xyz_all = np.asarray(mapdl.mesh.nodes)
    if len(xyz_all) != len(nnum):                # internal nodes appended (PIPE288 etc.)
        xyz = {i + 1: xyz_all[i] for i in range(len(xyz_all))}
    else:
        xyz = {int(n): xyz_all[i] for i, n in enumerate(nnum)}
    et_names: dict[int, str] = {}
    for line in deck_text.splitlines():
        if line.startswith("ET,"):
            parts = line.split(",")
            et_names[int(parts[1])] = parts[2].strip()
    enodes: dict[int, list[int]] = {}
    for e in mapdl.mesh.elem:
        e = np.asarray(e)
        name = et_names.get(int(e[1]), "")
        enodes[int(e[8])] = _structural_nodes([int(n) for n in e[10:] if int(n) > 0], name)
    rep.nodes, rep.elements = len(xyz), len(enodes)
    if not enodes:
        return rep

    # components
    parent: dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for nn in enodes.values():
        for a, b in zip(nn, nn[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    sizes: dict[int, int] = defaultdict(int)
    for nn in enodes.values():
        sizes[find(nn[0])] += 1
    rep.component_sizes = sorted(sizes.values(), reverse=True)
    rep.components = len(sizes)

    # coincident unmerged nodes
    ids = sorted(n for n in xyz if n in parent)     # nodes used by structural elements
    P = np.array([xyz[i] for i in ids])
    tree = cKDTree(P)
    pairs = tree.query_pairs(merge_tol)
    rep.unmerged_pairs = len(pairs)
    for a, b in list(pairs)[:3]:
        rep.samples.append(f"unmerged nodes {ids[a]}/{ids[b]} at {P[a].round(4).tolist()}")

    # hanging nodes: node within tol of an edge interior, not an edge end
    tol = merge_tol
    edges: set[tuple[int, int]] = set()
    for nn in enodes.values():
        if len(nn) == 2:
            edges.add((min(nn), max(nn)))
        else:
            for a, b in zip(nn, nn[1:] + nn[:1]):
                if a != b:
                    edges.add((min(a, b), max(a, b)))
    hanging: set[int] = set()
    for a, b in edges:
        pa, pb = xyz[a], xyz[b]
        d = pb - pa
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        for ci in tree.query_ball_point((pa + pb) / 2, L / 2 + tol):
            n = ids[ci]
            if n in (a, b):
                continue
            v = xyz[n] - pa
            t = float(v @ d) / (L * L)
            if tol / L < t < 1 - tol / L and float(np.linalg.norm(v - t * d)) <= tol:
                if n not in hanging and len(rep.samples) < 6:
                    rep.samples.append(f"node {n} at {xyz[n].round(3).tolist()} hangs on "
                                       f"edge {a}-{b}")
                hanging.add(n)
    rep.hanging_nodes = len(hanging)

    # areas / volumes that produced no elements (sliver or degenerate
    # panels): lost mass. With volumes present the elements belong to the
    # volumes (VMESH), so check those instead of their bounding areas.
    try:
        n_vol = int(mapdl.get_value("VOLU", 0, "NUM", "MAX"))
        empty = []
        if n_vol > 0:
            for v in range(1, n_vol + 1):
                mapdl.vsel("S", "VOLU", "", v)
                mapdl.eslv("S")
                if int(mapdl.get_value("ELEM", 0, "COUNT")) == 0:
                    empty.append(v)
            kind = "volumes"
        else:
            n_area = int(mapdl.get_value("AREA", 0, "NUM", "MAX"))
            for a in range(1, n_area + 1):
                mapdl.asel("S", "AREA", "", a)
                mapdl.esla("S")
                if int(mapdl.get_value("ELEM", 0, "COUNT")) == 0:
                    empty.append(a)
            kind = "areas"
        mapdl.allsel()
        rep.unmeshed_areas = len(empty)
        if empty:
            rep.samples.append(f"unmeshed {kind} {empty[:6]}")
    except Exception as exc:                          # pragma: no cover
        rep.samples.append(f"area check unavailable: {exc}")
        mapdl.allsel()
    return rep
