"""IR-level connectivity screening (runs before any solver).

Builds the undirected graph of analytical objects sharing pool nodes and
reports connected components. A component that contains no supported node
is a guaranteed free body in the assembled FEA model — caught here without
spending a solve. Complements (not replaces) the modal mechanism screen:
shared keypoints guarantee shared mesh nodes only when meshes are conformal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..model.ir import AnalyticalModel


@dataclass
class Component:
    object_count: int
    classes: dict[str, int]
    supported: bool
    sample_names: list[str] = field(default_factory=list)


def connectivity_report(model: AnalyticalModel) -> list[Component]:
    """Connected components of the member/surface/link graph via shared nodes."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    objects = []   # (node_ids, ifc_class, name)
    for m in model.members:
        objects.append(([m.start, m.end], m.prov.ifc_class, m.prov.name))
    for s in model.surfaces:
        objects.append((list(s.loop), s.prov.ifc_class, s.prov.name))
    for l in model.links:
        objects.append(([l.n1, l.n2], "SpringLink", ""))

    for nodes, _, _ in objects:
        for a, b in zip(nodes, nodes[1:]):
            union(a, b)

    supported_nodes: set[int] = set()
    for sup in model.supports:
        supported_nodes.update(sup.nodes)
        for k1, k2 in sup.edges:
            supported_nodes.update((k1, k2))

    groups: dict[int, list[tuple[list[int], str, str]]] = defaultdict(list)
    for obj in objects:
        groups[find(obj[0][0])].append(obj)

    supported_roots = {find(n) for n in supported_nodes if n in parent}
    out = []
    for root, objs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        classes: dict[str, int] = defaultdict(int)
        names = []
        for _, cls, name in objs:
            classes[cls] += 1
            if name and len(names) < 5:
                names.append(name)
        out.append(Component(len(objs), dict(classes), root in supported_roots, names))
    return out


@dataclass
class HangingNode:
    node: int
    edge: tuple[int, int]
    owner: str                       # object class owning the edge
    t: float                         # station along the edge (0..1)


def conformity_report(model: AnalyticalModel, tol: float | None = None) -> list[HangingNode]:
    """IR-level mesh-conformity screen.

    The APDL writer creates exactly one line per keypoint pair, so two
    objects share mesh nodes along an interface only when they share the
    same keypoint pair. A pool node lying in the INTERIOR of another object's
    edge (a surface boundary edge or a member axis) therefore means the two
    boundaries are distinct, overlapping lines: the meshes will not be
    conformal there (hanging node) unless divisions happen to coincide.
    Returns every such (node, edge) incidence.
    """
    import math

    tol = tol if tol is not None else max(model.nodes.tol * 10.0, 1e-3)
    n = len(model.nodes)
    if n == 0:
        return []
    pts = [model.nodes.xyz(i) for i in range(1, n + 1)]

    edges: dict[tuple[int, int], str] = {}
    for m in model.members:
        edges.setdefault((min(m.start, m.end), max(m.start, m.end)), m.prov.ifc_class)
    for s in model.surfaces:
        loop = s.loop
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if a != b:
                edges.setdefault((min(a, b), max(a, b)), s.prov.ifc_class)

    # coarse spatial bucketing of nodes
    cell = 1.0
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, p in enumerate(pts, 1):
        grid[(int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)),
              int(math.floor(p[2] / cell)))].append(i)

    out: list[HangingNode] = []
    for (a, b), owner in edges.items():
        pa, pb = pts[a - 1], pts[b - 1]
        d = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        L2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        if L2 < 1e-18:
            continue
        L = math.sqrt(L2)
        lo = [int(math.floor((min(pa[k], pb[k]) - tol) / cell)) for k in range(3)]
        hi = [int(math.floor((max(pa[k], pb[k]) + tol) / cell)) for k in range(3)]
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for iz in range(lo[2], hi[2] + 1):
                    for nid in grid.get((ix, iy, iz), ()):
                        if nid in (a, b):
                            continue
                        p = pts[nid - 1]
                        v = (p[0] - pa[0], p[1] - pa[1], p[2] - pa[2])
                        t = (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / L2
                        if not (tol / L < t < 1.0 - tol / L):
                            continue
                        perp = math.sqrt(sum((v[k] - t * d[k]) ** 2 for k in range(3)))
                        if perp <= tol:
                            out.append(HangingNode(nid, (a, b), owner, t))
    return out


@dataclass
class FreeJoint:
    node: int
    xyz: tuple[float, float, float]
    members: list[str]               # names of the (inclined) members meeting there


def free_joint_report(model: AnalyticalModel) -> list[FreeJoint]:
    """Brace joints that connect to nothing but braces.

    A brace end (IfcMember / inclined BEAM188) whose node is shared with no
    surface, no horizontal or vertical framing member, no link and no support
    is a floating workpoint: the brace hangs in space (typically a chevron
    apex or a diagonal end that missed its beam / column reconciliation).
    The connectivity screen cannot see this - the braces are graph-connected
    through each other - but the model is a mechanism there.
    """
    incident: dict[int, list] = defaultdict(list)
    for m in model.members:
        p1, p2 = model.nodes.xyz(m.start), model.nodes.xyz(m.end)
        inclined = (abs(p2[2] - p1[2]) > 1e-3
                    and (abs(p2[0] - p1[0]) > 1e-3 or abs(p2[1] - p1[1]) > 1e-3))
        kind = "brace" if (inclined or m.prov.ifc_class == "IfcMember") else "frame"
        incident[m.start].append((kind, m.prov.name))
        incident[m.end].append((kind, m.prov.name))
    anchored: set[int] = set()
    for s in model.surfaces:
        anchored.update(s.loop)
    for l in model.links:
        anchored.update((l.n1, l.n2))
    for sup in model.supports:
        anchored.update(sup.nodes)
        for a, b in sup.edges:
            anchored.update((a, b))
    out = []
    for nid, objs in incident.items():
        if nid in anchored:
            continue
        # a crossing of noded diagonals (X-bracing: 3-4 pieces meeting) is a
        # legitimate brace-only joint; a single end or a two-brace apex is not
        if all(kind == "brace" for kind, _ in objs) and len(objs) <= 2:
            out.append(FreeJoint(nid, model.nodes.xyz(nid), [n for _, n in objs]))
    return out
