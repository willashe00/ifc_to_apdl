"""Phase 1 — the IFC relationship graph.

Nodes are leaf IfcProduct instances; edges come from three evidence layers:

  E_A  aggregation/containment (spatial structure, system groups, assemblies)
  E_P  port connectivity (IfcRelConnectsPorts / IfcRelNests port ownership)
  E_G  geometric endpoint coincidence — the fallback layer the manuscript's
       port-only traversal lacks. Required in practice: the gas-pipe fixture
       carries no ports at all (legacy weakness W12). Also covers an endpoint
       landing on the interior of another straight run (branch welded into a
       header), which the piping assembler nodes as a junction.

Every edge is tagged with its layer so classification decisions are auditable.

Only E_P and E_G are physical continuity (``CONTINUITY_LAYERS``). E_A is
grouping evidence: one IfcDistributionSystem routinely spans several
physically separate runs (and the equipment between them), so it must never
join runs into one analysis model.
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
import numpy as np

from ..ingest.loader import DISTRIBUTION_CLASSES, IfcContext
from ..geometry.swept import extruded_axis, revolved_arc

#: evidence layers that establish physical (mechanical) continuity
CONTINUITY_LAYERS = frozenset({"E_P", "E_G"})


def _add_edge(g: nx.Graph, a: str, b: str, layer: str, via: str) -> None:
    """Record an evidence edge without discarding evidence already present.

    A pair of products is typically connected on several layers at once
    (ports AND coincident endpoints AND one system); ``Graph.add_edge`` on an
    existing edge would silently overwrite the earlier layer's attributes, so
    the layers are accumulated in ``layers`` and ``layer`` keeps the first."""
    if g.has_edge(a, b):
        data = g.edges[a, b]
        data.setdefault("layers", {data["layer"]}).add(layer)
        data.setdefault("vias", {data["via"]}).add(via)
    else:
        g.add_edge(a, b, layer=layer, via=via, layers={layer}, vias={via})


def edge_layers(g: nx.Graph, nodes) -> set[str]:
    """Union of evidence layers over the edges among ``nodes``."""
    out: set[str] = set()
    for _, _, d in g.edges(nodes, data=True):
        out |= d.get("layers", {d["layer"]})
    return out


def continuity_components(g: nx.Graph, nodes) -> list[set[str]]:
    """Connected components of ``nodes`` over physical continuity edges only
    (``CONTINUITY_LAYERS``); grouping-only E_A edges are ignored."""
    h = nx.Graph()
    h.add_nodes_from(nodes)
    h.add_edges_from((a, b) for a, b, d in g.subgraph(nodes).edges(data=True)
                     if d.get("layers", {d["layer"]}) & CONTINUITY_LAYERS)
    return list(nx.connected_components(h))


def build_graph(ctx: IfcContext, proximity_tol: float = 1e-3) -> nx.Graph:
    g = nx.Graph()
    for rec in ctx.products.values():
        g.add_node(rec.guid, ifc_class=rec.ifc_class, name=rec.name)

    # E_A: shared distribution-system membership
    by_system: dict[str, list[str]] = defaultdict(list)
    for rec in ctx.products.values():
        for sg in rec.system_guids:
            by_system[sg].append(rec.guid)
    for sg, members in by_system.items():
        for a, b in zip(members, members[1:]):
            _add_edge(g, a, b, "E_A", f"system:{sg[:8]}")

    # E_P: port connectivity
    port_owner: dict[int, str] = {}
    for rec in ctx.products.values():
        for rel in getattr(rec.entity, "IsNestedBy", None) or []:
            for obj in rel.RelatedObjects:
                if obj.is_a("IfcDistributionPort"):
                    port_owner[obj.id()] = rec.guid
        for rel in getattr(rec.entity, "HasPorts", None) or []:   # IFC2x3 path
            port_owner[rel.RelatingPort.id()] = rec.guid
    for rel in ctx.model.by_type("IfcRelConnectsPorts"):
        a = port_owner.get(rel.RelatingPort.id())
        b = port_owner.get(rel.RelatedPort.id())
        if a and b and a != b:
            _add_edge(g, a, b, "E_P", "ports")

    # E_G: endpoint coincidence between distribution elements
    endpoints: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    q = max(proximity_tol, 1e-9)
    centerlines = {rec.guid: _distribution_centerlines(ctx, rec)
                   for rec in ctx.products.values() if rec.ifc_class in DISTRIBUTION_CLASSES}
    for guid, lines in centerlines.items():
        for pt in (p for start, end, _ in lines for p in (start, end)):
            cell = (int(math.floor(pt[0] / q)), int(math.floor(pt[1] / q)),
                    int(math.floor(pt[2] / q)))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other in endpoints.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                            if other != guid:
                                _add_edge(g, guid, other, "E_G", "endpoint-coincidence")
            endpoints[cell].append(guid)

    # E_G: endpoint on the interior of another straight run (branch into a
    #      header without a tee fitting) — mirrors assemble.piping._node_junctions
    owners, pts = [], []
    for guid, lines in centerlines.items():
        for start, end, _ in lines:
            owners += [guid, guid]
            pts += [start, end]
    if pts:
        pts_arr = np.asarray(pts, dtype=float)
        for host, lines in centerlines.items():
            for start, end, straight in lines:
                if not straight:
                    continue
                p1, ab = np.asarray(start, dtype=float), np.subtract(end, start)
                denom = float(ab.dot(ab))
                if denom < 1e-18:
                    continue
                t = (pts_arr - p1) @ ab / denom
                off = np.linalg.norm(pts_arr - (p1 + np.outer(t, ab)), axis=1)
                hit = (t > 0.0) & (t < 1.0) & (off <= proximity_tol)
                for idx in np.flatnonzero(hit):
                    if owners[idx] != host:
                        _add_edge(g, owners[idx], host, "E_G", "endpoint-on-run")
    return g


def _distribution_centerlines(ctx: IfcContext, rec) -> list[tuple[tuple, tuple, bool]]:
    """(start, end, straight) centerline of each interpretable body item."""
    lines = []
    for ri in rec.body_items:
        try:
            if ri.item.is_a("IfcExtrudedAreaSolid"):
                ax = extruded_axis(ri.item, ri.matrix, ctx.length_scale)
                lines.append((ax.start, ax.end, True))
            elif ri.item.is_a("IfcRevolvedAreaSolid"):
                ax = revolved_arc(ri.item, ri.matrix, ctx.length_scale, ctx.angle_scale)
                lines.append((ax.start, ax.end, False))
        except Exception:
            continue
    return lines
