"""Phase 1 — the IFC relationship graph.

Nodes are leaf IfcProduct instances; edges come from three evidence layers:

  E_A  aggregation/containment (spatial structure, system groups, assemblies)
  E_P  port connectivity (IfcRelConnectsPorts / IfcRelNests port ownership)
  E_G  geometric endpoint coincidence — the fallback layer the manuscript's
       port-only traversal lacks. Required in practice: the gas-pipe fixture
       carries no ports at all (legacy weakness W12).

Every edge is tagged with its layer so classification decisions are auditable.
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx

from ..ingest.loader import DISTRIBUTION_CLASSES, IfcContext
from ..geometry.swept import extruded_axis, revolved_arc


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
    for rec in ctx.products.values():
        if rec.ifc_class not in DISTRIBUTION_CLASSES:
            continue
        for pt in _distribution_endpoints(ctx, rec):
            cell = (int(math.floor(pt[0] / q)), int(math.floor(pt[1] / q)),
                    int(math.floor(pt[2] / q)))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other in endpoints.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                            if other != rec.guid:
                                _add_edge(g, rec.guid, other, "E_G", "endpoint-coincidence")
            endpoints[cell].append(rec.guid)
    return g


def _distribution_endpoints(ctx: IfcContext, rec) -> list[tuple[float, float, float]]:
    pts = []
    for ri in rec.body_items:
        try:
            if ri.item.is_a("IfcExtrudedAreaSolid"):
                ax = extruded_axis(ri.item, ri.matrix, ctx.length_scale)
                pts += [ax.start, ax.end]
            elif ri.item.is_a("IfcRevolvedAreaSolid"):
                ax = revolved_arc(ri.item, ri.matrix, ctx.length_scale, ctx.angle_scale)
                pts += [ax.start, ax.end]
        except Exception:
            continue
    return pts
