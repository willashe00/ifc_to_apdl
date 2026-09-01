"""Conformity enforcement on the analytical IR.

The APDL writer emits exactly one line per keypoint pair and meshes areas on
those lines, so two objects are conformal along an interface only when they
share the same keypoint pairs. A pool node lying in the INTERIOR of a member
axis or a surface edge therefore signals two distinct, overlapping lines and
a non-conformal (hanging-node) mesh.

``enforce_conformity`` makes every such node a vertex: members are split at
interior nodes, surface loops get the node inserted between the edge's
endpoints. It is the last geometric pass of the building assembler, after the
domain-specific reconciliation steps have created every interface node.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

from ..model.ir import AnalyticalModel, Member1D


def _grid_index(model: AnalyticalModel, cell: float):
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    n = len(model.nodes)
    for nid in range(1, n + 1):
        p = model.nodes.xyz(nid)
        grid[(int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)),
              int(math.floor(p[2] / cell)))].append(nid)
    return grid


def interior_nodes(model: AnalyticalModel, grid, cell: float, a: int, b: int,
                   tol: float) -> list[tuple[float, int]]:
    """Pool nodes strictly inside segment a-b (within ``tol`` laterally),
    as (station, node id) sorted along the segment."""
    pa, pb = model.nodes.xyz(a), model.nodes.xyz(b)
    d = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
    L2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    if L2 < 1e-18:
        return []
    L = math.sqrt(L2)
    lo = [int(math.floor((min(pa[k], pb[k]) - tol) / cell)) for k in range(3)]
    hi = [int(math.floor((max(pa[k], pb[k]) + tol) / cell)) for k in range(3)]
    found: list[tuple[float, int]] = []
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            for iz in range(lo[2], hi[2] + 1):
                for nid in grid.get((ix, iy, iz), ()):
                    if nid in (a, b):
                        continue
                    p = model.nodes.xyz(nid)
                    v = (p[0] - pa[0], p[1] - pa[1], p[2] - pa[2])
                    t = (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / L2
                    if not (tol / L < t < 1.0 - tol / L):
                        continue
                    perp = math.sqrt(sum((v[k] - t * d[k]) ** 2 for k in range(3)))
                    if perp <= tol:
                        found.append((t, nid))
    found.sort()
    return found


def enforce_conformity(model: AnalyticalModel, tol: float):
    """Split members / refine surface loops at interior pool nodes.

    Returns ``(members_split, edges_refined, edge_map)`` where ``edge_map``
    maps an original keypoint pair to the list of sub-edges it became (used
    to keep support edges valid).
    """
    cell = 1.0
    grid = _grid_index(model, cell)
    split_of: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def split_edge(a: int, b: int) -> list[tuple[int, int]]:
        key = (a, b)
        if key in split_of:
            return split_of[key]
        inner = interior_nodes(model, grid, cell, a, b, tol)
        chain = [a] + [nid for _, nid in inner] + [b]
        # dedupe (a node could sit at both ends after tolerance games)
        out = [(x, y) for x, y in zip(chain, chain[1:]) if x != y]
        split_of[key] = out
        split_of[(b, a)] = [(y, x) for x, y in reversed(out)]
        return out

    # -- members ---------------------------------------------------------------
    # Two members on the same keypoint pair (a column stub overlapping the
    # storey piece above once both are split at the shared node) would be
    # ONE line in the deck, meshed once: keep the first, drop the twin so the
    # IR mass accounting matches the deck.
    n_split = 0
    new_members: list[Member1D] = []
    seen_pairs: set[tuple[int, int]] = set()
    next_id = max((m.id for m in model.members), default=0)

    def keep(mm: Member1D) -> bool:
        k = (min(mm.start, mm.end), max(mm.start, mm.end))
        if k in seen_pairs:
            return False
        seen_pairs.add(k)
        return True

    for m in model.members:
        pieces = split_edge(m.start, m.end)
        if len(pieces) <= 1:
            if keep(m):
                new_members.append(m)
            continue
        n_split += 1
        ev = dict(m.prov.evidence)
        ev["conformity"] = f"split into {len(pieces)} at coincident node(s)"
        for i, (a, b) in enumerate(pieces):
            mm = copy.copy(m)
            mm.prov = copy.copy(m.prov)
            mm.prov.evidence = dict(ev)
            mm.start, mm.end = a, b
            if i > 0:
                next_id += 1
                mm.id = next_id
            if keep(mm):
                new_members.append(mm)
    model.members = new_members

    # -- surfaces --------------------------------------------------------------
    n_ins = 0
    for s in model.surfaces:
        loop = s.loop
        present = set(loop)
        new_loop: list[int] = []
        changed = False
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if a == b:
                continue
            pieces = split_edge(a, b)
            new_loop.append(a)
            if len(pieces) > 1:
                # a loop vertex lying on another edge of the SAME loop marks
                # a degenerate (spiked) panel; inserting it again would fold
                # the loop back on itself - skip those, keep foreign nodes
                inner = [y for _, y in pieces[:-1] if y not in present]
                if inner:
                    changed = True
                    n_ins += 1
                    new_loop.extend(inner)
        if changed:
            s.loop = new_loop
            s.prov.evidence["conformity"] = "loop refined at coincident node(s)"

    def edge_map(key: tuple[int, int]) -> list[tuple[int, int]]:
        return split_edge(*key)

    return n_split, n_ins, edge_map
