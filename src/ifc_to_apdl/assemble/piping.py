"""Phase 4 — piping network assembly.

Network continuity comes from the shared NodePool: connected component
endpoints resolve to the same pool id when coincident within the piping
merge tolerance (manuscript union-find merge). Straight members become
PIPE288 lines, elbows become ELBOW290 arcs with the bend center/radius
recovered from the revolved solid. Leaf nodes (degree 1) become anchors
under the 'anchor-leaves' BC policy — logged as a heuristic.
"""

from __future__ import annotations

import math

from collections import Counter

import numpy as np

from ..assign.elements import resolve_element_type
from ..assign.materials import MaterialResolver
from ..classify.systems import SystemRecord
from ..config import ConversionConfig
from ..geometry.swept import extruded_axis, revolved_arc
from ..ingest.loader import IfcContext, is_hanger
from ..model.ir import (
    AnalyticalModel, Arc, Member1D, NodePool, Provenance, SpringLink, Support,
    _section_area,
)
from ..model.sections import PipeSection
from ..report.audit import AuditStatus

#: piping endpoint merge tolerance floor [m] (manuscript epsilon_p)
EPS_P_FLOOR = 1e-3


def assemble_piping(ctx: IfcContext, system: SystemRecord,
                    config: ConversionConfig,
                    resolver: MaterialResolver) -> AnalyticalModel:
    merge_tol = max(config.resolve_merge_tol(ctx.precision), EPS_P_FLOOR)
    model = AnalyticalModel(name=system.name, domain="piping",
                            nodes=NodePool(merge_tol))
    model.meta.update(source=ctx.path, schema=ctx.schema, merge_tol=merge_tol)

    member_id = 0
    hangers = []
    for guid in system.product_guids:
        rec = ctx.products[guid]
        if is_hanger(rec):
            hangers.append(rec)
            continue
        try:
            etype, ev = resolve_element_type(ctx, rec, "piping")
        except ValueError as exc:
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail=str(exc), system=system.name)
            continue

        # every interpretable body item becomes a member: straight segments and
        # elbows carry one solid; branch fittings (tees) carry one leg per solid
        axes = []
        for ri in rec.body_items:
            if ri.item.is_a("IfcExtrudedAreaSolid"):
                axes.append(extruded_axis(ri.item, ri.matrix, ctx.length_scale))
            elif ri.item.is_a("IfcRevolvedAreaSolid"):
                axes.append(revolved_arc(ri.item, ri.matrix, ctx.length_scale, ctx.angle_scale))
        if not axes:
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail="no interpretable body representation", system=system.name)
            continue

        if any(not isinstance(ax.profile.section, PipeSection) for ax in axes):
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.FLAGGED,
                             detail=f"non-circular profile '{axes[0].profile.kind}' on a piping "
                                    "member; converted with flag", system=system.name)

        mat = resolver.resolve(rec, "piping", "pipe")
        mat_id = model.add_material(mat)

        converted = 0
        sec_id = None
        for axis in axes:
            sec_id = model.add_section(axis.profile.section)
            n1 = model.nodes.get(axis.start)
            n2 = model.nodes.get(axis.end)
            if n1 == n2:
                continue
            member_id += 1
            prov = Provenance(guid=guid, ifc_class=rec.ifc_class, name=rec.name,
                              evidence={"element": ev,
                                        "centerline": "revolved-arc" if axis.arc_center else "extrusion-axis",
                                        "material": f"{mat.source}"
                                                    + (f" (s={mat.confidence:.2f})" if mat.confidence else "")})
            arc = Arc(center=axis.arc_center, radius=axis.arc_radius) if axis.arc_center else None
            m_etype = etype
            if arc is not None:
                # bend angle from chord and radius: a revolved body subtending
                # under 5 deg is a port-alignment kink -> straight pipe chord
                chord = math.dist(axis.start, axis.end)
                half = min(1.0, chord / (2.0 * axis.arc_radius)) if axis.arc_radius > 0 else 1.0
                bend = 2.0 * math.asin(half)
                if bend < math.radians(5) or m_etype != "ELBOW290":
                    if bend < math.radians(5):
                        prov.evidence["centerline"] = (f"revolved arc subtending "
                                                       f"{math.degrees(bend):.1f} deg -> straight chord")
                    arc = None
                    m_etype = "PIPE288"
            model.members.append(Member1D(member_id, prov, n1, n2, m_etype, sec_id, mat_id, arc=arc))
            converted += 1
        if converted == 0:
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail="degenerate member (coincident endpoints)", system=system.name)
            continue
        ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.CONVERTED,
                         detail=f"{etype}, section #{sec_id}, material #{mat_id}"
                                + (f"; {converted} body legs" if converted > 1 else ""),
                         system=system.name)

    # -- junction noding: an endpoint landing on another member's interior
    #    (tee branch on its run leg, branch line into a header) splits the
    #    host so the junction shares one analytical node ----------------------
    member_id = _node_junctions(model, member_id)

    # -- hangers: grounded axial springs tying the run to the structure -------
    if hangers:
        _attach_hangers(ctx, system, model, hangers, resolver, config, member_id)

    # -- boundary conditions: anchor network leaves ---------------------------
    #    (hanger ground nodes never appear in the member degree count, so
    #    grounding springs cannot masquerade as network anchors)
    if config.bcs.piping == "anchor-leaves":
        degree = Counter()
        for m in model.members:
            degree[m.start] += 1
            degree[m.end] += 1
        leaves = sorted(n for n, d in degree.items() if d == 1)
        if leaves:
            model.supports.append(Support(name="PIPE_ANCHORS", nodes=leaves,
                                          source="heuristic:anchor-leaves"))
            ctx.audit.event("boundary-condition",
                            f"{system.name}: fixed anchors applied at {len(leaves)} network "
                            "leaf node(s) [heuristic anchor-leaves]", severity="warning")
    return model


def _node_junctions(model: AnalyticalModel, member_id: int) -> int:
    """Split straight members wherever another member's endpoint lies on
    their interior (within the pool tolerance), so junctions share nodes."""
    import math as _math

    endpoints: dict[int, np.ndarray] = {}
    for m in model.members:
        for nid in (m.start, m.end):
            endpoints.setdefault(nid, np.asarray(model.nodes.xyz(nid), dtype=float))
    tol = model.nodes.tol
    for m in list(model.members):
        if m.arc is not None:
            continue
        p1 = np.asarray(model.nodes.xyz(m.start), dtype=float)
        p2 = np.asarray(model.nodes.xyz(m.end), dtype=float)
        ab = p2 - p1
        denom = float(ab.dot(ab))
        if denom < 1e-18:
            continue
        seg_len = _math.sqrt(denom)
        stations = []
        for nid, q in endpoints.items():
            if nid in (m.start, m.end):
                continue
            t = float((q - p1).dot(ab)) / denom
            if t <= 0.0 or t >= 1.0:
                continue
            if t * seg_len <= tol or (1.0 - t) * seg_len <= tol:
                continue
            if float(np.linalg.norm(q - (p1 + t * ab))) <= tol:
                stations.append((t, nid))
        if not stations:
            continue
        stations.sort()
        orig_end = m.end
        prev = m.start
        for idx, (t, nid) in enumerate(stations):
            if idx == 0:
                m.end = nid
            else:
                member_id += 1
                model.members.append(Member1D(member_id, m.prov, prev, nid, m.etype,
                                              m.section, m.material, orient=m.orient))
            prev = nid
        member_id += 1
        model.members.append(Member1D(member_id, m.prov, prev, orig_end, m.etype,
                                      m.section, m.material, orient=m.orient))
    return member_id


def _attach_hangers(ctx: IfcContext, system: SystemRecord, model: AnalyticalModel,
                    hangers: list, resolver: MaterialResolver,
                    config: ConversionConfig, member_id: int) -> None:
    """Idealize hanger rods as grounded axial springs (COMBIN14, k = EA/L).

    The rod's pipe-side end is projected onto the host pipe centerline (the
    host member is split there so a mesh node exists); the structure-side end
    becomes a grounded node — standard pipe-stress practice, where the
    supporting structure is taken as rigid relative to the hanger rod."""
    snap = config.resolve_snap_tol(model.nodes.tol)
    ground: list[int] = []
    link_id = 0
    for rec in hangers:
        axis = None
        for ri in rec.body_items:
            if ri.item.is_a("IfcExtrudedAreaSolid"):
                axis = extruded_axis(ri.item, ri.matrix, ctx.length_scale)
                break
        if axis is None:
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail="hanger without prismatic body", system=system.name)
            continue
        A = np.asarray(axis.start, dtype=float)
        B = np.asarray(axis.end, dtype=float)
        rod_len = float(np.linalg.norm(B - A))
        if rod_len < 1e-6:
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail="degenerate hanger rod", system=system.name)
            continue

        # nearest straight host member, judged from either rod end
        best = None            # (dist, member, t, ground_end, attach_point, r_o)
        for m in model.members:
            if m.arc is not None:
                continue
            p1 = np.asarray(model.nodes.xyz(m.start), dtype=float)
            p2 = np.asarray(model.nodes.xyz(m.end), dtype=float)
            ab = p2 - p1
            denom = float(ab.dot(ab))
            if denom < 1e-18:
                continue
            sec = model.sections[m.section]
            r_o = sec.od / 2.0 if isinstance(sec, PipeSection) else 0.05
            for pipe_end, gnd_end in ((A, B), (B, A)):
                t = max(0.0, min(1.0, float((pipe_end - p1).dot(ab)) / denom))
                q = p1 + t * ab
                d = float(np.linalg.norm(pipe_end - q))
                if best is None or d < best[0]:
                    best = (d, m, t, gnd_end, q, r_o)
        if best is None or best[0] > best[5] + max(snap, 0.05):
            gap = f"{best[0]:.3f} m" if best else "n/a"
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.FLAGGED,
                             detail=f"hanger rod end not on any pipe wall (stand-off {gap}); "
                                    "not converted", system=system.name)
            continue
        _, host, t, gnd_end, q, r_o = best

        # clamp collars (sleeves whose axis runs along the pipe) have BOTH ends
        # at the pipe wall; a rod interpretation would ground the run axially,
        # so the tie function is left to the companion rod spring instead
        hp1 = np.asarray(model.nodes.xyz(host.start), dtype=float)
        hp2 = np.asarray(model.nodes.xyz(host.end), dtype=float)
        hab = hp2 - hp1
        hdenom = float(hab.dot(hab))
        tg = max(0.0, min(1.0, float((np.asarray(gnd_end) - hp1).dot(hab)) / hdenom)) \
            if hdenom > 1e-18 else 0.0
        d_gnd = float(np.linalg.norm(np.asarray(gnd_end) - (hp1 + tg * hab)))
        if d_gnd <= r_o + max(snap, 0.05):
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.CONVERTED,
                             detail="hanger clamp collar at the pipe wall; tie function "
                                    "carried by the companion rod spring — no discrete "
                                    "element emitted", system=system.name)
            continue

        p1 = np.asarray(model.nodes.xyz(host.start), dtype=float)
        p2 = np.asarray(model.nodes.xyz(host.end), dtype=float)
        seg_len = float(np.linalg.norm(p2 - p1))
        if t * seg_len <= model.nodes.tol:
            n_att = host.start
        elif (1.0 - t) * seg_len <= model.nodes.tol:
            n_att = host.end
        else:
            n_att = model.nodes.get(tuple(q))
            if n_att not in (host.start, host.end):
                member_id += 1
                model.members.append(Member1D(member_id, host.prov, n_att, host.end,
                                              host.etype, host.section, host.material,
                                              orient=host.orient))
                host.end = n_att

        area = _section_area(axis.profile.section) if axis.profile.section else 0.0
        if area <= 0.0:
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.FLAGGED,
                             detail="hanger rod cross-section unavailable; spring stiffness "
                                    "cannot be derived — not converted", system=system.name)
            continue
        mat = resolver.resolve(rec, "piping", "hanger")
        k = mat.elastic_modulus * area / rod_len

        n_gnd = model.nodes.get(tuple(gnd_end))
        link_id += 1
        prov = Provenance(guid=rec.guid, ifc_class=rec.ifc_class, name=rec.name,
                          evidence={"idealization": "grounded axial spring (rod EA/L)",
                                    "attachment": f"host split at t={t:.3f}",
                                    "material": f"{mat.source}"
                                                + (f" (s={mat.confidence:.2f})"
                                                   if mat.confidence else "")})
        model.links.append(SpringLink(link_id, prov, n1=n_att, n2=n_gnd, k=k, dof="AXIAL"))
        ground.append(n_gnd)
        ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.CONVERTED,
                         detail=f"COMBIN14 grounded axial spring, k={k:.4g} N/m "
                                "(rod EA/L; supporting structure idealized rigid)",
                         system=system.name)
    if ground:
        model.supports.append(Support(name="HANGER_GROUND", nodes=sorted(set(ground)),
                                      source="heuristic:hanger-ground"))
        ctx.audit.event("boundary-condition",
                        f"{system.name}: {len(ground)} hanger structure-side node(s) "
                        "grounded [heuristic hanger-ground; structure-side flexibility "
                        "neglected in the decoupled per-system deck]", severity="warning")
