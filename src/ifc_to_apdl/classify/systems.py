"""Phase 1 — domain classification and functional isolation.

Domains are assigned by evidence scoring rather than a single brittle rule:
the containment fixture, for instance, authors its basemat as IfcWall, which
would defeat the strict slab+wall+roof class signature of the manuscript.

Outputs one SystemRecord per functionally isolated system; every product
lands in a system or is excluded with a cited rule (equipment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ingest.loader import (
    ACCESSORY_CLASSES, DISTRIBUTION_CLASSES, EQUIPMENT_CLASSES, IfcContext,
    STRUCTURAL_CLASSES, is_hanger,
)
from ..report.audit import AuditStatus
from .graph import build_graph, continuity_components, edge_layers

#: capture radius [m] for associating a hanger member with the piping run
#: it supports (typical rod-to-pipe stand-off is the pipe outer radius)
HANGER_REACH = 0.5


@dataclass
class SystemRecord:
    name: str                        # e.g. 'building-1', 'piping-1', 'containment-1'
    domain: str                      # 'building' | 'piping' | 'containment'
    product_guids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def classify_systems(ctx: IfcContext, proximity_tol: float = 1e-3) -> list[SystemRecord]:
    g = build_graph(ctx, proximity_tol)
    systems: list[SystemRecord] = []

    # -- equipment exclusion (cited rule, manuscript scope) -------------------
    for rec in ctx.products.values():
        if rec.ifc_class in EQUIPMENT_CLASSES:
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.EXCLUDED,
                             detail="equipment class — outside analytical conversion scope")

    # -- piping systems: one per physically continuous run ---------------------
    #    components over E_P|E_G only. Shared IfcDistributionSystem membership
    #    (E_A) is not continuity: a declared system spanning several separate
    #    runs is split at the run ends, so each deck is one continuous network
    dist_guids = [r.guid for r in ctx.products.values() if r.ifc_class in DISTRIBUTION_CLASSES]
    sub = g.subgraph(dist_guids)
    runs = sorted(continuity_components(g, dist_guids),
                  key=lambda c: (-len(c), min(c)))
    run_of = {guid: i for i, comp in enumerate(runs, 1) for guid in comp}
    declared: dict[str, set[int]] = {}
    for guid in dist_guids:
        for sg in ctx.products[guid].system_guids:
            declared.setdefault(sg, set()).add(run_of[guid])
    for sg, idx in sorted(declared.items()):
        if len(idx) > 1:
            ctx.audit.event("classification",
                            f"declared distribution system {sg[:8]} spans {len(idx)} "
                            "physically discontinuous runs; split at run ends into "
                            f"{', '.join(f'piping-{i}' for i in sorted(idx))}")
    for i, comp in enumerate(runs, 1):
        layers = edge_layers(sub, comp)
        rec = SystemRecord(name=f"piping-{i}", domain="piping",
                           product_guids=sorted(comp),
                           evidence=[f"connectivity layers: {sorted(layers) or ['isolated']}"])
        systems.append(rec)
        if "E_P" not in layers and len(comp) > 1:
            ctx.audit.event("classification",
                            f"{rec.name}: no port connectivity present; network continuity "
                            "established from geometric endpoint coincidence (E_G fallback)",
                            severity="warning")

    # -- hanger members: reassigned to the piping system they support ---------
    #    (a hanger bridges domains; per-system decks idealize it as a grounded
    #    spring on the piping side, so it must classify with the piping)
    hanger_guids: set[str] = set()
    piping_systems = [s for s in systems if s.domain == "piping"]
    if piping_systems:
        axes_of = {s.name: _straight_axes(ctx, s.product_guids) for s in piping_systems}
        for rec in ctx.products.values():
            if not is_hanger(rec):
                continue
            ends = _body_endpoints(ctx, rec)
            if not ends:
                continue
            best_sys, best_d = None, float("inf")
            for s in piping_systems:
                for seg in axes_of[s.name]:
                    for e in ends:
                        d = _point_segment_dist(e, seg)
                        if d < best_d:
                            best_sys, best_d = s, d
            if best_sys is not None and best_d <= HANGER_REACH:
                best_sys.product_guids.append(rec.guid)
                hanger_guids.add(rec.guid)
                ctx.audit.event(
                    "classification",
                    f"{best_sys.name}: hanger '{rec.name}' assigned to piping "
                    f"(min axis distance {best_d:.3f} m <= reach {HANGER_REACH:g} m)")
            else:
                fate = ("left in the structural system as a framing member"
                        if rec.ifc_class in STRUCTURAL_CLASSES
                        else "excluded as unassociated accessory hardware")
                ctx.audit.event(
                    "classification",
                    f"hanger '{rec.name}' not within {HANGER_REACH:g} m of any piping "
                    f"run; {fate}", severity="warning")
        for s in piping_systems:
            s.product_guids.sort()

    # -- accessory hardware that joined no piping system: outside scope -------
    #    (recorded so the coverage audit accounts for every ingested product)
    for rec in ctx.products.values():
        if rec.ifc_class in ACCESSORY_CLASSES and rec.guid not in hanger_guids:
            ctx.audit.record(rec.guid, rec.ifc_class, rec.name, AuditStatus.EXCLUDED,
                             detail="accessory hardware not associated with a piping "
                                    "run — outside analytical conversion scope")

    # -- structural systems: grouped per IfcBuilding via E_A ------------------
    struct = [r for r in ctx.products.values()
              if r.ifc_class in STRUCTURAL_CLASSES and r.guid not in hanger_guids]
    by_building: dict[str, list] = {}
    for r in struct:
        by_building.setdefault(r.building_guid or "_none", []).append(r)

    b_idx = c_idx = 0
    for bldg, recs in sorted(by_building.items()):
        classes = {r.ifc_class for r in recs}

        # containment evidence: products whose body is a vertical body of
        # revolution form the shell candidate set. Evidence is scored on the
        # RECOVERED form (wall cylinder, dome), never on the representation
        # type, so parametric, CSG and tessellated encodings of one structure
        # score alike. Equipment support racks are not structural framing and
        # are set aside before the framing test; any real frame inside a
        # containment stays a separate building system
        forms = {}
        for r in recs:
            if r.ifc_class not in FRAMING_CLASSES:
                fit = _revolution_of(r, ctx)
                if fit is not None:
                    forms[r.guid] = fit
        shells = [r for r in recs if r.guid in forms]
        rest = [r for r in recs if r.guid not in forms]
        racks, frame = _split_equipment_racks(rest, [f.params for f in forms.values()], ctx)

        score, why = 0, []
        if not any(r.ifc_class in FRAMING_CLASSES for r in frame):
            score += 1
            why.append("no framing classes" + (" besides equipment support racks" if racks else ""))
        walls = sorted({f.encoding for f in forms.values() if f.params.kind == "cylinder"
                        and f.params.params.get("r_inner", 0.0) > 0})
        if walls:
            score += 1; why.append(f"cylindrical wall shell [{'; '.join(walls)}]")
        domes = sorted({f.encoding for f in forms.values() if f.params.kind == "spherical_shell"})
        if domes:
            score += 1; why.append(f"spherical dome shell [{'; '.join(domes)}]")
        declared = " ".join([r.name for r in shells] + _building_labels(ctx, bldg)).lower()
        if any(tok in declared for tok in CONTAINMENT_TOKENS):
            score += 1; why.append("containment naming")

        if shells and score >= 2:
            c_idx += 1
            systems.append(SystemRecord(name=f"containment-{c_idx}", domain="containment",
                                        product_guids=[r.guid for r in shells],
                                        evidence=why))
            for r in racks:
                ctx.audit.record(r.guid, r.ifc_class, r.name, AuditStatus.EXCLUDED,
                                 detail="equipment support rack inside the containment "
                                        "shell - not a structural frame")
            if racks:
                ctx.audit.event("classification",
                                f"containment-{c_idx}: {len(racks)} equipment-support framing "
                                "member(s) discarded (rack inside the shell footprint / "
                                "rack naming)")
            if frame:
                b_idx += 1
                systems.append(SystemRecord(
                    name=f"building-{b_idx}", domain="building",
                    product_guids=[r.guid for r in frame],
                    evidence=[f"framing inside containment-{c_idx}: "
                              f"{sorted({r.ifc_class for r in frame})}"]))
        else:
            racks, frame = _split_equipment_racks(recs, [], ctx)
            for r in racks:
                ctx.audit.record(r.guid, r.ifc_class, r.name, AuditStatus.EXCLUDED,
                                 detail="equipment support rack (naming) - not a "
                                        "structural frame")
            if frame:
                b_idx += 1
                systems.append(SystemRecord(name=f"building-{b_idx}", domain="building",
                                            product_guids=[r.guid for r in frame],
                                            evidence=[f"framing classes {sorted(classes)}"]))

    for s in systems:
        ctx.audit.event("classification",
                        f"{s.name} ({s.domain}): {len(s.product_guids)} products; "
                        + "; ".join(s.evidence))
    return systems


#: linear framing classes (never containment shells)
FRAMING_CLASSES = ("IfcBeam", "IfcColumn", "IfcMember")

#: name tokens marking framing that only carries equipment (not a structure)
RACK_TOKENS = ("support rack", "equipment support", "support frame", "skid", "pipe rack")

#: declared containment semantics, matched in shell product names and in the
#: containing IfcBuilding's Name / LongName / ObjectType / Description
CONTAINMENT_TOKENS = ("containment", "dome", "basemat", "cylindrical")


def _split_equipment_racks(recs, shell_forms, ctx: IfcContext):
    """Separate equipment-support racks from real structural framing.

    A rack is framing (beams / columns / members) that either
      * is named as equipment support (``RACK_TOKENS``), or
      * lies entirely inside a containment shell's plan footprint while the
        framing set carries no slab, wall or plate of its own - a bare
        beam/column cage inside the containment is an equipment support,
        not part of the analysed structure.
    ``shell_forms`` are the recovered ``SolidParams`` of the shells.
    Returns ``(racks, frame)``.
    """
    framing = set(FRAMING_CLASSES)
    bare = not any(r.ifc_class not in framing for r in recs)
    footprints = [(p.params["cx"], p.params["cy"], p.params["r_outer"])
                  for p in shell_forms if p.kind == "cylinder"]
    racks, frame = [], []
    for r in recs:
        name = (r.name or "").lower()
        if r.ifc_class in framing and any(tok in name for tok in RACK_TOKENS):
            racks.append(r)
            continue
        if r.ifc_class in framing and bare and footprints:
            pts = _body_endpoints(ctx, r)
            if pts and all(any(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 <= r_out
                               for cx, cy, r_out in footprints) for p in pts):
                racks.append(r)
                continue
        frame.append(r)
    return racks, frame


def _revolution_of(rec, ctx: IfcContext):
    """``RevolutionFit`` of the first body item that is a vertical body of
    revolution in any encoding, or None."""
    from ..geometry.revolution import revolution_from_item
    for ri in rec.body_items:
        try:
            return revolution_from_item(ri.item, ri.matrix, ctx.length_scale, ctx.angle_scale)
        except Exception:
            continue
    return None


def _building_labels(ctx: IfcContext, building_guid: str) -> list[str]:
    """Declared labels of an IfcBuilding (Name, LongName, ObjectType,
    Description) - e.g. ObjectType 'REACTOR_CONTAINMENT'."""
    try:
        b = ctx.model.by_guid(building_guid)
    except Exception:
        return []
    return [str(v) for v in (getattr(b, a, None) for a in
                             ("Name", "LongName", "ObjectType", "Description")) if v]


def _body_endpoints(ctx: IfcContext, rec) -> list[tuple[float, float, float]]:
    """Extrusion-axis endpoints of a product's prismatic body items."""
    from ..geometry.swept import extruded_axis

    pts = []
    for ri in rec.body_items:
        if ri.item.is_a("IfcExtrudedAreaSolid"):
            try:
                ax = extruded_axis(ri.item, ri.matrix, ctx.length_scale)
            except Exception:
                continue
            pts += [ax.start, ax.end]
    return pts


def _straight_axes(ctx: IfcContext, guids: list[str]):
    """(start, end) centerline segments of the straight prismatic bodies in a
    product set (elbow arcs are irrelevant for hanger association)."""
    segs = []
    for guid in guids:
        rec = ctx.products.get(guid)
        if rec is None or rec.ifc_class not in DISTRIBUTION_CLASSES:
            continue
        pts = _body_endpoints(ctx, rec)
        for i in range(0, len(pts) - 1, 2):
            segs.append((pts[i], pts[i + 1]))
    return segs


def _point_segment_dist(p, seg) -> float:
    import numpy as np

    a, b = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
    q = np.asarray(p, dtype=float)
    ab = b - a
    denom = float(ab.dot(ab))
    if denom < 1e-18:
        return float(np.linalg.norm(q - a))
    t = max(0.0, min(1.0, float((q - a).dot(ab)) / denom))
    return float(np.linalg.norm(q - (a + t * ab)))
