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
    STRUCTURAL_CLASSES, is_footing, is_hanger,
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
    #: columns whose base plate was excluded: fixed at the foot in place of it
    fixed_columns: list[str] = field(default_factory=list)


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
        # column base plates are connection hardware, not frame: excluded
        # first (a leftover plate would also defeat the bare-cage rack rule),
        # and the columns standing on them are fixed at the foot instead
        plates, recs, seated = _split_base_plates(recs, ctx)
        for r, why in plates:
            ctx.audit.record(r.guid, r.ifc_class, r.name, AuditStatus.EXCLUDED,
                             detail=f"column base plate ({why}) - connection hardware; "
                                    "the column is fixed at its foot instead")
        if not recs:
            continue
        n_before = len(systems)
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
        # slabs on grade bear on the foundation, not on the frame (split
        # before the rack test for the same reason as base plates); shells
        # are left alone - a containment basemat is a shell, not a ground slab
        ground, rest = _split_ground_slabs(rest, recs, ctx)
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
            envelope, frame = _split_envelope(frame, ctx)
            if frame:
                b_idx += 1
                systems.append(SystemRecord(
                    name=f"building-{b_idx}", domain="building",
                    product_guids=[r.guid for r in frame],
                    evidence=[f"framing inside containment-{c_idx}: "
                              f"{sorted({r.ifc_class for r in frame})}"],
                    fixed_columns=[g for g in seated if g in {r.guid for r in frame}]))
        else:
            # an ordinary building: a round slab on grade is a ground slab too
            ground += _split_ground_slabs(shells, recs, ctx)[0]
            gone = {r.guid for r, _ in ground}
            racks, frame = _split_equipment_racks([r for r in recs if r.guid not in gone], [], ctx)
            for r in racks:
                ctx.audit.record(r.guid, r.ifc_class, r.name, AuditStatus.EXCLUDED,
                                 detail="equipment support rack (naming) - not a "
                                        "structural frame")
            envelope, frame = _split_envelope(frame, ctx)
            if frame:
                b_idx += 1
                systems.append(SystemRecord(name=f"building-{b_idx}", domain="building",
                                            product_guids=[r.guid for r in frame],
                                            evidence=[f"framing classes {sorted(classes)}"],
                                            fixed_columns=[g for g in seated
                                                           if g in {r.guid for r in frame}]))
        for r, why in ground:
            ctx.audit.record(r.guid, r.ifc_class, r.name, AuditStatus.EXCLUDED,
                             detail=f"ground slab ({why}) - rests on grade; the column / "
                                    "wall bases are fixed instead")
        for r, why in envelope:
            ctx.audit.record(r.guid, r.ifc_class, r.name, AuditStatus.EXCLUDED,
                             detail=f"non-load-bearing envelope ({why}) - neither its "
                                    "stiffness nor its mass enters the model")
        for s in systems[n_before:]:
            if envelope and s.domain == "building":
                s.evidence.append(f"{len(envelope)} non-load-bearing envelope element(s) "
                                  "excluded")
            if plates and s.domain == "building":
                s.evidence.append(f"{len(plates)} column base plate(s) excluded, "
                                  f"{len(s.fixed_columns)} column(s) fixed at the foot")
            if ground and s.domain == "building":
                s.evidence.append(f"{len(ground)} ground slab(s) excluded")

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

#: name tokens marking column base plates (plate name, ObjectType, or the
#: name of the element assembly the plate is aggregated into)
BASE_PLATE_TOKENS = ("base plate", "baseplate", "base_plate", "base-plate")
#: plan radius [m] around the column axis a base plate may occupy: a base
#: plate only cantilevers a short projection beyond the column section, so a
#: plate reaching further from the axis is a floor plate, not a base plate
BASE_PLATE_REACH = 1.0
#: tolerance [m] for a column foot seated on a plate face
BASE_PLATE_SEAT = 0.025
#: column feet within this height [m] of the lowest foot are at the support
#: level (same reach the building assembler uses for base supports)
SUPPORT_LEVEL_REACH = 0.5


def _split_base_plates(recs, ctx: IfcContext):
    """Separate column base plates from the structural products.

    A base plate is connection hardware between a column foot and its
    foundation: the analytical column ends at its foot and is fixed there,
    so the plate is not converted. Evidence cascade (first hit wins):
      * declared - PredefinedType BASE_PLATE on the occurrence or its type
        object (IFC4X3 IfcPlateTypeEnum);
      * named - the plate, its ObjectType, or the element assembly it is
        aggregated into carries a base-plate token (``BASE_PLATE_TOKENS``);
      * geometric - a thin horizontal plate carrying the foot of a column at
        the support level and lying within ``BASE_PLATE_REACH`` of that
        column's axis. Plates at column splices higher up are left alone:
        they are the only link between stacked column pieces.
    Returns ``(plates, rest, seated)``: ``(record, evidence)`` pairs, the
    remaining records, and the GUIDs of the columns found standing on an
    excluded plate at the support level.
    """
    import ifcopenshell.util.element as ioe

    feet = _column_feet(recs, ctx)
    support_z = min((p[2] for _, p in feet), default=0.0)
    plates, rest, seated = [], [], []
    for r in recs:
        if r.ifc_class != "IfcPlate":
            rest.append(r)
            continue
        why = None
        ptype = (ioe.get_predefined_type(r.entity) or "").upper()
        if ptype == "BASE_PLATE":
            why = "declared PredefinedType BASE_PLATE"
        else:
            agg = ioe.get_aggregate(r.entity)
            labels = [r.name, getattr(r.entity, "ObjectType", None) or "",
                      (getattr(agg, "Name", None) or "") if agg is not None else ""]
            if any(tok in " ".join(labels).lower() for tok in BASE_PLATE_TOKENS):
                why = "base-plate naming"
        # only feet at the support level are grounded: a column on a base
        # plate higher up (e.g. on a transfer girder) bears on the frame
        on = [(g, p) for g, p in _plate_seats(r, feet, ctx)
              if p[2] - support_z <= SUPPORT_LEVEL_REACH]
        if why is None and on:
            why = "thin horizontal plate under a column foot at the support level"
        if why is None:
            rest.append(r)
            continue
        plates.append((r, why))
        seated += [g for g, _ in on if g not in seated]
    return plates, rest, seated


#: tolerance [m] for a slab top face lying at the support level
GROUND_SLAB_SEAT = 0.025


def _split_ground_slabs(cands, recs, ctx: IfcContext):
    """Separate slabs on grade from the building frame.

    A slab bearing on the ground at the support level is not part of the
    analysed structure: the column and wall bases are fixed, which idealises
    the foundation (base plates and ground slab alike), so the slab is not
    converted (typical practice). Footings stay with ``footing_policy``.
    Evidence (first hit wins):
      * declared - PredefinedType BASESLAB (IFC: 'base slab, including slab
        on grade');
      * geometric - the slab's top face is at or below the support level,
        i.e. the lowest column foot or wall base among ``recs``. Walls
        reaching below a slab keep it (a podium slab over walls is carried
        by them), which errs on the side of converting a doubtful slab.
    Returns ``(slabs, rest)`` with slabs as ``(record, evidence)`` pairs.
    """
    import ifcopenshell.util.element as ioe

    bases = [p[2] for _, p in _column_feet(recs, ctx)]
    bases += [z[0] for z in (_z_range(r, ctx) for r in recs if r.ifc_class == "IfcWall") if z]
    support_z = min(bases, default=None)
    slabs, rest = [], []
    for r in cands:
        why = None
        if r.ifc_class == "IfcSlab" and not is_footing(r):
            if (ioe.get_predefined_type(r.entity) or "").upper() == "BASESLAB":
                why = "declared PredefinedType BASESLAB"
            elif support_z is not None:
                z = _z_range(r, ctx)
                if z and z[1] <= support_z + GROUND_SLAB_SEAT:
                    why = (f"top face z={z[1]:.3f} m at or below the support level "
                           f"z={support_z:.3f} m")
        if why is None:
            rest.append(r)
        else:
            slabs.append((r, why))
    return slabs, rest


#: IfcWall PredefinedTypes that are non-load-bearing / load-bearing by definition
NONBEARING_WALL_TYPES = ("PARTITIONING", "PARAPET")
BEARING_WALL_TYPES = ("SHEAR", "RETAININGWALL")
#: material families that carry no load in a wall or slab build-up
NONSTRUCTURAL_FAMILIES = ("insulation", "gypsum", "glass")
#: steel layers thinner than this [m] are sheet (cladding skins, decking)
SHEET_STEEL_MAX = 0.003
#: ACI 318-19 Table 11.3.1.1, bearing walls: thickness at least the greater
#: of 4 in and 1/25 of the lesser of the unsupported length and height
BEARING_WALL_T_MIN = 0.1016
BEARING_WALL_SLENDERNESS = 25.0
#: framing members at most this far apart [m] carry a sheet everywhere
DECK_SUPPORT_SPACING = 3.0
#: face-contact tolerance [m] between envelope elements and the frame
CONTACT_TOL = 0.025


def _split_envelope(recs, ctx: IfcContext):
    """Separate non-load-bearing envelope elements - cladding walls, roofing,
    decking - from the structural frame. Everything is kept unless the
    evidence says otherwise, so load-bearing walls and slabs survive:
      * declared (decisive either way): a LoadBearing property
        (Pset_WallCommon / Pset_SlabCommon / Pset_RoofCommon), or an IfcWall
        PredefinedType non-load-bearing (PARTITIONING, PARAPET) or
        load-bearing (SHEAR, RETAININGWALL) by definition;
      * walls - two independent signals must agree: the material build-up
        has no load-bearing layer (insulation / gypsum / glass and steel
        skins thinner than ``SHEET_STEEL_MAX`` only; unknown materials count
        as load-bearing), AND the whole wall is thinner than the ACI 318
        minimum for a bearing wall of its unsupported height and length;
      * slabs / roofs - two signals: the build-up has no concrete, masonry
        or timber body (metal sheet or plate only), AND framing members
        directly beneath carry it everywhere (``DECK_SUPPORT_SPACING``).
    A non-structural build-up without the second signal only earns a review
    warning; the element is kept.
    Returns ``(envelope, rest)`` with envelope as ``(record, evidence)`` pairs.
    """
    import ifcopenshell.util.element as ioe

    from shapely.geometry import MultiPoint

    if not any(r.ifc_class in ("IfcWall", "IfcSlab", "IfcRoof") for r in recs):
        return [], list(recs)
    clouds = {r.guid: _cloud(r, ctx) for r in recs}
    feet = {g: MultiPoint([tuple(p) for p in v[:, :2]]).convex_hull
            for g, v in clouds.items() if v is not None}
    envelope, rest = [], []
    for r in recs:
        why = None
        if r.ifc_class in ("IfcWall", "IfcSlab", "IfcRoof") and clouds[r.guid] is not None:
            declared, pset = _declared_load_bearing(r)
            ptype = (ioe.get_predefined_type(r.entity) or "").upper()
            if declared is not None:
                why = None if declared else f"declared {pset}.LoadBearing = FALSE"
            elif r.ifc_class == "IfcWall" and ptype in BEARING_WALL_TYPES:
                why = None
            elif r.ifc_class == "IfcWall" and ptype in NONBEARING_WALL_TYPES:
                why = f"declared PredefinedType {ptype}"
            else:
                signals = (_wall_signals(r, recs, clouds, feet, ctx) if r.ifc_class == "IfcWall"
                           else _deck_signals(r, recs, clouds, ctx))
                if all(signals):
                    why = "; ".join(signals)
                elif signals[0]:
                    # a non-structural build-up alone is worth a look; the
                    # geometric signal alone is normal (floors sit on beams)
                    ctx.audit.event("classification",
                                    f"'{r.name}' is possibly non-structural ({signals[0]}) but "
                                    "the second signal is missing - kept, review",
                                    guid=r.guid, severity="warning")
        if why is None:
            rest.append(r)
        else:
            envelope.append((r, why))
    return envelope, rest


def _declared_load_bearing(rec):
    """``(LoadBearing, pset name)`` from the element's property sets, or
    ``(None, None)`` when nothing is declared."""
    for name, props in rec.psets.items():
        val = props.get("LoadBearing") if isinstance(props, dict) else None
        if isinstance(val, bool):
            return val, name
    return None, None


def _build_up(rec, thickness: float, ctx: IfcContext):
    """``[(thickness [m], material name, layer category)]`` of an element's
    material build-up: the layers of a layer set, or one layer of the whole
    thickness for a single material; None when nothing is associated."""
    mat = rec.material
    if mat is None:
        return None
    if mat.is_a("IfcMaterialLayerSetUsage"):
        mat = mat.ForLayerSet
    if mat.is_a("IfcMaterialLayerSet"):
        return [(float(lay.LayerThickness) * ctx.length_scale,
                 (lay.Material.Name or "") if lay.Material else "",
                 getattr(lay, "Category", None) or "") for lay in mat.MaterialLayers]
    if mat.is_a("IfcMaterial"):
        return [(thickness, mat.Name or "", "")]
    return None


def _describe(layers) -> str:
    return " + ".join(f"{t * 1000:g} mm {name or '?'}" for t, name, _ in layers)


def _wall_signals(rec, recs, clouds, feet, ctx: IfcContext):
    """(build-up signal, code-minimum signal) of a wall: an evidence string
    for each signal that marks it non-load-bearing, else None."""
    from shapely.geometry import LineString, MultiPoint
    from ..assign.material_library import guess_family

    v = clouds[rec.guid]
    rect = MultiPoint([tuple(p) for p in v[:, :2]]).minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return None, None
    c = list(rect.exterior.coords)[:4]
    e1, e2 = LineString(c[:2]).length, LineString(c[1:3]).length
    t, length = min(e1, e2), max(e1, e2)
    ends = ((c[0], c[3]), (c[1], c[2])) if e1 >= e2 else ((c[0], c[1]), (c[3], c[2]))
    trace = LineString([((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in ends])
    z_lo, z_hi = float(v[:, 2].min()), float(v[:, 2].max())

    build = None
    layers = _build_up(rec, t, ctx)
    if layers:
        core = sum(th for th, name, cat in layers
                   if "loadbearing" in cat.lower().replace(" ", "").replace("_", "")
                   or not (guess_family(name) in NONSTRUCTURAL_FAMILIES
                           or (guess_family(name) == "steel" and th < SHEET_STEEL_MAX)))
        if core <= 0.0:
            build = f"no load-bearing layer in its build-up ({_describe(layers)})"

    # lateral supports: columns / walls / floors in contact with the wall
    u_st, z_st = {0.0, length}, {z_lo, z_hi}
    for o in recs:
        ov = clouds.get(o.guid)
        if o.guid == rec.guid or ov is None or feet[o.guid].distance(rect) > CONTACT_TOL:
            continue
        oz_lo, oz_hi = float(ov[:, 2].min()), float(ov[:, 2].max())
        if o.ifc_class in ("IfcColumn", "IfcWall") and oz_lo < z_hi and oz_hi > z_lo:
            u_st.add(min(max(trace.project(feet[o.guid].centroid), 0.0), length))
        elif o.ifc_class in ("IfcSlab", "IfcBeam", "IfcMember") and z_lo < oz_hi and oz_lo < z_hi:
            z_st.add(min(max((oz_lo + oz_hi) / 2.0, z_lo), z_hi))
    l_u = max(hi - lo for lo, hi in zip(sorted(u_st), sorted(u_st)[1:]))
    h_u = max(hi - lo for lo, hi in zip(sorted(z_st), sorted(z_st)[1:]))
    t_req = max(BEARING_WALL_T_MIN, min(l_u, h_u) / BEARING_WALL_SLENDERNESS)
    code = None
    if t < t_req:
        code = (f"{t * 1000:.0f} mm thick, below the ACI 318 bearing-wall minimum "
                f"{t_req * 1000:.0f} mm (unsupported height {h_u:.2f} m, length {l_u:.2f} m)")
    return build, code


def _deck_signals(rec, recs, clouds, ctx: IfcContext):
    """(build-up signal, support signal) of a slab / roof: metal-only body,
    and framing members directly beneath carrying it everywhere."""
    import numpy as np
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    from ..assign.material_library import guess_family

    v = clouds[rec.guid]
    centre = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - centre, full_matrices=False)
    n = vt[2] if vt[2][2] >= 0 else -vt[2]
    if n[2] < 0.5:
        return None, None                           # not a floor / roof plate
    e1, e2 = vt[0], np.cross(n, vt[0])
    s = (v - centre) @ n
    t, bottom = float(np.ptp(s)), float(s.min())

    build = None
    layers = _build_up(rec, t, ctx)
    body = [f for f in (guess_family(name) for _, name, _ in layers or [])
            if f not in NONSTRUCTURAL_FAMILIES]
    if body and all(f in ("steel", "aluminum") for f in body):
        build = f"metal-only build-up ({_describe(layers)}), no concrete / masonry / timber body"

    outline = MultiPoint([((p - centre) @ e1, (p - centre) @ e2) for p in v]).convex_hull
    carried = []
    for o in recs:
        ov = clouds.get(o.guid)
        if o.ifc_class not in ("IfcBeam", "IfcMember") or ov is None:
            continue
        if abs(float(((ov - centre) @ n).max()) - bottom) > CONTACT_TOL:
            continue                                # top not at the underside
        strip = MultiPoint([((p - centre) @ e1, (p - centre) @ e2) for p in ov]).convex_hull
        if strip.intersects(outline):
            carried.append(strip.buffer(DECK_SUPPORT_SPACING / 2.0))
    support = None
    if carried and outline.area > 0:
        covered = unary_union(carried).intersection(outline).area / outline.area
        if covered >= 0.95:
            support = (f"carried everywhere by {len(carried)} framing member(s) directly "
                       f"beneath (spacing <= {DECK_SUPPORT_SPACING:g} m)")
    return build, support


def _cloud(rec, ctx: IfcContext):
    """Global body vertex cloud [m] of a product (all encodings), or None."""
    import numpy as np
    from ..geometry.revolution import vertex_cloud

    parts = []
    for ri in rec.body_items:
        try:
            parts.append(vertex_cloud(ri.item, ri.matrix, ctx.length_scale)[0])
        except Exception:
            continue
    parts = [p for p in parts if len(p)]
    return np.vstack(parts) if parts else None


def _z_range(rec, ctx: IfcContext):
    """``(z_min, z_max)`` [m] of a product body in any encoding, or None."""
    from ..geometry.revolution import vertex_cloud

    zs = []
    for ri in rec.body_items:
        try:
            verts, _ = vertex_cloud(ri.item, ri.matrix, ctx.length_scale)
        except Exception:
            continue
        if len(verts):
            zs += [float(verts[:, 2].min()), float(verts[:, 2].max())]
    return (min(zs), max(zs)) if zs else None


def _column_feet(recs, ctx: IfcContext):
    """``(guid, lowest axis point)`` of every column: extrusion axis, else the
    Axis representation."""
    from ..geometry.axes import axis_rep_endpoints

    feet = []
    for r in recs:
        if r.ifc_class != "IfcColumn":
            continue
        pts = _body_endpoints(ctx, r) or list(axis_rep_endpoints(r.axis_items,
                                                                 ctx.length_scale) or [])
        if pts:
            feet.append((r.guid, min(pts, key=lambda p: p[2])))
    return feet


def _plate_seats(rec, feet, ctx: IfcContext):
    """Column feet ``(guid, point)`` that stand on a plate: the plate is thin
    and horizontal, the foot lies on its plan outline between its faces
    (within ``BASE_PLATE_SEAT``) and the whole plate is local to that column
    (within ``BASE_PLATE_REACH`` of its axis)."""
    from shapely.geometry import Point, Polygon

    from ..geometry.swept import plate_from_extrusion

    out = []
    for ri in rec.body_items:
        if not ri.item.is_a("IfcExtrudedAreaSolid"):
            continue
        try:
            pg = plate_from_extrusion(ri.item, ri.matrix, ctx.length_scale)
        except Exception:
            continue
        if abs(pg.normal[2]) < 0.99 or len(pg.loop) < 3:
            continue
        mid = sum(p[2] for p in pg.loop) / len(pg.loop)
        lo, hi = mid - pg.thickness / 2.0, mid + pg.thickness / 2.0
        outline = Polygon([(p[0], p[1]) for p in pg.loop]).buffer(BASE_PLATE_SEAT)
        for guid, (x, y, z) in feet:
            if (lo - BASE_PLATE_SEAT <= z <= hi + BASE_PLATE_SEAT
                    and outline.contains(Point(x, y))
                    and all(((p[0] - x) ** 2 + (p[1] - y) ** 2) ** 0.5 <= BASE_PLATE_REACH
                            for p in pg.loop)):
                out.append((guid, (x, y, z)))
    return out


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
