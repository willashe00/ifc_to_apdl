"""Phase 4 — conventional building structure assembly.

The compatibility pipeline (manuscript Section: Global Compatibility,
formalized with a planar arrangement instead of ad-hoc rules):

  1. interpret members/plates (axis-rep first, cardinal-point aware)
  1c. wall-corner reconciliation: a wall whose end face abuts another wall
     stops half a thickness short of that wall's mid-plane; the analytical
     trace is extended to the mid-plane intersection (face-gap criterion)
  1d. wall-end reconciliation to columns / beams: a wall whose end face abuts
     a column (boundary element, infill wall between columns) or a beam's
     side face is extended to that member's axis, so the column line becomes
     the wall's vertical edge (embedded boundary column)
  2. snap beam endpoints to column axes / wall mid-planes (face -> axis merge)
  3. lift beam axes into the slab centroidal plane (delta = d_b/2 + t_s/2),
     compensated by SECOFFSET carried on the section
  3b. wall vertical extents reconciled to slab planes: tops/bases within the
     slab thickness of a level (Revit level-to-level walls, walls to the
     soffit) land ON the slab mid-plane; per-storey walls stacked across a
     slab therefore share the slab's keypoints
  4. per-level planar arrangement (shapely, 1 mm precision model): beams,
     wall traces (dangling ends extended to the next line so a wall under a
     slab interior still partitions it), slab boundaries and openings are
     noded together; beam segments and slab bay panels emerge from the same
     linework, so shared keypoints/lines are guaranteed. Every arrangement
     vertex on a wall trace becomes a wall partition station.
  4a. brace workpoints to column axes / beam planes; crossing or interrupted
     diagonals (X-bracing) noded at their common point
  5. walls partitioned at storey planes, crossing framing lines, other wall
     traces, arrangement vertices and member attachment points
  5b. conformity enforcement: any pool node lying on the interior of a member
     axis or a surface edge is inserted (member split / loop refined) - the
     writer emits one line per keypoint pair, so this guarantees conformal
     meshes along every interface
  6. supports: lowest-level column bases + wall base edges (heuristic, logged)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, substring, unary_union

from ..assign.elements import resolve_element_type
from ..assign.materials import MaterialResolver
from ..classify.systems import SystemRecord
from ..config import ConversionConfig
from ..geometry.axes import axis_rep_endpoints
from ..geometry.profiles import parse_profile
from ..geometry.swept import extruded_axis, plate_from_extrusion
from ..ingest.loader import IfcContext, ProductRecord
from ..model.ir import AnalyticalModel, Member1D, NodePool, Provenance, Support
from ..model.sections import (PipeSection, RectSection, Section, ShellSection,
                              TubeSection)


def _as_beam_section(sec: Section | None) -> Section | None:
    """Framing members are BEAM188: the PIPE section category is illegal
    there, so circular hollows become the BEAM,CTUBE subtype."""
    if isinstance(sec, PipeSection):
        return TubeSection(name=sec.name, od=sec.od, t=sec.t)
    return sec
from ..report.audit import AuditStatus

Vec3 = tuple[float, float, float]
LEVEL_TOL = 1e-3


# ---------------------------------------------------------------------------
# interpreted intermediates
# ---------------------------------------------------------------------------

@dataclass
class RawBeam:
    rec: ProductRecord
    start: np.ndarray
    end: np.ndarray
    section: Section
    depth: float
    evidence: dict[str, str] = field(default_factory=dict)
    lifted_offset: float = 0.0     # SECOFFSET y after slab lift


@dataclass
class RawColumn:
    rec: ProductRecord
    base: np.ndarray
    top: np.ndarray
    section: Section
    plan_halfwidth: float
    orient: Vec3
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass
class RawWall:
    rec: ProductRecord
    p1: np.ndarray                 # mid-plane trace endpoints (z = base)
    p2: np.ndarray
    z_lo: float
    z_hi: float
    thickness: float
    evidence: dict[str, str] = field(default_factory=dict)

    def trace2d(self) -> LineString:
        return LineString([(self.p1[0], self.p1[1]), (self.p2[0], self.p2[1])])


@dataclass
class RawSlab:
    rec: ProductRecord
    poly: Polygon                  # footprint at mid-plane
    z_mid: float
    thickness: float
    evidence: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# interpretation
# ---------------------------------------------------------------------------

def _cardinal_offset(rec: ProductRecord, depth: float, audit) -> float:
    """Vertical offset from the axis line to the section centroid implied by
    IfcMaterialProfileSetUsage.CardinalPoint (evidence-first eccentricity)."""
    assoc = rec.material
    cp = getattr(assoc, "CardinalPoint", None) if assoc is not None else None
    if cp is None:
        return 0.0
    cp = int(cp)
    if cp in (1, 2, 3):        # bottom row: centroid is d/2 above the axis line
        return +depth / 2.0
    if cp in (7, 8, 9):        # top row: centroid is d/2 below
        return -depth / 2.0
    if cp in (4, 5, 6):
        return 0.0
    audit.event("geometry", f"unsupported cardinal point {cp}; no axis offset applied",
                guid=rec.guid, severity="warning")
    return 0.0


def _beam_from(rec: ProductRecord, ctx: IfcContext) -> RawBeam | None:
    solid = next((ri for ri in rec.body_items if ri.item.is_a("IfcExtrudedAreaSolid")), None)
    if solid is None:
        return _beam_from_tessellated(rec, ctx)
    ax_body = extruded_axis(solid.item, solid.matrix, ctx.length_scale)
    section = _as_beam_section(ax_body.profile.section)
    if section is None:
        return None
    depth = getattr(section, "depth", 0.0) or 0.0
    ev = {"section": f"{ax_body.profile.kind}-profile '{ax_body.profile.name}'"}

    ends = axis_rep_endpoints(rec.axis_items, ctx.length_scale)
    if ends is not None:
        dz = _cardinal_offset(rec, depth, ctx.audit)
        start = np.array(ends[0]) + np.array([0, 0, dz])
        end = np.array(ends[1]) + np.array([0, 0, dz])
        ev["centerline"] = "axis-representation" + (f" (cardinal dz={dz:+.4f})" if dz else "")
    else:
        start, end = np.array(ax_body.start), np.array(ax_body.end)
        ev["centerline"] = "extrusion-axis"
    return RawBeam(rec, start, end, section, depth, ev)


def _beam_from_tessellated(rec: ProductRecord, ctx: IfcContext) -> RawBeam | None:
    """Computational branch of the evidence cascade: recover the section and
    centerline of a tessellated member (validated in experiments/geom_extraction)."""
    item = next((ri for ri in rec.body_items
                 if ri.item.is_a("IfcTriangulatedFaceSet")
                 or ri.item.is_a("IfcPolygonalFaceSet")), None)
    if item is None:
        return None
    try:
        from ..geometry.tessellated import extract_section, mesh_from_faceset
        verts, faces = mesh_from_faceset(item.item, ctx.length_scale, item.matrix)
        res = extract_section(verts, faces)
    except Exception as exc:
        ctx.audit.event("geometry", f"tessellated extraction failed: {exc}",
                        guid=rec.guid, severity="warning")
        return None
    section = _as_beam_section(res.section)
    if section is None:
        return None
    depth = getattr(section, "depth", 0.0) or 0.0
    ev = {"section": f"computational: tessellated {res.shape} "
                     f"({res.stations} station median)"}
    ends = axis_rep_endpoints(rec.axis_items, ctx.length_scale)
    if ends is not None:
        start, end = np.array(ends[0]), np.array(ends[1])
        ev["centerline"] = "axis-representation"
    else:
        start, end = np.array(res.start), np.array(res.end)
        ev["centerline"] = "tessellated mesh axis"
    return RawBeam(rec, start, end, section, depth, ev)


def _column_from(rec: ProductRecord, ctx: IfcContext) -> RawColumn | None:
    solid = next((ri for ri in rec.body_items if ri.item.is_a("IfcExtrudedAreaSolid")), None)
    if solid is None:
        return None
    ax = extruded_axis(solid.item, solid.matrix, ctx.length_scale)
    if ax.profile.section is None:
        return None
    base, top = np.array(ax.start), np.array(ax.end)
    if base[2] > top[2]:
        base, top = top, base
    sec = _as_beam_section(ax.profile.section)
    halfwidth = max(getattr(sec, "width", 0.0), getattr(sec, "depth", 0.0),
                    getattr(sec, "od", 0.0), getattr(sec, "diameter", 0.0)) / 2.0
    frame_x = solid.matrix[:3, 0]
    orient = tuple(frame_x / (np.linalg.norm(frame_x) or 1.0))
    return RawColumn(rec, base, top, sec, halfwidth, orient,
                     {"centerline": "extrusion-axis",
                      "section": f"{ax.profile.kind}-profile '{ax.profile.name}'"})


def _wall_from(rec: ProductRecord, ctx: IfcContext) -> RawWall | None:
    solid = next((ri for ri in rec.body_items if ri.item.is_a("IfcExtrudedAreaSolid")), None)
    if solid is None:
        return None
    prof = parse_profile(solid.item.SweptArea, ctx.length_scale)
    d_local = np.array(solid.item.ExtrudedDirection.DirectionRatios, dtype=float)
    from ..ingest.transforms import apply, apply_dir, axis2placement3d
    frame = solid.matrix @ axis2placement3d(getattr(solid.item, "Position", None), ctx.length_scale)
    d_glob = apply_dir(frame, tuple(d_local))
    depth = float(solid.item.Depth) * ctx.length_scale

    if abs(d_glob[2]) > 0.99 and prof.poly:
        # vertical extrusion: footprint polygon -> mid-line by rotated bounding box
        pts = [apply(frame, apply(prof.position, (px, py, 0.0))) for px, py in prof.poly]
        poly2 = Polygon([(p[0], p[1]) for p in pts])
        obb = poly2.minimum_rotated_rectangle
        corners = list(obb.exterior.coords)[:4]
        e1 = np.array(corners[1]) - np.array(corners[0])
        e2 = np.array(corners[2]) - np.array(corners[1])
        if np.linalg.norm(e1) < np.linalg.norm(e2):
            # c0-c1 is the short (thickness) edge; length runs along e2
            t_vec, l_vec, origin = e1, e2, np.array(corners[0])
        else:
            # c1-c2 is the short edge; length runs BACK along -e1 from c1
            t_vec, l_vec, origin = e2, -e1, np.array(corners[1])
        thickness = float(np.linalg.norm(t_vec))
        mid0 = origin + t_vec / 2.0
        mid1 = mid0 + l_vec
        z0 = pts[0][2]
        p1 = np.array([mid0[0], mid0[1], z0])
        p2 = np.array([mid1[0], mid1[1], z0])
        return RawWall(rec, p1, p2, z0, z0 + depth * d_glob[2], thickness,
                       {"midplane": "vertical-extrusion footprint OBB"})

    # horizontal extrusion of an elevation profile (rect): use plate helper
    try:
        plate = plate_from_extrusion(solid.item, solid.matrix, ctx.length_scale)
    except ValueError:
        return None
    zs = [p[2] for p in plate.loop]
    lo, hi = min(zs), max(zs)
    bottom = [p for p in plate.loop if abs(p[2] - lo) < LEVEL_TOL]
    if len(bottom) < 2:
        return None
    p1, p2 = np.array(bottom[0]), np.array(bottom[1])
    return RawWall(rec, p1, p2, lo, hi, plate.thickness,
                   {"midplane": "plate-extrusion"})


def _slab_from(rec: ProductRecord, ctx: IfcContext) -> RawSlab | None:
    solid = next((ri for ri in rec.body_items if ri.item.is_a("IfcExtrudedAreaSolid")), None)
    if solid is None:
        return None
    try:
        plate = plate_from_extrusion(solid.item, solid.matrix, ctx.length_scale)
    except ValueError:
        return None
    holes = [[(p[0], p[1]) for p in h] for h in plate.holes] if plate.holes else None
    poly = Polygon([(p[0], p[1]) for p in plate.loop], holes)
    if not poly.is_valid or poly.area < 1e-6:
        return None
    z_mid = float(np.mean([p[2] for p in plate.loop]))
    ev = {"midplane": "extrusion mid-depth"}
    if holes:
        ev["voids"] = f"{len(holes)} opening(s) preserved"
    return RawSlab(rec, poly, z_mid, plate.thickness, ev)


def _snap_ring_to_line(ring, line: LineString, tol: float):
    """Project ring vertices lying within ``tol`` of ``line`` onto it."""
    out = []
    for x, y in list(ring.coords):
        p = Point(x, y)
        if line.distance(p) <= tol:
            q = line.interpolate(line.project(p))
            out.append((q.x, q.y))
        else:
            out.append((x, y))
    from shapely.geometry import LinearRing
    return LinearRing(out)


def _snap_ring_to_trace(ring, p1, p2, tol: float, fixed: set | None = None):
    """Project ring vertices onto a wall mid-plane LINE (not segment), with a
    jog where the ring edge runs on past the wall's end.

    An architectural slab edge running parallel to a wall sits half a
    thickness off the mid-plane. Vertices within ``tol`` laterally and within
    the trace span (± ``tol``, so corner vertices converge onto the
    intersection of two traces under sequential application) are moved onto
    the trace, so the level arrangement nodes wall and slab together. Where
    an edge continues well past a wall end, it is split there: one vertex
    stays on the original edge (pinned in ``fixed`` so a neighbouring wall's
    end window cannot capture it), one sits on the trace. Without the jog the
    unsnapped far vertex tilts the whole edge, producing wedge-shaped sliver
    panels that break meshing and mass accounting.

    Returns ``(ring, fixed)``.
    """
    from shapely.geometry import LinearRing
    fixed = set(fixed or ())
    a = np.array([p1[0], p1[1]], dtype=float)
    b = np.array([p2[0], p2[1]], dtype=float)
    d = b - a
    length = float(np.hypot(*d))
    if length < 1e-9:
        return ring, fixed
    d /= length
    coords = list(ring.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    def key(pt):
        return (round(pt[0], 6), round(pt[1], 6))

    def station(x, y):
        v = np.array([x, y]) - a
        t = float(v @ d)
        perp = float(np.hypot(*(v - t * d)))
        return t, perp

    def inside(t, perp):
        return perp <= tol and -tol <= t <= length + tol

    def alongside(i, j):
        """Edge i->j runs beside the trace over a positive length of its span."""
        ti, qi = station(*coords[i])
        tj, qj = station(*coords[j])
        if max(qi, qj) > tol:
            return False
        lo, hi = min(ti, tj), max(ti, tj)
        return min(hi, length) - max(lo, 0.0) > tol

    out = []
    n = len(coords)
    for i in range(n):
        x, y = coords[i]
        xn, yn = coords[(i + 1) % n]
        t0, q0 = station(x, y)
        t1, q1 = station(xn, yn)
        # a vertex is moved onto the trace only when one of its edges actually
        # runs along the wall (or it already sits on the mid-plane line): a
        # slab corner at a wall END whose edges lead away from the wall stays
        # put, otherwise projecting it tilts the edge that leaves the wall
        touches = q0 <= tol / 6.0
        eligible = inside(t0, q0) and (touches or alongside(i, (i + 1) % n)
                                       or alongside((i - 1) % n, i))
        if eligible and key((x, y)) not in fixed:
            qpt = a + t0 * d
            out.append((float(qpt[0]), float(qpt[1])))
        else:
            out.append((x, y))
        # edge running alongside the trace and continuing well past a span
        # end: insert the jog at that end
        if max(q0, q1) <= tol and abs(t1 - t0) > 1e-9:
            lo, hi = min(t0, t1), max(t0, t1)
            crossings = []
            if lo < -tol and hi > 0.0:
                crossings.append(0.0)
            if hi > length + tol and lo < length:
                crossings.append(length)
            crossings.sort(key=lambda ts: abs(ts - t0))     # in edge direction
            for ts in crossings:
                f = (ts - t0) / (t1 - t0)
                if not (0.0 < f < 1.0):
                    continue
                ex, ey = x + f * (xn - x), y + f * (yn - y)   # point on the edge
                tr = a + ts * d                               # point on the trace
                on_trace = (float(tr[0]), float(tr[1]))
                on_edge = (float(ex), float(ey))
                fixed.add(key(on_edge))
                leaving = inside(t0, q0) or (-tol <= t0 <= length + tol)
                out.extend([on_trace, on_edge] if leaving else [on_edge, on_trace])
    cleaned = []
    for pt in out:
        if not cleaned or math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(pt)
    if len(cleaned) > 1 and math.hypot(cleaned[0][0] - cleaned[-1][0],
                                       cleaned[0][1] - cleaned[-1][1]) <= 1e-9:
        cleaned.pop()
    if len(cleaned) < 3:
        return ring, fixed
    return LinearRing(cleaned), fixed


def _clean_polygon(poly: Polygon, tol: float) -> Polygon:
    """Drop zero-width spikes / hairs left by snapping (buffer(0)); keep the
    largest part if the outline fell apart."""
    try:
        fixed = poly.buffer(0)
    except Exception:
        return poly
    if fixed.is_empty:
        return poly
    if fixed.geom_type == "MultiPolygon":
        fixed = max(fixed.geoms, key=lambda g: g.area)
    if fixed.geom_type != "Polygon":
        return poly
    return fixed


def _seg_intersection(p1, p2, q1, q2, tol: float):
    """Intersection point of segments p1-p2 and q1-q2 (plan), accepting hits
    within ``tol`` beyond either segment's ends (tolerant, unlike shapely's
    exact predicates which miss corners reconciled to 1e-15 gaps)."""
    p1 = np.asarray(p1[:2], float); p2 = np.asarray(p2[:2], float)
    q1 = np.asarray(q1[:2], float); q2 = np.asarray(q2[:2], float)
    d = p2 - p1
    e = q2 - q1
    denom = d[0] * e[1] - d[1] * e[0]
    lp, le = float(np.hypot(*d)), float(np.hypot(*e))
    if lp < 1e-12 or le < 1e-12 or abs(denom) < 1e-9 * lp * le:
        return None
    r = q1 - p1
    t = (r[0] * e[1] - r[1] * e[0]) / denom
    u = (r[0] * d[1] - r[1] * d[0]) / denom
    if -tol / lp <= t <= 1 + tol / lp and -tol / le <= u <= 1 + tol / le:
        return p1 + t * d
    return None


def _extend_dangles(traces: list, others, bounds, tol: float) -> list:
    """For each trace endpoint not touching any other linework, extend the
    trace along its direction until it meets the nearest other line (cut
    line only; the wall itself keeps its true length). Without this a wall
    ending inside a slab bay is a dangle that polygonize drops, leaving the
    slab un-partitioned along the wall."""
    out = []
    diag = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]) + 1.0
    for tr in traces:
        (x1, y1), (x2, y2) = tr.coords[0], tr.coords[-1]
        d = np.array([x2 - x1, y2 - y1])
        L = float(np.hypot(*d))
        if L < 1e-9:
            continue
        d /= L
        for end, sign in (((x1, y1), -1.0), ((x2, y2), +1.0)):
            p = Point(end)
            if others.distance(p) <= tol:
                continue                                    # already touching
            ray = LineString([end, (end[0] + sign * d[0] * diag, end[1] + sign * d[1] * diag)])
            hit = ray.intersection(others)
            if hit.is_empty:
                continue
            pts = [hit] if hit.geom_type == "Point" else [g for g in getattr(hit, "geoms", [])
                                                           if g.geom_type == "Point"]
            if not pts:
                continue
            q = min(pts, key=lambda g: p.distance(g))
            if p.distance(q) > tol:
                out.append(LineString([end, (q.x, q.y)]))
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def assemble_building(ctx: IfcContext, system: SystemRecord,
                      config: ConversionConfig,
                      resolver: MaterialResolver) -> AnalyticalModel:
    # node identity in a building is physically meaningless below ~1 mm;
    # flooring the pool tolerance there absorbs authoring jitter that the
    # file's nominal precision does not admit to
    merge_tol = max(config.resolve_merge_tol(ctx.precision), 1e-3)
    snap_tol = config.resolve_snap_tol(merge_tol)
    model = AnalyticalModel(name=system.name, domain="building",
                            nodes=NodePool(merge_tol))
    model.meta.update(source=ctx.path, schema=ctx.schema, merge_tol=merge_tol)

    beams: list[RawBeam] = []
    columns: list[RawColumn] = []
    walls: list[RawWall] = []
    slabs: list[RawSlab] = []

    # -- interpretation pass --------------------------------------------------
    for guid in system.product_guids:
        rec = ctx.products[guid]
        name_l = rec.name.lower()
        if rec.ifc_class == "IfcFooting" or (rec.ifc_class == "IfcSlab" and "footing" in name_l):
            if config.footing_policy == "support":
                ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.EXCLUDED,
                                 detail="footing idealized as rigid support (footing_policy="
                                        "'support'); column/wall base fixity represents it",
                                 system=system.name)
                continue
        obj = None
        if rec.ifc_class in ("IfcBeam", "IfcMember"):
            obj = _beam_from(rec, ctx)
            if obj:
                beams.append(obj)
        elif rec.ifc_class == "IfcColumn":
            obj = _column_from(rec, ctx)
            if obj:
                columns.append(obj)
        elif rec.ifc_class == "IfcWall":
            obj = _wall_from(rec, ctx)
            if obj:
                walls.append(obj)
        elif rec.ifc_class in ("IfcSlab", "IfcPlate", "IfcRoof"):
            obj = _slab_from(rec, ctx)
            if obj:
                slabs.append(obj)
        if obj is None:
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail="no interpretable representation for building conversion",
                             system=system.name)

    # -- step 1b: reconcile column stacks — per-storey column pieces authored
    #    at one gridpoint (with independent placement noise) collapse onto a
    #    single shared plan position, so the analytical column axis is
    #    continuous across storeys and beams snap to one line, not many -----
    clusters: list[list[RawColumn]] = []
    for c in columns:
        for cl in clusters:
            if math.hypot(c.base[0] - cl[0].base[0],
                          c.base[1] - cl[0].base[1]) <= max(snap_tol, c.plan_halfwidth):
                cl.append(c)
                break
        else:
            clusters.append([c])
    for cl in clusters:
        if len(cl) > 1:
            mx = sum(c.base[0] for c in cl) / len(cl)
            my = sum(c.base[1] for c in cl) / len(cl)
            for c in cl:
                c.base[0] = c.top[0] = mx
                c.base[1] = c.top[1] = my
                c.evidence["stack"] = f"plan position reconciled over {len(cl)} pieces"

    # -- step 1c: wall-corner reconciliation — a wall whose end FACE abuts
    #    another wall's face stops half that wall's thickness short of its
    #    mid-plane (mid-surface idealization gap). Extend the analytical
    #    trace to the mid-plane intersection under the same face-gap
    #    criterion used for beam endpoints, so corner traces intersect and
    #    the wall partitioning nodes them together.
    n_wall_ext = 0
    for w in walls:
        d = np.array([w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]], dtype=float)
        length = float(np.hypot(*d))
        if length < LEVEL_TOL:
            continue
        d /= length
        for endpoint, sign in ((w.p1, -1.0), (w.p2, +1.0)):
            out_dir = sign * d                      # outward along the trace
            best = None                             # (s, other-wall name)
            for w2 in walls:
                if w2 is w:
                    continue
                if min(w.z_hi, w2.z_hi) - max(w.z_lo, w2.z_lo) <= LEVEL_TOL:
                    continue
                e = np.array([w2.p2[0] - w2.p1[0], w2.p2[1] - w2.p1[1]],
                             dtype=float)
                len2 = float(np.hypot(*e))
                if len2 < LEVEL_TOL:
                    continue
                e /= len2
                denom = out_dir[0] * e[1] - out_dir[1] * e[0]
                if abs(denom) < 0.1:                # near-parallel: no corner
                    continue
                rel = np.array([w2.p1[0] - endpoint[0],
                                w2.p1[1] - endpoint[1]])
                s = (rel[0] * e[1] - rel[1] * e[0]) / denom
                t = (rel[0] * out_dir[1] - rel[1] * out_dir[0]) / denom
                reach = w.thickness / 2.0 + snap_tol
                if not (merge_tol < s <= w2.thickness / 2.0 + snap_tol):
                    continue                        # extension only, face-gap bound
                if not (-reach <= t <= len2 + reach):
                    continue                        # must land on w2's extent
                if best is None or s < best[0]:
                    best = (s, w2.rec.name)
            if best is not None:
                endpoint[0] += best[0] * out_dir[0]
                endpoint[1] += best[0] * out_dir[1]
                w.evidence["corner"] = (f"end extended {best[0]:.3f} m to "
                                        f"mid-plane of '{best[1]}' (face-gap)")
                n_wall_ext += 1
    if n_wall_ext:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_wall_ext} wall end(s) extended to "
                        "intersecting wall mid-planes [mid-surface corner "
                        "reconciliation, face-gap criterion]")

    # -- step 1d: wall ends abutting a column / beam face -> that member's
    #    axis. A wall between columns (infill shear wall, boundary element)
    #    stops at the column FACE; the analytical wall edge must coincide
    #    with the column axis so the column line is embedded in the wall edge.
    n_wall_mem = 0
    for w in walls:
        d = np.array([w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]], dtype=float)
        length = float(np.hypot(*d))
        if length < LEVEL_TOL:
            continue
        d /= length
        for endpoint, sign in ((w.p1, -1.0), (w.p2, +1.0)):
            out_dir = sign * d
            best = None                             # (|face gap|, target xy, why)
            for c in columns:
                if min(w.z_hi, c.top[2]) - max(w.z_lo, c.base[2]) <= LEVEL_TOL:
                    continue
                rel = np.array([c.base[0] - endpoint[0], c.base[1] - endpoint[1]])
                s_al = float(rel @ out_dir)
                lat = float(np.hypot(*(rel - s_al * out_dir)))
                if lat > snap_tol:
                    continue                        # axis not on the mid-plane
                if not (-(c.plan_halfwidth + snap_tol) <= s_al <= c.plan_halfwidth + snap_tol):
                    continue
                gap = abs(abs(s_al) - c.plan_halfwidth)
                if gap <= snap_tol and (best is None or gap < best[0]):
                    best = (gap, (c.base[0], c.base[1]),
                            f"end -> column axis '{c.rec.name}'")
            if best is None:
                for b in beams:
                    b_lo = min(b.start[2], b.end[2]) - b.depth / 2.0
                    b_hi = max(b.start[2], b.end[2]) + b.depth / 2.0
                    if min(w.z_hi, b_hi) - max(w.z_lo, b_lo) <= LEVEL_TOL:
                        continue
                    half_w = max(getattr(b.section, "width", 0.0), 0.05) / 2.0
                    probe = np.array([endpoint[0], endpoint[1]]) + out_dir * (half_w + snap_tol)
                    hit = _seg_intersection(endpoint, probe, b.start, b.end, snap_tol)
                    if hit is None:
                        continue
                    s_al = float((hit - np.array([endpoint[0], endpoint[1]])) @ out_dir)
                    if s_al <= merge_tol:
                        continue                    # extension only
                    gap = abs(s_al - half_w)
                    if gap <= snap_tol and (best is None or gap < best[0]):
                        best = (gap, (float(hit[0]), float(hit[1])),
                                f"end -> beam axis '{b.rec.name}'")
            if best is not None:
                endpoint[0], endpoint[1] = best[1]
                w.evidence["end"] = f"{best[2]} (face gap {best[0]:.4f})"
                n_wall_mem += 1
    if n_wall_mem:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_wall_mem} wall end(s) reconciled to "
                        "abutting column / beam axes [face-gap criterion]")
    # wall traces on the node-identity grid: exact-arithmetic noding (shapely)
    # must see reconciled corners as coincident, not 1e-15 apart
    for w in walls:
        for pt in (w.p1, w.p2):
            pt[0] = round(pt[0] / merge_tol) * merge_tol
            pt[1] = round(pt[1] / merge_tol) * merge_tol

    # -- step 2: snap beam endpoints to the analytical axis of the member
    #    whose FACE the end touches (column / wall / supporting beam).
    #    Candidates are scored by |face gap| — the distance from the endpoint
    #    to the target's physical face — because a beam frames into the member
    #    its end face abuts, not merely the nearest analytical plane. --------
    for b in beams:
        for pt in (b.start, b.end):
            candidates = []   # (|face_gap|, snapped_xy, evidence)
            for c in columns:
                r = math.hypot(pt[0] - c.base[0], pt[1] - c.base[1])
                if r <= c.plan_halfwidth + snap_tol:
                    candidates.append((abs(r - c.plan_halfwidth),
                                       (c.base[0], c.base[1]),
                                       "endpoint -> column axis"))
            for w in walls:
                if not (w.z_lo - snap_tol <= pt[2] <= w.z_hi + snap_tol):
                    continue
                tr = w.trace2d()
                p2 = Point(pt[0], pt[1])
                d = tr.distance(p2)
                if d <= w.thickness / 2.0 + snap_tol:
                    near = tr.interpolate(tr.project(p2))
                    candidates.append((abs(d - w.thickness / 2.0),
                                       (near.x, near.y),
                                       "endpoint -> wall mid-plane"))
            b_dir = np.array([b.end[0] - b.start[0], b.end[1] - b.start[1]])
            b_len = float(np.hypot(*b_dir))
            for ob in beams:
                if ob is b:
                    continue
                oz = (ob.start[2] + ob.end[2]) / 2.0
                if abs(oz - pt[2]) > max(0.3, snap_tol):
                    continue
                # a collinear neighbour (the next span of the same beam line)
                # cannot support this end; only a crossing / skewed beam can
                o_dir = np.array([ob.end[0] - ob.start[0], ob.end[1] - ob.start[1]])
                o_len = float(np.hypot(*o_dir))
                if b_len > 1e-9 and o_len > 1e-9:
                    cosang = abs(float(b_dir @ o_dir)) / (b_len * o_len)
                    if cosang > math.cos(math.radians(10.0)):
                        continue
                line = LineString([(ob.start[0], ob.start[1]), (ob.end[0], ob.end[1])])
                p2 = Point(pt[0], pt[1])
                d = line.distance(p2)
                half_w = max(getattr(ob.section, "width", 0.0), 0.05) / 2.0
                if 0 < d <= half_w + snap_tol:
                    near = line.interpolate(line.project(p2))
                    candidates.append((abs(d - half_w),
                                       (near.x, near.y),
                                       "endpoint -> supporting beam axis"))
            if candidates:
                # an end inside a column footprint frames into that column:
                # columns take precedence, then walls, then supporting beams
                rank = {"endpoint -> column axis": 0, "endpoint -> wall mid-plane": 1,
                        "endpoint -> supporting beam axis": 2}
                gap, (x, y), why = min(candidates, key=lambda c: (rank[c[2]], c[0]))
                pt[0], pt[1] = x, y
                b.evidence["snap"] = f"{why} (face gap {gap:.4f})"

    # -- step 3: lift beams into slab centroidal planes -----------------------
    n_lifted = 0
    for b in beams:
        z_c = (b.start[2] + b.end[2]) / 2.0
        for s in slabs:
            slab_bottom = s.z_mid - s.thickness / 2.0
            if abs((z_c + b.depth / 2.0) - slab_bottom) <= max(snap_tol, config.beam_lift_tolerance):
                delta = s.z_mid - z_c
                b.start[2] = b.end[2] = s.z_mid
                b.lifted_offset = -delta
                b.evidence["elevation"] = (f"lifted to slab mid-plane z={s.z_mid:g} "
                                           f"(delta={delta:g}, SECOFFSET compensated)")
                n_lifted += 1
                break
    if n_lifted:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_lifted} beam axes lifted into slab centroidal "
                        "planes with SECOFFSET compensation")

    # -- levels: cluster beam elevations + slab mid-planes --------------------
    level_zs: list[float] = []
    for z in ([s.z_mid for s in slabs]
              + [b.start[2] for b in beams if abs(b.start[2] - b.end[2]) < LEVEL_TOL]):
        if not any(abs(z - lz) < max(LEVEL_TOL, snap_tol) for lz in level_zs):
            level_zs.append(z)
    level_zs.sort()

    # -- step 3b: wall vertical extents -> slab planes. A wall top or base
    #    within the slab thickness of a level plane (walls modelled level-to-
    #    level, or to the soffit) is embedded in that slab: the analytical
    #    edge lands ON the mid-plane. Beyond that, a top up to 0.5 m below a
    #    plane / a base up to 0.5 m above one (walls stopping at a soffit or
    #    starting on a slab top) is extended, mirroring column reach.
    slab_half: dict[float, float] = defaultdict(float)
    for s in slabs:
        lz = next((z for z in level_zs if abs(z - s.z_mid) < max(LEVEL_TOL, snap_tol)), None)
        if lz is not None:
            slab_half[lz] = max(slab_half[lz], s.thickness / 2.0)
    WALL_REACH = 0.5
    n_wall_z = 0
    for w in walls:
        top_c = [z for z in level_zs
                 if abs(z - w.z_hi) <= slab_half[z] + snap_tol or w.z_hi < z <= w.z_hi + WALL_REACH]
        if top_c:
            z = min(top_c)
            if abs(z - w.z_hi) > LEVEL_TOL:
                w.evidence["top"] = f"top {w.z_hi:g} -> slab plane {z:g}"
                w.z_hi = z
                n_wall_z += 1
        base_c = [z for z in level_zs
                  if abs(z - w.z_lo) <= slab_half[z] + snap_tol or w.z_lo - WALL_REACH <= z < w.z_lo]
        if base_c:
            z = max(base_c)
            if abs(z - w.z_lo) > LEVEL_TOL and z < w.z_hi - LEVEL_TOL:
                w.evidence["base"] = f"base {w.z_lo:g} -> slab plane {z:g}"
                w.z_lo = z
                n_wall_z += 1
    if n_wall_z:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_wall_z} wall top/base edge(s) reconciled to "
                        "slab mid-planes")

    member_id = 0
    surface_id = 0

    def add_member(rec, n1, n2, sec_id, mat_id, ev, orient=None):
        nonlocal member_id
        member_id += 1
        prov = Provenance(rec.guid, rec.ifc_class, rec.name, dict(ev))
        model.members.append(Member1D(member_id, prov, n1, n2, "BEAM188",
                                      sec_id, mat_id, orient=orient))

    def add_surface(rec, loop_kps, sec_id, mat_id, ev):
        nonlocal surface_id
        surface_id += 1
        from ..model.ir import Surface2D
        prov = Provenance(rec.guid, rec.ifc_class, rec.name, dict(ev))
        model.surfaces.append(Surface2D(surface_id, prov, loop_kps, "SHELL181",
                                        sec_id, mat_id))

    # -- step 4: per-level planar arrangement ---------------------------------
    converted: set[str] = set()
    horiz: list[RawBeam] = []
    incl: list[RawBeam] = []
    for b in beams:
        (horiz if abs(b.start[2] - b.end[2]) < LEVEL_TOL else incl).append(b)

    # -- step 4a: brace workpoint reconciliation ------------------------------
    #    an inclined member's end must land on the analytical object it
    #    braces: a column axis, or a beam axis in its (lifted) level plane.
    #    Beam attachment points are injected into the level arrangement so
    #    the beam is noded there (conformal chevron/V apexes).
    injections: dict[float, list[tuple[float, float]]] = defaultdict(list)
    n_reconciled = 0
    for b in incl:
        for pt in (b.start, b.end):
            snapped = False
            for c in columns:
                r = math.hypot(pt[0] - c.base[0], pt[1] - c.base[1])
                if (r <= c.plan_halfwidth + snap_tol
                        and c.base[2] - 0.5 <= pt[2] <= c.top[2] + 0.5):
                    pt[0], pt[1] = c.base[0], c.base[1]
                    b.evidence["workpoint"] = "brace end -> column axis"
                    snapped = True
                    break
            if snapped:
                n_reconciled += 1
                continue
            best = None
            for hb in horiz:
                lz = hb.start[2]
                z_orig = lz + hb.lifted_offset        # centroid before the slab lift
                if not (abs(pt[2] - lz) <= snap_tol or abs(pt[2] - z_orig) <= snap_tol):
                    continue
                line = LineString([(hb.start[0], hb.start[1]), (hb.end[0], hb.end[1])])
                d = line.distance(Point(pt[0], pt[1]))
                if d <= snap_tol and (best is None or d < best[0]):
                    q = line.interpolate(line.project(Point(pt[0], pt[1])))
                    best = (d, q.x, q.y, lz)
            if best is not None:
                pt[0], pt[1], pt[2] = best[1], best[2], best[3]
                b.evidence["workpoint"] = "brace end -> beam axis plane"
                lz_key = next((z for z in level_zs
                               if abs(z - best[3]) < max(LEVEL_TOL, snap_tol)), best[3])
                injections[lz_key].append((best[1], best[2]))
                n_reconciled += 1
    if n_reconciled:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_reconciled} inclined-member workpoint(s) "
                        "reconciled to column axes / beam planes")
    # -- step 4b: junction noding of inclined members. X-bracing (one
    #    continuous diagonal, one interrupted at the crossing) and braces
    #    meeting at gussets must share the common point: split every inclined
    #    member at any other inclined member's endpoint lying on its interior.
    incl_pts = [np.array(p) for b in incl for p in (b.start, b.end)]
    n_junction = 0
    for b in incl:
        a, e = np.array(b.start), np.array(b.end)
        d = e - a
        L2 = float(d @ d)
        stations = []
        if L2 > 0:
            L = math.sqrt(L2)
            for q in incl_pts:
                t = float((q - a) @ d) / L2
                if merge_tol / L < t < 1 - merge_tol / L:
                    perp = float(np.linalg.norm(q - (a + t * d)))
                    if perp <= max(snap_tol, merge_tol * 10):
                        stations.append(t)
        mat = resolver.resolve(b.rec, "building", "beam")
        mat_id = model.add_material(mat)
        sec_id = model.add_section(b.section)
        ts = [0.0] + sorted(set(round(t, 9) for t in stations)) + [1.0]
        pts = [a + t * d for t in ts]
        if len(ts) > 2:
            b.evidence["junction"] = f"noded at {len(ts) - 2} crossing/gusset point(s)"
            n_junction += len(ts) - 2
        for pa, pb in zip(pts, pts[1:]):
            n1 = model.nodes.get(tuple(pa))
            n2 = model.nodes.get(tuple(pb))
            if n1 != n2:
                add_member(b.rec, n1, n2, sec_id, mat_id, b.evidence)
                converted.add(b.rec.guid)
    if n_junction:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_junction} inclined-member junction(s) noded "
                        "(crossing / interrupted diagonals)")

    beams_by_level: dict[float, list[RawBeam]] = defaultdict(list)
    for b in horiz:
        lz = next((z for z in level_zs if abs(z - b.start[2]) < max(LEVEL_TOL, snap_tol)),
                  round(b.start[2], 4))
        beams_by_level[lz].append(b)

    all_levels = sorted(set(level_zs) | set(beams_by_level))
    #: wall id -> trace stations of arrangement vertices (become panel cuts)
    wall_stations: dict[int, set[float]] = defaultdict(set)
    for lz in all_levels:
        lv_beams = beams_by_level.get(lz, [])
        lv_slabs = [s for s in slabs if abs(s.z_mid - lz) < max(LEVEL_TOL, snap_tol)]
        if not lv_beams and not lv_slabs:
            continue
        beam_lines = [LineString([(b.start[0], b.start[1]), (b.end[0], b.end[1])])
                      for b in lv_beams]
        # walls whose (reconciled) span reaches this plane
        lv_walls = [w for w in walls if w.z_lo - snap_tol <= lz <= w.z_hi + snap_tol]
        wall_traces = [w.trace2d() for w in lv_walls]
        # brace apexes split their host beam line so the arrangement nodes there
        inj = injections.get(lz, [])
        geoms: list = []
        for bl in beam_lines:
            on = [Point(q) for q in inj if bl.distance(Point(q)) <= merge_tol * 10]
            if on:
                ds = sorted({0.0, bl.length, *(bl.project(p) for p in on)})
                geoms.extend(substring(bl, a, e) for a, e in zip(ds, ds[1:])
                             if e - a > merge_tol)
            else:
                geoms.append(bl)
        geoms += wall_traces
        # column gridlines partition slabs that bear directly on columns
        # (flat-slab equivalent-frame idealization); where beams already run
        # on the grid these lines are absorbed by the union
        lv_cols = [c for c in columns if c.base[2] - LEVEL_TOL <= lz <= c.top[2] + 0.5]
        if lv_slabs and lv_cols:
            minx = min(s.poly.bounds[0] for s in lv_slabs) - 0.1
            miny = min(s.poly.bounds[1] for s in lv_slabs) - 0.1
            maxx = max(s.poly.bounds[2] for s in lv_slabs) + 0.1
            maxy = max(s.poly.bounds[3] for s in lv_slabs) + 0.1
            for x in sorted({round(c.base[0], 6) for c in lv_cols}):
                geoms.append(LineString([(x, miny), (x, maxy)]))
            for y in sorted({round(c.base[1], 6) for c in lv_cols}):
                geoms.append(LineString([(minx, y), (maxx, y)]))
        for s in lv_slabs:
            # snap the slab boundary AND opening rings onto wall mid-planes and
            # beam axes so the arrangement produces shared nodes along those
            # interfaces (the architectural slab edge / shaft edge sits at the
            # wall FACE, half a thickness off the analytical mid-plane)
            def _snap(ring):
                fixed: set = set()
                for w in lv_walls:
                    ring, fixed = _snap_ring_to_trace(ring, w.p1, w.p2,
                                                      w.thickness / 2.0 + snap_tol, fixed)
                for bl in beam_lines:
                    ring = _snap_ring_to_line(ring, bl, snap_tol)
                return ring
            boundary = _snap(s.poly.exterior)
            holes = [list(_snap(r).coords) for r in s.poly.interiors]
            s.poly = _clean_polygon(Polygon(boundary, holes), merge_tol)
            geoms.append(s.poly.exterior)
            geoms.extend(s.poly.interiors)
        # 1 mm precision model: reconciled corners meeting to 1e-15 must node
        geoms = [shapely.set_precision(g, merge_tol) for g in geoms]
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        network = unary_union(geoms)
        # a wall trace ending inside a slab bay is a dangle polygonize would
        # drop: extend such ends (cut lines only) to the next line
        if wall_traces and lv_slabs:
            bounds = unary_union([s.poly for s in lv_slabs]).bounds
            # stacked storeys contribute identical traces at a shared plane:
            # keep one of each, or every trace "touches" its twin and no
            # dangle is ever extended
            seen: set = set()
            snapped_traces = []
            for t in wall_traces:
                t = shapely.set_precision(t, merge_tol)
                if t.is_empty or t.geom_type != "LineString":
                    continue
                c = [tuple(round(v, 4) for v in pt) for pt in t.coords]
                k = tuple(sorted((c[0], c[-1])))
                if k in seen:
                    continue
                seen.add(k)
                snapped_traces.append(t)
            cuts = []
            for i, tr in enumerate(snapped_traces):
                if tr.is_empty or tr.geom_type != "LineString":
                    continue
                if not any(s.poly.buffer(snap_tol).intersects(tr) for s in lv_slabs):
                    continue
                twins = [shapely.set_precision(t, merge_tol) for t in wall_traces]
                others = unary_union([g for g in geoms if not any(g.equals(t) for t in twins)]
                                     + [t for j, t in enumerate(snapped_traces) if j != i])
                cuts.extend(_extend_dangles([tr], others, bounds, merge_tol * 2))
            if cuts:
                network = unary_union([network] + [shapely.set_precision(c, merge_tol)
                                                   for c in cuts])
        # every arrangement vertex on a wall trace is a wall partition station
        for seg in getattr(network, "geoms", [network]):
            if seg.geom_type != "LineString":
                continue
            for x, y in seg.coords:
                for w in lv_walls:
                    tr = w.trace2d()
                    if tr.distance(Point(x, y)) <= merge_tol * 2:
                        wall_stations[id(w)].add(round(tr.project(Point(x, y)), 6))

        # beam segments from the noded network
        segments = list(getattr(network, "geoms", [network]))
        for seg in segments:
            if seg.geom_type != "LineString":
                continue
            coords = list(seg.coords)
            for a, bpt in zip(coords, coords[1:]):
                mid = ((a[0] + bpt[0]) / 2.0, (a[1] + bpt[1]) / 2.0)
                owner = None
                for b, bl in zip(lv_beams, beam_lines):
                    if bl.distance(Point(mid)) < merge_tol * 10:
                        owner = b
                        break
                if owner is None:
                    continue                    # wall-trace or slab-edge piece
                mat = resolver.resolve(owner.rec, "building", "beam")
                mat_id = model.add_material(mat)
                sec = owner.section
                if owner.lifted_offset:
                    import copy
                    sec = copy.copy(sec)
                    sec.offset_y = owner.lifted_offset
                sec_id = model.add_section(sec)
                n1 = model.nodes.get((a[0], a[1], lz))
                n2 = model.nodes.get((bpt[0], bpt[1], lz))
                if n1 != n2:
                    add_member(owner.rec, n1, n2, sec_id, mat_id, owner.evidence)
                    converted.add(owner.rec.guid)

        # slab bay panels from the same arrangement
        for s in lv_slabs:
            mat = resolver.resolve(s.rec, "building", "shell")
            mat_id = model.add_material(mat)
            sec_id = model.add_section(
                ShellSection(name=s.rec.name[:8] or "slab", layers=[(s.thickness, mat_id)]))
            panels = [g for g in polygonize(network) if s.poly.buffer(snap_tol).contains(
                g.representative_point())]
            # ring-shaped faces (perimeter overhang around the beam grid) must
            # be subdivided into simple panels: cut radially from each interior
            # ring vertex to the exterior ring, then re-polygonize locally
            simple: list = []
            for panel in panels:
                if not panel.interiors:
                    simple.append(panel)
                    continue
                cuts = [panel.exterior]
                cuts.extend(panel.interiors)
                for ring in panel.interiors:
                    for x, y in list(ring.coords)[:-1]:
                        p = Point(x, y)
                        q = panel.exterior.interpolate(panel.exterior.project(p))
                        cuts.append(LineString([(x, y), (q.x, q.y)]))
                for sub in polygonize(unary_union(cuts)):
                    if panel.buffer(snap_tol).contains(sub.representative_point()):
                        simple.append(sub)
            for panel in simple:
                coords = list(panel.exterior.coords)[:-1]
                kps = [model.nodes.get((x, y, lz)) for x, y in coords]
                kps = [k for i, k in enumerate(kps) if k != kps[i - 1]]
                if len(kps) >= 3:
                    add_surface(s.rec, kps, sec_id, mat_id,
                                {**s.evidence, "partition": f"bay panel of {len(simple)}"})
            converted.add(s.rec.guid)

    # -- step 5: wall partitioning --------------------------------------------
    #    Only walls standing on the structure's lowest support level contribute
    #    base edges to the fixity (foundation walls may reach up to
    #    SUPPORT_REACH below/above the column bases). Per-storey walls stacked
    #    on upper levels rest on the slab below, not on the ground - fixing
    #    their base edges would clamp the building at every floor
    #    (defect found by verification_tests/mesh_size/experiment_set_01:
    #    an 8-storey core tower lost its sway modes, f1 13.7 Hz instead of 3 Hz).
    SUPPORT_REACH = 0.5
    wall_base_edges: list[tuple[int, int]] = []
    support_z = min([c.base[2] for c in columns] + [w.z_lo for w in walls])         if (columns or walls) else 0.0
    for w in walls:
        on_ground = w.z_lo <= support_z + max(snap_tol, SUPPORT_REACH)
        mat = resolver.resolve(w.rec, "building", "shell")
        mat_id = model.add_material(mat)
        sec_id = model.add_section(
            ShellSection(name=w.rec.name[:8] or "wall", layers=[(w.thickness, mat_id)]))
        tr = w.trace2d()
        length = tr.length
        # vertical cuts: wall ends + crossing beam axes + other wall traces
        # (tolerant intersections) + other wall ends landing on this trace
        # (T-junctions) + slab boundary vertices + arrangement vertices
        us = {0.0, length}
        for b in beams:
            hit = _seg_intersection(w.p1, w.p2, b.start, b.end, snap_tol)
            if hit is not None:
                us.add(tr.project(Point(hit[0], hit[1])))
        for w2 in walls:
            if w2 is w:
                continue
            hit = _seg_intersection(w.p1, w.p2, w2.p1, w2.p2, snap_tol)
            if hit is not None:
                us.add(tr.project(Point(hit[0], hit[1])))
            for pt in (w2.p1, w2.p2):
                p = Point(pt[0], pt[1])
                if tr.distance(p) <= snap_tol:
                    us.add(tr.project(p))
        for s in slabs:
            for ring in (s.poly.exterior, *s.poly.interiors):
                for x, y in list(ring.coords):
                    p = Point(x, y)
                    if tr.distance(p) <= snap_tol:
                        us.add(tr.project(p))
        us.update(wall_stations.get(id(w), ()))
        # member attachment points on the wall (beams framing into the wall
        # at mid-height, coupling beams below the slab): cut at (u, z)
        vs = {w.z_lo, w.z_hi}
        for z in level_zs:
            if w.z_lo + LEVEL_TOL < z < w.z_hi - LEVEL_TOL:
                vs.add(z)
        for m in model.members:
            for nid in (m.start, m.end):
                x, y, z = model.nodes.xyz(nid)
                if w.z_lo - LEVEL_TOL <= z <= w.z_hi + LEVEL_TOL:
                    p = Point(x, y)
                    if tr.distance(p) <= merge_tol * 2:
                        us.add(tr.project(p))
                        if w.z_lo + LEVEL_TOL < z < w.z_hi - LEVEL_TOL:
                            vs.add(z)
        u_cuts = sorted(u for u in us if -LEVEL_TOL <= u <= length + LEVEL_TOL)
        u_cuts = [u_cuts[0]] + [u for a, u in zip(u_cuts, u_cuts[1:])
                                if u - a > max(LEVEL_TOL, merge_tol * 2)]
        if length - u_cuts[-1] > LEVEL_TOL:
            u_cuts.append(length)
        elif len(u_cuts) > 1:
            u_cuts[-1] = length
        z_hi = w.z_hi
        v_cuts = sorted(vs)
        for i in range(len(u_cuts) - 1):
            for j in range(len(v_cuts) - 1):
                pa = tr.interpolate(u_cuts[i])
                pb = tr.interpolate(u_cuts[i + 1])
                k1 = model.nodes.get((pa.x, pa.y, v_cuts[j]))
                k2 = model.nodes.get((pb.x, pb.y, v_cuts[j]))
                k3 = model.nodes.get((pb.x, pb.y, v_cuts[j + 1]))
                k4 = model.nodes.get((pa.x, pa.y, v_cuts[j + 1]))
                add_surface(w.rec, [k1, k2, k3, k4], sec_id, mat_id,
                            {**w.evidence,
                             "partition": f"panel u[{i}] v[{j}] of "
                                          f"{len(u_cuts) - 1}x{len(v_cuts) - 1}"})
                if j == 0 and on_ground:
                    wall_base_edges.append((k1, k2))
        converted.add(w.rec.guid)

    # -- columns: split at attachment elevations; extend the analytical axis
    #    to reach beam planes lifted slightly beyond the architectural extent -
    COLUMN_REACH = 0.5
    for c in columns:
        mat = resolver.resolve(c.rec, "building", "beam")
        mat_id = model.add_material(mat)
        sec_id = model.add_section(c.section)
        base_z, top_z = c.base[2], c.top[2]
        attach = []
        for m in model.members:
            for nid in (m.start, m.end):
                p = model.nodes.xyz(nid)
                if math.hypot(p[0] - c.base[0], p[1] - c.base[1]) <= merge_tol * 10:
                    if base_z - COLUMN_REACH <= p[2] <= top_z + COLUMN_REACH:
                        attach.append(p[2])
        # slabs bearing directly on the column (flat-slab framing): panel
        # corners at the column point are attachment elevations too
        for s2 in model.surfaces:
            for nid in s2.loop:
                p = model.nodes.xyz(nid)
                if math.hypot(p[0] - c.base[0], p[1] - c.base[1]) <= merge_tol * 10:
                    if base_z - COLUMN_REACH <= p[2] <= top_z + COLUMN_REACH:
                        attach.append(p[2])
        if attach:
            if max(attach) > top_z + LEVEL_TOL:
                c.evidence["extension"] = (f"top extended {max(attach) - top_z:.3f} m to the "
                                           "lifted beam plane")
                top_z = max(attach)
            if min(attach) < base_z - LEVEL_TOL:
                base_z = min(attach)
        zs = {base_z, top_z}
        zs.update(z for z in attach if base_z + LEVEL_TOL < z < top_z - LEVEL_TOL)
        z_cuts = sorted(zs)
        for z1, z2 in zip(z_cuts, z_cuts[1:]):
            n1 = model.nodes.get((c.base[0], c.base[1], z1))
            n2 = model.nodes.get((c.base[0], c.base[1], z2))
            if n1 != n2:
                add_member(c.rec, n1, n2, sec_id, mat_id,
                           {**c.evidence, "split": f"storey segment z[{z1:g},{z2:g}]"},
                           orient=c.orient)
        converted.add(c.rec.guid)

    # -- audit records --------------------------------------------------------
    for guid in system.product_guids:
        rec = ctx.products[guid]
        if guid in converted:
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.CONVERTED,
                             system=system.name)

    # -- step 5b: conformity enforcement ---------------------------------------
    from .conformity import enforce_conformity
    n_split, n_ins, edge_map = enforce_conformity(model, merge_tol * 2)
    if n_split or n_ins:
        ctx.audit.event("geometry",
                        f"{system.name}: conformity enforcement split {n_split} member(s) "
                        f"and refined {n_ins} surface edge(s) at coincident nodes")
    wall_base_edges = [e for k in wall_base_edges for e in edge_map(k)]

    # -- step 6: supports ------------------------------------------------------
    if config.bcs.building == "base-fixed" and (columns or wall_base_edges):
        # columns anchor at their own lowest base cluster; walls at their own
        # base edges — a single global minimum would leave the frame floating
        # when foundation walls reach deeper than the column bases
        col_nodes: list[int] = []
        if columns:
            col_base_z = min(c.base[2] for c in columns)
            col_nodes = sorted({model.nodes.get((c.base[0], c.base[1], c.base[2]))
                                for c in columns if abs(c.base[2] - col_base_z) < snap_tol})
        model.supports.append(Support(name="BASE_FIX", nodes=col_nodes,
                                      edges=wall_base_edges,
                                      source="heuristic:base-fixed"))
        ctx.audit.event("boundary-condition",
                        f"{system.name}: fixed base at {len(col_nodes)} column base node(s)"
                        + (f" (z={col_base_z:g})" if col_nodes else "")
                        + f" and {len(wall_base_edges)} wall base edge(s) "
                        "[heuristic base-fixed]", severity="warning")
    return model
