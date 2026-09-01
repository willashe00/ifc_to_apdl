"""Phase 3 — computational parameter extraction from tessellated solids.

Implements the manuscript's procedure for geometry that carries no design
parameters (IfcTriangulatedFaceSet / IfcPolygonalFaceSet):

  Step 1  decomposition: local frame from ObjectPlacement, mesh from the
          coordinate list
  Step 2  cross-section boundary extraction: slice the mesh with a plane
          normal to the member axis; boundary loops = section outline(s)
  Step 3  shape classification: circularity C = 4*pi*A/P^2, convexity
          chi = A/A_hull, reflection symmetry about the section axes,
          void indicator nu
  Step 4  measurement operators: coordinate extents, mean vertex radius,
          and the material-chord function c(y)

Hardening beyond the manuscript (validated in experiments/geom_extraction):
  - sections are sampled at several stations along the axis and parameters
    are taken as the median vote, which absorbs end effects and local
    tessellation defects;
  - section loops are projected onto the deterministic member section frame
    (x_hat, y_hat) rather than an arbitrary library projection frame, which
    made recovery orientation-dependent;
  - open (web-bearing) sections are canonicalized in-plane: aligned to the
    minimum-area bounding-rectangle axes, web set vertical by the minimal-
    central-chord rule, flange side set upward by the chord-dominance rule;
  - the web-to-flange transition height y* is refined by bisection instead
    of a fixed 60-step scan (removes a d/120 thickness quantization bias);
  - collinear slice vertices are removed before measurement: the slicing
    plane also crosses the diagonal edges of triangulated side facets,
    injecting mid-facet points that biased the median-vertex-radius
    operator by up to half the facet sagitta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

import numpy as np
from shapely.geometry import LineString, Polygon

from ..model.sections import (
    ChannelSection, CircSection, ISection, LSection, PipeSection,
    RectHollowSection, RectSection, Section, TSection,
)

SYM_TOL = 0.98        # manuscript tau_s
CIRC_TOL = 0.99
CONV_TOL = 0.99


@dataclass
class TessellatedResult:
    shape: str
    section: Section | None
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    stations: int = 0
    notes: list[str] = field(default_factory=list)


def mesh_from_faceset(item, length_scale: float, matrix: np.ndarray):
    """(vertices Nx3 [m, global], faces Mx3) from an IfcTriangulatedFaceSet."""
    coords = np.array(item.Coordinates.CoordList, dtype=float) * length_scale
    verts = (matrix[:3, :3] @ coords.T).T + matrix[:3, 3]
    faces = np.array(item.CoordIndex, dtype=int) - 1
    return verts, faces


def extract_section(verts: np.ndarray, faces: np.ndarray,
                    axis: np.ndarray | None = None,
                    n_stations: int = 5,
                    canonicalize: bool = True) -> TessellatedResult:
    """Recover the constant cross-section of a prismatic tessellated member."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    if axis is None:
        # PCA fallback: dominant extent direction is the member axis
        centered = verts - verts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    t = verts @ axis
    t_lo, t_hi = float(t.min()), float(t.max())
    length = t_hi - t_lo
    origin = verts.mean(axis=0)

    # section frame (x_hat, y_hat) perpendicular to the axis: y_hat carries
    # the vertical for horizontal-ish members; near-vertical members align
    # the section frame with the global plan axes (x_hat ~ +X, y_hat ~ +Y)
    if abs(axis[2]) < 0.9:
        x_hat = np.cross(np.array([0.0, 0.0, 1.0]), axis)
        x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(axis, x_hat)
    else:
        y_hat = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        y_hat /= np.linalg.norm(y_hat)
        x_hat = np.cross(y_hat, axis)

    stations = np.linspace(t_lo + 0.15 * length, t_hi - 0.15 * length, n_stations)
    votes: list[dict] = []
    for st in stations:
        plane_origin = origin + (st - float(origin @ axis)) * axis
        try:
            sec = mesh.section(plane_origin=plane_origin, plane_normal=axis)
            if sec is None:
                continue
            polys = _section_polygons(sec, plane_origin, x_hat, y_hat)
        except Exception:
            continue
        if not polys:
            continue
        poly = max(polys, key=lambda p: p.area)
        try:
            votes.append(_measure(poly, canonicalize=canonicalize))
        except Exception:
            continue                    # unmeasurable station: leave to the vote

    if not votes:
        return TessellatedResult("unknown", None, tuple(origin), tuple(origin),
                                 notes=["no valid section could be sliced"])

    shape = _majority([v["shape"] for v in votes])
    same = [v for v in votes if v["shape"] == shape]
    params = {k: median(v[k] for v in same) for k in same[0] if k != "shape"}
    section = _to_section(shape, params)

    p_start = origin + (t_lo - float(origin @ axis)) * axis
    p_end = origin + (t_hi - float(origin @ axis)) * axis
    return TessellatedResult(shape, section, tuple(p_start), tuple(p_end),
                             stations=len(same))


def _section_polygons(path3d, plane_origin: np.ndarray,
                      x_hat: np.ndarray, y_hat: np.ndarray) -> list[Polygon]:
    """Project the slice loops onto the member section frame and nest them.

    The projection frame is the deterministic (x_hat, y_hat) pair built from
    the member axis and the global vertical, so the recovered polygon carries
    the profile's as-modelled roll instead of an arbitrary library frame.
    """
    rings: list[np.ndarray] = []
    for line in path3d.discrete:
        pts = np.asarray(line, dtype=float)
        if len(pts) < 4:
            continue
        rel = pts - plane_origin
        rings.append(np.stack([rel @ x_hat, rel @ y_hat], axis=1))
    if not rings:
        return []
    loops = sorted((Polygon(r) for r in rings), key=lambda p: p.area, reverse=True)
    loops = [p if p.is_valid else p.buffer(0) for p in loops]
    outer = loops[0]
    holes = [p for p in loops[1:] if outer.contains(p.representative_point())]
    return [Polygon(outer.exterior.coords,
                    [h.exterior.coords for h in holes])]


def _majority(items: list[str]) -> str:
    return max(set(items), key=items.count)


def _measure(poly: Polygon, canonicalize: bool = True) -> dict:
    """Classify one section polygon and measure its parameters (manuscript
    Steps 3-4). Returns a vote dict with a 'shape' key."""
    # repair invalid (self-touching / crossing) loops from degraded meshes
    if not poly.is_valid:
        fixed = poly.buffer(0)
        if fixed.geom_type == "MultiPolygon":
            fixed = max(fixed.geoms, key=lambda g: g.area)
        if fixed.is_empty or fixed.geom_type != "Polygon":
            raise ValueError("unmeasurable section polygon")
        poly = fixed

    # drop collinear vertices injected by facet-diagonal crossings
    eps = 1e-9 * math.sqrt(max(poly.area, 1e-30))
    simp = poly.simplify(eps)
    if simp.is_valid and not simp.is_empty and simp.area > 0:
        poly = simp

    # center on the area centroid
    cx, cy = poly.centroid.x, poly.centroid.y
    from shapely.affinity import translate
    poly = translate(poly, -cx, -cy)

    if canonicalize:
        poly = _align_minimal(poly)

    # classification measures are evaluated on the OUTER boundary loop
    # (manuscript: "all classification measures are evaluated on S")
    outer = Polygon(poly.exterior.coords)
    A = outer.area
    P = outer.exterior.length
    nu = len(poly.interiors) > 0
    C = 4 * math.pi * A / (P * P)
    chi = A / outer.convex_hull.area

    if C >= CIRC_TOL:
        r_out = median(math.hypot(x, y) for x, y in poly.exterior.coords)
        if nu:
            r_in = median(math.hypot(x, y) for x, y in poly.interiors[0].coords)
            return {"shape": "hollow_circle", "od": 2 * r_out, "t": r_out - r_in}
        return {"shape": "circle", "d": 2 * r_out}

    if chi >= CONV_TOL:
        xs = [p[0] for p in poly.exterior.coords]
        ys = [p[1] for p in poly.exterior.coords]
        b_ext = max(xs) - min(xs)
        d_ext = max(ys) - min(ys)
        if nu:
            t = _chord(poly, 0.0) / 2.0
            return {"shape": "hollow_rect", "b": b_ext, "d": d_ext, "t": t}
        return {"shape": "rect", "b": b_ext, "d": d_ext}

    # open (web-bearing) section: set the web vertical and the flange up
    if canonicalize:
        poly = _orient_open_section(poly)
    xs = [p[0] for p in poly.exterior.coords]
    ys = [p[1] for p in poly.exterior.coords]
    b_ext = max(xs) - min(xs)
    d_ext = max(ys) - min(ys)

    sym_x = _symmetric(poly, "x")
    sym_y = _symmetric(poly, "y")
    tw = _chord(poly, 0.0)
    y_star = _flange_transition(poly, tw, d_ext)
    tf = max(ys) - y_star if y_star is not None else tw

    if sym_x and sym_y:
        return {"shape": "i", "b": b_ext, "d": d_ext, "tw": tw, "tf": tf}
    if sym_y and not sym_x:
        return {"shape": "t", "b": b_ext, "d": d_ext, "tw": tw, "tf": tf}
    if sym_x and not sym_y:
        return {"shape": "channel", "b": b_ext, "d": d_ext, "tw": tw, "tf": tf}
    return {"shape": "angle", "b1": d_ext, "b2": b_ext, "t": tw}


def _mrr_angle(poly: Polygon) -> float:
    """Orientation [deg] of the minimum-area bounding rectangle's first edge."""
    mrr = poly.minimum_rotated_rectangle
    if mrr.is_empty or not hasattr(mrr, "exterior"):
        return 0.0
    (x0, y0), (x1, y1) = list(mrr.exterior.coords)[:2]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _align_minimal(poly: Polygon) -> Polygon:
    """Rotate by the smallest angle that aligns the minimum-area bounding
    rectangle with the frame axes (delta constrained to (-45, 45] deg), so
    a rolled profile is squared up without discarding the roll reference."""
    from shapely.affinity import rotate
    theta = _mrr_angle(poly)
    delta = -theta % 90.0
    if delta > 45.0:
        delta -= 90.0
    if abs(delta) < 1e-12:
        return poly
    return rotate(poly, delta, origin=(0, 0))


def _orient_open_section(poly: Polygon) -> Polygon:
    """Canonical in-plane orientation for web-bearing sections.

    Among the four axis-aligned rotations, the web is vertical where the
    central chord c(0) is smallest; the flange side is set upward where the
    dominant chord lies (validated on I / T / channel / angle families).
    """
    from shapely.affinity import rotate
    best, best_chord = poly, _chord(poly, 0.0)
    for k in (90.0, 180.0, 270.0):
        cand = rotate(poly, k, origin=(0, 0))
        c = _chord(cand, 0.0)
        if c < best_chord - 1e-12:
            best, best_chord = cand, c
    ys = [p[1] for p in best.exterior.coords]
    half = (max(ys) - min(ys)) / 2.0
    grid = [half * (i + 1) / 20.0 for i in range(20)]
    top = max(_chord(best, y) for y in grid)
    bot = max(_chord(best, -y) for y in grid)
    if bot > top * (1.0 + 1e-9):
        best = rotate(best, 180.0, origin=(0, 0))
    return best


def _symmetric(poly: Polygon, about: str) -> bool:
    from shapely.affinity import scale
    mirrored = scale(poly, xfact=1 if about == "x" else -1,
                     yfact=-1 if about == "x" else 1, origin=(0, 0))
    inter = poly.intersection(mirrored).area
    return inter / poly.area >= SYM_TOL


def _chord(poly: Polygon, y: float) -> float:
    """Total material width c(y): length of the horizontal line at height y
    inside the section (manuscript Eq. geometry_integration)."""
    xs = [p[0] for p in poly.exterior.coords]
    span = max(xs) - min(xs)
    line = LineString([(-2 * span, y), (2 * span, y)])
    inter = poly.intersection(line)
    return inter.length if not inter.is_empty else 0.0


def _flange_transition(poly: Polygon, web_chord: float, d_ext: float):
    """Height y* where c(y) steps from the web plateau to the flange width,
    bracketed by a coarse scan and refined by bisection."""
    n = 60
    prev_y, prev_c = None, None
    for i in range(n + 1):
        y = d_ext / 2.0 * (i / n)
        c = _chord(poly, y)
        if (prev_c is not None and c > 2.5 * web_chord
                and prev_c <= 2.5 * web_chord):
            lo, hi = prev_y, y
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if _chord(poly, mid) > 2.5 * web_chord:
                    hi = mid
                else:
                    lo = mid
            return hi
        prev_y, prev_c = y, c
    return None


def _to_section(shape: str, p: dict) -> Section | None:
    if shape == "circle":
        return CircSection(diameter=p["d"])
    if shape == "hollow_circle":
        return PipeSection(od=p["od"], t=p["t"])
    if shape == "rect":
        return RectSection(width=p["b"], depth=p["d"])
    if shape == "hollow_rect":
        return RectHollowSection(width=p["b"], depth=p["d"], t=p["t"])
    if shape == "i":
        return ISection(depth=p["d"], width=p["b"], tw=p["tw"], tf=p["tf"])
    if shape == "t":
        return TSection(depth=p["d"], width=p["b"], tw=p["tw"], tf=p["tf"])
    if shape == "channel":
        return ChannelSection(depth=p["d"], width=p["b"], tw=p["tw"], tf=p["tf"])
    if shape == "angle":
        return LSection(leg1=p["b1"], leg2=p["b2"], t=p["t"])
    return None


# ---------------------------------------------------------------------------
# bodies of revolution (containment shells authored as tessellations)
# ---------------------------------------------------------------------------

def _bands(values: np.ndarray, gap: float) -> list[np.ndarray]:
    """Split sorted values into clusters separated by more than ``gap``."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        return []
    cuts = np.where(np.diff(v) > gap)[0]
    return np.split(v, cuts + 1)


def revolution_params(verts: np.ndarray, faces: np.ndarray | None = None,
                      rel_tol: float = 0.01):
    """Recover a vertical body of revolution from a tessellated shell.

    Recognised forms (all centred on a vertical axis through the plan
    centroid of the vertex cloud):
      * solid / hollow cylinder - radial distances fall into one or two tight
        bands, elevations into two (basemat disc, cylindrical wall);
      * hemispherical shell - a sphere centred on the axis fits every vertex,
        distances fall into two bands (inner / outer surface) and the shell
        starts at the sphere's equator (dome on a cylinder).
    Returns ``SolidParams`` (same contract as the parametric extractors in
    ``geometry.swept``) or ``None`` when the mesh is not one of these.
    """
    from .swept import SolidParams

    v = np.asarray(verts, dtype=float)
    if len(v) < 12:
        return None
    # axis: algebraic circle fit of the plan projection (mixed radii bias the
    # radius, not the centre of a symmetric ring)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = np.c_[2 * x, 2 * y, np.ones(len(v))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    rho = np.hypot(x - cx, y - cy)
    r_max = float(rho.max())
    if r_max < 1e-6:
        return None
    tol = max(rel_tol * r_max, 1e-3)
    z_min, z_max = float(z.min()), float(z.max())

    def covers_circle(mask, max_gap_deg: float = 45.0) -> bool:
        """Vertices in ``mask`` go all the way round the axis (a box's corners
        are equidistant from its centre too, but leave 90+ degree gaps)."""
        ang = np.sort(np.arctan2(y[mask] - cy, x[mask] - cx))
        if len(ang) < 8:
            return False
        gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        return float(np.degrees(gaps.max())) <= max_gap_deg

    # -- cylinder / disc -------------------------------------------------------
    zb = _bands(z, tol)
    rb_all = _bands(rho, tol)
    if len(zb) == 2 and 1 <= len(rb_all) <= 2 and all(np.ptp(bnd) <= tol for bnd in rb_all):
        radii = sorted(float(np.median(bnd)) for bnd in rb_all)
        radii = [0.0 if r <= tol else r for r in radii]
        rings = [r for r in radii if r > 0]
        if not rings or not all(covers_circle(np.abs(rho - r) <= tol) for r in rings):
            return None
        if len(rings) == 2:
            r_in, r_out = rings
        else:
            r_in, r_out = 0.0, rings[0]
        return SolidParams("cylinder", {
            "r_outer": r_out, "r_inner": r_in,
            "z_min": z_min, "z_max": z_max, "cx": cx, "cy": cy,
        })

    # -- spherical shell -------------------------------------------------------
    # rho^2 + z^2 = 2 cz z + (R^2 - cz^2): linear in (cz, K)
    A = np.c_[2 * z, np.ones(len(v))]
    b = rho * rho + z * z
    (cz, K), *_ = np.linalg.lstsq(A, b, rcond=None)
    dist = np.sqrt(rho * rho + (z - cz) ** 2)
    db = _bands(dist, tol)
    if len(db) == 2 and all(np.ptp(bnd) <= tol for bnd in db):
        r_in, r_out = sorted(float(np.median(bnd)) for bnd in db)
        # refine the centre elevation from the outer band alone
        outer = dist >= (r_in + r_out) / 2.0
        A2 = np.c_[2 * z[outer], np.ones(int(outer.sum()))]
        b2 = rho[outer] ** 2 + z[outer] ** 2
        (cz2, K2), *_ = np.linalg.lstsq(A2, b2, rcond=None)
        cz = float(cz2)
        r_out = float(np.sqrt(max(K2 + cz * cz, 0.0)))
        inner = ~outer
        if inner.any():
            r_in = float(np.median(np.sqrt(rho[inner] ** 2 + (z[inner] - cz) ** 2)))
        equator = (np.abs(z - cz) <= max(5 * tol, 0.05)) & outer
        if (abs(z_min - cz) <= max(5 * tol, 0.05)
                and abs(z_max - (cz + r_out)) <= max(5 * tol, 0.05)
                and covers_circle(equator)):
            return SolidParams("spherical_shell", {
                "r_outer": r_out, "r_inner": r_in, "cz": cz,
                "cx": cx, "cy": cy, "hemisphere": 1.0,
            })
    return None
