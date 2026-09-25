"""Bodies of revolution from any IFC body encoding (containment shells).

One dispatch serves classification (Phase 1 containment evidence) and
assembly (Phase 4 primitives), so the *recovered form* decides, never the
representation type. Evidence cascade, first success wins:

  1. parametric (exact): extruded circular / circular-hollow profile ->
     cylinder; revolved annular-arc profile or CSG sphere Boolean ->
     spherical shell;
  2. explicit tessellation (exact vertices): IfcTriangulatedFaceSet,
     IfcPolygonalFaceSet, IfcFacetedBrep -> ``revolution_params`` fit;
     arbitrary-outline extrusions contribute their extruded outline;
  3. kernel triangulation of any other solid (revolved rectangles,
     non-sphere Booleans, advanced B-reps) -> ``revolution_params`` fit.

A prism of a parameterized non-circular profile (rectangle, I, ...) can
never be a body of revolution and is rejected without tessellating, which
keeps the classifier cheap on ordinary walls and slabs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..ingest.transforms import apply
from .profiles import parse_profile
from .swept import SolidParams, _solid_frame, cylinder_params, dome_params
from .tessellated import revolution_params

CIRCULAR_PROFILES = ("IfcCircleProfileDef", "IfcCircleHollowProfileDef")


class RevolutionFit(NamedTuple):
    params: SolidParams
    encoding: str               # how the form was recovered, e.g. 'CSG sphere boolean'
    detail: str                 # provenance text for the audit / deck comments


def revolution_from_item(item, matrix: np.ndarray, length_scale: float,
                         angle_scale: float) -> RevolutionFit:
    """Recovered vertical body of revolution (cylinder / disc / hemispherical
    shell) of one body item; raises ``ValueError`` when it is none."""
    if item.is_a("IfcCsgSolid"):
        item = item.TreeRootExpression

    # -- 1. parametric -----------------------------------------------------
    if item.is_a("IfcExtrudedAreaSolid"):
        prof = item.SweptArea
        if prof.is_a() in CIRCULAR_PROFILES:
            return RevolutionFit(cylinder_params(item, matrix, length_scale),
                                 "extruded circular profile",
                                 "extruded circular profile -> cylinder primitive")
        if not prof.is_a("IfcArbitraryClosedProfileDef"):
            raise ValueError(f"extruded {prof.is_a()} is not a body of revolution")
    elif item.is_a("IfcRevolvedAreaSolid"):
        try:
            return RevolutionFit(dome_params(item, matrix, length_scale, angle_scale),
                                 "revolved annular profile",
                                 "revolved annular profile -> spherical shell")
        except ValueError:
            pass                # e.g. a revolved rectangle (cylinder wall)
    elif item.is_a("IfcBooleanResult"):
        try:
            return RevolutionFit(dome_params(item, matrix, length_scale, angle_scale),
                                 "CSG sphere boolean",
                                 "CSG sphere boolean -> spherical shell")
        except ValueError:
            base = _boolean_base(item)
            if (base.is_a("IfcExtrudedAreaSolid")
                    and base.SweptArea.is_a() not in CIRCULAR_PROFILES
                    and not base.SweptArea.is_a("IfcArbitraryClosedProfileDef")):
                raise ValueError("boolean of an extruded "
                                 f"{base.SweptArea.is_a()} is not a body of revolution")

    # -- 2./3. computational fit on the vertex cloud ----------------------
    verts, how = _vertex_cloud(item, matrix, length_scale)
    prm = revolution_params(verts)
    if prm is None:
        raise ValueError(f"{how} is not a vertical body of revolution "
                         "(cylinder / disc / hemispherical shell)")
    return RevolutionFit(prm, how, f"{how} ({len(verts)} vertices) -> {prm.kind}")


def _boolean_base(item):
    """First-operand solid at the root of a Boolean tree (the body that
    DIFFERENCE / INTERSECTION / clipping operations carve)."""
    while item.is_a("IfcBooleanResult"):
        item = item.FirstOperand
    return item


def _vertex_cloud(item, matrix: np.ndarray, length_scale: float) -> tuple[np.ndarray, str]:
    """Global vertex cloud [m] of a body item and a label of how it was
    obtained. Fitting a body of revolution needs vertices only (no faces)."""
    if item.is_a("IfcTriangulatedFaceSet") or item.is_a("IfcPolygonalFaceSet"):
        local = np.array(item.Coordinates.CoordList, dtype=float) * length_scale
        how = "tessellated face set"
    elif item.is_a("IfcFacetedBrep"):
        shells = [item.Outer] + list(getattr(item, "Voids", None) or [])
        pts = {p.id(): p.Coordinates for sh in shells for face in sh.CfsFaces
               for bound in face.Bounds for p in bound.Bound.Polygon}
        local = np.array(list(pts.values()), dtype=float) * length_scale
        how = "faceted B-rep"
    elif item.is_a("IfcExtrudedAreaSolid"):
        # arbitrary outline (e.g. an annulus of arc segments): the sampled
        # outline and voids at both ends of the extrusion
        prof = parse_profile(item.SweptArea, length_scale)
        ring = [apply(prof.position, p) for p in [*prof.poly, *(q for h in prof.holes for q in h)]]
        d = np.array(item.ExtrudedDirection.DirectionRatios, dtype=float)
        d = d / np.linalg.norm(d) * float(item.Depth) * length_scale
        frame = _solid_frame(item, matrix, length_scale)
        pts = [apply(frame, np.add(p, off)) for off in (np.zeros(3), d) for p in ring]
        return np.array(pts, dtype=float), "extruded arbitrary outline"
    else:
        import ifcopenshell.geom
        try:
            shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), item)
        except Exception as exc:
            raise ValueError(f"{item.is_a()} could not be triangulated ({exc})") from exc
        local = np.array(shape.verts, dtype=float).reshape(-1, 3)   # metres
        how = f"{item.is_a()} kernel triangulation"
    return (matrix[:3, :3] @ local.T).T + matrix[:3, 3], how
