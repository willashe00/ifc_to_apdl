"""Parametric swept-solid interpretation (manuscript Phase 3, direct branch).

Recovers analytical centerlines / mid-surfaces / solid parameters from
IfcExtrudedAreaSolid, IfcRevolvedAreaSolid and dome encodings (both the
CSG sphere-boolean form and the revolved-annular-profile form observed in
``containment_structure_revolved_dome.ifc``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..ingest.transforms import apply, apply_dir, axis2placement3d
from .profiles import ProfileInfo, parse_profile

Vec3 = tuple[float, float, float]


@dataclass
class MemberAxis:
    start: Vec3
    end: Vec3
    profile: ProfileInfo
    direction: Vec3                  # unit extrusion/sweep direction at start
    arc_center: Optional[Vec3] = None
    arc_radius: Optional[float] = None
    sweep_angle: Optional[float] = None


@dataclass
class PlateGeom:
    loop: list[Vec3]                 # mid-surface boundary corners (ordered)
    thickness: float
    normal: Vec3
    holes: list = None               # void loops (list[list[Vec3]]) or None


def _solid_frame(solid, matrix: np.ndarray, length_scale: float) -> np.ndarray:
    pos = getattr(solid, "Position", None)
    return matrix @ axis2placement3d(pos, length_scale)


def extruded_axis(solid, matrix: np.ndarray, length_scale: float) -> MemberAxis:
    """Centerline of a linear prismatic member from IfcExtrudedAreaSolid."""
    frame = _solid_frame(solid, matrix, length_scale)
    prof = parse_profile(solid.SweptArea, length_scale)
    depth = float(solid.Depth) * length_scale
    ex, ey = prof.centroid2d()
    # profile centroid within the solid XY plane (profile Position offset applies)
    c2 = apply(prof.position, (ex, ey, 0.0))
    d_local = np.array(solid.ExtrudedDirection.DirectionRatios, dtype=float)
    d_local = d_local / np.linalg.norm(d_local)
    start = apply(frame, c2)
    end = apply(frame, (c2[0] + d_local[0] * depth,
                        c2[1] + d_local[1] * depth,
                        c2[2] + d_local[2] * depth))
    direction = apply_dir(frame, d_local)
    return MemberAxis(start=start, end=end, profile=prof, direction=direction)


def revolved_arc(solid, matrix: np.ndarray, length_scale: float,
                 angle_scale: float) -> MemberAxis:
    """Curved-member arc from IfcRevolvedAreaSolid (elbows).

    The revolution axis lives in the solid's Position frame; the bend radius
    is the distance from the profile centroid to the axis line.
    """
    frame = _solid_frame(solid, matrix, length_scale)
    prof = parse_profile(solid.SweptArea, length_scale)
    angle = float(solid.Angle) * angle_scale

    ex, ey = prof.centroid2d()
    c_local = np.array(apply(prof.position, (ex, ey, 0.0)))

    ax = solid.Axis
    a_loc = np.array(list(ax.Location.Coordinates) + [0.0])[:3] * length_scale
    a_dir = np.array(ax.Axis.DirectionRatios, dtype=float)
    a_dir = a_dir / np.linalg.norm(a_dir)

    # closest point on the axis line to the profile centroid = arc center
    w = c_local - a_loc
    center_local = a_loc + np.dot(w, a_dir) * a_dir
    radius = float(np.linalg.norm(c_local - center_local))

    # rotate the start point about the axis by the sweep angle (Rodrigues)
    v = c_local - center_local
    k = a_dir
    v_rot = (v * math.cos(angle) + np.cross(k, v) * math.sin(angle)
             + k * np.dot(k, v) * (1 - math.cos(angle)))
    end_local = center_local + v_rot

    start = apply(frame, tuple(c_local))
    end = apply(frame, tuple(end_local))
    center = apply(frame, tuple(center_local))
    tangent = apply_dir(frame, tuple(np.cross(k, v)))   # sweep tangent at start
    return MemberAxis(start=start, end=end, profile=prof, direction=tangent,
                      arc_center=center, arc_radius=radius, sweep_angle=angle)


def plate_from_extrusion(solid, matrix: np.ndarray, length_scale: float) -> PlateGeom:
    """Mid-surface of a plate-like extrusion (slab or wall).

    The solid has three characteristic dims (profile x, profile y, depth);
    the smallest is the plate thickness and the mid-surface is spanned by
    the other two. Arbitrary profiles are treated as footprints (thickness =
    extrusion depth), which matches slab authoring.
    """
    frame = _solid_frame(solid, matrix, length_scale)
    prof = parse_profile(solid.SweptArea, length_scale)
    depth = float(solid.Depth) * length_scale
    d_local = np.array(solid.ExtrudedDirection.DirectionRatios, dtype=float)
    d_local = d_local / np.linalg.norm(d_local)

    if prof.kind == "rect":
        x, y = prof.dims["x"], prof.dims["y"]
        dims = {"x": x, "y": y, "d": depth}
        thick_key = min(dims, key=dims.get)
        t = dims[thick_key]
        # local corner coordinates of the mid-surface rectangle
        hx, hy = x / 2.0, y / 2.0
        if thick_key == "d":            # flat plate: footprint at mid-depth
            corners = [(-hx, -hy, depth / 2), (hx, -hy, depth / 2),
                       (hx, hy, depth / 2), (-hx, hy, depth / 2)]
            n_local = (0.0, 0.0, 1.0)
        elif thick_key == "y":          # wall-like: mid-plane at y=0, spans x & depth
            corners = [(-hx, 0.0, 0.0), (hx, 0.0, 0.0),
                       (hx, 0.0, depth), (-hx, 0.0, depth)]
            n_local = (0.0, 1.0, 0.0)
        else:                           # thick_key == 'x'
            corners = [(0.0, -hy, 0.0), (0.0, hy, 0.0),
                       (0.0, hy, depth), (0.0, -hy, depth)]
            n_local = (1.0, 0.0, 0.0)
        pos = prof.position
        loop = [apply(frame, apply(pos, c)) if thick_key == "d" else
                apply(frame, _offset2d(pos, c)) for c in corners]
        return PlateGeom(loop=loop, thickness=t, normal=apply_dir(frame, n_local))

    if prof.kind == "arbitrary" and prof.poly:
        loop = [apply(frame, apply(prof.position, (px, py, depth / 2.0)))
                for (px, py) in prof.poly]
        holes = [[apply(frame, apply(prof.position, (px, py, depth / 2.0)))
                  for (px, py) in h] for h in prof.holes] or None
        return PlateGeom(loop=loop, thickness=depth,
                         normal=apply_dir(frame, tuple(d_local)), holes=holes)

    raise ValueError(f"plate interpretation unsupported for profile kind '{prof.kind}'")


def _offset2d(pos2d: np.ndarray, c: tuple[float, float, float]) -> Vec3:
    """Apply a 2-D profile placement to (x, y) and keep the local z."""
    p = apply(pos2d, (c[0], c[1], 0.0))
    return (p[0], p[1], c[2])


# ---------------------------------------------------------------------------
# Containment primitives
# ---------------------------------------------------------------------------

@dataclass
class SolidParams:
    kind: str          # 'cylinder' | 'spherical_shell'
    params: dict


def cylinder_params(solid, matrix: np.ndarray, length_scale: float) -> SolidParams:
    """Vertical (hollow) cylinder from an extruded (hollow) circular profile."""
    frame = _solid_frame(solid, matrix, length_scale)
    prof = parse_profile(solid.SweptArea, length_scale)
    depth = float(solid.Depth) * length_scale
    base = apply(frame, apply(prof.position, (0.0, 0.0, 0.0)))
    d = apply_dir(frame, tuple(np.array(solid.ExtrudedDirection.DirectionRatios, dtype=float)))
    z0, z1 = base[2], base[2] + d[2] * depth
    if prof.kind == "circle":
        r_out, r_in = prof.dims["r"], 0.0
    elif prof.kind == "circle_hollow":
        r_out = prof.dims["r"]
        r_in = r_out - prof.dims["t"]
    else:
        raise ValueError("cylinder interpretation requires a circular profile")
    return SolidParams("cylinder", {
        "r_outer": r_out, "r_inner": r_in,
        "z_min": min(z0, z1), "z_max": max(z0, z1),
        "cx": base[0], "cy": base[1],
    })


def dome_params(solid, matrix: np.ndarray, length_scale: float,
                angle_scale: float) -> SolidParams:
    """Spherical-shell dome from either encoding.

    (a) IfcBooleanResult of two spheres (manuscript CSG form);
    (b) IfcRevolvedAreaSolid of an annular-arc profile revolved 2*pi
        (form observed in the containment fixture).
    """
    if solid.is_a("IfcBooleanResult"):
        first = solid.FirstOperand
        r_out = float(first.FirstOperand.Radius) * length_scale
        r_in = float(first.SecondOperand.Radius) * length_scale
        cz = float(first.FirstOperand.Position.Location.Coordinates[2]) * length_scale
        return SolidParams("spherical_shell",
                           {"r_outer": r_out, "r_inner": r_in, "cz": cz,
                            "cx": 0.0, "cy": 0.0, "hemisphere": 1.0})

    if solid.is_a("IfcRevolvedAreaSolid"):
        frame = _solid_frame(solid, matrix, length_scale)
        radii: list[float] = []
        centers: list[tuple[float, float]] = []
        outer = solid.SweptArea.OuterCurve
        segments = outer.Segments if outer.is_a("IfcCompositeCurve") else []
        for seg in segments:
            pc = seg.ParentCurve
            if pc.is_a("IfcTrimmedCurve") and pc.BasisCurve.is_a("IfcCircle"):
                radii.append(float(pc.BasisCurve.Radius) * length_scale)
                loc = pc.BasisCurve.Position.Location.Coordinates
                centers.append((float(loc[0]) * length_scale, float(loc[1]) * length_scale))
        if len(radii) < 2:
            raise ValueError("revolved dome profile lacks two arc segments")
        r_out, r_in = max(radii), min(radii)
        # profile-plane arc center: its local y maps to the global axis elevation
        cy = centers[0][1]
        axis_loc = solid.Axis.Location.Coordinates
        base = apply(frame, (float(axis_loc[0]) * length_scale,
                             float(axis_loc[1]) * length_scale,
                             float(axis_loc[2]) * length_scale if len(axis_loc) > 2 else 0.0))
        return SolidParams("spherical_shell",
                           {"r_outer": r_out, "r_inner": r_in,
                            "cz": base[2] + cy, "cx": base[0], "cy_": base[1],
                            "cy": base[1], "hemisphere": 1.0})

    raise ValueError(f"unsupported dome encoding {solid.is_a()}")
