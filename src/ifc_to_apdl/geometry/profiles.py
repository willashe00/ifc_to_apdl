"""IfcProfileDef -> typed section parameters (full subtype coverage).

The legacy converter handled I / hollow-circle / rectangle / arbitrary only;
channels, tees, angles, hollow rectangles and solid circles fell through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..model.sections import (
    ChannelSection, CircSection, ISection, LSection, PipeSection,
    RectHollowSection, RectSection, Section, TSection,
)
from ..ingest.transforms import axis2placement2d


@dataclass
class ProfileInfo:
    kind: str                       # 'i','t','u','l','rect','rect_hollow','circle','circle_hollow','arbitrary'
    section: Optional[Section]      # beam-usable section (None for 'arbitrary')
    poly: Optional[list[tuple[float, float]]]   # outer polygon, profile plane [m]
    position: np.ndarray            # 2-D placement of the profile within the solid XY plane
    dims: dict = field(default_factory=dict)
    name: str = ""
    #: inner void polygons (IfcArbitraryProfileDefWithVoids), profile plane [m]
    holes: list = field(default_factory=list)

    def centroid2d(self) -> tuple[float, float]:
        """Profile-frame centroid of the *bounding definition* (profile origin
        for parameterized shapes; polygon centroid for arbitrary)."""
        if self.poly:
            xs = [p[0] for p in self.poly]
            ys = [p[1] for p in self.poly]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return (0.0, 0.0)


def _curve_points(curve, s: float) -> list[tuple[float, float]]:
    """Sample a bounded 2-D curve to a polygon point list [m]."""
    if curve.is_a("IfcPolyline"):
        pts = [(float(p.Coordinates[0]) * s, float(p.Coordinates[1]) * s) for p in curve.Points]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return pts
    if curve.is_a("IfcIndexedPolyCurve"):
        coords = curve.Points.CoordList
        pts = [(float(c[0]) * s, float(c[1]) * s) for c in coords]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return pts
    if curve.is_a("IfcCompositeCurve"):
        pts: list[tuple[float, float]] = []
        for seg in curve.Segments:
            pts.extend(_curve_points_segment(seg.ParentCurve, s))
        return pts
    if curve.is_a("IfcTrimmedCurve"):
        return _curve_points_segment(curve, s)
    raise ValueError(f"unsupported profile curve {curve.is_a()}")


def _curve_points_segment(curve, s: float, n: int = 24) -> list[tuple[float, float]]:
    if curve.is_a("IfcPolyline"):
        return [(float(p.Coordinates[0]) * s, float(p.Coordinates[1]) * s) for p in curve.Points]
    if curve.is_a("IfcTrimmedCurve"):
        basis = curve.BasisCurve
        if basis.is_a("IfcCircle"):
            cx, cy = (float(v) * s for v in basis.Position.Location.Coordinates[:2])
            r = float(basis.Radius) * s
            t1, t2 = _trim_angles(curve)
            if curve.SenseAgreement is False and t2 > t1:
                t1, t2 = t2, t1 + 2 * math.pi if t2 > t1 else t2
            if t2 < t1:
                t2 += 2 * math.pi
            return [(cx + r * math.cos(t), cy + r * math.sin(t))
                    for t in np.linspace(t1, t2, n)]
        if basis.is_a("IfcLine"):
            return _curve_points_segment(basis, s)
    raise ValueError(f"unsupported composite segment {curve.is_a()}")


def _trim_angles(trimmed) -> tuple[float, float]:
    def angle_of(sel):
        for item in sel:
            if hasattr(item, "is_a") and item.is_a("IfcCartesianPoint"):
                basis = trimmed.BasisCurve
                cx, cy = (float(v) for v in basis.Position.Location.Coordinates[:2])
                x, y = (float(v) for v in item.Coordinates[:2])
                return math.atan2(y - cy, x - cx)
            if hasattr(item, "wrappedValue"):
                return float(item.wrappedValue)   # parameter value: radians for circles
        return 0.0
    return angle_of(trimmed.Trim1), angle_of(trimmed.Trim2)


def parse_profile(profile, length_scale: float) -> ProfileInfo:
    s = length_scale
    pos = axis2placement2d(getattr(profile, "Position", None), s)
    name = getattr(profile, "ProfileName", None) or ""
    cls = profile.is_a()

    if cls == "IfcIShapeProfileDef":
        sec = ISection(name=name, depth=float(profile.OverallDepth) * s,
                       width=float(profile.OverallWidth) * s,
                       tw=float(profile.WebThickness) * s,
                       tf=float(profile.FlangeThickness) * s)
        return ProfileInfo("i", sec, None, pos, name=name)

    if cls == "IfcTShapeProfileDef":
        sec = TSection(name=name, depth=float(profile.Depth) * s,
                       width=float(profile.FlangeWidth) * s,
                       tw=float(profile.WebThickness) * s,
                       tf=float(profile.FlangeThickness) * s)
        return ProfileInfo("t", sec, None, pos, name=name)

    if cls == "IfcUShapeProfileDef":
        sec = ChannelSection(name=name, depth=float(profile.Depth) * s,
                             width=float(profile.FlangeWidth) * s,
                             tw=float(profile.WebThickness) * s,
                             tf=float(profile.FlangeThickness) * s)
        return ProfileInfo("u", sec, None, pos, name=name)

    if cls == "IfcLShapeProfileDef":
        depth = float(profile.Depth) * s
        width = float(profile.Width) * s if profile.Width else depth
        sec = LSection(name=name, leg1=depth, leg2=width, t=float(profile.Thickness) * s)
        return ProfileInfo("l", sec, None, pos, name=name)

    if cls == "IfcRectangleHollowProfileDef":
        sec = RectHollowSection(name=name, width=float(profile.XDim) * s,
                                depth=float(profile.YDim) * s,
                                t=float(profile.WallThickness) * s)
        return ProfileInfo("rect_hollow", sec, None, pos, name=name)

    if cls == "IfcRectangleProfileDef":
        x, y = float(profile.XDim) * s, float(profile.YDim) * s
        sec = RectSection(name=name, width=x, depth=y)
        poly = [(-x / 2, -y / 2), (x / 2, -y / 2), (x / 2, y / 2), (-x / 2, y / 2)]
        return ProfileInfo("rect", sec, poly, pos, dims={"x": x, "y": y}, name=name)

    if cls == "IfcCircleHollowProfileDef":
        r = float(profile.Radius) * s
        t = float(profile.WallThickness) * s
        sec = PipeSection(name=name, od=2.0 * r, t=t)
        return ProfileInfo("circle_hollow", sec, None, pos, dims={"r": r, "t": t}, name=name)

    if cls == "IfcCircleProfileDef":
        r = float(profile.Radius) * s
        sec = CircSection(name=name, diameter=2.0 * r)
        return ProfileInfo("circle", sec, None, pos, dims={"r": r}, name=name)

    if profile.is_a("IfcArbitraryClosedProfileDef"):
        poly = _curve_points(profile.OuterCurve, s)
        holes = []
        if cls == "IfcArbitraryProfileDefWithVoids":
            holes = [_curve_points(c, s) for c in profile.InnerCurves]
        return ProfileInfo("arbitrary", None, poly, pos, name=name, holes=holes)

    raise ValueError(f"unsupported profile type {cls}")
