"""Analytical centerline extraction — evidence-first.

When the exporter provides an ``Axis`` representation (Revit does), that IS
the analytical centerline and takes precedence over geometric inference from
the body solid (legacy weakness W3/W5).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..ingest.loader import ResolvedItem
from ..ingest.transforms import apply

Vec3 = tuple[float, float, float]


def axis_rep_endpoints(axis_items: list[ResolvedItem],
                       length_scale: float) -> Optional[tuple[Vec3, Vec3]]:
    """Endpoints of the first line-like Axis representation item, global metres."""
    for ri in axis_items:
        item, m = ri.item, ri.matrix
        if item.is_a("IfcPolyline"):
            pts = [apply(m, np.array(p.Coordinates, dtype=float) * length_scale)
                   for p in item.Points]
            if len(pts) >= 2:
                return pts[0], pts[-1]
        elif item.is_a("IfcTrimmedCurve") and item.BasisCurve.is_a("IfcLine"):
            basis = item.BasisCurve
            p0 = np.array(basis.Pnt.Coordinates, dtype=float) * length_scale
            vec = np.array(basis.Dir.Orientation.DirectionRatios, dtype=float)
            mag = float(basis.Dir.Magnitude) * length_scale
            ends = []
            for trim in (item.Trim1, item.Trim2):
                pt = _trim_point(trim, p0, vec, mag, length_scale)
                if pt is None:
                    ends = []
                    break
                ends.append(pt)
            if len(ends) == 2:
                return apply(m, ends[0]), apply(m, ends[1])
    return None


def _trim_point(trim, p0: np.ndarray, direction: np.ndarray, magnitude: float,
                length_scale: float):
    """Resolve one trim selector to a point on the line (prefers Cartesian)."""
    param = None
    for sel in trim:
        if hasattr(sel, "is_a") and sel.is_a("IfcCartesianPoint"):
            return np.array(sel.Coordinates, dtype=float) * length_scale
        if hasattr(sel, "wrappedValue"):
            param = float(sel.wrappedValue)
    if param is not None:
        # IfcLine parameterization: point = Pnt + param * Dir (Dir has Magnitude)
        return p0 + param * direction * magnitude
    return None
