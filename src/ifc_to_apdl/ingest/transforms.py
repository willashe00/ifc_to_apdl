"""Homogeneous-transform helpers for IFC placements and representation items.

All functions return 4x4 numpy arrays acting on column vectors; translations
are converted to metres by the caller-supplied length scale.
"""

from __future__ import annotations

import numpy as np


def axis2placement3d(placement, length_scale: float = 1.0) -> np.ndarray:
    """IfcAxis2Placement3D -> 4x4 matrix (Z = Axis, X = RefDirection)."""
    m = np.eye(4)
    if placement is None:
        return m
    loc = np.array(placement.Location.Coordinates, dtype=float) * length_scale
    z = np.array(placement.Axis.DirectionRatios, dtype=float) if placement.Axis else np.array([0.0, 0.0, 1.0])
    x = (np.array(placement.RefDirection.DirectionRatios, dtype=float)
         if placement.RefDirection else np.array([1.0, 0.0, 0.0]))
    z = z / np.linalg.norm(z)
    x = x - np.dot(x, z) * z            # project X off Z
    n = np.linalg.norm(x)
    if n < 1e-12:                        # degenerate RefDirection parallel to Axis
        x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = x - np.dot(x, z) * z
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, loc
    return m


def axis2placement2d(placement, length_scale: float = 1.0) -> np.ndarray:
    """IfcAxis2Placement2D -> 4x4 matrix in the profile plane (Z identity)."""
    m = np.eye(4)
    if placement is None:
        return m
    loc = np.array(list(placement.Location.Coordinates) + [0.0], dtype=float) * length_scale
    if placement.RefDirection:
        x2 = np.array(placement.RefDirection.DirectionRatios, dtype=float)
        x = np.array([x2[0], x2[1], 0.0])
        x = x / np.linalg.norm(x)
    else:
        x = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, loc
    return m


def local_placement(placement, length_scale: float = 1.0) -> np.ndarray:
    """Recursively compose an IfcLocalPlacement chain into a global 4x4."""
    if placement is None:
        return np.eye(4)
    rel = axis2placement3d(placement.RelativePlacement, length_scale)
    if getattr(placement, "PlacementRelTo", None) is not None:
        return local_placement(placement.PlacementRelTo, length_scale) @ rel
    return rel


def cartesian_transform(op, length_scale: float = 1.0) -> np.ndarray:
    """IfcCartesianTransformationOperator3D (mapped-item target) -> 4x4."""
    m = np.eye(4)
    if op is None:
        return m
    origin = np.array(op.LocalOrigin.Coordinates, dtype=float) * length_scale
    x = np.array(op.Axis1.DirectionRatios, dtype=float) if getattr(op, "Axis1", None) else np.array([1.0, 0.0, 0.0])
    y = np.array(op.Axis2.DirectionRatios, dtype=float) if getattr(op, "Axis2", None) else np.array([0.0, 1.0, 0.0])
    z = np.array(op.Axis3.DirectionRatios, dtype=float) if getattr(op, "Axis3", None) else np.cross(x, y)
    scale = float(op.Scale) if getattr(op, "Scale", None) else 1.0
    x, y, z = (v / np.linalg.norm(v) for v in (x, y, z))
    m[:3, 0], m[:3, 1], m[:3, 2] = x * scale, y * scale, z * scale
    m[:3, 3] = origin
    return m


def apply(m: np.ndarray, p) -> tuple[float, float, float]:
    """Transform a 3-vector (or 2-vector, zero-padded) by a 4x4 matrix."""
    v = np.asarray(p, dtype=float)
    if v.shape[0] == 2:
        v = np.array([v[0], v[1], 0.0])
    out = m @ np.append(v, 1.0)
    return (float(out[0]), float(out[1]), float(out[2]))


def apply_dir(m: np.ndarray, d) -> tuple[float, float, float]:
    """Transform a direction (no translation), normalized."""
    v = np.asarray(d, dtype=float)
    if v.shape[0] == 2:
        v = np.array([v[0], v[1], 0.0])
    out = m[:3, :3] @ v
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return (float(out[0]), float(out[1]), float(out[2]))
