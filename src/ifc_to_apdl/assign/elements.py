"""Phase 2 — semantic class -> APDL element formulation.

Direct subclass mapping where the structural role is unambiguous; pipe
fittings are disambiguated by port alignment when ports exist, with a
geometric fallback (revolved solid => elbow) for port-less exports.
"""

from __future__ import annotations

import math

import numpy as np

from ..ingest.loader import IfcContext, ProductRecord
from ..ingest.transforms import local_placement


def resolve_element_type(ctx: IfcContext, rec: ProductRecord, domain: str) -> tuple[str, str]:
    """Return (apdl_element, evidence)."""
    cls = rec.ifc_class

    if domain == "containment":
        return "SOLID187", "containment component -> 10-node tetrahedral solid"

    if cls in ("IfcBeam", "IfcColumn", "IfcMember"):
        return "BEAM188", "framing class -> Timoshenko beam"
    if cls in ("IfcSlab", "IfcWall", "IfcPlate"):
        return "SHELL181", "planar class -> 4-node shell"
    if cls == "IfcFooting":
        return "SHELL181", "footing -> shell (policy-dependent; may become support)"
    if cls in ("IfcPipeSegment", "IfcFlowSegment", "IfcValve", "IfcFlowController"):
        return "PIPE288", "straight distribution member -> 3-D pipe"

    if cls in ("IfcPipeFitting", "IfcFlowFitting"):
        dirs = _port_dirs(ctx, rec)
        if len(dirs) >= 3:
            # branch fitting (tee/wye): each prismatic body leg becomes a
            # straight member meeting at the shared junction node
            return "PIPE288", f"{len(dirs)}-port fitting -> branch junction"
        if len(dirs) == 2:
            theta = _angle(dirs[0], dirs[1])
            # (anti-)parallel ports => straight coupling. Exporters orient port
            # axes either outward (coupling: theta ~ pi) or along the flow
            # (coupling: theta ~ 0); a bend under 5 deg either way is a
            # port-alignment kink, not an elbow - a 2 deg ELBOW290 would mesh
            # into millimetre elements that fail the solver shape check
            bend = min(theta, math.pi - theta)
            if bend < math.radians(5):
                return "PIPE288", f"port alignment theta={math.degrees(theta):.1f} deg -> coupling"
            return "ELBOW290", f"port alignment theta={math.degrees(theta):.1f} deg -> elbow"
        # geometric fallback (no ports authored)
        for ri in rec.body_items:
            if ri.item.is_a("IfcRevolvedAreaSolid"):
                return "ELBOW290", "no ports; revolved body geometry -> elbow"
        n_bodies = sum(1 for ri in rec.body_items if ri.item.is_a("IfcExtrudedAreaSolid"))
        if n_bodies >= 2:
            return "PIPE288", f"no ports; {n_bodies} prismatic bodies -> branch junction"
        return "PIPE288", "no ports; prismatic body geometry -> coupling"

    raise ValueError(f"no element mapping for {cls} in domain {domain}")


def _port_dirs(ctx: IfcContext, rec: ProductRecord) -> list:
    """Orientation vectors (port Z axes) of the ports nested on an element."""
    dirs = []
    for rel in getattr(rec.entity, "IsNestedBy", None) or []:
        for obj in rel.RelatedObjects:
            if obj.is_a("IfcDistributionPort") and obj.ObjectPlacement is not None:
                m = local_placement(obj.ObjectPlacement, ctx.length_scale)
                dirs.append(m[:3, 2])
    return dirs


def _angle(d1, d2) -> float:
    a = d1 / np.linalg.norm(d1)
    b = d2 / np.linalg.norm(d2)
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))
