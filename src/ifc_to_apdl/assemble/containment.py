"""Phase 4 — nuclear containment structure assembly.

The structure is carried parametrically in the IR (cylinder / spherical-shell
primitives recovered in Phase 3) and emitted as APDL solid primitives glued
with VADD; compatibility is a product of the single conformal VMESH
(manuscript strategy), with element size driven by the wall thickness rule
rather than a hard-coded SMRTSIZE level. Coordinates keep the IFC frame —
no origin shift (legacy weakness W10).
"""

from __future__ import annotations

from ..assign.materials import MaterialResolver
from ..classify.systems import SystemRecord
from ..config import ConversionConfig
from ..geometry.revolution import revolution_from_item
from ..ingest.loader import IfcContext
from ..model.ir import AnalyticalModel, NodePool, Provenance, Support, Volume3D
from ..report.audit import AuditStatus


def assemble_containment(ctx: IfcContext, system: SystemRecord,
                         config: ConversionConfig,
                         resolver: MaterialResolver) -> AnalyticalModel:
    merge_tol = config.resolve_merge_tol(ctx.precision)
    model = AnalyticalModel(name=system.name, domain="containment",
                            nodes=NodePool(merge_tol))
    model.meta.update(source=ctx.path, schema=ctx.schema, merge_tol=merge_tol)

    vol_id = 0
    for guid in system.product_guids:
        rec = ctx.products[guid]
        mat = resolver.resolve(rec, "containment (nuclear)", "solid")
        mat_id = model.add_material(mat)

        # one dispatch for every encoding (parametric, CSG, tessellated,
        # B-rep) - the same one that classified the product as a shell
        params = None
        detail = "no body items"
        for ri in rec.body_items:
            try:
                params, _, detail = revolution_from_item(ri.item, ri.matrix, ctx.length_scale,
                                                         ctx.angle_scale)
                break
            except ValueError as exc:
                detail = str(exc)

        if params is None:
            ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                             detail=f"no containment primitive recovered ({detail})",
                             system=system.name)
            continue

        vol_id += 1
        prov = Provenance(rec.guid, rec.ifc_class, rec.name,
                          {"geometry": detail,
                           "material": mat.source
                                       + (f" (s={mat.confidence:.2f})" if mat.confidence else "")})
        model.volumes.append(Volume3D(vol_id, prov, params.kind, params.params,
                                      "SOLID187", mat_id))
        ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.CONVERTED,
                         detail=f"{params.kind}: " + ", ".join(
                             f"{k}={v:g}" for k, v in params.params.items()
                             if isinstance(v, (int, float))),
                         system=system.name)

    # -- interface reconciliation: stacked bodies of revolution must share
    #    exact faces or VADD leaves them as separate (unconnected) volumes.
    #    Tessellation fits carry millimetre noise; snap a dome's equator to
    #    the wall top, and equalise radii / elevations that agree within the
    #    snap tolerance (mid-surface compatibility, same rule as buildings).
    snap = config.resolve_snap_tol(merge_tol)
    n_rec = 0
    cyls = [v for v in model.volumes if v.kind == "cylinder"]
    for v in model.volumes:
        p = v.params
        for c in cyls:
            if c is v:
                continue
            q = c.params
            for key in ("cx", "cy"):
                if key in p and key in q and 0 < abs(p[key] - q[key]) <= snap:
                    p[key] = q[key]; n_rec += 1
            if v.kind == "spherical_shell":
                if 0 < abs(p["cz"] - q["z_max"]) <= max(snap, 0.05):
                    p["cz"] = q["z_max"]; n_rec += 1
                for key in ("r_outer", "r_inner"):
                    if 0 < abs(p[key] - q[key]) <= max(snap, 0.05):
                        p[key] = q[key]; n_rec += 1
            elif v.kind == "cylinder":
                if 0 < abs(p["z_min"] - q["z_max"]) <= max(snap, 0.05):
                    p["z_min"] = q["z_max"]; n_rec += 1
                for key in ("r_outer", "r_inner"):
                    if key in p and key in q and 0 < abs(p[key] - q[key]) <= max(snap, 0.05):
                        p[key] = q[key]; n_rec += 1
    if n_rec:
        ctx.audit.event("geometry",
                        f"{system.name}: {n_rec} interface parameter(s) reconciled between "
                        "stacked bodies of revolution (shared faces for VADD)")

    if config.bcs.containment == "basemat-bottom" and model.volumes:
        model.supports.append(Support(name="BASEMAT_FIX", source="heuristic:basemat-bottom"))
        zmin = min(v.params["z_min"] for v in model.volumes if v.kind == "cylinder")
        ctx.audit.event("boundary-condition",
                        f"{system.name}: basemat bottom face fixed at z={zmin:g} "
                        "[heuristic basemat-bottom]", severity="warning")
    return model
