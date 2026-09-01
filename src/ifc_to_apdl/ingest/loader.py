"""Phase 0 — IFC ingestion and normalization.

Produces an :class:`IfcContext`: the opened model plus everything downstream
phases need in normalized SI form — unit scales (resolved from the unit
assignment actually referenced by IfcProject, guarding against orphaned
duplicates), model precision, storey elevations, product indexes, material
associations, psets, and resolved representations (mapped items composed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ifcopenshell
import ifcopenshell.util.element as ioe
import numpy as np

from ..report.audit import AuditLedger
from . import transforms as T

#: physical product classes we consider (order matters nowhere; membership does)
STRUCTURAL_CLASSES = (
    "IfcBeam", "IfcColumn", "IfcSlab", "IfcWall", "IfcRoof", "IfcFooting",
    "IfcMember", "IfcPlate",
)
DISTRIBUTION_CLASSES = (
    "IfcPipeSegment", "IfcPipeFitting", "IfcValve", "IfcFlowSegment",
    "IfcFlowFitting", "IfcFlowController",
)
EQUIPMENT_CLASSES = (
    "IfcTank", "IfcPump", "IfcBoiler", "IfcChiller", "IfcUnitaryEquipment",
    "IfcHeatExchanger", "IfcEvaporator", "IfcCondenser", "IfcCoolingTower",
    "IfcElectricMotor", "IfcEngine", "IfcFan",
    "IfcCompressor", "IfcEnergyConversionDevice", "IfcFlowMovingDevice",
    "IfcFlowStorageDevice", "IfcElectricAppliance",
)
ACCESSORY_CLASSES = (
    "IfcDiscreteAccessory", "IfcMechanicalFastener",
)


@dataclass
class ResolvedItem:
    """One representation item with its full item->global transform."""
    item: Any                      # the IFC geometry item (e.g. IfcExtrudedAreaSolid)
    matrix: np.ndarray             # global 4x4, translations in metres


@dataclass
class ProductRecord:
    entity: Any
    guid: str
    ifc_class: str
    name: str
    placement: np.ndarray                       # object placement, global, metres
    body_items: list[ResolvedItem] = field(default_factory=list)
    axis_items: list[ResolvedItem] = field(default_factory=list)
    material: Any = None                        # IfcMaterial / usage / None
    psets: dict[str, dict] = field(default_factory=dict)
    storey_guid: str = ""
    building_guid: str = ""
    system_guids: list[str] = field(default_factory=list)


def is_hanger(rec: "ProductRecord") -> bool:
    """Pipe-support hanger components: IfcMember or accessory hardware whose
    name or ObjectType carries a hanger designation (IFC4 has no HANGER member
    type; BIM tools export rods/clamps as USERDEFINED members named
    'Hanger ...' or as IfcDiscreteAccessory assemblies)."""
    if rec.ifc_class != "IfcMember" and rec.ifc_class not in ACCESSORY_CLASSES:
        return False
    tokens = f"{rec.name} {getattr(rec.entity, 'ObjectType', '') or ''}".lower()
    return "hanger" in tokens


@dataclass
class IfcContext:
    path: str
    model: Any
    schema: str
    length_scale: float                          # project length unit -> metres
    angle_scale: float                           # project angle unit -> radians
    precision: Optional[float]                   # metres, from representation context
    storeys: dict[str, float]                    # storey GUID -> elevation [m]
    buildings: list[str]                         # building GUIDs
    products: dict[str, ProductRecord]           # GUID -> record
    audit: AuditLedger = field(default_factory=AuditLedger)

    def by_class(self, *classes: str) -> list[ProductRecord]:
        return [p for p in self.products.values() if p.ifc_class in classes]


def _unit_scale(model, unit_type: str, default: float) -> float:
    try:
        import ifcopenshell.util.unit as iou
        s = iou.calculate_unit_scale(model, unit_type)
        return float(s) if s else default
    except Exception:
        return default


def _precision(model, length_scale: float) -> Optional[float]:
    for ctx in model.by_type("IfcGeometricRepresentationContext"):
        p = getattr(ctx, "Precision", None)
        if p:
            return float(p) * length_scale
    return None


def _resolve_items(rep, obj_matrix: np.ndarray, length_scale: float) -> list[ResolvedItem]:
    """Flatten a shape representation, composing IfcMappedItem transforms."""
    out: list[ResolvedItem] = []
    for item in rep.Items:
        if item.is_a("IfcMappedItem"):
            src = item.MappingSource
            origin = T.axis2placement3d(src.MappingOrigin, length_scale)
            target = T.cartesian_transform(item.MappingTarget, length_scale)
            inner_matrix = obj_matrix @ target @ origin
            for inner in src.MappedRepresentation.Items:
                out.append(ResolvedItem(inner, inner_matrix))
        else:
            out.append(ResolvedItem(item, obj_matrix))
    return out


def _spatial_parents(entity) -> tuple[str, str]:
    """(storey_guid, building_guid) via containment/aggregation walk-up."""
    storey_guid = building_guid = ""
    node = entity
    for _ in range(12):
        parent = None
        for rel in getattr(node, "ContainedInStructure", None) or []:
            parent = rel.RelatingStructure
        if parent is None:
            for rel in getattr(node, "Decomposes", None) or []:
                parent = rel.RelatingObject
        if parent is None:
            break
        if parent.is_a("IfcBuildingStorey") and not storey_guid:
            storey_guid = parent.GlobalId
        if parent.is_a("IfcBuilding") and not building_guid:
            building_guid = parent.GlobalId
            break
        node = parent
    return storey_guid, building_guid


def _system_memberships(entity) -> list[str]:
    guids = []
    for rel in getattr(entity, "HasAssignments", None) or []:
        if rel.is_a("IfcRelAssignsToGroup"):
            grp = rel.RelatingGroup
            if grp.is_a("IfcSystem") or grp.is_a("IfcDistributionSystem"):
                guids.append(grp.GlobalId)
    return guids


def load_ifc(path: str | Path) -> IfcContext:
    path = str(path)
    model = ifcopenshell.open(path)
    schema = model.schema

    length_scale = _unit_scale(model, "LENGTHUNIT", 1.0)
    angle_scale = _unit_scale(model, "PLANEANGLEUNIT", 1.0)
    precision = _precision(model, length_scale)

    audit = AuditLedger(source_file=Path(path).name)
    audit.event("units", f"length scale {length_scale:g} m/unit, angle scale {angle_scale:g} rad/unit"
                + (f", precision {precision:g} m" if precision else ", no precision declared"))

    # Guard against multiple unit assignments (observed in gas_pipe_model.ifc):
    if len(model.by_type("IfcUnitAssignment")) > 1:
        audit.event("units", "multiple IfcUnitAssignment entities present; using the one "
                    "referenced by IfcProject", severity="warning")

    storeys = {
        s.GlobalId: (float(s.Elevation) * length_scale if s.Elevation is not None else 0.0)
        for s in model.by_type("IfcBuildingStorey")
    }
    buildings = [b.GlobalId for b in model.by_type("IfcBuilding")]

    products: dict[str, ProductRecord] = {}
    wanted = (STRUCTURAL_CLASSES + DISTRIBUTION_CLASSES + EQUIPMENT_CLASSES
              + ACCESSORY_CLASSES)
    for cls in wanted:
        try:
            entities = model.by_type(cls)
        except Exception:
            continue
        for e in entities:
            if e.GlobalId in products:
                continue
            # subclass queries can re-return entities; keep the most-derived class name
            rec = ProductRecord(
                entity=e,
                guid=e.GlobalId,
                ifc_class=e.is_a(),
                name=e.Name or "",
                placement=T.local_placement(e.ObjectPlacement, length_scale),
            )
            if e.Representation:
                for rep in e.Representation.Representations:
                    ident = (rep.RepresentationIdentifier or "").lower()
                    items = _resolve_items(rep, rec.placement, length_scale)
                    if ident == "body":
                        rec.body_items.extend(items)
                    elif ident == "axis":
                        rec.axis_items.extend(items)
            rec.material = ioe.get_material(e)
            try:
                rec.psets = ioe.get_psets(e)
            except Exception:
                rec.psets = {}
            rec.storey_guid, rec.building_guid = _spatial_parents(e)
            rec.system_guids = _system_memberships(e)
            products[e.GlobalId] = rec

    audit.event("classification", f"ingested {len(products)} physical products "
                f"({len(storeys)} storeys, {len(buildings)} buildings)")
    return IfcContext(
        path=path, model=model, schema=schema,
        length_scale=length_scale, angle_scale=angle_scale, precision=precision,
        storeys=storeys, buildings=buildings, products=products, audit=audit,
    )
