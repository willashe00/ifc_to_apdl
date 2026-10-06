"""Minimal IFC authoring (metres, one IfcBuilding) for self-contained tests."""

import ifcopenshell
import ifcopenshell.guid as guid
import numpy as np


class Author:

    def __init__(self, building_type=None, schema="IFC4"):
        f = self.f = ifcopenshell.file(schema=schema)
        unit = f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE")
        self.ctx = f.createIfcGeometricRepresentationContext(
            None, "Model", 3, 1e-5, self.a2p((0.0, 0.0, 0.0)), None)
        project = f.createIfcProject(guid.new(), None, "p", None, None, None, None,
                                     (self.ctx,), f.createIfcUnitAssignment((unit,)))
        self.building = f.create_entity("IfcBuilding", GlobalId=guid.new(), Name="Building",
                                        ObjectType=building_type,
                                        ObjectPlacement=f.createIfcLocalPlacement(None, self.a2p((0, 0, 0))))
        f.createIfcRelAggregates(guid.new(), None, None, None, project, (self.building,))
        self.products = []
        self.concrete = self.material("Concrete", 30e9, 0.2, 2400.0)
        self.steel = self.material("Steel", 210e9, 0.3, 7850.0)
        self.mat_of = {}

    def a2p(self, loc, z=None, x=None):
        f = self.f

        def d(v):
            return f.createIfcDirection(tuple(float(c) for c in v)) if v is not None else None
        return f.createIfcAxis2Placement3D(
            f.createIfcCartesianPoint(tuple(float(v) for v in loc)), d(z), d(x))

    def material(self, name, e, nu, rho):
        f = self.f
        m = f.createIfcMaterial(name, None, None)
        props = [f.createIfcPropertySingleValue(n, None, f.create_entity("IfcReal", v), None)
                 for n, v in (("YoungModulus", e), ("PoissonRatio", nu), ("MassDensity", rho))]
        f.createIfcMaterialProperties("Pset_MaterialMechanical", None, props, m)
        return m

    def product(self, cls, name, loc, items, rep_type, material, contained=True):
        f = self.f
        shape = f.createIfcProductDefinitionShape(None, None, (
            f.createIfcShapeRepresentation(self.ctx, "Body", rep_type, tuple(items)),))
        e = f.create_entity(cls, GlobalId=guid.new(), Name=name, Representation=shape,
                            ObjectPlacement=f.createIfcLocalPlacement(None, self.a2p(loc)))
        if contained:
            self.products.append(e)
        self.mat_of.setdefault(material, []).append(e)
        return e

    def save(self, path):
        f = self.f
        f.createIfcRelContainedInSpatialStructure(guid.new(), None, None, None,
                                                  self.products, self.building)
        for m, objs in self.mat_of.items():
            f.createIfcRelAssociatesMaterial(guid.new(), None, None, None, objs, m)
        f.write(str(path))
        return path

    # -- framing -------------------------------------------------------------
    def rect_member(self, cls, name, p0, p1, b=0.3, d=0.3):
        f = self.f
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        ref = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
        solid = f.createIfcExtrudedAreaSolid(
            f.createIfcRectangleProfileDef("AREA", None, None, b, d),
            self.a2p((0, 0, 0), tuple(axis), ref), f.createIfcDirection((0.0, 0.0, 1.0)),
            float(np.linalg.norm(p1 - p0)))
        return self.product(cls, name, p0, [solid], "SweptSolid", self.steel)

    def i_beam(self, name, p0, p1, d=0.4, bf=0.2, tw=0.01, tf=0.015, cls="IfcBeam"):
        """IfcBeam (or ``cls``, e.g. an IfcMember truss chord) of an I profile
        (depth vertical) along a horizontal axis."""
        f = self.f
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        ref = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
        solid = f.createIfcExtrudedAreaSolid(
            f.createIfcIShapeProfileDef("AREA", None, None, bf, d, tw, tf, None),
            self.a2p((0, 0, 0), tuple(axis), ref), f.createIfcDirection((0.0, 0.0, 1.0)),
            float(np.linalg.norm(p1 - p0)))
        return self.product(cls, name, p0, [solid], "SweptSolid", self.steel)

    def rack(self, x0, y0):
        """Named 4-column equipment support rack (2.5 m square, 4 m tall)."""
        corners = [(x0, y0), (x0 + 2.5, y0), (x0 + 2.5, y0 + 2.5), (x0, y0 + 2.5)]
        for i, (x, y) in enumerate(corners, 1):
            self.rect_member("IfcColumn", f"Pump Support Rack - Column {i}", (x, y, 0), (x, y, 4))
        for i, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1]), 1):
            self.rect_member("IfcBeam", f"Pump Support Rack - Beam {i}", (*a, 4), (*b, 4))

    def layer_usage(self, layers):
        """IfcMaterialLayerSetUsage of [(thickness, material name)] layers
        (materials carry no properties: their family comes from the name)."""
        f = self.f
        mls = [f.create_entity("IfcMaterialLayer", Material=f.createIfcMaterial(name, None, None),
                               LayerThickness=t) for t, name in layers]
        return f.create_entity("IfcMaterialLayerSetUsage",
                               ForLayerSet=f.create_entity("IfcMaterialLayerSet", MaterialLayers=mls),
                               LayerSetDirection="AXIS2", DirectionSense="POSITIVE",
                               OffsetFromReferenceLine=0.0)

    def pset(self, e, name, props):
        """Property set of single values (bool -> IfcBoolean, else IfcLabel).
        A type object holds its sets directly in HasPropertySets, which is
        where get_psets looks when it inherits from the type; an occurrence
        gets one through IfcRelDefinesByProperties."""
        f = self.f
        vals = [f.createIfcPropertySingleValue(
            k, None, f.create_entity("IfcBoolean" if isinstance(v, bool) else "IfcLabel", v), None)
            for k, v in props.items()]
        ps = f.createIfcPropertySet(guid.new(), None, name, None, vals)
        if e.is_a("IfcTypeObject"):
            e.HasPropertySets = tuple(e.HasPropertySets or ()) + (ps,)
        else:
            f.createIfcRelDefinesByProperties(guid.new(), None, None, None, (e,), ps)
        return ps

    def slab(self, name, x0, y0, x1, y1, z0, t, predefined_type=None, layers=None):
        """Rectangular IfcSlab from (x0, y0) to (x1, y1), bottom face at z0."""
        f = self.f
        solid = f.createIfcExtrudedAreaSolid(
            f.createIfcRectangleProfileDef("AREA", None, None, x1 - x0, y1 - y0),
            self.a2p((0, 0, 0)), f.createIfcDirection((0.0, 0.0, 1.0)), t)
        e = self.product("IfcSlab", name, ((x0 + x1) / 2, (y0 + y1) / 2, z0), [solid],
                         "SweptSolid", self.layer_usage(layers) if layers else self.concrete)
        e.PredefinedType = predefined_type
        return e

    def wall(self, name, p0, p1, z0, height, t, layers=None, predefined_type=None):
        """Straight IfcWall along p0 -> p1 (plan), base at z0."""
        f = self.f
        (x0, y0), (x1, y1) = p0, p1
        length = float(np.hypot(x1 - x0, y1 - y0))
        solid = f.createIfcExtrudedAreaSolid(
            f.createIfcRectangleProfileDef(
                "AREA", None, f.createIfcAxis2Placement2D(
                    f.createIfcCartesianPoint((length / 2, 0.0)), None), length, t),
            self.a2p((0, 0, 0)), f.createIfcDirection((0.0, 0.0, 1.0)), height)
        e = self.product("IfcWall", name, (x0, y0, z0), [solid], "SweptSolid",
                         self.layer_usage(layers) if layers else self.concrete)
        e.PredefinedType = predefined_type
        e.ObjectPlacement.RelativePlacement.RefDirection = f.createIfcDirection(
            ((x1 - x0) / length, (y1 - y0) / length, 0.0))
        e.ObjectPlacement.RelativePlacement.Axis = f.createIfcDirection((0.0, 0.0, 1.0))
        return e

    def plate(self, name, centre, x, y, t, predefined_type=None, assembly=None, fasteners=0):
        """Horizontal rectangular IfcPlate (x by y, thickness t) whose bottom
        face is centred on ``centre``; optionally typed through an IfcPlateType
        and/or aggregated into a named IfcElementAssembly (with ``fasteners``
        bolts)."""
        f = self.f
        solid = f.createIfcExtrudedAreaSolid(
            f.createIfcRectangleProfileDef("AREA", None, None, x, y),
            self.a2p((0, 0, 0)), f.createIfcDirection((0.0, 0.0, 1.0)), t)
        e = self.product("IfcPlate", name, centre, [solid], "SweptSolid", self.steel,
                         contained=assembly is None)
        return self._plate_links(e, centre, predefined_type, assembly, fasteners)

    def web_plate(self, name, centre, along, x, y, t, plan_outline=False,
                  predefined_type=None, assembly=None, fasteners=0):
        """Upright rectangular IfcPlate centred on ``centre``: ``x`` along the
        horizontal unit vector ``along``, ``y`` high, ``t`` thick across. As a
        rectangle extruded through the thickness, or (``plan_outline``) as the
        same solid written as its plan outline extruded up the height."""
        f = self.f
        along = np.asarray(along, float)
        across = np.cross(along, (0.0, 0.0, 1.0))
        if plan_outline:
            pts = [(-x / 2, -t / 2), (x / 2, -t / 2), (x / 2, t / 2), (-x / 2, t / 2), (-x / 2, -t / 2)]
            profile = f.createIfcArbitraryClosedProfileDef("AREA", None, f.createIfcPolyline(
                [f.createIfcCartesianPoint(p) for p in pts]))
            solid = f.createIfcExtrudedAreaSolid(
                profile, self.a2p((0.0, 0.0, -y / 2), (0.0, 0.0, 1.0), tuple(along)),
                f.createIfcDirection((0.0, 0.0, 1.0)), y)
        else:
            solid = f.createIfcExtrudedAreaSolid(
                f.createIfcRectangleProfileDef("AREA", None, None, x, y),
                self.a2p(tuple(-across * t / 2), tuple(across), tuple(along)),
                f.createIfcDirection((0.0, 0.0, 1.0)), t)
        e = self.product("IfcPlate", name, centre, [solid], "SweptSolid", self.steel,
                         contained=assembly is None)
        return self._plate_links(e, centre, predefined_type, assembly, fasteners)

    def fastener(self, name, centre, contained=True):
        """IfcMechanicalFastener (bolt) without a body."""
        f = self.f
        e = f.create_entity("IfcMechanicalFastener", GlobalId=guid.new(), Name=name,
                            PredefinedType="BOLT",
                            ObjectPlacement=f.createIfcLocalPlacement(None, self.a2p(centre)))
        if contained:
            self.products.append(e)
        return e

    def _plate_links(self, e, centre, predefined_type, assembly, fasteners):
        f = self.f
        if predefined_type:
            ptype = f.create_entity("IfcPlateType", GlobalId=guid.new(), Name="Plate type",
                                    PredefinedType=predefined_type)
            f.createIfcRelDefinesByType(guid.new(), None, None, None, (e,), ptype)
        if assembly:
            asm = f.create_entity("IfcElementAssembly", GlobalId=guid.new(), Name=assembly,
                                  ObjectPlacement=f.createIfcLocalPlacement(None, self.a2p(centre)))
            bolts = [self.fastener(f"{assembly} - Bolt {i}", centre, contained=False)
                     for i in range(1, fasteners + 1)]
            f.createIfcRelAggregates(guid.new(), None, None, None, asm, (e, *bolts))
            self.products.append(asm)
        return e

    def _prism(self, cls, name, origin, dx, dy, dz, material=None):
        """Box of (dx, dy, dz) from ``origin``: a dx-by-dy plan outline
        extruded +Z by dz, centred on ``origin`` in plan with its underside at
        its z. The outline is an IfcArbitraryClosedProfileDef over an
        IfcIndexedPolyCurve, which is how Alchemy authors a panel skin - a
        ribbed skin has no rectangular profile to author. The encoding matters
        to a reader: plate_from_extrusion takes the smallest of the three
        characteristic dimensions as the thickness of a *rectangle* profile,
        but an arbitrary profile is treated as a footprint and its extrusion
        depth as the thickness.

        Not contained in the spatial structure: a panel piece belongs to its
        panel.
        """
        f = self.f
        hx, hy = dx / 2.0, dy / 2.0
        outline = f.create_entity(
            "IfcIndexedPolyCurve",
            Points=f.create_entity("IfcCartesianPointList2D", CoordList=(
                (-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy))),
            SelfIntersect=False)
        solid = f.createIfcExtrudedAreaSolid(
            f.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA",
                            OuterCurve=outline),
            self.a2p((0, 0, 0)), f.createIfcDirection((0.0, 0.0, 1.0)), dz)
        return self.product(cls, name, origin, [solid], "SweptSolid",
                            material or self.steel, contained=False)

    def panel(self, cls, name, origin, dx, dy, dz, layers, load_bearing=False):
        """Insulated metal panel as Alchemy authors one: an IfcWall or IfcRoof
        with no body of its own, decomposed through IfcRelAggregates into two
        IfcPlate skins and an IfcBuildingElementPart core per inner layer.

        The build-up and the LoadBearing declaration sit on the type, where
        get_material and get_psets both find them by inheritance. ``layers``
        is laid out along whichever edge of (dx, dy, dz) is thinnest, so a
        wall panel (thin in y) and a roof panel (thin in z) both work.

        Returns ``(container, pieces)``.
        """
        f = self.f
        size = [float(dx), float(dy), float(dz)]
        axis = min(range(3), key=lambda i: size[i])     # the thickness direction

        container = f.create_entity(
            cls, GlobalId=guid.new(), Name=name,
            ObjectPlacement=f.createIfcLocalPlacement(None, self.a2p(origin)))
        wall = cls == "IfcWall"
        ptype = f.create_entity(f"{cls}Type", GlobalId=guid.new(), Name=f"{name} type",
                                PredefinedType="ELEMENTEDWALL" if wall else "FLAT_ROOF",
                                ElementType="INSULATEDMETALPANEL")
        f.createIfcRelDefinesByType(guid.new(), None, None, None, (container,), ptype)
        self.pset(ptype, "Pset_WallCommon" if wall else "Pset_RoofCommon",
                  {"LoadBearing": load_bearing})
        self.mat_of.setdefault(self.layer_usage(layers), []).append(ptype)

        pieces, offset = [], 0.0
        for i, (t, mat_name) in enumerate(layers):
            at = list(origin)
            at[axis] += offset
            box = list(size)
            box[axis] = t
            role = ("Outer Skin" if i == 0 else
                    "Inner Liner" if i == len(layers) - 1 else "Core")
            pieces.append(self._prism(
                "IfcPlate" if role != "Core" else "IfcBuildingElementPart",
                f"{name} - {role}", at, *box,
                material=None if role != "Core" else f.createIfcMaterial(mat_name, None, None)))
            offset += t
        f.createIfcRelAggregates(guid.new(), None, None, None, container, tuple(pieces))
        self.products.append(container)
        return container, pieces
