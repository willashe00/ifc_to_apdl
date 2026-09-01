"""Author realistic multi-storey IFC4 building models with different lateral
systems, for compatibility (global connectivity) testing of the converter.

    python tools/author_lateral_models.py [out_dir]      (default: test_models/lateral_systems)

All models follow as-designed architectural conventions - physical bodies,
no overlapping geometry:
  * levels are top-of-slab; slabs are extruded down from the level;
  * beams end at column / wall FACES, beam top flush with the slab soffit;
  * per-storey columns and walls span from a level (top of slab) to the next
    slab soffit, so the slab passes continuously over them (LS1, LS2), or
    columns are continuous with splices above floor level (LS3);
  * walls between columns stop at the column faces; walls carrying a slab
    edge have the slab edge on their outer face; coupled piers are separate
    walls with a coupling beam spanning the opening between pier end faces;
  * braces carry a trimmed body (gusset cut-back) plus an Axis representation
    running workpoint to workpoint; X-bracing is one continuous diagonal and
    one interrupted diagonal in two pieces, meeting at the crossing.

Models
  LS1_dual_wall_frame   RC frame + perimeter shear walls between columns + a
                        stair enclosure (U-walls) straddling a column gridline,
                        slab opening inside the enclosure. Per-storey members.
  LS2_coupled_core      RC frame (continuous columns) + closed core with coupled
                        piers, interior cross wall (T-junctions), stair landing
                        beam + landing slab framing into the core at mid-storey.
  LS3_braced_steel      Steel frame, spliced columns, deep girders, infill
                        beams, composite slab; X-braced, chevron-braced and
                        single-diagonal bays.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import ifcopenshell
import ifcopenshell.guid as guid
import numpy as np

Vec3 = tuple[float, float, float]


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def section_frame(direction):
    a = np.asarray(direction, dtype=float)
    a = a / np.linalg.norm(a)
    if abs(a[2]) < 0.9:
        x_hat = np.cross(np.array([0.0, 0.0, 1.0]), a)
        x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(a, x_hat)
    else:
        y_hat = np.cross(a, np.array([1.0, 0.0, 0.0]))
        y_hat /= np.linalg.norm(y_hat)
        x_hat = np.cross(y_hat, a)
    return x_hat, y_hat


class Author:
    """Minimal IFC4 authoring with architectural (physical-body) conventions."""

    def __init__(self, name: str):
        self.f = f = ifcopenshell.file(schema="IFC4")
        units = [f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE"),
                 f.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE"),
                 f.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE"),
                 f.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")]
        origin = f.createIfcCartesianPoint((0.0, 0.0, 0.0))
        wcs = f.createIfcAxis2Placement3D(origin, None, None)
        self.context = f.createIfcGeometricRepresentationContext(
            None, "Model", 3, 1e-4, wcs, None)
        self.project = f.createIfcProject(guid.new(), None, name, None, None, None, None,
                                          (self.context,), f.createIfcUnitAssignment(units))
        site_pl = f.createIfcLocalPlacement(None, self._a2p((0, 0, 0)))
        self.site = f.createIfcSite(guid.new(), None, "Site", None, None, site_pl,
                                    None, None, "ELEMENT", None, None, None, None, None)
        f.createIfcRelAggregates(guid.new(), None, None, None, self.project, (self.site,))
        self.building = f.createIfcBuilding(
            guid.new(), None, name, None, None,
            f.createIfcLocalPlacement(site_pl, self._a2p((0, 0, 0))),
            None, None, "ELEMENT", None, None, None)
        f.createIfcRelAggregates(guid.new(), None, None, None, self.site, (self.building,))
        self.storeys: list[tuple[float, object]] = []
        self._contained: dict[object, list] = {}
        self._mat_uses: dict[object, list] = {}

    # -- spatial ------------------------------------------------------------
    def add_storey(self, name, elevation):
        s = self.f.createIfcBuildingStorey(
            guid.new(), None, name, None, None,
            self.f.createIfcLocalPlacement(self.building.ObjectPlacement,
                                           self._a2p((0, 0, elevation))),
            None, None, "ELEMENT", float(elevation))
        self.f.createIfcRelAggregates(guid.new(), None, None, None, self.building, (s,))
        self.storeys.append((elevation, s))
        return s

    def storey_for(self, z):
        below = [(e, s) for e, s in self.storeys if e <= z + 1e-6]
        return (below[-1] if below else self.storeys[0])[1]

    # -- primitives ---------------------------------------------------------
    def _pt(self, p):
        return self.f.createIfcCartesianPoint(tuple(float(v) for v in p))

    def _dir(self, d):
        return self.f.createIfcDirection(tuple(float(v) for v in d))

    def _a2p(self, loc, axis=None, ref=None):
        return self.f.createIfcAxis2Placement3D(
            self._pt(loc), self._dir(axis) if axis is not None else None,
            self._dir(ref) if ref is not None else None)

    def _shape(self, items, axis_pts=None):
        reps = []
        if axis_pts is not None:
            poly = self.f.createIfcPolyline(tuple(self._pt(p) for p in axis_pts))
            reps.append(self.f.createIfcShapeRepresentation(
                self.context, "Axis", "Curve3D", (poly,)))
        reps.append(self.f.createIfcShapeRepresentation(
            self.context, "Body", "SweptSolid", tuple(items)))
        return self.f.createIfcProductDefinitionShape(None, None, tuple(reps))

    def _element(self, cls, name, shape, predefined, material, z):
        e = self.f.create_entity(
            cls, GlobalId=guid.new(), Name=name,
            ObjectPlacement=self.f.createIfcLocalPlacement(None, self._a2p((0, 0, 0))),
            Representation=shape, PredefinedType=predefined)
        self._contained.setdefault(self.storey_for(z), []).append(e)
        if material is not None:
            self._mat_uses.setdefault(material, []).append(e)
        return e

    def material(self, name, E, nu, rho):
        f = self.f
        mat = f.createIfcMaterial(name, None, None)
        f.createIfcMaterialProperties(
            "Pset_MaterialCommon", None,
            (f.createIfcPropertySingleValue("MassDensity", None,
                                            f.createIfcMassDensityMeasure(rho), None),), mat)
        f.createIfcMaterialProperties(
            "Pset_MaterialMechanical", None,
            (f.createIfcPropertySingleValue("YoungModulus", None,
                                            f.createIfcModulusOfElasticityMeasure(E), None),
             f.createIfcPropertySingleValue("PoissonRatio", None,
                                            f.createIfcPositiveRatioMeasure(nu), None)), mat)
        return mat

    def profile(self, family, p, name=""):
        f = self.f
        if family == "rect":
            return f.create_entity("IfcRectangleProfileDef", ProfileType="AREA",
                                   ProfileName=name, XDim=p["width"], YDim=p["depth"])
        if family == "i":
            return f.create_entity("IfcIShapeProfileDef", ProfileType="AREA",
                                   ProfileName=name, OverallWidth=p["width"],
                                   OverallDepth=p["depth"], WebThickness=p["tw"],
                                   FlangeThickness=p["tf"])
        if family == "rect_hollow":
            return f.create_entity("IfcRectangleHollowProfileDef", ProfileType="AREA",
                                   ProfileName=name, XDim=p["width"], YDim=p["depth"],
                                   WallThickness=p["t"])
        raise ValueError(family)

    def _extruded(self, profile, start, direction, length):
        x_hat, y_hat = section_frame(direction)
        return self.f.createIfcExtrudedAreaSolid(
            profile, self._a2p(start, axis=unit(direction), ref=tuple(x_hat)),
            self._dir((0.0, 0.0, 1.0)), float(length))

    # -- members ------------------------------------------------------------
    def beam(self, name, p1, p2, family, params, material, profile_name=""):
        d = tuple(b - a for a, b in zip(p1, p2))
        solid = self._extruded(self.profile(family, params, profile_name), p1, unit(d),
                               math.dist(p1, p2))
        return self._element("IfcBeam", name, self._shape([solid]), "BEAM", material, p1[2])

    def column(self, name, base, height, family, params, material, profile_name=""):
        solid = self._extruded(self.profile(family, params, profile_name), base, (0, 0, 1),
                               height)
        return self._element("IfcColumn", name, self._shape([solid]), "COLUMN", material,
                             base[2])

    def brace(self, name, wp1, wp2, family, params, material, cutback=0.3, profile_name=""):
        d = unit(tuple(b - a for a, b in zip(wp1, wp2)))
        length = math.dist(wp1, wp2)
        b1 = tuple(a + cutback * dc for a, dc in zip(wp1, d))
        solid = self._extruded(self.profile(family, params, profile_name), b1, d,
                               max(length - 2 * cutback, 0.1))
        return self._element("IfcMember", name, self._shape([solid], axis_pts=[wp1, wp2]),
                             "BRACE", material, min(wp1[2], wp2[2]))

    def slab(self, name, polygon, z_top, thickness, material, holes=None):
        f = self.f
        p2 = lambda p: f.createIfcCartesianPoint((float(p[0]), float(p[1])))
        outer = f.createIfcPolyline(tuple([p2(p) for p in polygon] + [p2(polygon[0])]))
        if holes:
            inner = tuple(f.createIfcPolyline(tuple([p2(p) for p in h] + [p2(h[0])]))
                          for h in holes)
            prof = f.createIfcArbitraryProfileDefWithVoids("AREA", "Slab", outer, inner)
        else:
            prof = f.createIfcArbitraryClosedProfileDef("AREA", "Slab", outer)
        solid = f.createIfcExtrudedAreaSolid(prof, self._a2p((0, 0, z_top - thickness)),
                                             self._dir((0.0, 0.0, 1.0)), float(thickness))
        return self._element("IfcSlab", name, self._shape([solid]), "FLOOR", material,
                             z_top - thickness)

    def wall(self, name, p1, p2, z_base, height, thickness, material):
        """Wall by mid-plane trace (p1 -> p2, plan) from z_base, extruded up."""
        d = (p2[0] - p1[0], p2[1] - p1[1])
        length = math.hypot(*d)
        center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, z_base)
        ref = (d[0] / length, d[1] / length, 0.0)
        solid = self.f.createIfcExtrudedAreaSolid(
            self.f.createIfcRectangleProfileDef("AREA", "Wall", None, length, thickness),
            self._a2p(center, axis=(0, 0, 1), ref=ref),
            self._dir((0.0, 0.0, 1.0)), float(height))
        return self._element("IfcWall", name, self._shape([solid]), "SOLIDWALL", material,
                             z_base)

    def save(self, path):
        f = self.f
        for storey, elems in self._contained.items():
            f.createIfcRelContainedInSpatialStructure(guid.new(), None, None, None,
                                                      tuple(elems), storey)
        for mat, elems in self._mat_uses.items():
            f.createIfcRelAssociatesMaterial(guid.new(), None, None, None, tuple(elems), mat)
        f.write(str(path))
        return path


# ---------------------------------------------------------------------------
# helpers shared by the RC models
# ---------------------------------------------------------------------------

CONCRETE = ("Concrete C30/37", 33.0e9, 0.2, 2500.0)
STEEL = ("Structural Steel ASTM A992", 200.0e9, 0.3, 7850.0)


def _rc_beam(a, name, p1, p2, z_top, b, d, mat, trim1, trim2):
    """Rectangular RC beam between two plan points, trimmed back from each end
    by the abutting member half-width, top of beam at z_top."""
    d2 = unit((p2[0] - p1[0], p2[1] - p1[1], 0.0))
    q1 = (p1[0] + trim1 * d2[0], p1[1] + trim1 * d2[1], z_top - d / 2)
    q2 = (p2[0] - trim2 * d2[0], p2[1] - trim2 * d2[1], z_top - d / 2)
    return a.beam(name, q1, q2, "rect", {"width": b, "depth": d}, mat,
                  profile_name=f"RC {int(b*1000)}x{int(d*1000)}")


def _wall_between(a, name, p1, p2, z_base, height, t, mat, trim1, trim2):
    d2 = unit((p2[0] - p1[0], p2[1] - p1[1], 0.0))
    q1 = (p1[0] + trim1 * d2[0], p1[1] + trim1 * d2[1])
    q2 = (p2[0] - trim2 * d2[0], p2[1] - trim2 * d2[1])
    return a.wall(name, q1, q2, z_base, height, t, mat)


# ---------------------------------------------------------------------------
# LS1 - RC dual system: moment frame + perimeter shear walls + stair enclosure
# ---------------------------------------------------------------------------

def build_ls1(path: Path):
    a = Author("LS1_dual_wall_frame")
    conc = a.material(*CONCRETE)
    xs = [0.0, 6.0, 12.0, 18.0, 24.0]
    ys = [0.0, 6.0, 12.0, 18.0]
    storey_h, n_st = 4.0, 4
    levels = [storey_h * (i + 1) for i in range(n_st)]          # top-of-slab
    t_slab, b_beam, d_beam, c_col, t_wall = 0.20, 0.30, 0.60, 0.40, 0.30
    a.add_storey("Ground", 0.0)
    for i, z in enumerate(levels[:-1]):
        a.add_storey(f"Level {i + 2}", z)
    a.add_storey("Roof", levels[-1])

    # wall bays: (axis, fixed coordinate, from, to) - between column faces
    wall_bays = [("x", 0.0, 6.0, 12.0), ("x", 0.0, 12.0, 18.0),
                 ("x", 18.0, 6.0, 12.0), ("x", 18.0, 12.0, 18.0),
                 ("y", 0.0, 6.0, 12.0), ("y", 24.0, 6.0, 12.0)]

    def is_wall_bay(axis, fixed, u0, u1):
        return any(ax == axis and abs(fx - fixed) < 1e-9 and abs(a0 - u0) < 1e-9
                   for ax, fx, a0, _ in wall_bays)

    # stair enclosure straddling gridline x = 12: U-walls at x = 10.5, 13.5
    # (y 7.5 -> 10.5) and y = 7.5 (x 10.5 -> 13.5); slab opening inside
    enc_x0, enc_x1, enc_y0, enc_y1 = 10.5, 13.5, 7.5, 10.5
    hole = [(enc_x0 + t_wall / 2, enc_y0 + t_wall / 2), (enc_x1 - t_wall / 2, enc_y0 + t_wall / 2),
            (enc_x1 - t_wall / 2, enc_y1), (enc_x0 + t_wall / 2, enc_y1)]

    for s in range(n_st):
        z0 = 0.0 if s == 0 else levels[s - 1]        # top of slab below
        z_top = levels[s]                            # top of this storey's slab
        soffit = z_top - t_slab
        h_clear = soffit - z0
        # columns (per storey, slab passes over)
        for x in xs:
            for y in ys:
                a.column(f"C-{x:g}-{y:g}-S{s + 1}", (x, y, z0), h_clear, "rect",
                         {"width": c_col, "depth": c_col}, conc, "RC 400x400")
        # beams on gridlines (omitted in wall bays), ends at column faces
        for y in ys:
            for x0, x1 in zip(xs, xs[1:]):
                if is_wall_bay("x", y, x0, x1):
                    continue
                _rc_beam(a, f"BX-{x0:g}-{y:g}-S{s + 1}", (x0, y), (x1, y), soffit,
                         b_beam, d_beam, conc, c_col / 2, c_col / 2)
        for x in xs:
            for y0, y1 in zip(ys, ys[1:]):
                if is_wall_bay("y", x, y0, y1):
                    continue
                _rc_beam(a, f"BY-{x:g}-{y0:g}-S{s + 1}", (x, y0), (x, y1), soffit,
                         b_beam, d_beam, conc, c_col / 2, c_col / 2)
        # perimeter shear walls between column faces
        for ax, fx, u0, u1 in wall_bays:
            p1 = (u0, fx) if ax == "x" else (fx, u0)
            p2 = (u1, fx) if ax == "x" else (fx, u1)
            _wall_between(a, f"SW-{ax}{fx:g}-{u0:g}-S{s + 1}", p1, p2, z0, h_clear,
                          t_wall, conc, c_col / 2, c_col / 2)
        # stair enclosure walls (free ends, not on the column grid)
        a.wall(f"ST-W-S{s + 1}", (enc_x0, enc_y0 - t_wall / 2), (enc_x0, enc_y1), z0,
               h_clear, t_wall, conc)
        a.wall(f"ST-E-S{s + 1}", (enc_x1, enc_y0 - t_wall / 2), (enc_x1, enc_y1), z0,
               h_clear, t_wall, conc)
        a.wall(f"ST-S-S{s + 1}", (enc_x0 + t_wall / 2, enc_y0), (enc_x1 - t_wall / 2, enc_y0),
               z0, h_clear, t_wall, conc)
        # slab: edge flush with the perimeter beam / wall outer face
        e = b_beam / 2
        outline = [(xs[0] - e, ys[0] - e), (xs[-1] + e, ys[0] - e),
                   (xs[-1] + e, ys[-1] + e), (xs[0] - e, ys[-1] + e)]
        a.slab(f"Slab-L{s + 1}", outline, z_top, t_slab, conc,
               holes=None if s == n_st - 1 else [hole])
    return a.save(path)


# ---------------------------------------------------------------------------
# LS2 - RC frame with a coupled core, T-junction cross wall, stair landing
# ---------------------------------------------------------------------------

def build_ls2(path: Path):
    a = Author("LS2_coupled_core")
    conc = a.material(*CONCRETE)
    xs = [0.0, 7.5, 15.0, 22.5]
    ys = [0.0, 7.5, 15.0, 22.5]
    storey_h, n_st = 3.5, 6
    levels = [storey_h * (i + 1) for i in range(n_st)]
    t_slab, b_beam, d_beam, c_col, t_wall = 0.20, 0.30, 0.55, 0.50, 0.30
    a.add_storey("Ground", 0.0)
    for i, z in enumerate(levels[:-1]):
        a.add_storey(f"Level {i + 2}", z)
    a.add_storey("Roof", levels[-1])

    # continuous columns, full height to the roof soffit
    for x in xs:
        for y in ys:
            a.column(f"C-{x:g}-{y:g}", (x, y, 0.0), levels[-1] - t_slab, "rect",
                     {"width": c_col, "depth": c_col}, conc, "RC 500x500")

    # core: mid-planes x = 9, 14 ; y = 9, 14 (inside bay 7.5..15 both ways)
    cx0, cx1, cy0, cy1 = 9.0, 14.0, 9.0, 14.0
    door_x0, door_x1 = 10.8, 12.2                 # coupled piers on the south face
    cross_y = 11.5                                # interior cross wall (T-junctions)

    for s in range(n_st):
        z0 = 0.0 if s == 0 else levels[s - 1]
        z_top = levels[s]
        soffit = z_top - t_slab
        h_clear = soffit - z0
        # gridline beams, ends at column faces
        for y in ys:
            for x0, x1 in zip(xs, xs[1:]):
                _rc_beam(a, f"BX-{x0:g}-{y:g}-S{s + 1}", (x0, y), (x1, y), soffit,
                         b_beam, d_beam, conc, c_col / 2, c_col / 2)
        for x in xs:
            for y0, y1 in zip(ys, ys[1:]):
                _rc_beam(a, f"BY-{x:g}-{y0:g}-S{s + 1}", (x, y0), (x, y1), soffit,
                         b_beam, d_beam, conc, c_col / 2, c_col / 2)
        # core walls (per storey): west / east / north full; south = 2 piers
        a.wall(f"CW-W-S{s + 1}", (cx0, cy0 - t_wall / 2), (cx0, cy1 + t_wall / 2), z0,
               h_clear, t_wall, conc)
        a.wall(f"CW-E-S{s + 1}", (cx1, cy0 - t_wall / 2), (cx1, cy1 + t_wall / 2), z0,
               h_clear, t_wall, conc)
        a.wall(f"CW-N-S{s + 1}", (cx0 + t_wall / 2, cy1), (cx1 - t_wall / 2, cy1), z0,
               h_clear, t_wall, conc)
        a.wall(f"CW-S1-S{s + 1}", (cx0 + t_wall / 2, cy0), (door_x0, cy0), z0,
               h_clear, t_wall, conc)
        a.wall(f"CW-S2-S{s + 1}", (door_x1, cy0), (cx1 - t_wall / 2, cy0), z0,
               h_clear, t_wall, conc)
        # coupling beam over the door: 300 x 600, top at the soffit, between pier ends
        a.beam(f"CB-S{s + 1}", (door_x0, cy0, soffit - 0.3), (door_x1, cy0, soffit - 0.3),
               "rect", {"width": t_wall, "depth": 0.6}, conc, "RC 300x600 coupling")
        # interior cross wall between west and east walls (T-junctions)
        a.wall(f"CW-X-S{s + 1}", (cx0 + t_wall / 2, cross_y), (cx1 - t_wall / 2, cross_y),
               z0, h_clear, t_wall, conc)
        # stair landing at mid-storey in the north half of the core: landing beam
        # spanning west -> east wall faces, landing slab north of it to the
        # north wall face
        z_land = z0 + storey_h / 2                 # landing top of slab
        lb_d, l_t = 0.40, 0.18
        y_lb = cross_y + 1.6
        a.beam(f"LB-S{s + 1}", (cx0 + t_wall / 2, y_lb, z_land - l_t - lb_d / 2),
               (cx1 - t_wall / 2, y_lb, z_land - l_t - lb_d / 2),
               "rect", {"width": 0.25, "depth": lb_d}, conc, "RC 250x400 landing")
        a.slab(f"Landing-S{s + 1}",
               [(cx0 + t_wall / 2, y_lb - 0.125), (cx1 - t_wall / 2, y_lb - 0.125),
                (cx1 - t_wall / 2, cy1 - t_wall / 2), (cx0 + t_wall / 2, cy1 - t_wall / 2)],
               z_land, l_t, conc)
        # floor slab with the core opening (stair/lift shaft) south of the cross wall
        e = b_beam / 2
        outline = [(xs[0] - e, ys[0] - e), (xs[-1] + e, ys[0] - e),
                   (xs[-1] + e, ys[-1] + e), (xs[0] - e, ys[-1] + e)]
        shaft = [(cx0 + t_wall / 2, cy0 + t_wall / 2), (cx1 - t_wall / 2, cy0 + t_wall / 2),
                 (cx1 - t_wall / 2, cross_y - t_wall / 2), (cx0 + t_wall / 2, cross_y - t_wall / 2)]
        a.slab(f"Slab-L{s + 1}", outline, z_top, t_slab, conc, holes=[shaft])
    return a.save(path)


# ---------------------------------------------------------------------------
# LS3 - braced steel frame: spliced columns, X / chevron / single diagonals
# ---------------------------------------------------------------------------

W14X90 = {"depth": 0.356, "width": 0.369, "tw": 0.011, "tf": 0.018}
W24X76 = {"depth": 0.608, "width": 0.228, "tw": 0.011, "tf": 0.017}
W16X31 = {"depth": 0.403, "width": 0.140, "tw": 0.007, "tf": 0.011}
HSS8 = {"width": 0.203, "depth": 0.203, "t": 0.0125}


def build_ls3(path: Path):
    a = Author("LS3_braced_steel")
    steel = a.material(*STEEL)
    conc = a.material("Normal Weight Concrete 4000 psi", 24.9e9, 0.2, 2400.0)
    xs = [0.0, 8.0, 16.0, 24.0, 32.0]
    ys = [0.0, 8.0, 16.0]
    storey_h, n_st = 4.0, 5
    levels = [storey_h * (i + 1) for i in range(n_st)]
    t_slab = 0.15
    a.add_storey("Ground", 0.0)
    for i, z in enumerate(levels[:-1]):
        a.add_storey(f"Level {i + 2}", z)
    a.add_storey("Roof", levels[-1])
    col_half = W14X90["depth"] / 2                  # face trim for beams
    g_d = W24X76["depth"]

    # spliced columns: splices 1.2 m above levels 2 and 4
    top_of_steel_roof = levels[-1] - t_slab
    splices = [0.0, levels[1] + 1.2, levels[3] + 1.2, top_of_steel_roof]
    for x in xs:
        for y in ys:
            for i, (z0, z1) in enumerate(zip(splices, splices[1:])):
                a.column(f"C-{x:g}-{y:g}-P{i + 1}", (x, y, z0), z1 - z0, "i", W14X90,
                         steel, "W14X90")

    def beam_centroid(z_top):
        return z_top - t_slab - g_d / 2

    for s in range(n_st):
        z_top = levels[s]
        tos = z_top - t_slab                        # top of steel
        zc = beam_centroid(z_top)
        # girders on gridlines, ends at column faces
        for y in ys:
            for x0, x1 in zip(xs, xs[1:]):
                a.beam(f"G-{x0:g}-{y:g}-S{s + 1}", (x0 + col_half, y, zc), (x1 - col_half, y, zc),
                       "i", W24X76, steel, "W24X76")
        for x in xs:
            for y0, y1 in zip(ys, ys[1:]):
                a.beam(f"G-{x:g}-{y0:g}-S{s + 1}", (x, y0 + col_half, zc), (x, y1 - col_half, zc),
                       "i", W24X76, steel, "W24X76")
        # infill beams at bay mid (x = 4, 12, ...), top flush, framing into girders
        zi = tos - W16X31["depth"] / 2
        for x0, x1 in zip(xs, xs[1:]):
            xm = (x0 + x1) / 2
            for y0, y1 in zip(ys, ys[1:]):
                a.beam(f"B-{xm:g}-{y0:g}-S{s + 1}",
                       (xm, y0 + W24X76["width"] / 2, zi), (xm, y1 - W24X76["width"] / 2, zi),
                       "i", W16X31, steel, "W16X31")
        # composite slab to the perimeter girder outer faces
        e = W24X76["width"] / 2
        a.slab(f"Slab-L{s + 1}", [(xs[0] - e, ys[0] - e), (xs[-1] + e, ys[0] - e),
                                  (xs[-1] + e, ys[-1] + e), (xs[0] - e, ys[-1] + e)],
               z_top, t_slab, conc)

        # braces (workpoints at column CL x beam CL)
        z_lo = 0.0 if s == 0 else beam_centroid(levels[s - 1])
        z_hi = zc
        for y in (ys[0], ys[-1]):
            # X-bracing in bay 0..8: continuous diagonal + interrupted diagonal
            xa, xb = xs[0], xs[1]
            xm, zm = (xa + xb) / 2, (z_lo + z_hi) / 2
            a.brace(f"XB1-y{y:g}-S{s + 1}", (xa, y, z_lo), (xb, y, z_hi), "rect_hollow", HSS8,
                    steel, profile_name="HSS8x8x1/2")
            a.brace(f"XB2a-y{y:g}-S{s + 1}", (xb, y, z_lo), (xm, y, zm), "rect_hollow", HSS8,
                    steel, profile_name="HSS8x8x1/2")
            a.brace(f"XB2b-y{y:g}-S{s + 1}", (xm, y, zm), (xa, y, z_hi), "rect_hollow", HSS8,
                    steel, profile_name="HSS8x8x1/2")
            # chevron bracing in bay 24..32, apex at girder mid-span
            xa, xb = xs[3], xs[4]
            xm = (xa + xb) / 2
            a.brace(f"CV1-y{y:g}-S{s + 1}", (xa, y, z_lo), (xm, y, z_hi), "rect_hollow", HSS8,
                    steel, profile_name="HSS8x8x1/2")
            a.brace(f"CV2-y{y:g}-S{s + 1}", (xb, y, z_lo), (xm, y, z_hi), "rect_hollow", HSS8,
                    steel, profile_name="HSS8x8x1/2")
        # single diagonals in the end bays y 0..8 on x = 0 and x = 32 (alternating)
        for x in (xs[0], xs[-1]):
            ya, yb = (ys[0], ys[1]) if s % 2 == 0 else (ys[1], ys[0])
            a.brace(f"D-x{x:g}-S{s + 1}", (x, ya, z_lo), (x, yb, z_hi), "rect_hollow", HSS8,
                    steel, profile_name="HSS8x8x1/2")
    return a.save(path)


BUILDERS = {"LS1_dual_wall_frame": build_ls1,
            "LS2_coupled_core": build_ls2,
            "LS3_braced_steel": build_ls3}


def main(argv):
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "test_models" / "lateral_systems"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        p = fn(out / f"{name}.ifc")
        print("wrote", p)


if __name__ == "__main__":
    main(sys.argv)
