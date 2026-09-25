"""Containment recognition is representation-agnostic: one containment
(basemat disc + cylindrical wall + hemispherical dome, off-origin, generic
product names, a named equipment rack inside) authored as tessellations,
B-reps, parametric sweeps and CSG must classify and assemble identically."""

import math
from pathlib import Path

import numpy as np
import pytest

from ifc_to_apdl.assemble.containment import assemble_containment
from ifc_to_apdl.assign.materials import MaterialResolver
from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_authoring import Author

CX, CY = 150.0, 50.0                  # plan centre of the containment axis
R_IN, R_OUT, H, T_BASE = 28.0, 29.0, 32.5, 3.0
NUCLEAR_ISLAND = Path(__file__).resolve().parents[1] / "test_models" / "nuclear_island.ifc"


def _revolve(profile, n=64):
    """Closed triangulated surface of revolution of an (r, z) profile loop
    about the local z axis; r == 0 points collapse to a pole vertex."""
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    verts, rings = [], []
    for r, z in profile:
        if r == 0.0:
            rings.append([len(verts)] * n)
            verts.append((0.0, 0.0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + n)))
            verts += [(r * math.cos(a), r * math.sin(a), z) for a in ang]
    faces = []
    for a, b in zip(rings, rings[1:] + rings[:1]):
        for j in range(n):
            k = (j + 1) % n
            for tri in ((a[j], a[k], b[k]), (a[j], b[k], b[j])):
                if len(set(tri)) == 3:
                    faces.append(tri)
    return verts, faces


def _dome_profile(m=12):
    phis = np.linspace(0.0, np.pi / 2, m)
    return ([(R_OUT * math.cos(p), R_OUT * math.sin(p)) for p in phis[:-1]] + [(0.0, R_OUT)]
            + [(0.0, R_IN)] + [(R_IN * math.cos(p), R_IN * math.sin(p)) for p in phis[::-1][1:]])


BODIES = {   # (class, generic name, placement, (r, z) profile)
    "basemat": ("IfcSlab", "IfcSlab #3", (CX, CY, -T_BASE),
                [(0.0, 0.0), (R_OUT, 0.0), (R_OUT, T_BASE), (0.0, T_BASE)]),
    "wall": ("IfcWall", "IfcWall #5", (CX, CY, 0.0),
             [(R_IN, 0.0), (R_OUT, 0.0), (R_OUT, H), (R_IN, H)]),
    "dome": ("IfcRoof", "IfcRoof #1", (CX, CY, H), _dome_profile()),
}


def _tessellated_item(a, verts, faces, kind):
    f = a.f
    if kind == "brep":
        pts = [f.createIfcCartesianPoint(v) for v in verts]
        cfs = [f.createIfcFace((f.createIfcFaceOuterBound(
            f.createIfcPolyLoop([pts[i] for i in tri]), True),)) for tri in faces]
        return f.createIfcFacetedBrep(f.createIfcClosedShell(cfs)), "Brep"
    coords = f.createIfcCartesianPointList3D(verts)
    idx = [[i + 1 for i in tri] for tri in faces]
    if kind == "polygonal":
        return f.createIfcPolygonalFaceSet(
            coords, True, [f.createIfcIndexedPolygonalFace(t) for t in idx], None), "Tessellation"
    return f.createIfcTriangulatedFaceSet(coords, None, True, idx, None), "Tessellation"


def _parametric_item(a, key):
    """Circular extrusions for basemat and wall; CSG sphere Boolean dome."""
    f = a.f
    up = f.createIfcDirection((0.0, 0.0, 1.0))
    if key == "basemat":
        return f.createIfcExtrudedAreaSolid(f.createIfcCircleProfileDef("AREA", None, None, R_OUT),
                                            a.a2p((0, 0, 0)), up, T_BASE), "SweptSolid"
    if key == "wall":
        return f.createIfcExtrudedAreaSolid(
            f.createIfcCircleHollowProfileDef("AREA", None, None, R_OUT, R_OUT - R_IN),
            a.a2p((0, 0, 0)), up, H), "SweptSolid"
    shell = f.createIfcBooleanResult("DIFFERENCE", f.createIfcSphere(a.a2p((0, 0, 0)), R_OUT),
                                     f.createIfcSphere(a.a2p((0, 0, 0)), R_IN))
    below = f.createIfcHalfSpaceSolid(f.createIfcPlane(a.a2p((0, 0, 0))), True)
    return f.createIfcBooleanResult("DIFFERENCE", shell, below), "CSG"


def _revolved_wall(a):
    """Cylinder wall as a revolved rectangle (no parametric cylinder reading:
    recovered through the kernel-triangulation tier)."""
    f = a.f
    prof = f.createIfcRectangleProfileDef(
        "AREA", None, f.createIfcAxis2Placement2D(
            f.createIfcCartesianPoint(((R_IN + R_OUT) / 2, H / 2)), None), R_OUT - R_IN, H)
    axis = f.createIfcAxis1Placement(f.createIfcCartesianPoint((0.0, 0.0, 0.0)),
                                     f.createIfcDirection((0.0, 1.0, 0.0)))
    return f.createIfcRevolvedAreaSolid(prof, a.a2p((0, 0, 0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
                                        axis, 2 * math.pi), "SweptSolid"


ENCODINGS = ("triangulated", "polygonal", "brep", "parametric", "revolved")


def _containment(tmp_path, encoding, building_type=None):
    a = Author(building_type)
    for key, (cls, name, loc, profile) in BODIES.items():
        if encoding == "parametric" or (encoding == "revolved" and key != "wall"):
            item, rep = _parametric_item(a, key)
        elif encoding == "revolved":
            item, rep = _revolved_wall(a)
        else:
            item, rep = _tessellated_item(a, *_revolve(profile), encoding)
        a.product(cls, name, loc, [item], rep, a.concrete)
    a.rack(CX - 5.0, CY - 1.0)          # equipment support inside the shell
    return a.save(tmp_path / f"containment_{encoding}.ifc")


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_every_encoding_classifies_as_containment(tmp_path, encoding):
    ctx = load_ifc(_containment(tmp_path, encoding))
    systems = classify_systems(ctx)
    assert [(s.domain, len(s.product_guids)) for s in systems] == [("containment", 3)]
    ev = systems[0].evidence
    assert ev[0] == "no framing classes besides equipment support racks"
    assert any(e.startswith("cylindrical wall shell") for e in ev)
    assert any(e.startswith("spherical dome shell") for e in ev)
    excluded = [e for e in ctx.audit.entries.values() if e.status == "excluded"]
    assert len(excluded) == 8 and all("Support Rack" in e.name for e in excluded)


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_every_encoding_assembles_the_same_primitives(tmp_path, encoding):
    ctx = load_ifc(_containment(tmp_path, encoding))
    system = classify_systems(ctx)[0]
    cfg = ConversionConfig()
    model = assemble_containment(ctx, system, cfg, MaterialResolver(cfg.materials, ctx.audit))
    vols = {v.prov.ifc_class: v for v in model.volumes}
    assert sorted(vols) == ["IfcRoof", "IfcSlab", "IfcWall"]
    base, wall, dome = vols["IfcSlab"].params, vols["IfcWall"].params, vols["IfcRoof"].params
    assert vols["IfcSlab"].kind == vols["IfcWall"].kind == "cylinder"
    assert vols["IfcRoof"].kind == "spherical_shell"
    for p in (base, wall, dome):          # every body sits on the placed axis
        assert (p["cx"], p["cy"]) == pytest.approx((CX, CY), abs=1e-6)
    assert (base["r_inner"], base["r_outer"], base["z_min"], base["z_max"]) == \
        pytest.approx((0.0, R_OUT, -T_BASE, 0.0), abs=1e-6)
    assert (wall["r_inner"], wall["r_outer"], wall["z_min"], wall["z_max"]) == \
        pytest.approx((R_IN, R_OUT, 0.0, H), abs=1e-6)
    assert (dome["r_inner"], dome["r_outer"], dome["cz"]) == pytest.approx((R_IN, R_OUT, H), abs=1e-6)


def test_framed_building_with_a_round_wall_stays_a_building(tmp_path):
    """One recovered cylinder in a framed building is not enough evidence."""
    a = Author()
    verts, faces = _revolve(BODIES["wall"][3])
    item, rep = _tessellated_item(a, verts, faces, "triangulated")
    a.product("IfcWall", "Stair core", (CX, CY, 0.0), [item], rep, a.concrete)
    for i, (x, y) in enumerate([(0, 0), (6, 0), (6, 6), (0, 6)], 1):
        a.rect_member("IfcColumn", f"Column {i}", (x, y, 0), (x, y, 4))
    a.rect_member("IfcBeam", "Beam 1", (0, 0, 4), (6, 0, 4))
    systems = classify_systems(load_ifc(a.save(tmp_path / "framed.ifc")))
    assert [(s.domain, len(s.product_guids)) for s in systems] == [("building", 6)]


@pytest.mark.skipif(not NUCLEAR_ISLAND.exists(), reason="nuclear_island.ifc not present")
def test_nuclear_island_containment_is_isolated():
    ctx = load_ifc(NUCLEAR_ISLAND)
    systems = classify_systems(ctx)
    domains = sorted(s.domain for s in systems)
    assert domains == ["building", "containment"] + ["piping"] * 6
    cont = next(s for s in systems if s.domain == "containment")
    classes = sorted(ctx.products[g].ifc_class for g in cont.product_guids)
    assert classes == ["IfcRoof", "IfcSlab", "IfcWall"]
    assert "containment naming" in cont.evidence       # IfcBuilding ObjectType REACTOR_CONTAINMENT
    cfg = ConversionConfig()
    model = assemble_containment(ctx, cont, cfg, MaterialResolver(cfg.materials, ctx.audit))
    vols = {v.prov.ifc_class: v.params for v in model.volumes}
    base, wall, dome = vols["IfcSlab"], vols["IfcWall"], vols["IfcRoof"]
    # stacked interfaces coincide exactly, so VADD fuses one body
    assert base["z_max"] == wall["z_min"] and dome["cz"] == wall["z_max"]
    assert base["r_outer"] == wall["r_outer"] == dome["r_outer"]
    assert dome["r_inner"] == wall["r_inner"]
    assert len({(p["cx"], p["cy"]) for p in vols.values()}) == 1
