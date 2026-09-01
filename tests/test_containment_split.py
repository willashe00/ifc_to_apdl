"""Model splitting around a tessellated containment: shells become the
containment system, equipment support racks inside the shell are discarded,
real framing (carrying slabs / walls, or outside the shell) is kept."""

import importlib.util
import sys
from pathlib import Path

import ifcopenshell.guid as guid
import numpy as np
import pytest

from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.ingest.loader import load_ifc

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "author_lateral_models.py"


@pytest.fixture(scope="module")
def authoring():
    spec = importlib.util.spec_from_file_location("author_lateral_models", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ring(r, z, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(r * np.cos(t), r * np.sin(t), z) for t in a]


def _add_cylinder_faceset(a, name, cls, cx, cy, r_in, r_out, z0, z1, material):
    """Tessellated hollow cylinder (inner + outer skins, no caps needed for
    the body-of-revolution fit) as an IfcTriangulatedFaceSet product."""
    f = a.f
    verts, faces = [], []
    for r in (r_in, r_out):
        base = len(verts)
        lo, hi = _ring(r, z0), _ring(r, z1)
        n = len(lo)
        verts += [(x + cx, y + cy, z) for x, y, z in lo] + [(x + cx, y + cy, z) for x, y, z in hi]
        for i in range(n):
            j = (i + 1) % n
            faces.append((base + i + 1, base + j + 1, base + n + i + 1))
            faces.append((base + j + 1, base + n + j + 1, base + n + i + 1))
    coords = f.createIfcCartesianPointList3D([tuple(float(v) for v in p) for p in verts])
    fs = f.createIfcTriangulatedFaceSet(coords, None, None, faces, None)
    shape = f.createIfcProductDefinitionShape(None, None, (
        f.createIfcShapeRepresentation(a.context, "Body", "Tessellation", (fs,)),))
    e = f.create_entity(cls, GlobalId=guid.new(), Name=name,
                        ObjectPlacement=f.createIfcLocalPlacement(None, a._a2p((0, 0, 0))),
                        Representation=shape)
    a._contained.setdefault(a.storey_for(z0), []).append(e)
    a._mat_uses.setdefault(material, []).append(e)
    return e


def _model(authoring, tmp_path, rack_named: bool, rack_with_slab: bool, outside_frame: bool):
    a = authoring.Author("containment-split")
    conc = a.material("Concrete C40/50", 35e9, 0.2, 2500.0)
    steel = a.material("Structural Steel S355", 210e9, 0.3, 7850.0)
    a.add_storey("Ground", 0.0)
    # containment: basemat disc + wall centred at (30, 30)
    _add_cylinder_faceset(a, "Basemat", "IfcSlab", 30, 30, 0.0, 20.0, -2.0, 0.0, conc)
    _add_cylinder_faceset(a, "Containment Wall", "IfcWall", 30, 30, 19.0, 20.0, 0.0, 30.0, conc)
    # an equipment support cage inside the shell
    tag = "Pump Support Rack - " if rack_named else "Frame "
    col = {"width": 0.3, "depth": 0.3}
    for i, (x, y) in enumerate([(28, 28), (32, 28), (32, 32), (28, 32)]):
        a.column(f"{tag}Column {i + 1}", (x, y, 0.0), 4.0, "rect", col, steel)
    a.beam(f"{tag}Beam 1", (28.15, 28, 3.8), (31.85, 28, 3.8), "rect", {"width": 0.2, "depth": 0.4}, steel)
    a.beam(f"{tag}Beam 2", (28.15, 32, 3.8), (31.85, 32, 3.8), "rect", {"width": 0.2, "depth": 0.4}, steel)
    if rack_with_slab:
        a.slab("Internal Floor", [(27.9, 27.9), (32.1, 27.9), (32.1, 32.1), (27.9, 32.1)],
               4.2, 0.2, conc)
    if outside_frame:
        # a real frame outside the shell footprint
        for i, (x, y) in enumerate([(60, 0), (66, 0), (66, 6), (60, 6)]):
            a.column(f"Aux Column {i + 1}", (x, y, 0.0), 4.0, "rect", col, steel)
        a.beam("Aux Beam", (60.15, 0, 3.8), (65.85, 0, 3.8), "rect", {"width": 0.2, "depth": 0.4}, steel)
    return a.save(tmp_path / "split.ifc")


def _domains(path):
    ctx = load_ifc(path)
    systems = classify_systems(ctx)
    excluded = {e.name for e in ctx.audit.entries.values() if e.status == "excluded"}
    return {s.name: (s.domain, len(s.product_guids)) for s in systems}, excluded


def test_named_rack_inside_shell_is_discarded(authoring, tmp_path):
    systems, excluded = _domains(_model(authoring, tmp_path, True, False, False))
    assert [d for d, _ in systems.values()] == ["containment"]
    assert len(excluded) == 6


def test_bare_cage_inside_shell_is_discarded(authoring, tmp_path):
    systems, excluded = _domains(_model(authoring, tmp_path, False, False, False))
    assert [d for d, _ in systems.values()] == ["containment"]
    assert len(excluded) == 6


def test_internal_frame_with_slab_is_kept(authoring, tmp_path):
    systems, excluded = _domains(_model(authoring, tmp_path, False, True, False))
    assert sorted(d for d, _ in systems.values()) == ["building", "containment"]
    assert not excluded


def test_frame_outside_shell_is_kept_while_rack_is_discarded(authoring, tmp_path):
    systems, excluded = _domains(_model(authoring, tmp_path, True, False, True))
    doms = {d: n for d, n in systems.values()}
    assert doms == {"containment": 2, "building": 5}
    assert len(excluded) == 6
