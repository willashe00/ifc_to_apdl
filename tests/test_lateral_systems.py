"""Global compatibility of building lateral systems.

The three IFC models are authored on the fly by ``tools/author_lateral_models.py``
(realistic as-designed geometry: per-storey walls/columns, walls between
column faces, coupled piers, T-junction walls, stair landings, spliced
columns, X / chevron / single-diagonal bracing). Each converted building
system must be ONE connected body with conformal interfaces everywhere and
no floating brace joints.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.model.ir import AnalyticalModel, Member1D, Provenance, Surface2D
from ifc_to_apdl.model.materials import MaterialDef
from ifc_to_apdl.model.sections import PipeSection, ShellSection
from ifc_to_apdl.pipeline import convert_file
from ifc_to_apdl.verify.connectivity import (conformity_report, connectivity_report,
                                             free_joint_report)

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "author_lateral_models.py"


@pytest.fixture(scope="module")
def authoring():
    spec = importlib.util.spec_from_file_location("author_lateral_models", TOOLS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", params=["LS1_dual_wall_frame", "LS2_coupled_core",
                                        "LS3_braced_steel"])
def converted(request, authoring, tmp_path_factory):
    name = request.param
    root = tmp_path_factory.mktemp(name)
    ifc = authoring.BUILDERS[name](root / f"{name}.ifc")
    results = convert_file(ifc, root / "out", ConversionConfig())
    building = [r for r in results if r.system.domain == "building"]
    assert len(building) == 1, "expected exactly one building system"
    return name, building[0]


def test_single_connected_body(converted):
    name, r = converted
    comps = connectivity_report(r.model)
    assert len(comps) == 1, f"{name}: {len(comps)} components"
    assert comps[0].supported


def test_conformal_interfaces(converted):
    name, r = converted
    hang = conformity_report(r.model)
    assert not hang, f"{name}: {len(hang)} node(s) hanging on another object's edge"


def test_no_floating_brace_joints(converted):
    name, r = converted
    assert free_joint_report(r.model) == []


def test_every_product_converted(converted):
    name, r = converted
    counts = r.audit.counts()
    assert set(counts) == {"converted"}, counts


def test_reconciliation_evidence(converted):
    """The domain rules that make the systems compatible leave evidence."""
    name, r = converted
    ev = [o.prov.evidence for o in (*r.model.members, *r.model.surfaces)]
    if name == "LS1_dual_wall_frame":
        # walls between columns: ends moved onto the column axes
        assert any("column axis" in e.get("end", "") for e in ev)
        # per-storey walls: bases lowered / tops raised to slab mid-planes
        assert any("base" in e for e in ev) and any("top" in e for e in ev)
        # wall vertical edges coincide with column lines -> embedded columns
        cols = [m for m in r.model.members if m.prov.ifc_class == "IfcColumn"]
        col_nodes = {n for m in cols for n in (m.start, m.end)}
        wall_nodes = {n for s in r.model.surfaces if s.prov.ifc_class == "IfcWall"
                      for n in s.loop}
        assert len(col_nodes & wall_nodes) >= 4 * 6 * 2
    if name == "LS2_coupled_core":
        # coupling beams lifted into the slab plane, landing beams into the
        # landing plane, both sharing nodes with the wall panels
        cb = [m for m in r.model.members if m.prov.name.startswith("CB-")]
        lb = [m for m in r.model.members if m.prov.name.startswith("LB-")]
        assert cb and lb
        wall_nodes = {n for s in r.model.surfaces if s.prov.ifc_class == "IfcWall"
                      for n in s.loop}
        for m in cb + lb:
            assert m.start in wall_nodes and m.end in wall_nodes
    if name == "LS3_braced_steel":
        assert any("junction" in e for e in ev)               # X-brace crossings
        assert any("beam axis plane" in e.get("workpoint", "") for e in ev)   # chevron apex


def test_enforce_conformity_splits_and_refines():
    from ifc_to_apdl.assemble.conformity import enforce_conformity

    m = AnalyticalModel("t", "building")
    mid = m.add_material(MaterialDef(0, "s", 2e11, 0.3, 7850.0))
    sid = m.add_section(PipeSection(od=0.2, t=0.01))
    ssid = m.add_section(ShellSection(name="w", layers=[(0.2, mid)]))
    a = m.nodes.get((0, 0, 0))
    b = m.nodes.get((4, 0, 0))
    c = m.nodes.get((4, 0, 3))
    d = m.nodes.get((0, 0, 3))
    m.surfaces.append(Surface2D(1, Provenance("w", "IfcWall"), [a, b, c, d], "SHELL181", ssid, mid))
    # a beam ending on the wall's top edge interior and a node on its base edge
    e = m.nodes.get((2, 0, 3))
    f = m.nodes.get((2, 5, 3))
    m.members.append(Member1D(1, Provenance("b", "IfcBeam"), e, f, "BEAM188", sid, mid))
    g = m.nodes.get((1, 0, 0))
    # a column passing through the wall corner mid-height node
    h = m.nodes.get((4, 0, 1.5))
    m.members.append(Member1D(2, Provenance("c", "IfcColumn"), b, c, "BEAM188", sid, mid))

    assert conformity_report(m)
    n_split, n_ins, edge_map = enforce_conformity(m, 2e-3)
    assert n_split == 1 and n_ins >= 2
    assert conformity_report(m) == []
    assert m.surfaces[0].loop == [a, g, b, h, c, e, d]
    assert edge_map((a, b)) == [(a, g), (g, b)]
    col = [x for x in m.members if x.prov.ifc_class == "IfcColumn"]
    assert {(x.start, x.end) for x in col} == {(b, h), (h, c)}
