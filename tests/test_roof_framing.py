"""Roof framing forms one continuous load path: truss web members cut to the
chord faces land on the chord axes, purlins bearing on top of chords / eave
beams are noded at the crossings, eave beams running through interior
columns attach to them, and a framing group with flush tops but different
depths shares its top-of-steel plane."""

from pathlib import Path

import pytest

from ifc_to_apdl.assemble.building import assemble_building
from ifc_to_apdl.assign.materials import MaterialResolver
from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_to_apdl.verify.connectivity import (conformity_report, connectivity_report,
                                             free_joint_report)
from ifc_authoring import Author

TOP = 6.0                     # top of steel of the roof framing
NUCLEAR_ISLAND = Path(__file__).resolve().parents[1] / "test_models" / "nuclear_island.ifc"
MODEL_33 = Path(__file__).resolve().parents[1] / "test_models" / "model (33).ifc"


def _assemble(path):
    ctx = load_ifc(path)
    system = next(s for s in classify_systems(ctx) if s.domain == "building")
    cfg = ConversionConfig()
    return assemble_building(ctx, system, cfg, MaterialResolver(cfg.materials, ctx.audit))


def _one_rigid_body(model):
    comps = connectivity_report(model)
    assert len(comps) == 1 and comps[0].supported
    assert not free_joint_report(model)
    assert not conformity_report(model)


def _centroids_kept(model, a):
    """Every member's node plane + SECOFFSET equals its authored centroid."""
    for m in model.members:
        if m.prov.ifc_class != "IfcBeam":
            continue
        z_node = model.nodes.xyz(m.start)[2]
        z_true = next(v for k, v in a.centroid_z.items() if k == m.prov.name)
        assert z_node + model.sections[m.section].offset_y == pytest.approx(z_true, abs=1e-3)


def test_truss_purlins_and_eave_form_one_body(tmp_path):
    """Pratt truss (web cut to chord faces) on columns at y=0, an eave beam
    at y=4 running through a middle column, purlins bearing on both."""
    a = Author()
    for x, y in ((0, 0), (12, 0), (0, 4), (6, 4), (12, 4)):
        a.rect_member("IfcColumn", f"Column {x}-{y}", (x, y, 0.0), (x, y, TOP))
    d = 0.3                                                   # chord / eave depth
    a.rect_member("IfcMember", "Top chord", (0.15, 0, TOP - d / 2), (11.85, 0, TOP - d / 2), d=d)
    a.rect_member("IfcMember", "Bottom chord", (2, 0, 4 - d / 2), (10, 0, 4 - d / 2), d=d)
    for x in (2, 4, 6, 8, 10):                                # verticals face to face
        a.rect_member("IfcMember", f"Vertical {x}", (x, 0, 4.0), (x, 0, TOP - d), b=0.15, d=0.15)
    for x0, x1 in ((0.15, 2), (2, 4), (4, 6)):                # diagonals face to face
        a.rect_member("IfcMember", f"Diagonal {x0}", (x0, 0, TOP - d), (x1, 0, 4.0), b=0.15, d=0.15)
    for x0, x1 in ((6, 8), (8, 10), (10, 11.85)):
        a.rect_member("IfcMember", f"Diagonal {x0}", (x0, 0, 4.0), (x1, 0, TOP - d), b=0.15, d=0.15)
    a.rect_member("IfcBeam", "Eave", (0.15, 4, TOP - d / 2), (11.85, 4, TOP - d / 2), d=d)
    a.centroid_z = {"Eave": TOP - d / 2}
    for k, x in enumerate((1.0, 3.0, 5.0, 7.0, 9.0, 11.0)):
        a.rect_member("IfcBeam", f"Purlin {k}", (x, -0.3, TOP + 0.1), (x, 4.3, TOP + 0.1),
                      b=0.1, d=0.2)
        a.centroid_z[f"Purlin {k}"] = TOP + 0.1
    model = _assemble(a.save(tmp_path / "truss.ifc"))
    _one_rigid_body(model)
    _centroids_kept(model, a)
    # the middle column is noded where the eave beam runs through it
    mid = [m for m in model.members if m.prov.name == "Column 6-4"]
    col_zs = {round(model.nodes.xyz(n)[2], 4) for m in mid for n in (m.start, m.end)}
    assert round(TOP - d / 2, 4) in col_zs
    eave_nodes = {n for m in model.members if m.prov.name == "Eave" for n in (m.start, m.end)}
    assert eave_nodes & {n for m in mid for n in (m.start, m.end)}


def test_flush_top_framing_group_shares_top_of_steel(tmp_path):
    """A deep girder and a shallow eave beam, tops flush: purlins bearing on
    both put the whole group on the top-of-steel plane."""
    a = Author()
    for x, y in ((0, 0), (12, 0), (0, 5), (12, 5)):
        a.rect_member("IfcColumn", f"Column {x}-{y}", (x, y, 0.0), (x, y, TOP))
    a.rect_member("IfcBeam", "Girder", (0.15, 0, TOP - 0.3), (11.85, 0, TOP - 0.3), b=0.2, d=0.6)
    a.rect_member("IfcBeam", "Eave", (0.15, 5, TOP - 0.15), (11.85, 5, TOP - 0.15), d=0.3)
    a.centroid_z = {"Girder": TOP - 0.3, "Eave": TOP - 0.15}
    for k, x in enumerate((1.5, 4.5, 7.5, 10.5)):
        a.rect_member("IfcBeam", f"Purlin {k}", (x, -0.3, TOP + 0.1), (x, 5.3, TOP + 0.1),
                      b=0.1, d=0.2)
        a.centroid_z[f"Purlin {k}"] = TOP + 0.1
    model = _assemble(a.save(tmp_path / "flush.ifc"))
    _one_rigid_body(model)
    _centroids_kept(model, a)
    assert {round(model.nodes.xyz(m.start)[2], 4) for m in model.members
            if m.prov.ifc_class == "IfcBeam"} == {TOP}


@pytest.mark.parametrize("path", [NUCLEAR_ISLAND, MODEL_33], ids=["nuclear_island", "model33"])
def test_turbine_hall_frame_is_one_body(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    model = _assemble(path)
    _one_rigid_body(model)
    assert not model.surfaces                                # envelope and ground slab gone
    fix = next(s for s in model.supports if s.name == "BASE_FIX")
    assert not fix.edges and len(fix.nodes) == len(
        {m.prov.guid for m in model.members if m.prov.ifc_class == "IfcColumn"})
