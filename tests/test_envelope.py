"""Non-load-bearing envelope elements (cladding walls, metal roofing) are
excluded only on declared data or two agreeing signals; load-bearing walls
and slabs are kept."""

from pathlib import Path

import pytest

from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_authoring import Author

IMP = [(0.0005, "Coated carbon steel"), (0.1, "Closed-cell PIR foam"),
       (0.0004, "Coated carbon steel")]                   # insulated metal panel
NUCLEAR_ISLAND = Path(__file__).resolve().parents[1] / "test_models" / "nuclear_island.ifc"
MODEL_33 = Path(__file__).resolve().parents[1] / "test_models" / "model (33).ifc"


def _hall(height=8.0):
    """Portal of four columns and eave beams, purlins at 1.5 m bearing on
    the eave beams (top of steel at ``height``)."""
    a = Author()
    for i, (x, y) in enumerate([(0, 0), (12, 0), (12, 9), (0, 9)], 1):
        a.rect_member("IfcColumn", f"Column {i}", (x, y, 0.0), (x, y, height))
    for i, (y) in enumerate((0.0, 9.0), 1):
        a.rect_member("IfcBeam", f"Eave {i}", (0.15, y, height - 0.15), (11.85, y, height - 0.15))
    for k in range(9):
        x = 0.0 + 1.5 * k
        a.rect_member("IfcBeam", f"Purlin {k}", (x, -0.2, height + 0.1), (x, 9.2, height + 0.1),
                      b=0.1, d=0.2)
    return a


def _classify(a, path):
    ctx = load_ifc(a.save(path))
    classify_systems(ctx)
    excluded = {e.name: e.detail for e in ctx.audit.entries.values() if "envelope" in e.detail}
    warned = [e.message for e in ctx.audit.events if "possibly non-structural" in e.message]
    return excluded, warned


def test_metal_envelope_excluded(tmp_path):
    a = _hall()
    a.wall("Cladding south", (-0.3, -0.25), (12.3, -0.25), 0.0, 8.2, 0.1009, layers=IMP)
    a.slab("Roof sheet", -0.3, -0.3, 12.3, 9.3, 8.2, 0.02, "ROOF",
           layers=[(0.02, "Coated carbon steel")])
    excluded, _ = _classify(a, tmp_path / "hall.ifc")
    assert set(excluded) == {"Cladding south", "Roof sheet"}
    assert "no load-bearing layer" in excluded["Cladding south"]
    assert "below the ACI 318 bearing-wall minimum" in excluded["Cladding south"]
    assert "metal-only build-up" in excluded["Roof sheet"]
    assert "carried everywhere" in excluded["Roof sheet"]


def test_load_bearing_walls_and_slabs_kept(tmp_path):
    """RC shear walls (plain and layered with veneer + insulation), a
    composite slab on the purlins, and a thick insulated panel all stay."""
    a = _hall(height=4.0)
    a.wall("RC wall", (0.0, 0.0), (0.0, 9.0), 0.0, 4.0, 0.3)
    a.wall("Layered wall", (12.0, 0.0), (12.0, 9.0), 0.0, 4.0, 0.36,
           layers=[(0.1, "Brick veneer"), (0.06, "Mineral insulation"), (0.2, "Concrete C30/37")])
    a.slab("Composite deck", -0.1, -0.1, 12.1, 9.1, 4.2, 0.13,
           layers=[(0.001, "Metal deck"), (0.129, "Concrete C30/37")])
    excluded, warned = _classify(a, tmp_path / "rc.ifc")
    assert not excluded and not warned


def test_one_signal_only_is_kept_with_warning(tmp_path):
    """A steel sheet spanning 12 m with no framing beneath it is not
    'carried everywhere' - kept, but flagged for review."""
    a = Author()
    for i, x in enumerate((0.0, 12.0), 1):
        a.rect_member("IfcColumn", f"Column {i}", (x, 0.0, 0.0), (x, 0.0, 4.0))
    a.rect_member("IfcBeam", "Beam", (0.15, 0.0, 3.85), (11.85, 0.0, 3.85))
    a.slab("Steel sheet", 0.0, -3.0, 12.0, 3.0, 4.0, 0.01, layers=[(0.01, "Steel plate")])
    excluded, warned = _classify(a, tmp_path / "sheet.ifc")
    assert not excluded
    assert len(warned) == 1 and "Steel sheet" in warned[0]


@pytest.mark.parametrize("decl, excluded_expected", [
    ({"pset": ("Pset_WallCommon", {"LoadBearing": True})}, False),         # IMP wall kept
    ({"ptype": "SHEAR"}, False),
    ({"pset": ("Pset_WallCommon", {"LoadBearing": False}), "rc": True}, True),
    ({"ptype": "PARTITIONING", "rc": True}, True),
])
def test_declared_role_decides(tmp_path, decl, excluded_expected):
    a = _hall()
    rc = decl.get("rc", False)
    w = a.wall("Wall", (-0.3, -0.25), (12.3, -0.25), 0.0, 8.2, 0.3 if rc else 0.1009,
               layers=None if rc else IMP, predefined_type=decl.get("ptype"))
    if "pset" in decl:
        a.pset(w, *decl["pset"])
    excluded, _ = _classify(a, tmp_path / "declared.ifc")
    assert ("Wall" in excluded) is excluded_expected
    if excluded_expected:
        assert "declared" in excluded["Wall"]


@pytest.mark.parametrize("path", [NUCLEAR_ISLAND, MODEL_33], ids=["nuclear_island", "model33"])
def test_turbine_hall_envelope(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    ctx = load_ifc(path)
    systems = classify_systems(ctx)
    excluded = {e.name for e in ctx.audit.entries.values() if "envelope" in e.detail}
    assert excluded == {"Wall 1", "Wall 2", "Wall 3", "Wall 4", "Roof Slab"}
    bldg = next(s for s in systems if s.domain == "building")
    assert not {ctx.products[g].ifc_class for g in bldg.product_guids} & {"IfcWall", "IfcSlab"}
