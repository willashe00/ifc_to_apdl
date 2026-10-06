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


def _panel_hall(tmp_path, name="panels.ifc", load_bearing=False, height=8.0):
    """A hall with an insulated metal panel wall and roof, each decomposed
    into skins and a core the way Alchemy authors them."""
    a = _hall(height=height)
    wall = a.panel("IfcWall", "Panel wall", (6.0, -0.25, 0.0),
                   12.6, 0.1009, height + 0.2, IMP, load_bearing=load_bearing)
    roof = a.panel("IfcRoof", "Panel roof", (6.0, 4.5, height + 0.2),
                   12.6, 9.6, 0.1009, IMP, load_bearing=load_bearing)
    ctx = load_ifc(a.save(tmp_path / name))
    systems = classify_systems(ctx)
    return ctx, systems, [wall, roof]


def test_decomposed_panel_and_its_parts_excluded(tmp_path):
    """A panel carries no body of its own: its skins and core do. Excluding
    the panel alone leaves the pieces holding the geometry to convert as
    structure on their own, so the parts have to inherit its role."""
    ctx, systems, panels = _panel_hall(tmp_path)
    excluded = {e.name: e.detail for e in ctx.audit.entries.values() if "envelope" in e.detail}

    assert "declared Pset_WallCommon.LoadBearing = FALSE" in excluded["Panel wall"]
    assert "declared Pset_RoofCommon.LoadBearing = FALSE" in excluded["Panel roof"]
    for container, pieces in panels:
        for skin in [p for p in pieces if p.is_a("IfcPlate")]:
            assert f"part of '{container.Name}'" in excluded[skin.Name]
            assert "LoadBearing = FALSE" in excluded[skin.Name]
    converted = {g for s in systems for g in s.product_guids}
    assert not converted & {p.GlobalId for _, pieces in panels for p in pieces}

    # the foam cores never reach classification at all: IfcBuildingElementPart
    # is in no ingested class list, so they are dropped before this point and
    # the coverage audit does not account for them
    cores = [p for _, pieces in panels for p in pieces if p.is_a("IfcBuildingElementPart")]
    assert cores and not any(p.GlobalId in ctx.products for p in cores)


def test_decomposed_load_bearing_panel_and_parts_kept(tmp_path):
    """The inherited role follows the declaration both ways: a decomposed
    element declared load-bearing keeps its parts in the frame."""
    ctx, systems, panels = _panel_hall(tmp_path, "bearing.ifc", load_bearing=True)
    assert not [e for e in ctx.audit.entries.values() if "envelope" in e.detail]
    converted = {g for s in systems for g in s.product_guids}
    skins = {p.GlobalId for _, pieces in panels for p in pieces if p.is_a("IfcPlate")}
    assert skins <= converted


def test_decomposed_panel_leaves_no_envelope_shell(tmp_path):
    """What the deck shows. A skin is a thin plan sliver extruded by the
    panel height, and _slab_from reads an extrusion depth as a thickness, so
    a converted skin becomes a shell as thick as the panel is tall. Asserting
    on classes alone does not catch that - the 15.2 m shell on a generated
    clear-span building passed a class-only assertion."""
    from ifc_to_apdl.config import ConversionConfig
    from ifc_to_apdl.pipeline import convert_file

    a = _hall(height=4.0)
    a.panel("IfcWall", "Panel wall", (6.0, -0.25, 0.0), 12.6, 0.1009, 4.2, IMP)
    a.slab("Composite deck", -0.1, -0.1, 12.1, 9.1, 4.2, 0.13,
           layers=[(0.001, "Metal deck"), (0.129, "Concrete C30/37")])
    res = convert_file(a.save(tmp_path / "deck.ifc"), tmp_path / "out", ConversionConfig())
    text = next(r for r in res if r.system.domain == "building").deck.read_text()

    thicknesses, shell = [], False
    for line in text.splitlines():
        if line.startswith("SECTYPE"):
            shell = ",SHELL," in line
        elif shell and line.startswith("SECDATA"):
            thicknesses.append(float(line.split(",")[1]))
    assert thicknesses, "no shell section emitted; the assertion below would pass on nothing"
    assert max(thicknesses) < 0.5, f"envelope shell in the deck: {thicknesses}"


@pytest.mark.parametrize("path", [NUCLEAR_ISLAND, MODEL_33], ids=["nuclear_island", "model33"])
def test_turbine_hall_envelope(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    ctx = load_ifc(path)
    systems = classify_systems(ctx)
    excluded = {e.name for e in ctx.audit.entries.values() if "envelope" in e.detail}
    assert excluded == {"Wall 1", "Wall 2", "Wall 3", "Wall 4", "Roof Slab"}
    bldg = next(s for s in systems if s.domain == "building")
    assert not ({ctx.products[g].ifc_class for g in bldg.product_guids}
                & {"IfcWall", "IfcSlab", "IfcRoof", "IfcPlate"})
