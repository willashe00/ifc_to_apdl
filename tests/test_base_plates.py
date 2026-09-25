"""Column base plates are connection hardware: classification excludes them
(declared type, naming, or geometry) and the column standing on each plate is
fixed at its foot instead."""

from pathlib import Path

import pytest

from ifc_to_apdl.assemble.building import assemble_building
from ifc_to_apdl.assign.materials import MaterialResolver
from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_authoring import Author

T_PLATE = 0.02
GRID = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
NUCLEAR_ISLAND = Path(__file__).resolve().parents[1] / "test_models" / "nuclear_island.ifc"


def _portal(tmp_path, plate_kw=lambda i: {}, schema="IFC4", thickness=lambda i: T_PLATE):
    """Four columns standing on 450x550 base plates, tied by four beams."""
    a = Author(schema=schema)
    for i, (x, y) in enumerate(GRID, 1):
        t = thickness(i)
        a.plate(f"IfcPlate #{i}", (x, y, 0.0), 0.45, 0.55, t, **plate_kw(i))
        a.rect_member("IfcColumn", f"Column {i}", (x, y, t), (x, y, 4.0))
    for i, (p, q) in enumerate(zip(GRID, GRID[1:] + GRID[:1]), 1):
        a.rect_member("IfcBeam", f"Beam {i}", (*p, 4.0), (*q, 4.0))
    return a


def _classify(path):
    ctx = load_ifc(path)
    systems = classify_systems(ctx)
    plates = {e.name: e.detail for e in ctx.audit.entries.values()
              if e.status == "excluded" and e.ifc_class == "IfcPlate"}
    return ctx, systems, plates


@pytest.mark.parametrize("evidence, schema, plate_kw", [
    ("declared PredefinedType BASE_PLATE", "IFC4X3", lambda i: {"predefined_type": "BASE_PLATE"}),
    ("base-plate naming", "IFC4", lambda i: {"assembly": f"Base Plate for Column {i}"}),
    ("thin horizontal plate under a column foot", "IFC4", lambda i: {}),
])
def test_base_plates_excluded_and_columns_recorded(tmp_path, evidence, schema, plate_kw):
    path = _portal(tmp_path, plate_kw, schema).save(tmp_path / "portal.ifc")
    ctx, systems, plates = _classify(path)
    assert [(s.domain, len(s.product_guids)) for s in systems] == [("building", 8)]
    assert len(plates) == 4 and all(evidence in d for d in plates.values())
    columns = {g for g in systems[0].product_guids if ctx.products[g].ifc_class == "IfcColumn"}
    assert set(systems[0].fixed_columns) == columns


def test_non_base_plates_are_kept(tmp_path):
    """Without declared type or naming, only a column-local plate at the
    support level is a base plate: a splice plate between stacked column
    pieces and a floor-size plate under a column foot stay structure."""
    a = Author()
    a.plate("Plate 1", (0.0, 0.0, 0.0), 0.45, 0.55, T_PLATE)
    a.rect_member("IfcColumn", "Column 1", (0.0, 0.0, T_PLATE), (0.0, 0.0, 4.0))
    a.plate("Splice plate", (0.0, 0.0, 4.0), 0.45, 0.55, T_PLATE)
    a.rect_member("IfcColumn", "Column 1 upper", (0.0, 0.0, 4.0 + T_PLATE), (0.0, 0.0, 8.0))
    a.plate("Floor plate", (6.0, 0.0, -T_PLATE), 3.0, 3.0, T_PLATE)
    a.rect_member("IfcColumn", "Column 2", (6.0, 0.0, 0.0), (6.0, 0.0, 4.0))
    a.rect_member("IfcBeam", "Beam", (0.0, 0.0, 4.0), (6.0, 0.0, 4.0))
    ctx, systems, plates = _classify(a.save(tmp_path / "kept.ifc"))
    assert list(plates) == ["Plate 1"]
    kept = {ctx.products[g].name for g in systems[0].product_guids
            if ctx.products[g].ifc_class == "IfcPlate"}
    assert kept == {"Splice plate", "Floor plate"}
    assert [ctx.products[g].name for g in systems[0].fixed_columns] == ["Column 1"]


def test_columns_fixed_at_their_foot_without_plates(tmp_path):
    """Column 1 stands on a 60 mm plate, the others on 20 mm plates: its foot
    is outside the lowest-base cluster, yet it is fixed because its plate was
    excluded. No plate reaches the analytical model."""
    a = _portal(tmp_path, lambda i: {"assembly": f"Base Plate for Column {i}"},
                thickness=lambda i: 0.06 if i == 1 else T_PLATE)
    ctx, systems, _ = _classify(a.save(tmp_path / "mixed.ifc"))
    cfg = ConversionConfig()
    model = assemble_building(ctx, systems[0], cfg, MaterialResolver(cfg.materials, ctx.audit))
    assert not [s for s in model.surfaces if s.prov.ifc_class == "IfcPlate"]
    fix = next(s for s in model.supports if s.name == "BASE_FIX")
    feet = sorted(tuple(round(v, 6) for v in model.nodes.xyz(n)) for n in fix.nodes)
    assert feet == sorted([(0.0, 0.0, 0.06), (6.0, 0.0, T_PLATE),
                           (6.0, 6.0, T_PLATE), (0.0, 6.0, T_PLATE)])
    # every fixed node is the foot of a column member
    col_nodes = {n for m in model.members if m.prov.ifc_class == "IfcColumn"
                 for n in (m.start, m.end)}
    assert set(fix.nodes) <= col_nodes


@pytest.mark.skipif(not NUCLEAR_ISLAND.exists(), reason="nuclear_island.ifc not present")
def test_nuclear_island_turbine_base_plates():
    ctx, systems, plates = _classify(NUCLEAR_ISLAND)
    bldg = next(s for s in systems if s.domain == "building")
    assert not [g for g in bldg.product_guids if ctx.products[g].ifc_class == "IfcPlate"]
    assert len(plates) == 24
    assert all("declared PredefinedType BASE_PLATE" in d for d in plates.values())
    assert len(bldg.fixed_columns) == 24
    assert {ctx.products[g].ifc_class for g in bldg.fixed_columns} == {"IfcColumn"}
