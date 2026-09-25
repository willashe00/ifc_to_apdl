"""Slabs on grade rest on the foundation, not on the frame: classification
excludes them (declared BASESLAB or top face at the support level) while
suspended slabs, podium slabs over walls and footings are left alone."""

from pathlib import Path

import pytest

from ifc_to_apdl.assemble.building import assemble_building
from ifc_to_apdl.assign.materials import MaterialResolver
from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_authoring import Author

GRID = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
NUCLEAR_ISLAND = Path(__file__).resolve().parents[1] / "test_models" / "nuclear_island.ifc"


def _frame(z_foot=0.0, top=4.0):
    """Four columns from z_foot to top tied by four beams."""
    a = Author()
    for i, (x, y) in enumerate(GRID, 1):
        a.rect_member("IfcColumn", f"Column {i}", (x, y, z_foot), (x, y, top))
    for i, (p, q) in enumerate(zip(GRID, GRID[1:] + GRID[:1]), 1):
        a.rect_member("IfcBeam", f"Beam {i}", (*p, top), (*q, top))
    return a


def _classify(a, path):
    ctx = load_ifc(a.save(path))
    systems = classify_systems(ctx)
    ground = {e.name: e.detail for e in ctx.audit.entries.values()
              if e.status == "excluded" and "ground slab" in e.detail}
    kept = {ctx.products[g].name for s in systems for g in s.product_guids
            if ctx.products[g].ifc_class == "IfcSlab"}
    return ctx, systems, ground, kept


@pytest.mark.parametrize("ptype, evidence", [
    ("BASESLAB", "declared PredefinedType BASESLAB"),
    ("FLOOR", "at or below the support level"),
    (None, "at or below the support level"),
])
def test_slab_on_grade_excluded(tmp_path, ptype, evidence):
    a = _frame()
    a.slab("Ground slab", -1.0, -1.0, 7.0, 7.0, -0.2, 0.2, ptype)
    a.slab("Roof", -0.2, -0.2, 6.2, 6.2, 4.0, 0.2, "FLOOR")
    _, systems, ground, kept = _classify(a, tmp_path / "sog.ifc")
    assert list(ground) == ["Ground slab"] and evidence in ground["Ground slab"]
    assert kept == {"Roof"}
    assert "1 ground slab(s) excluded" in systems[0].evidence


def test_suspended_and_podium_slabs_kept(tmp_path):
    """A slab carried by walls below it (podium under a frame), a slab
    raised off the wall base, and a footing slab are not ground slabs."""
    a = _frame(z_foot=4.0, top=8.0)
    for i, (p, q) in enumerate(zip(GRID, GRID[1:] + GRID[:1]), 1):
        a.wall(f"Podium wall {i}", p, q, 0.0, 3.8, 0.3)
    a.slab("Podium slab", -0.2, -0.2, 6.2, 6.2, 3.8, 0.2, "FLOOR")
    a.slab("Raised slab", 1.0, 1.0, 5.0, 5.0, 0.0, 0.3, "FLOOR")      # top 0.3 m above wall base
    a.slab("M_Footing-Rectangular:1800 x 1200", -0.9, -0.6, 0.9, 0.6, -0.9, 0.45, "BASESLAB")
    _, _, ground, kept = _classify(a, tmp_path / "podium.ifc")
    assert not ground
    assert kept == {"Podium slab", "Raised slab", "M_Footing-Rectangular:1800 x 1200"}


def test_columns_stay_fixed_without_ground_slab(tmp_path):
    a = _frame()
    a.slab("Ground slab", -1.0, -1.0, 7.0, 7.0, -0.2, 0.2, "BASESLAB")
    ctx, systems, _, _ = _classify(a, tmp_path / "fixed.ifc")
    cfg = ConversionConfig()
    model = assemble_building(ctx, systems[0], cfg, MaterialResolver(cfg.materials, ctx.audit))
    assert not model.surfaces
    fix = next(s for s in model.supports if s.name == "BASE_FIX")
    assert sorted(tuple(round(v, 6) for v in model.nodes.xyz(n)) for n in fix.nodes) == \
        sorted((x, y, 0.0) for x, y in GRID)


@pytest.mark.skipif(not NUCLEAR_ISLAND.exists(), reason="nuclear_island.ifc not present")
def test_nuclear_island_ground_slab():
    ctx = load_ifc(NUCLEAR_ISLAND)
    systems = classify_systems(ctx)
    ground = {e.name: e.detail for e in ctx.audit.entries.values()
              if e.status == "excluded" and "ground slab" in e.detail}
    assert list(ground) == ["IfcSlab #1"]
    assert "declared PredefinedType BASESLAB" in ground["IfcSlab #1"]
    cont = next(s for s in systems if s.domain == "containment")
    assert "IfcSlab" in {ctx.products[g].ifc_class for g in cont.product_guids}   # basemat kept
