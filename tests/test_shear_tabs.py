"""Shear tabs are connection hardware: classification excludes them (naming,
or a thin upright plate lapping the web at one beam end, in any body
encoding), the beam frames directly into its support, and their bolts are
excluded with them."""

from pathlib import Path

import pytest

from ifc_to_apdl.assemble.building import assemble_building
from ifc_to_apdl.assign.materials import MaterialResolver
from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_to_apdl.verify.connectivity import connectivity_report
from ifc_authoring import Author

TW, D = 0.01, 0.4                      # beam web thickness, depth
TAB = dict(x=0.114, y=0.229, t=0.0095)  # model (2) shear tab
FACE = 0.15                            # column face offset from its axis
MODEL_2 = Path(__file__).resolve().parents[1] / "test_models" / "model (2).ifc"


def _tab_at(a, name, x, **kw):
    """Shear tab lapping the +y face of the web of a beam along x at z=4."""
    return a.web_plate(name, (x, TW / 2 + TAB["t"] / 2, 4.0), (1.0, 0.0, 0.0), **TAB, **kw)


def _bay():
    """Two columns at x=0 and x=6 (IFC4X3: WEB_PLATE is declarable)."""
    a = Author(schema="IFC4X3")
    a.rect_member("IfcColumn", "Column 1", (0.0, 0.0, 0.0), (0.0, 0.0, 4.2))
    a.rect_member("IfcColumn", "Column 2", (6.0, 0.0, 0.0), (6.0, 0.0, 4.2))
    return a


def _audit(ctx, text):
    return {e.name: e.detail for e in ctx.audit.entries.values()
            if e.status == "excluded" and text in e.detail}


@pytest.mark.parametrize("names, kw, evidence", [
    (lambda end: f"IfcPlate {end}", lambda end: {"assembly": f"Shear Tab at {end} of B1",
                                          "predefined_type": "WEB_PLATE"}, "shear-tab naming"),
    (lambda end: f"Fin plate {end}", lambda end: {}, "shear-tab naming"),
    (lambda end: f"Plate {end}", lambda end: {}, "lapping the web at an end of beam 'B1'"),
    (lambda end: f"Plate {end}", lambda end: {"plan_outline": True},
     "lapping the web at an end of beam 'B1'"),
])
def test_shear_tabs_excluded(tmp_path, names, kw, evidence):
    a = _bay()
    a.i_beam("B1", (FACE, 0.0, 4.0), (6.0 - FACE, 0.0, 4.0), d=D, tw=TW)
    for end, x in (("Start", FACE + TAB["x"] / 2), ("End", 6.0 - FACE - TAB["x"] / 2)):
        _tab_at(a, names(end), x, **kw(end))
    ctx = load_ifc(a.save(tmp_path / "bay.ifc"))
    systems = classify_systems(ctx)
    tabs = _audit(ctx, "shear tab (")
    assert len(tabs) == 2 and all(evidence in d for d in tabs.values())
    assert [(s.domain, sorted(ctx.products[g].name for g in s.product_guids))
            for s in systems] == [("building", ["B1", "Column 1", "Column 2"])]


def test_plates_that_are_not_shear_tabs_are_kept(tmp_path):
    """Without naming, only a thin upright plate lapping the web at ONE beam
    end is a shear tab. Each plate here fails exactly one of those tests and
    stays structure: a web splice plate (laps two beam ends - the only link
    between them), a web doubler at mid-span (declared WEB_PLATE is no
    evidence), a gusset reaching below the beam, a plate skewed to the web, a
    plate standing clear of the flange tips, a chunky bearing block and a
    flat plate on the top flange."""
    a = _bay()
    a.i_beam("B1a", (FACE, 0.0, 4.0), (2.995, 0.0, 4.0), d=D, tw=TW)
    a.i_beam("B1b", (3.005, 0.0, 4.0), (6.0 - FACE, 0.0, 4.0), d=D, tw=TW)
    _tab_at(a, "Splice plate", 3.0)
    _tab_at(a, "Web doubler", 1.5, predefined_type="WEB_PLATE")
    a.web_plate("Gusset", (FACE + 0.25, TW / 2 + 0.005, 3.6), (1.0, 0.0, 0.0), 0.6, 0.6, 0.01)
    a.web_plate("Skewed plate", (6.0 - FACE - 0.06, 0.0, 4.0), (0.866, 0.5, 0.0), **TAB)
    a.web_plate("Side plate", (6.0 - FACE - 0.06, 0.4, 4.0), (1.0, 0.0, 0.0), **TAB)
    a.web_plate("Bearing block", (FACE + 0.05, TW / 2 + 0.025, 4.0), (1.0, 0.0, 0.0),
                0.1, 0.15, 0.05)
    a.plate("Flange plate", (FACE + 0.15, 0.0, 4.0 + D / 2), 0.3, 0.15, 0.012)
    ctx = load_ifc(a.save(tmp_path / "kept.ifc"))
    systems = classify_systems(ctx)
    assert not _audit(ctx, "shear tab (")
    kept = {ctx.products[g].name for g in systems[0].product_guids
            if ctx.products[g].ifc_class == "IfcPlate"}
    assert kept == {"Splice plate", "Web doubler", "Gusset", "Skewed plate", "Side plate",
                    "Bearing block", "Flange plate"}


def test_fasteners_follow_their_connection_plate(tmp_path):
    """Bolts aggregated with an excluded shear tab or base plate are excluded
    with it, citing the plate's assembly; a loose bolt keeps the generic
    out-of-scope rule."""
    a = Author()
    a.plate("Base plate", (0.0, 0.0, 0.0), 0.45, 0.55, 0.02,
            assembly="Base Plate for Column 1", fasteners=4)
    a.rect_member("IfcColumn", "Column 1", (0.0, 0.0, 0.02), (0.0, 0.0, 4.2))
    a.rect_member("IfcColumn", "Column 2", (6.0, 0.0, 0.0), (6.0, 0.0, 4.2))
    a.i_beam("B1", (FACE, 0.0, 4.0), (6.0 - FACE, 0.0, 4.0), d=D, tw=TW)
    _tab_at(a, "Plate", FACE + TAB["x"] / 2, assembly="Shear Tab at Start of B1", fasteners=3)
    a.fastener("Loose bolt", (3.0, 3.0, 0.0))
    ctx = load_ifc(a.save(tmp_path / "bolts.ifc"))
    systems = classify_systems(ctx)
    hw = _audit(ctx, "connection hardware of the excluded")
    assert {n: d.split(" - ")[0] for n, d in hw.items()} == {
        **{f"Base Plate for Column 1 - Bolt {i}":
           "connection hardware of the excluded column base plate 'Base Plate for Column 1'"
           for i in range(1, 5)},
        **{f"Shear Tab at Start of B1 - Bolt {i}":
           "connection hardware of the excluded shear tab 'Shear Tab at Start of B1'"
           for i in range(1, 4)}}
    assert "piping run" in _audit(ctx, "accessory hardware")["Loose bolt"]
    assert "7 connection fastener(s) excluded with their plates" in systems[0].evidence


@pytest.mark.skipif(not MODEL_2.exists(), reason="model (2).ifc not present")
def test_model_2_turbine_shear_tabs():
    ctx = load_ifc(MODEL_2)
    systems = classify_systems(ctx)
    tabs = _audit(ctx, "shear tab (")
    assert len(tabs) == 24 and all("shear-tab naming" in d for d in tabs.values())
    # every bolt is excluded with the tab it is assembled into
    bolts = {n: d for n, d in _audit(ctx, "excluded shear tab").items()}
    assert len(bolts) == 72
    assert all(d.startswith(f"connection hardware of the excluded shear tab '{n.split(' - ')[0]}'")
               for n, d in bolts.items())
    assert len(_audit(ctx, "excluded column base plate")) == 144
    bldg = next(s for s in systems if s.domain == "building")
    assert not [g for g in bldg.product_guids if ctx.products[g].ifc_class == "IfcPlate"]
    cfg = ConversionConfig()
    model = assemble_building(ctx, bldg, cfg, MaterialResolver(cfg.materials, ctx.audit))
    assert not model.surfaces
    comps = connectivity_report(model)
    assert len(comps) == 1 and comps[0].supported
    assert not [e for e in ctx.audit.entries.values() if e.status == "unhandled"]
