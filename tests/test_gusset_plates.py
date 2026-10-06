"""Gusset plates are connection hardware: classification excludes them
(declared GUSSET_PLATE or naming), and the building assembler extends the
truss web members that stop on them through the gusset clearance to the
chord faces, so they frame into the chord axes at concentric work points.
Bottom chords stopping short of the columns hang on the web members and run
work point to work point: the overhang to the edge of the end gusset is
trimmed unless something frames into it."""

from collections import defaultdict
from pathlib import Path

import ifcopenshell.util.element as ioe
import numpy as np
import pytest

from ifc_to_apdl.assemble.building import assemble_building
from ifc_to_apdl.assign.materials import MaterialResolver
from ifc_to_apdl.classify.systems import classify_systems
from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.ingest.loader import load_ifc
from ifc_to_apdl.verify.connectivity import connectivity_report, free_joint_report
from ifc_authoring import Author

FACE = 0.15                            # column face offset from its axis
ZT, ZB, DC = 4.0, 2.5, 0.3             # top / bottom chord centroids, chord depth
TF, BF = ZT - DC / 2, ZB + DC / 2      # chord faces the work points lie on
PP = [FACE, 1.575, 3.0, 4.425, 6.0 - FACE]   # panel points (4-panel Pratt truss)
CLEAR = 0.15                           # gusset clearance at every web member end
GW, GH = 0.6, 0.4                      # gusset width along the chord, height
OVERHANG = 0.3                         # bottom chord beyond its end panel points
MODEL_42 = Path(__file__).resolve().parents[1] / "test_models" / "model (42).ifc"


def _web(a, name, p, q):
    """Web member between work points p and q, trimmed back by the gusset
    clearance at both ends."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    a.rect_member("IfcMember", name, p + CLEAR * u, q - CLEAR * u, b=0.1, d=0.1)


def _truss(gusset=None):
    """Two columns carrying a Pratt truss: top chord into the column faces,
    bottom chord short of the columns, web members stopping short of the
    chord faces. ``gusset(i, top)`` returns ``(name, web_plate kwargs)`` for
    the gusset at panel point i, seated on the chord face; None omits them."""
    a = Author(schema="IFC4X3")
    a.rect_member("IfcColumn", "Column 1", (0.0, 0.0, 0.0), (0.0, 0.0, 4.2))
    a.rect_member("IfcColumn", "Column 2", (6.0, 0.0, 0.0), (6.0, 0.0, 4.2))
    a.i_beam("Top chord", (FACE, 0.0, ZT), (6.0 - FACE, 0.0, ZT), d=DC, cls="IfcMember")
    a.i_beam("Bottom chord", (PP[1] - OVERHANG, 0.0, ZB), (PP[3] + OVERHANG, 0.0, ZB),
             d=DC, cls="IfcMember")
    for i in (1, 2, 3):
        _web(a, f"Vertical {i}", (PP[i], 0.0, BF), (PP[i], 0.0, TF))
    for i, (top, bottom) in enumerate(((0, 1), (1, 2), (3, 2), (4, 3)), 1):
        _web(a, f"Diagonal {i}", (PP[top], 0.0, TF), (PP[bottom], 0.0, BF))
    if gusset:
        joints = [(i, True) for i in range(5)] + [(i, False) for i in (1, 2, 3)]
        for i, top in joints:
            x = min(max(PP[i], FACE + GW / 2), 6.0 - FACE - GW / 2)
            z = TF - GH / 2 if top else BF + GH / 2
            name, kw = gusset(i, top)
            a.web_plate(name, (x, 0.0, z), (1.0, 0.0, 0.0), GW, GH, 0.01, **kw)
    return a


def _audit(ctx, text):
    return {e.name: e.detail for e in ctx.audit.entries.values()
            if e.status == "excluded" and text in e.detail}


def _assemble(ctx, system):
    cfg = ConversionConfig()
    return assemble_building(ctx, system, cfg, MaterialResolver(cfg.materials, ctx.audit))


def _web_ends(model, is_web):
    """``{web name: [end node ids]}``: the nodes of each web member that
    belong to only one of its pieces."""
    count = defaultdict(lambda: defaultdict(int))
    for m in model.members:
        if is_web(m.prov.name):
            for n in (m.start, m.end):
                count[m.prov.name][n] += 1
    return {name: [n for n, k in c.items() if k == 1] for name, c in count.items()}


def _incident(model):
    inc = defaultdict(set)
    for m in model.members:
        inc[m.start].add(m.prov.name)
        inc[m.end].add(m.prov.name)
    return inc


def _x_range(model, name):
    xs = [model.nodes.xyz(n)[0] for m in model.members if m.prov.name == name
          for n in (m.start, m.end)]
    return round(min(xs), 3), round(max(xs), 3)


def _gusset_by_type(i, top):
    return f"Plate {i}{'T' if top else 'B'}", {"predefined_type": "GUSSET_PLATE"}


@pytest.mark.parametrize("gusset, evidence, n_bolts", [
    (_gusset_by_type, "declared PredefinedType GUSSET_PLATE", 0),
    (lambda i, top: (f"Gusset Plate at {'Top' if top else 'Bottom'} Chord Panel Point {i}", {}),
     "gusset-plate naming", 0),
    (lambda i, top: (f"Plate {i}{'T' if top else 'B'}",
                     {"assembly": f"Gusset plate {i}{'T' if top else 'B'}", "fasteners": 2}),
     "gusset-plate naming", 16),
])
def test_gusset_plates_excluded_and_webs_framed_into_chords(tmp_path, gusset, evidence, n_bolts):
    ctx = load_ifc(_truss(gusset).save(tmp_path / "truss.ifc"))
    systems = classify_systems(ctx)
    gussets = _audit(ctx, "gusset plate (")
    assert len(gussets) == 8 and all(evidence in d for d in gussets.values())
    assert len(_audit(ctx, "connection hardware of the excluded gusset plate")) == n_bolts
    assert [(s.domain, sorted(ctx.products[g].name for g in s.product_guids))
            for s in systems] == [("building", [
                "Bottom chord", "Column 1", "Column 2", *(f"Diagonal {i}" for i in range(1, 5)),
                "Top chord", *(f"Vertical {i}" for i in range(1, 4))])]
    assert len(systems[0].gusset_plates) == 8

    model = _assemble(ctx, systems[0])
    comps = connectivity_report(model)
    assert len(comps) == 1 and comps[0].supported
    # every web end lands on a chord at a concentric work point: verticals
    # and diagonals meeting at a panel point share the chord node there
    is_web = lambda n: n.startswith(("Vertical", "Diagonal"))
    inc = _incident(model)
    ends = _web_ends(model, is_web)
    assert len(ends) == 7 and all(len(e) == 2 for e in ends.values())
    for nodes in ends.values():
        assert all(inc[n] & {"Top chord", "Bottom chord"} for n in nodes)
    work_points = {(round(x, 3), round(z, 3)) for e in ends.values()
                   for x, _, z in map(model.nodes.xyz, e)}
    assert work_points == ({(round(x, 3), ZT) for x in PP}
                           | {(round(x, 3), ZB) for x in PP[1:4]})
    assert all("gusset" in m.prov.evidence for m in model.members if is_web(m.prov.name))
    # the bottom chord stays clear of the columns, hanging on the web
    # members, and runs work point to work point: no overhang, no free end
    assert _x_range(model, "Bottom chord") == (PP[1], PP[3])
    assert _x_range(model, "Top chord") == (0.0, 6.0)
    assert not free_joint_report(model)
    assert not [e for e in ctx.audit.entries.values() if e.status == "unhandled"]


def test_overhang_something_frames_into_is_kept(tmp_path):
    """A strut framing into the tip of one bottom-chord overhang keeps that
    overhang; the free one at the other end is still trimmed."""
    a = _truss(_gusset_by_type)
    tip = PP[1] - OVERHANG
    a.rect_member("IfcMember", "Bottom chord strut", (tip, -2.0, ZB), (tip, -0.1, ZB),
                  b=0.1, d=0.1)
    ctx = load_ifc(a.save(tmp_path / "strut.ifc"))
    model = _assemble(ctx, classify_systems(ctx)[0])
    assert _x_range(model, "Bottom chord") == (round(tip, 3), PP[3])
    strut = {n for m in model.members if m.prov.name == "Bottom chord strut"
             for n in (m.start, m.end)}
    assert "Bottom chord" in set().union(*(_incident(model)[n] for n in strut))
    assert any("overhang past its outermost gusset work point kept - 'Bottom chord strut'"
               in e.message for e in ctx.audit.events)


def test_web_members_are_extended_only_through_gussets(tmp_path):
    """Without gusset plates the same trimmed web members are left as
    authored: the clearance is closed only where a gusset bridges it."""
    ctx = load_ifc(_truss().save(tmp_path / "bare.ifc"))
    systems = classify_systems(ctx)
    assert not systems[0].gusset_plates
    model = _assemble(ctx, systems[0])
    assert not [m for m in model.members if "gusset" in m.prov.evidence]
    assert len(connectivity_report(model)) > 1


@pytest.mark.skipif(not MODEL_42.exists(), reason="model (42).ifc not present")
def test_model_42_pratt_trusses():
    ctx = load_ifc(MODEL_42)
    systems = classify_systems(ctx)
    # gusset names repeat across the five trusses: count audit entries
    excluded = lambda text: [e.detail for e in ctx.audit.entries.values()
                             if e.status == "excluded" and text in e.detail]
    gussets = excluded("gusset plate (")
    assert len(gussets) == 120
    assert all("declared PredefinedType GUSSET_PLATE" in d for d in gussets)
    # the other connection hardware is excluded exactly as before
    assert len(excluded("shear tab (")) == 30
    assert len(excluded("column base plate (")) == 22
    assert len(excluded("connection hardware of the excluded")) == 266
    bldg = next(s for s in systems if s.domain == "building")
    assert not [g for g in bldg.product_guids if ctx.products[g].ifc_class == "IfcPlate"]

    model = _assemble(ctx, bldg)
    comps = connectivity_report(model)
    assert len(comps) == 1 and comps[0].supported
    assert not [e for e in ctx.audit.entries.values() if e.status == "unhandled"]
    role = {r.name: (ioe.get_predefined_type(r.entity) or "") for r in ctx.products.values()
            if r.ifc_class == "IfcMember"}
    chords = {n for n, t in role.items() if t.endswith("CHORD")}
    webs = {n for n, t in role.items() if t == "MEMBER"}
    assert len(chords) == 10 and len(webs) == 115
    inc = _incident(model)
    ends = _web_ends(model, webs.__contains__)
    assert len(ends) == 115 and all(len(e) == 2 for e in ends.values())
    assert all(inc[n] & chords for e in ends.values() for n in e)
    # chords run work point to work point: no overhang left as a free end
    assert not free_joint_report(model)
    for name in (n for n, t in role.items() if t == "BOTTOM_CHORD"):
        ext = _x_range(model, name)
        tips = [n for m in model.members if m.prov.name == name for n in (m.start, m.end)
                if round(model.nodes.xyz(n)[0], 3) in ext]
        assert all(inc[n] & webs for n in tips)
    # bottom chords stop short of the columns (Pratt truss configuration)
    col_xy = {model.nodes.xyz(n)[:2] for m in model.members if m.prov.ifc_class == "IfcColumn"
              for n in (m.start, m.end)}
    bottom = [model.nodes.xyz(n) for m in model.members
              if role.get(m.prov.name) == "BOTTOM_CHORD" for n in (m.start, m.end)]
    assert col_xy and bottom
    assert not [p for p in bottom if any(np.hypot(p[0] - x, p[1] - y) < 0.01 for x, y in col_xy)]
