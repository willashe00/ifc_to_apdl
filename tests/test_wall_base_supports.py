"""Per-storey walls must be fixed at the ground only.

Regression for the defect found by verification_tests/mesh_size/experiment_set_01:
the base-fixed heuristic collected the base edge of EVERY wall, so a building
authored with level-to-level walls was clamped at every floor (an 8-storey
core tower lost its sway modes). The authoring rig of that study is reused
to build a minimal 3-storey core.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTHORING = ROOT / "verification_tests" / "mesh_size" / "experiment_set_01" / "scripts" / "authoring.py"


def _load_author():
    spec = importlib.util.spec_from_file_location("mesh_authoring", AUTHORING)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mesh_authoring"] = mod
    spec.loader.exec_module(mod)
    return mod.Author


@pytest.mark.skipif(not AUTHORING.exists(), reason="authoring rig not present")
def test_per_storey_walls_fixed_at_ground_only(tmp_path):
    Author = _load_author()
    a = Author("wall_support_check")
    conc = a.material("Concrete C30/37", 33e9, 0.2, 2500.0)
    xs, ys = [0.0, 6.0, 12.0], [0.0, 6.0, 12.0]
    t_slab, storey_h, n_st, t_wall, c_col = 0.2, 3.3, 3, 0.3, 0.4
    levels = [storey_h * (i + 1) for i in range(n_st)]
    a.add_storey("Ground", 0.0)
    for s in range(n_st):
        z0 = 0.0 if s == 0 else levels[s - 1]
        z_top = levels[s]
        soffit = z_top - t_slab
        for x in xs:
            for y in ys:
                a.column(f"C{x:g}{y:g}S{s}", (x, y, z0), soffit - z0, "rect",
                         {"width": c_col, "depth": c_col}, conc)
        for y in ys:
            for x0, x1 in zip(xs, xs[1:]):
                a.beam(f"BX{x0:g}{y:g}S{s}", (x0 + c_col / 2, y, soffit - 0.25),
                       (x1 - c_col / 2, y, soffit - 0.25), "rect",
                       {"width": 0.3, "depth": 0.5}, conc)
        for x in xs:
            for y0, y1 in zip(ys, ys[1:]):
                a.beam(f"BY{x:g}{y0:g}S{s}", (x, y0 + c_col / 2, soffit - 0.25),
                       (x, y1 - c_col / 2, soffit - 0.25), "rect",
                       {"width": 0.3, "depth": 0.5}, conc)
        # per-storey wall between two columns on gridline y = 6, bay 0..6
        a.wall(f"W-S{s}", (c_col / 2, 6.0), (6.0 - c_col / 2, 6.0), z0, soffit - z0,
               t_wall, conc)
        e = 0.15
        a.slab(f"Slab{s}", [(-e, -e), (12 + e, -e), (12 + e, 12 + e), (-e, 12 + e)],
               z_top, t_slab, conc)
    ifc = a.save(tmp_path / "walls.ifc")

    from ifc_to_apdl.config import ConversionConfig
    from ifc_to_apdl.pipeline import convert_file

    res = convert_file(ifc, tmp_path / "out", ConversionConfig())
    assert len(res) == 1
    m = res[0].model
    sup = next(s for s in m.supports if s.name == "BASE_FIX")
    zs = Counter(round(m.nodes.xyz(k1)[2], 3) for k1, k2 in sup.edges)
    assert sup.edges, "ground wall must contribute base edges"
    assert set(zs) == {0.0}, f"wall base edges fixed above ground: {dict(zs)}"
    assert {round(m.nodes.xyz(n)[2], 3) for n in sup.nodes} == {0.0}
