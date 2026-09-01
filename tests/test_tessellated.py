"""Tessellated-recovery tests on synthetic prismatic meshes.

No IFC fixture exercises the tessellated path, so these synthesize meshes
(extruded profiles triangulated by trimesh) and assert that the recovered
shape class and parameters match the generating profile.
"""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from ifc_to_apdl.geometry.tessellated import extract_section
from ifc_to_apdl.model import sections as S


def _extrude(profile_2d: Polygon, length: float = 3.0):
    mesh = trimesh.creation.extrude_polygon(profile_2d, height=length)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _i_profile(d=0.4, b=0.2, tw=0.01, tf=0.016) -> Polygon:
    h = d / 2
    return Polygon([
        (-b / 2, -h), (b / 2, -h), (b / 2, -h + tf), (tw / 2, -h + tf),
        (tw / 2, h - tf), (b / 2, h - tf), (b / 2, h), (-b / 2, h),
        (-b / 2, h - tf), (-tw / 2, h - tf), (-tw / 2, -h + tf), (-b / 2, -h + tf),
    ])


def test_i_shape_recovery():
    verts, faces = _extrude(_i_profile())
    res = extract_section(verts, faces, axis=np.array([0.0, 0.0, 1.0]))
    assert res.shape == "i"
    assert isinstance(res.section, S.ISection)
    assert res.section.depth == pytest.approx(0.4, abs=1e-3)
    assert res.section.width == pytest.approx(0.2, abs=1e-3)
    assert res.section.tw == pytest.approx(0.01, abs=2e-3)
    assert res.section.tf == pytest.approx(0.016, abs=5e-3)


def test_hollow_circle_recovery():
    outer = Polygon([( 0.06 * np.cos(a), 0.06 * np.sin(a))
                     for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)])
    inner = Polygon([(0.05 * np.cos(a), 0.05 * np.sin(a))
                     for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)])
    verts, faces = _extrude(Polygon(outer.exterior.coords, [inner.exterior.coords]))
    res = extract_section(verts, faces, axis=np.array([0.0, 0.0, 1.0]))
    assert res.shape == "hollow_circle"
    assert isinstance(res.section, S.PipeSection)
    assert res.section.od == pytest.approx(0.12, rel=0.01)
    assert res.section.t == pytest.approx(0.01, rel=0.05)


def test_rect_recovery():
    verts, faces = _extrude(Polygon([(-0.15, -0.25), (0.15, -0.25),
                                     (0.15, 0.25), (-0.15, 0.25)]))
    res = extract_section(verts, faces, axis=np.array([0.0, 0.0, 1.0]))
    assert res.shape == "rect"
    assert res.section.width == pytest.approx(0.3, abs=1e-3)
    assert res.section.depth == pytest.approx(0.5, abs=1e-3)


def test_axis_pca_fallback():
    verts, faces = _extrude(_i_profile(), length=5.0)
    res = extract_section(verts, faces, axis=None)   # must find Z by PCA
    assert res.shape == "i"
    span = np.linalg.norm(np.array(res.end) - np.array(res.start))
    assert span == pytest.approx(5.0, abs=1e-2)
