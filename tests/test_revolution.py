"""Tessellated bodies of revolution (containment shells authored as meshes)."""

import numpy as np
import pytest

from ifc_to_apdl.geometry.tessellated import revolution_params


def _ring(r, z, n=96):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]


def test_hollow_cylinder():
    v = np.vstack([_ring(28.0, 0.0), _ring(29.0, 0.0), _ring(28.0, 36.5), _ring(29.0, 36.5)])
    v[:, :2] += (50.0, 50.0)
    p = revolution_params(v)
    assert p is not None and p.kind == "cylinder"
    assert p.params["r_inner"] == pytest.approx(28.0, abs=1e-6)
    assert p.params["r_outer"] == pytest.approx(29.0, abs=1e-6)
    assert (p.params["z_min"], p.params["z_max"]) == (0.0, 36.5)
    assert (p.params["cx"], p.params["cy"]) == pytest.approx((50.0, 50.0), abs=1e-6)


def test_solid_disc():
    v = np.vstack([_ring(29.0, -3.0), _ring(29.0, 0.0), [[0, 0, -3.0], [0, 0, 0.0]]])
    p = revolution_params(v)
    assert p is not None and p.kind == "cylinder"
    assert p.params["r_inner"] == 0.0
    assert p.params["r_outer"] == pytest.approx(29.0, abs=1e-6)


def test_hemispherical_shell():
    cz = 36.5
    rings = []
    for r in (28.0, 29.0):
        for phi in np.linspace(0, np.pi / 2, 14):          # equator -> pole
            rings.append(_ring(r * np.cos(phi), cz + r * np.sin(phi), n=64))
    v = np.vstack(rings)
    v[:, :2] += (50.0, 50.0)
    p = revolution_params(v)
    assert p is not None and p.kind == "spherical_shell"
    assert p.params["cz"] == pytest.approx(cz, abs=1e-6)
    assert p.params["r_outer"] == pytest.approx(29.0, abs=1e-6)
    assert p.params["r_inner"] == pytest.approx(28.0, abs=1e-6)
    assert p.params["hemisphere"] == 1.0


def test_prismatic_member_is_not_a_body_of_revolution():
    v = np.array([[x, y, z] for x in (0, 6) for y in (-0.1, 0.1) for z in (-0.2, 0.2)], float)
    v = np.vstack([v] * 2)
    assert revolution_params(v) is None
