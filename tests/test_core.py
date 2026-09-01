"""Core IR / config / audit unit tests (no Ansys, no IFC files)."""

import math

import pytest

from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.model.ir import AnalyticalModel, Member1D, NodePool, Provenance
from ifc_to_apdl.model.materials import MaterialDef
from ifc_to_apdl.model.sections import ISection, PipeSection, ShellSection
from ifc_to_apdl.report.audit import AuditLedger, AuditStatus


def test_nodepool_merges_within_tolerance():
    pool = NodePool(tol=1e-3)
    a = pool.get((0.0, 0.0, 0.0))
    b = pool.get((0.0004, 0.0, 0.0))       # within tol -> same node
    c = pool.get((0.01, 0.0, 0.0))         # outside -> new node
    assert a == b
    assert c != a
    assert len(pool) == 2


def test_section_dedup_including_offsets():
    m = AnalyticalModel("t", "building")
    s1 = ISection(depth=0.4, width=0.2, tw=0.01, tf=0.016)
    s2 = ISection(depth=0.4, width=0.2, tw=0.01, tf=0.016)
    s3 = ISection(depth=0.4, width=0.2, tw=0.01, tf=0.016, offset_y=-0.3)
    assert m.add_section(s1) == m.add_section(s2)
    assert m.add_section(s3) != m.add_section(s1)


def test_expected_mass_pipe():
    m = AnalyticalModel("t", "piping")
    mid = m.add_material(MaterialDef(0, "steel", 2.1e11, 0.3, 7850.0))
    sid = m.add_section(PipeSection(od=0.2, t=0.01))
    n1 = m.nodes.get((0, 0, 0))
    n2 = m.nodes.get((10, 0, 0))
    m.members.append(Member1D(1, Provenance("g", "IfcPipeSegment"), n1, n2,
                              "PIPE288", sid, mid))
    area = math.pi * (0.1**2 - 0.09**2)
    assert m.expected_mass() == pytest.approx(7850.0 * area * 10.0, rel=1e-9)


def test_audit_accounts_everything():
    led = AuditLedger("x.ifc")
    led.record("g1", "IfcBeam", "b", AuditStatus.CONVERTED)
    led.record("g2", "IfcTank", "t", AuditStatus.EXCLUDED, detail="equipment")
    assert led.unaccounted({"g1", "g2", "g3"}) == {"g3"}
    assert led.counts() == {"converted": 1, "excluded": 1}


def test_shell_section_apdl_layers():
    s = ShellSection(id=3, layers=[(0.21, 2)])
    lines = s.apdl()
    assert lines[0].startswith("SECTYPE,3,SHELL")
    assert "SECDATA,0.21,2" in lines[1]


def test_config_tolerance_derivation():
    cfg = ConversionConfig()
    assert cfg.resolve_merge_tol(1e-4) == 1e-4
    assert cfg.resolve_merge_tol(1e-9) == 1e-6      # floored
    assert cfg.resolve_merge_tol(None) == 1e-4      # default
    cfg2 = ConversionConfig(merge_tolerance=0.005)
    assert cfg2.resolve_merge_tol(1e-4) == 0.005
