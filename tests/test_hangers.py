"""Hanger (COMBIN14 grounded spring) and branch-fitting conversion tests."""

import math

import ifcopenshell
import ifcopenshell.guid as guid
import pytest

from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.model.ir import AnalyticalModel, Member1D, NodePool, Provenance, SpringLink, Support
from ifc_to_apdl.model.materials import MaterialDef
from ifc_to_apdl.model.sections import PipeSection
from ifc_to_apdl.pipeline import convert_file


class _MiniAuthor:
    """Just enough IFC4 authoring for pipeline tests (SI metres)."""

    def __init__(self, name):
        self.f = f = ifcopenshell.file(schema="IFC4")
        units = [f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE"),
                 f.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")]
        origin = f.createIfcCartesianPoint((0.0, 0.0, 0.0))
        wcs = f.createIfcAxis2Placement3D(origin, None, None)
        self.ctx = f.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-4, wcs, None)
        self.project = f.createIfcProject(guid.new(), None, name, None, None, None, None,
                                          (self.ctx,), f.createIfcUnitAssignment(units))
        self.building = f.createIfcBuilding(
            guid.new(), None, "B1", None, None,
            f.createIfcLocalPlacement(None, self._a2p((0, 0, 0))), None, None,
            "ELEMENT", None, None, None)
        f.createIfcRelAggregates(guid.new(), None, None, None, self.project, (self.building,))
        self.storey = f.createIfcBuildingStorey(
            guid.new(), None, "L0", None, None,
            f.createIfcLocalPlacement(self.building.ObjectPlacement, self._a2p((0, 0, 0))),
            None, None, "ELEMENT", 0.0)
        f.createIfcRelAggregates(guid.new(), None, None, None, self.building, (self.storey,))
        self.elements = []
        self.mat_uses = {}

    def _a2p(self, loc, axis=None, ref=None):
        f = self.f
        return f.createIfcAxis2Placement3D(
            f.createIfcCartesianPoint(tuple(float(v) for v in loc)),
            f.createIfcDirection(tuple(float(v) for v in axis)) if axis else None,
            f.createIfcDirection(tuple(float(v) for v in ref)) if ref else None)

    def material(self, name, E, nu, rho):
        f = self.f
        mat = f.createIfcMaterial(name, None, None)
        f.createIfcMaterialProperties(
            "Pset_MaterialCommon", None,
            (f.createIfcPropertySingleValue("MassDensity", None,
                                            f.createIfcMassDensityMeasure(rho), None),), mat)
        f.createIfcMaterialProperties(
            "Pset_MaterialMechanical", None,
            (f.createIfcPropertySingleValue("YoungModulus", None,
                                            f.createIfcModulusOfElasticityMeasure(E), None),
             f.createIfcPropertySingleValue("PoissonRatio", None,
                                            f.createIfcPositiveRatioMeasure(nu), None)), mat)
        self.mat_uses[mat] = []
        return mat

    def _extruded(self, profile, p1, axis, length):
        up = (0.0, 0.0, 1.0)
        ref = (axis[1] * up[2] - axis[2] * up[1],
               axis[2] * up[0] - axis[0] * up[2],
               axis[0] * up[1] - axis[1] * up[0])
        n = math.sqrt(sum(c * c for c in ref))
        ref = tuple(c / n for c in ref) if n > 1e-9 else (1.0, 0.0, 0.0)
        return self.f.createIfcExtrudedAreaSolid(
            profile, self._a2p(p1, axis=axis, ref=ref),
            self.f.createIfcDirection((0.0, 0.0, 1.0)), float(length))

    def _make(self, ifc_class, name, solids, material=None, object_type=None,
              predefined=None):
        f = self.f
        rep = f.createIfcShapeRepresentation(self.ctx, "Body", "SweptSolid", tuple(solids))
        kw = dict(GlobalId=guid.new(), Name=name,
                  ObjectPlacement=f.createIfcLocalPlacement(None, self._a2p((0, 0, 0))),
                  Representation=f.createIfcProductDefinitionShape(None, None, (rep,)))
        if object_type:
            kw["ObjectType"] = object_type
        if predefined:
            kw["PredefinedType"] = predefined
        e = f.create_entity(ifc_class, **kw)
        self.elements.append(e)
        if material is not None:
            self.mat_uses[material].append(e)
        return e

    def pipe(self, name, p1, p2, od, t, material=None):
        d = tuple(b - a for a, b in zip(p1, p2))
        length = math.sqrt(sum(c * c for c in d))
        axis = tuple(c / length for c in d)
        prof = self.f.createIfcCircleHollowProfileDef("AREA", "P", None, od / 2.0, t)
        return self._make("IfcPipeSegment", name,
                          [self._extruded(prof, p1, axis, length)], material,
                          predefined="RIGIDSEGMENT")

    def rod(self, name, p1, p2, d, material=None, object_type="Hanger Rod"):
        vec = tuple(b - a for a, b in zip(p1, p2))
        length = math.sqrt(sum(c * c for c in vec))
        axis = tuple(c / length for c in vec)
        prof = self.f.createIfcCircleProfileDef("AREA", "ROD", None, d / 2.0)
        return self._make("IfcMember", name, [self._extruded(prof, p1, axis, length)],
                          material, object_type=object_type, predefined="USERDEFINED")

    def beam(self, name, p1, p2, material=None, ifc_class="IfcBeam"):
        d = tuple(b - a for a, b in zip(p1, p2))
        length = math.sqrt(sum(c * c for c in d))
        axis = tuple(c / length for c in d)
        prof = self.f.createIfcIShapeProfileDef("AREA", "IPE200", None,
                                                0.1, 0.2, 0.0056, 0.0085, None)
        return self._make(ifc_class, name, [self._extruded(prof, p1, axis, length)], material)

    def column(self, name, base, h, material=None):
        prof = self.f.createIfcIShapeProfileDef("AREA", "HEB200", None,
                                                0.2, 0.2, 0.009, 0.015, None)
        return self._make("IfcColumn", name, [self._extruded(prof, base, (0, 0, 1), h)],
                          material)

    def tank(self, name, base):
        prof = self.f.createIfcCircleProfileDef("AREA", "TK", None, 0.5)
        return self._make("IfcTank", name, [self._extruded(prof, base, (0, 0, 1), 1.5)])

    def tee(self, name, center, run_dir, branch_dir, half_run, branch_len, od, t,
            material=None):
        prof = self.f.createIfcCircleHollowProfileDef("AREA", "TEE", None, od / 2.0, t)
        p_start = tuple(c - half_run * d for c, d in zip(center, run_dir))
        run = self._extruded(prof, p_start, run_dir, 2 * half_run)
        prof2 = self.f.createIfcCircleHollowProfileDef("AREA", "TEEB", None, od / 2.0, t)
        branch = self._extruded(prof2, center, branch_dir, branch_len)
        e = self._make("IfcPipeFitting", name, [run, branch], material, predefined="JUNCTION")
        ports = []
        for pos, direction in (
            (p_start, tuple(-d for d in run_dir)),
            (tuple(c + half_run * d for c, d in zip(center, run_dir)), run_dir),
            (tuple(c + branch_len * d for c, d in zip(center, branch_dir)), branch_dir),
        ):
            ports.append(self.f.createIfcDistributionPort(
                guid.new(), None, None, None, None,
                self.f.createIfcLocalPlacement(None, self._a2p(pos, axis=direction)),
                None, None, "PIPE"))
        self.f.createIfcRelNests(guid.new(), None, None, None, e, tuple(ports))
        return e

    def save(self, path):
        f = self.f
        f.createIfcRelContainedInSpatialStructure(
            guid.new(), None, None, None, tuple(self.elements), self.storey)
        for mat, els in self.mat_uses.items():
            if els:
                f.createIfcRelAssociatesMaterial(guid.new(), None, None, None,
                                                 tuple(els), mat)
        f.write(str(path))
        return path


ROD_D = 0.012
ROD_L = 0.4
E_STEEL = 2.1e11
PIPE_OD = 0.1683
PIPE_RO = PIPE_OD / 2.0


@pytest.fixture()
def federated_results(tmp_path):
    a = _MiniAuthor("hanger-fixture")
    steel = a.material("Structural Steel S355", E_STEEL, 0.3, 7850.0)
    a.pipe("Run-1", (0, 0, 3), (4, 0, 3), PIPE_OD, 0.0071, steel)
    a.rod("Hanger-1", (2, 0, 3 + PIPE_RO), (2, 0, 3 + PIPE_RO + ROD_L), ROD_D, steel)
    a.beam("B-1", (10, 0, 3), (14, 0, 3), steel)
    a.column("C-1", (10, 0, 0), 3.0, steel)
    a.column("C-2", (14, 0, 0), 3.0, steel)
    a.tank("TK-1", (6, 3, 0))
    path = a.save(tmp_path / "hanger.ifc")
    return convert_file(path, tmp_path / "out", audit_dir=tmp_path / "out")


def test_hanger_routed_to_piping_not_building(federated_results):
    by_domain = {r.system.domain: r for r in federated_results}
    assert set(by_domain) == {"piping", "building"}
    building_guids = {m.prov.name for m in by_domain["building"].model.members}
    assert "Hanger-1" not in building_guids
    assert len(by_domain["piping"].model.links) == 1


def test_hanger_spring_stiffness_and_split(federated_results):
    piping = next(r for r in federated_results if r.system.domain == "piping").model
    link = piping.links[0]
    area = math.pi * ROD_D**2 / 4.0
    assert link.k == pytest.approx(E_STEEL * area / ROD_L, rel=1e-9)
    # host pipe split at mid-span attachment -> two members sharing the node
    assert len(piping.members) == 2
    shared = {piping.members[0].start, piping.members[0].end} & \
             {piping.members[1].start, piping.members[1].end}
    assert link.n1 in shared
    # ground node at the rod's structure-side end
    gx, gy, gz = piping.nodes.xyz(link.n2)
    assert (gx, gy) == pytest.approx((2.0, 0.0))
    assert gz == pytest.approx(3 + PIPE_RO + ROD_L, abs=1e-6)
    assert any(s.name == "HANGER_GROUND" for s in piping.supports)


def test_hanger_deck_emission(federated_results):
    piping = next(r for r in federated_results if r.system.domain == "piping")
    text = piping.deck.read_text()
    assert "ET,2,COMBIN14" in text or "COMBIN14" in text
    assert "CM,HANGER_GROUND,NODE" in text
    assert "E,NATT_,NGRND0+1" in text
    area = math.pi * ROD_D**2 / 4.0
    assert f"R,1,{E_STEEL * area / ROD_L:g}" in text


def test_equipment_excluded_evidentially(federated_results, tmp_path):
    import json
    audit = json.loads((tmp_path / "out" / "hanger_audit.json").read_text())
    tank = [r for r in audit["entries"] if r["ifc_class"] == "IfcTank"]
    assert tank and tank[0]["status"] == "excluded"


def test_tee_branch_junction(tmp_path):
    a = _MiniAuthor("tee-fixture")
    steel = a.material("Structural Steel S355", E_STEEL, 0.3, 7850.0)
    a.pipe("Run-A", (0, 0, 1), (2, 0, 1), PIPE_OD, 0.0071, steel)
    a.tee("Tee-1", (2.2, 0, 1), (1, 0, 0), (0, 1, 0), 0.2, 0.25, PIPE_OD, 0.0071, steel)
    a.pipe("Run-B", (2.4, 0, 1), (4.4, 0, 1), PIPE_OD, 0.0071, steel)
    a.pipe("Run-C", (2.2, 0.25, 1), (2.2, 2.25, 1), PIPE_OD, 0.0071, steel)
    path = a.save(tmp_path / "tee.ifc")
    results = convert_file(path, tmp_path / "out")
    piping = next(r for r in results if r.system.domain == "piping").model
    # 3 straight runs + tee run leg split at the branch + tee branch leg
    assert len(piping.members) == 6
    tee_members = [m for m in piping.members if m.prov.name == "Tee-1"]
    assert len(tee_members) == 3
    assert all(m.etype == "PIPE288" for m in tee_members)
    assert "3-port fitting" in tee_members[0].prov.evidence["element"]
    # the junction node is shared by both run-leg halves and the branch
    from collections import Counter
    counts = Counter()
    for m in tee_members:
        counts[m.start] += 1
        counts[m.end] += 1
    assert max(counts.values()) == 3


def test_chs_brace_emits_ctube_not_pipe(tmp_path):
    """BEAM188 framing members with circular-hollow profiles must carry a
    BEAM,CTUBE section — SECTYPE,PIPE is illegal on beam elements."""
    a = _MiniAuthor("chs-brace")
    steel = a.material("Structural Steel S355", E_STEEL, 0.3, 7850.0)
    a.column("C-1", (0, 0, 0), 4.0, steel)
    a.column("C-2", (6, 0, 0), 4.0, steel)
    a.beam("B-1", (0.1, 0, 4), (5.9, 0, 4), steel)
    # CHS diagonal via the beam helper with a hollow-circle profile
    prof = a.f.createIfcCircleHollowProfileDef("AREA", "CHS", None,
                                               PIPE_OD / 2.0, 0.0071)
    d = (6.0, 0.0, 4.0)
    length = math.sqrt(sum(c * c for c in d))
    axis = tuple(c / length for c in d)
    a._make("IfcMember", "BR-1", [a._extruded(prof, (0, 0, 0), axis, length)],
            steel, predefined="BRACE")
    path = a.save(tmp_path / "chs.ifc")
    results = convert_file(path, tmp_path / "out")
    building = next(r for r in results if r.system.domain == "building")
    text = building.deck.read_text()
    assert "BEAM,CTUBE" in text
    for line in text.splitlines():
        assert not line.startswith("SECTYPE") or ",PIPE," not in line


def test_writer_handles_links_without_members_context():
    m = AnalyticalModel("t", "piping", nodes=NodePool(1e-3))
    mid = m.add_material(MaterialDef(0, "steel", 2.1e11, 0.3, 7850.0))
    sid = m.add_section(PipeSection(od=0.2, t=0.01))
    n1 = m.nodes.get((0, 0, 0))
    n2 = m.nodes.get((5, 0, 0))
    m.members.append(Member1D(1, Provenance("g", "IfcPipeSegment"), n1, n2,
                              "PIPE288", sid, mid))
    na = m.nodes.get((2.5, 0, 0))
    ng = m.nodes.get((2.5, 0, 1.0))
    m.links.append(SpringLink(1, Provenance("h", "IfcMember", "HangerX"),
                              na, ng, 1.0e7))
    m.supports.append(Support(name="HANGER_GROUND", nodes=[ng],
                              source="heuristic:hanger-ground"))
    from ifc_to_apdl.emit.apdl_writer import DeckWriter
    text = DeckWriter(m, ConversionConfig()).build()
    assert "COMBIN14" in text
    assert "R,1,1e+07" in text
    assert text.index("NUMMRG,NODE") < text.index("CM,HANGER_GROUND,NODE")
