"""Deck contract: generated APDL models are preprocessing-only.

Element types, materials, sections, geometry, mesh and boundary conditions
are emitted; no solution, load-case, or postprocessing commands may appear.
The verification runner supplies those at run time instead.
"""

from ifc_to_apdl.config import ConversionConfig
from ifc_to_apdl.emit.apdl_writer import DeckWriter
from ifc_to_apdl.model.ir import AnalyticalModel, Member1D, Provenance, Support
from ifc_to_apdl.model.materials import MaterialDef
from ifc_to_apdl.model.sections import PipeSection
from ifc_to_apdl.verify.runner import verification_commands

ANALYSIS_TOKENS = ("/SOLU", "ANTYPE", "SOLVE", "/POST1", "ACEL", "MODOPT",
                   "MXPAND", "*GET", "NSORT", "FSUM", "SET,LAST")


def _tiny_model() -> AnalyticalModel:
    m = AnalyticalModel("t", "piping", meta={"source": "t.ifc", "merge_tol": 1e-4})
    mid = m.add_material(MaterialDef(0, "steel", 2.1e11, 0.3, 7850.0))
    sid = m.add_section(PipeSection(od=0.2, t=0.01))
    n1 = m.nodes.get((0, 0, 0))
    n2 = m.nodes.get((10, 0, 0))
    m.members.append(Member1D(1, Provenance("g", "IfcPipeSegment", "seg"),
                              n1, n2, "PIPE288", sid, mid))
    m.supports.append(Support(name="ANCHOR_1", nodes=[n1], dofs="ALL",
                              source="heuristic:anchor-leaves"))
    return m


def test_deck_is_preprocessing_only():
    text = DeckWriter(_tiny_model(), ConversionConfig()).build()
    body = "\n".join(l for l in text.splitlines() if not l.startswith("/COM"))
    for tok in ANALYSIS_TOKENS:
        assert tok not in body, f"analysis command {tok!r} leaked into the deck"
    # preprocessing content is still all there
    assert "/PREP7" in body
    assert "ET,1,PIPE288" in body
    assert "MP,EX,1," in body
    assert "LMESH,ALL" in body
    assert "NUMMRG,NODE,MERGE_TOL" in body
    assert "CM,ANCHOR_1,NODE" in body and "D,ALL,ALL,0" in body
    assert body.rstrip().endswith("SAVE\nFINISH")


def test_verifier_owns_the_solves():
    cfg = ConversionConfig()
    cmds = verification_commands(cfg)
    assert "ANTYPE,STATIC" in cmds and "ANTYPE,MODAL" in cmds
    assert f"MODOPT,LANB,{cfg.verify.modal_modes}" in cmds
    assert "ACEL,0,9.80665,0" in cmds            # Y-vertical default frame

    cfg.vertical_axis = "z"
    cfg.verify.modal = False
    cmds = verification_commands(cfg)
    assert "ACEL,0,0,9.80665" in cmds
    assert "ANTYPE,MODAL" not in cmds
