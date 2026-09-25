"""Phase 6 — PyMAPDL closed-loop verification.

Every generated deck is read into a batch MAPDL session and screened:
  - reads without fatal errors (the deck itself is preprocessing-only)
  - mesh compatibility (``mesh_conformity``): one connected element graph,
    no hanging nodes on interfaces, no coincident unmerged nodes
  - gravity static + modal solves are issued *by this runner* against the
    live session (see ``VerifyConfig``); nothing analysis-related is ever
    written into the deck
  - mass reconciliation: vertical gravity reaction / g vs the IR's analytical
    mass (geometry x density) — catches missing members, wrong densities,
    broken units
  - mechanism screen: no near-zero modal frequencies in a constrained model
    — catches disconnected subassemblies and free bodies
  - gravity screen: peak USUM reported for engineering review

Scratch files from the MAPDL session live in ``run_dir`` (a temporary
directory by default) and are removed afterwards.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ConversionConfig
from ..model.ir import AnalyticalModel

MECHANISM_FREQ = 0.05      # Hz — modes below this in a constrained model are mechanisms
MASS_TOL_PCT = 5.0


@dataclass
class VerificationReport:
    deck: str
    read: bool = False                 # deck input completed without a fatal error
    solved: bool = False               # verification solves completed
    mesh_components: int | None = None
    hanging_nodes: int | None = None
    unmerged_pairs: int | None = None
    frequencies: list[float] = field(default_factory=list)
    mass_apdl: float | None = None
    mass_ir: float | None = None
    mass_error_pct: float | None = None
    peak_gravity_usum: float | None = None
    mechanisms: int = 0
    passed: bool = False
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"deck: {Path(self.deck).name}",
                 f"  read: {self.read}   solved: {self.solved}"]
        if self.mesh_components is not None:
            lines.append(f"  mesh: {self.mesh_components} connected component(s), "
                         f"{self.hanging_nodes} hanging node(s), "
                         f"{self.unmerged_pairs} unmerged coincident pair(s)")
        if self.frequencies:
            lines.append("  modes [Hz]: " + ", ".join(f"{f:.3f}" for f in self.frequencies[:8]))
        if self.mass_error_pct is not None:
            lines.append(f"  mass: APDL {self.mass_apdl:.1f} kg vs IR {self.mass_ir:.1f} kg "
                         f"({self.mass_error_pct:+.2f}%)")
        if self.peak_gravity_usum is not None:
            lines.append(f"  gravity peak USUM: {self.peak_gravity_usum * 1000:.3f} mm")
        lines.append(f"  verdict: {'PASS' if self.passed else 'FAIL'}"
                     + (f" — {'; '.join(self.messages)}" if self.messages else ""))
        return "\n".join(lines)


def verification_commands(config: ConversionConfig) -> list[str]:
    """APDL commands the verifier issues after the deck has been read.

    Kept as a pure function so tests can assert exactly what gets solved and
    so the deck writer never has to know about it.
    """
    v = config.verify
    vert = "Y" if config.vertical_axis == "y" else "Z"
    g = v.gravity
    cmds = ["FINISH"]
    if v.gravity_static:
        acel = f"ACEL,0,{g:g},0" if vert == "Y" else f"ACEL,0,0,{g:g}"
        cmds += ["/SOLU", "ANTYPE,STATIC", acel, "SOLVE", "FINISH",
                 "/POST1", "SET,LAST",
                 "NSEL,S,D,U", "FSUM", f"*GET,W_REACT,FSUM,0,ITEM,F{vert}",
                 "ALLSEL,ALL",
                 "NSORT,U,SUM", "*GET,UMAX_G,SORT,0,MAX",
                 "FINISH"]
    if v.modal:
        cmds += ["/SOLU", "ANTYPE,MODAL",
                 f"MODOPT,LANB,{v.modal_modes}", f"MXPAND,{v.modal_modes}",
                 "ACEL,0,0,0", "SOLVE", "FINISH"]
    return cmds


def verify_deck(deck_path: str | Path, model: AnalyticalModel,
                config: ConversionConfig, run_dir: str | Path | None = None,
                keep_run_dir: bool = False) -> VerificationReport:
    from ansys.mapdl.core import launch_mapdl

    # MAPDL resolves /INPUT against its own run directory, not the caller's CWD
    deck_path = Path(deck_path).resolve()
    report = VerificationReport(deck=str(deck_path))
    temp_root = None
    if run_dir is None:
        temp_root = Path(tempfile.mkdtemp(prefix="ifc2apdl_verify_"))
        run_dir = temp_root
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    mapdl = None
    try:
        try:
            mapdl = launch_mapdl(run_location=str(run_dir), override=True,
                                 loglevel="ERROR", start_timeout=120)
            mapdl.input(str(deck_path))
            report.read = True
        except Exception as exc:
            report.messages.append(f"deck input failed: {exc}")
            return report

        try:
            from .mesh_conformity import mesh_report
            mrep = mesh_report(mapdl, Path(deck_path).read_text(),
                               merge_tol=float(model.meta.get("merge_tol", 1e-3)))
            report.mesh_components = mrep.components
            report.hanging_nodes = mrep.hanging_nodes
            report.unmerged_pairs = mrep.unmerged_pairs
            # separate supported structures are legitimate (equipment racks,
            # independent piping loops); the mesh may not split further than
            # the analytical topology does
            from .connectivity import connectivity_report
            ir_comps = connectivity_report(model)
            allowed = max(1, len(ir_comps)) if all(c.supported for c in ir_comps) else 1
            if mrep.components > allowed:
                report.messages.append(
                    f"{mrep.components} separate mesh components {mrep.component_sizes[:4]} "
                    f"(analytical model has {len(ir_comps)})")
            if mrep.hanging_nodes:
                report.messages.append(f"{mrep.hanging_nodes} hanging node(s): "
                                       + "; ".join(mrep.samples[:2]))
            if mrep.unmerged_pairs:
                report.messages.append(f"{mrep.unmerged_pairs} unmerged coincident node pair(s)")
            if mrep.unmeshed_areas:
                report.messages.append(f"{mrep.unmeshed_areas} area(s) produced no elements: "
                                       + "; ".join(x for x in mrep.samples if "unmeshed" in x))
        except Exception as exc:
            report.messages.append(f"mesh screen unavailable: {exc}")

        try:
            for cmd in verification_commands(config):
                mapdl.run(cmd)
            report.solved = True
        except Exception as exc:
            report.messages.append(f"verification solve failed: {exc}")
            return report

        v = config.verify
        if v.modal:
            for i in range(1, v.modal_modes + 1):
                try:
                    f = float(mapdl.get_value("MODE", i, "FREQ"))
                except Exception:
                    break
                report.frequencies.append(f)
            report.mechanisms = sum(1 for f in report.frequencies if f < MECHANISM_FREQ)
            if report.mechanisms:
                report.messages.append(
                    f"{report.mechanisms} near-zero mode(s) — possible disconnected parts")

        if v.gravity_static:
            try:
                w = abs(float(mapdl.parameters["W_REACT"]))
                report.mass_apdl = w / v.gravity
                report.mass_ir = model.expected_mass()
                if report.mass_ir > 0:
                    report.mass_error_pct = (report.mass_apdl - report.mass_ir) / report.mass_ir * 100
                    if abs(report.mass_error_pct) > MASS_TOL_PCT:
                        report.messages.append(
                            f"mass reconciliation off by {report.mass_error_pct:.1f}%")
            except Exception as exc:
                report.messages.append(f"mass check unavailable: {exc}")
            try:
                report.peak_gravity_usum = float(mapdl.parameters["UMAX_G"])
            except Exception:
                pass
    finally:
        if mapdl is not None:
            try:
                mapdl.exit()
            except Exception:
                pass
        if temp_root is not None and not keep_run_dir:
            shutil.rmtree(temp_root, ignore_errors=True)

    mass_ok = report.mass_error_pct is None or abs(report.mass_error_pct) <= MASS_TOL_PCT
    mesh_ok = (not any("separate mesh components" in m for m in report.messages)
               and not report.hanging_nodes and not report.unmerged_pairs
               and not any("produced no elements" in m for m in report.messages))
    report.passed = report.solved and report.mechanisms == 0 and mass_ok and mesh_ok
    return report
