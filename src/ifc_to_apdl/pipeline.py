"""End-to-end conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assign.materials import MaterialResolver
from .classify.systems import SystemRecord, classify_systems
from .config import ConversionConfig
from .emit.apdl_writer import write_deck
from .ingest.loader import load_ifc
from .model.ir import AnalyticalModel
from .report.audit import AuditLedger, AuditStatus


DECK_SUFFIX = ".txt"        # APDL macro files


@dataclass
class ConversionResult:
    system: SystemRecord
    model: AnalyticalModel
    deck: Path
    audit: AuditLedger          # coverage ledger of the whole run (shared by all systems)


def convert_file(ifc_path: str | Path, out_dir: str | Path,
                 config: ConversionConfig | None = None,
                 audit_dir: str | Path | None = None) -> list[ConversionResult]:
    """Convert one IFC file into one APDL macro (``<stem>_<system>.txt``) per
    classified system in ``out_dir``.

    The coverage audit stays in memory (``result.audit``); pass ``audit_dir``
    to also write it out as JSON + Markdown.
    """
    config = config or ConversionConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_ifc(ifc_path)
    systems = classify_systems(ctx)
    resolver = MaterialResolver(config.materials, ctx.audit)

    results: list[ConversionResult] = []
    for system in systems:
        model = _assemble(ctx, system, config, resolver)
        if model is None:
            continue

        # pre-solver screens: free bodies, non-conformal interfaces and
        # floating brace joints are conversion defects
        if model.members or model.surfaces:
            from .verify.connectivity import (conformity_report, connectivity_report,
                                              free_joint_report)
            comps = connectivity_report(model)
            free = [c for c in comps if not c.supported]
            ctx.audit.event("connectivity",
                            f"{system.name}: {len(comps)} connected component(s), "
                            f"{len(free)} without support")
            for c in free:
                ctx.audit.event("connectivity",
                                f"{system.name}: FREE BODY with {c.object_count} object(s) "
                                f"{c.classes} e.g. {c.sample_names[:2]}", severity="warning")
            if len(comps) > 1 and system.domain == "building":
                ctx.audit.event("connectivity",
                                f"{system.name}: {len(comps)} separate supported components "
                                "- check whether the structure should be one body",
                                severity="warning" if free else "info")
            hang = conformity_report(model)
            if hang:
                nodes = sorted({h.node for h in hang})
                ctx.audit.event("connectivity",
                                f"{system.name}: {len(nodes)} node(s) lie on the interior of "
                                f"another object's edge (non-conformal interface), e.g. "
                                f"{[tuple(round(v, 3) for v in model.nodes.xyz(n)) for n in nodes[:3]]}",
                                severity="warning")
            for fj in (free_joint_report(model) if system.domain == "building" else []):
                ctx.audit.event("connectivity",
                                f"{system.name}: floating brace joint at "
                                f"{tuple(round(v, 3) for v in fj.xyz)} ({fj.members})",
                                severity="warning")

        deck = write_deck(model, config,
                          out_dir / f"{Path(str(ifc_path)).stem}_{system.name}{DECK_SUFFIX}")
        results.append(ConversionResult(system, model, deck, ctx.audit))

    # coverage integrity: nothing may leave the pipeline unaccounted
    for guid in sorted(ctx.audit.unaccounted(set(ctx.products))):
        rec = ctx.products[guid]
        ctx.audit.record(guid, rec.ifc_class, rec.name, AuditStatus.UNHANDLED,
                         detail="product not reached by any conversion path")

    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(ifc_path)).stem
        ctx.audit.to_json(audit_dir / f"{stem}_audit.json")
        ctx.audit.to_markdown(audit_dir / f"{stem}_audit.md")
    return results


def convert_and_verify(ifc_path: str | Path, out_dir: str | Path,
                       config: ConversionConfig | None = None):
    """Convert, then batch-solve every generated deck under PyMAPDL QA.

    The decks stay preprocessing-only; the verifier issues the gravity and
    modal solves against a live session and works in a temporary directory,
    so nothing but the decks lands in ``out_dir``.
    """
    from .verify.runner import verify_deck

    config = config or ConversionConfig()
    results = convert_file(ifc_path, out_dir, config)
    reports = [verify_deck(r.deck, r.model, config) for r in results]
    return results, reports


def _assemble(ctx, system, config, resolver):
    if system.domain == "piping":
        from .assemble.piping import assemble_piping
        return assemble_piping(ctx, system, config, resolver)
    if system.domain == "building":
        from .assemble.building import assemble_building
        return assemble_building(ctx, system, config, resolver)
    if system.domain == "containment":
        from .assemble.containment import assemble_containment
        return assemble_containment(ctx, system, config, resolver)
    return None
