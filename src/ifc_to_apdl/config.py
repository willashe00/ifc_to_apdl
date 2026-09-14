"""Conversion configuration.

Every constant that the legacy converter hard-coded (mesh sizes, selection
tolerances, BC strategy, axis convention) is a validated, documented field
here. Defaults reproduce the conventions of the reference decks in
``test_models`` so results stay comparable.

Emitted decks are preprocessing-only (element types, materials, sections,
geometry, mesh, supports). The solve settings in ``VerifyConfig`` are used
exclusively by the optional PyMAPDL verification pass and never written into
a deck.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MeshConfig(BaseModel):
    """Meshing policy shared by all domains (single abstraction, unlike the
    legacy converter's split absolute-size / division-count schemes)."""

    target_size: float = Field(0.5, gt=0, description="Global target element size [m]")
    arc_min_divisions: int = Field(8, ge=2, description="Minimum divisions along a curved member")
    straight_min_divisions: int = Field(2, ge=1, description="Minimum divisions along a straight member")
    containment_smartsize: int = Field(
        2, ge=1, le=10,
        description="SMRTSIZE level for containment solid meshing (2 reproduces the "
        "verified Liu-benchmark discretization; global/modal response converges there)",
    )
    containment_strict_thickness: bool = Field(
        False,
        description="Strict through-thickness meshing: uniform ESIZE = thinnest shell "
        "wall / solid_through_thickness. Accurate for through-wall stress gradients but "
        "orders of magnitude more elements — use for stress runs, not modal screening.",
    )
    solid_through_thickness: int = Field(
        3, ge=1, description="Solid elements across the thinnest containment wall when "
        "containment_strict_thickness is enabled"
    )


class MaterialConfig(BaseModel):
    similarity_threshold: float = Field(
        0.45, ge=-1, le=1,
        description="Cosine-similarity acceptance threshold (embedding backend). "
        "Validated for all-MiniLM-L6-v2 on a 73-name realistic corpus "
        "(experiments/material_tests/experiment_02): 93.9% acceptance of informative "
        "names at 100% family precision with every placeholder and out-of-scope "
        "name flagged (vague names below tau flag conservatively); 0.45 sits "
        "centered in the empirical operating window [0.43, 0.47]. Thresholds are "
        "encoder-specific; recalibrate on the corpus when the embedding model "
        "changes.",
    )
    fallback_threshold: float = Field(
        0.50, ge=0, le=1, description="Acceptance threshold for the lexical fallback scorer"
    )
    validate_declared: bool = Field(
        True, description="Physical-plausibility validation of IFC-declared properties"
    )
    allow_unit_reinterpretation: bool = Field(
        True,
        description=(
            "If declared values fail plausibility in SI, retry interpreting them in the "
            "mm-tonne-s engineering system (a common authoring-unit defect) before "
            "falling through to retrieval"
        ),
    )


class BCConfig(BaseModel):
    """Boundary-condition strategy per domain. Every heuristic application is
    logged to the audit ledger."""

    building: Literal["base-fixed", "none"] = "base-fixed"
    piping: Literal["anchor-leaves", "none"] = "anchor-leaves"
    containment: Literal["basemat-bottom", "none"] = "basemat-bottom"


class VerifyConfig(BaseModel):
    """Solves run by the optional PyMAPDL verification pass (Phase 6).

    These are issued to a live MAPDL session *after* the deck has been read;
    they are never part of the generated APDL model, which contains only
    preprocessing commands.
    """

    gravity_static: bool = Field(True, description="Gravity static solve for the mass "
                                 "reconciliation and peak-displacement screen")
    modal: bool = Field(True, description="Modal solve for the mechanism screen")
    modal_modes: int = Field(10, ge=1)
    gravity: float = Field(9.80665, gt=0, description="Gravitational acceleration [m/s^2]")


class ConversionConfig(BaseModel):
    """Top-level conversion settings."""

    vertical_axis: Literal["y", "z"] = Field(
        "y",
        description=(
            "'y': IFC Z-up is emitted with APDL Y vertical by a proper rotation about X "
            "(APDL X,Y,Z = IFC X,Z,-Y; a plain Y/Z swap would mirror the plan). "
            "'z': keep the IFC frame unchanged."
        ),
    )
    merge_tolerance: Optional[float] = Field(
        None,
        description="Node/keypoint merge tolerance [m]; None derives it from the IFC "
        "geometric context precision (min 1e-6).",
    )
    snap_tolerance: Optional[float] = Field(
        None,
        description="Face/axis snapping tolerance for building compatibility [m]; "
        "None -> 10x merge tolerance.",
    )
    coordinate_decimals: int = Field(
        6, ge=3, le=12, description="Rounding of emitted coordinates (kills authoring float noise)"
    )
    beam_lift_tolerance: float = Field(
        0.3, ge=0,
        description="Capture window [m] between beam top and slab soffit for lifting the "
        "beam axis into the slab centroidal plane. Real exports carry deck build-ups and "
        "authoring offsets of 0.05-0.3 m; the SECOFFSET compensation always uses the "
        "actual measured shift, so a generous window stays physically consistent.",
    )
    footing_policy: Literal["support", "shell"] = Field(
        "support",
        description="'support': footings become support locations (excluded with reason); "
        "'shell': converted as thick shells.",
    )
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    materials: MaterialConfig = Field(default_factory=MaterialConfig)
    bcs: BCConfig = Field(default_factory=BCConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)

    def resolve_merge_tol(self, ifc_precision: Optional[float]) -> float:
        if self.merge_tolerance is not None:
            return self.merge_tolerance
        if ifc_precision and ifc_precision > 0:
            return max(float(ifc_precision), 1e-6)
        return 1e-4

    def resolve_snap_tol(self, merge_tol: float) -> float:
        if self.snap_tolerance is not None:
            return self.snap_tolerance
        # default: an erection-tolerance-scale capture window, never tighter
        # than 10x the node merge tolerance
        return max(10.0 * merge_tol, 0.025)
