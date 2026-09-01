"""Phase 2 — material resolution via the evidence cascade.

Order of authority (each decision is logged to the audit ledger):
  1. IFC-declared mechanical properties (Pset_MaterialMechanical /
     Pset_MaterialCommon on the associated IfcMaterial), validated against
     physical plausibility windows for the inferred material family.
  1b. If SI interpretation fails plausibility, reinterpret the raw numbers in
     the mm-tonne-s engineering system (E in MPa, rho in tonne/mm^3) — a
     common authoring-unit defect (present in the gas-pipe fixture).
  2. Hybrid retrieval against the curated regulatory library: dense
     embedding similarity ranks candidates; exact designation tokens in the
     name (C30/37, S355, A992, '4000 psi', ...) pin the grade within the
     candidate pool; the designation-grammar family, when the library
     carries it, is appended to the query as a hint. Acceptance is always
     gated by the embedding similarity at the configured threshold. A name
     carrying explicit evidence of a material family the library does not
     cover (timber, polymer, masonry, ...) is never accepted on similarity
     alone — out-of-scope veto. An element with no material association at
     all is never accepted either: structural-context similarity is not
     material-identity evidence.
  3. Below threshold (or vetoed): best candidate is instantiated but FLAGGED
     for review (never a silent default).
"""

from __future__ import annotations

from ..config import MaterialConfig
from ..model.materials import MaterialDef
from ..report.audit import AuditLedger
from .material_library import LIBRARY_FAMILIES, guess_family, plausible
from .retrieval import Retriever


def _material_of(assoc):
    """Primary IfcMaterial from whatever get_material returned."""
    if assoc is None:
        return None
    if assoc.is_a("IfcMaterial"):
        return assoc
    if assoc.is_a("IfcMaterialLayerSetUsage"):
        layers = assoc.ForLayerSet.MaterialLayers
        return layers[0].Material if layers else None
    if assoc.is_a("IfcMaterialLayerSet"):
        return assoc.MaterialLayers[0].Material if assoc.MaterialLayers else None
    if assoc.is_a("IfcMaterialProfileSetUsage"):
        profs = assoc.ForProfileSet.MaterialProfiles
        return profs[0].Material if profs else None
    if assoc.is_a("IfcMaterialProfileSet"):
        return assoc.MaterialProfiles[0].Material if assoc.MaterialProfiles else None
    if assoc.is_a("IfcMaterialList"):
        return assoc.Materials[0] if assoc.Materials else None
    return None


def designation_terms(name: str) -> tuple[str, ...]:
    """Exact designation tokens in a material name (EN concrete classes,
    EN/ASTM steel grades, US strength callouts). Regulatory designations are
    exact identifiers — their verbatim presence in a library descriptor pins
    the grade within the retrieval candidate pool (see Retriever.query)."""
    import re

    t = name.lower().replace("_", " ")     # underscores are separators in
    terms: list[str] = []                  # authoring conventions (MAT_, M_)
    m = re.search(r"\blc\s?(\d{2})\s?[/-]\s?(\d{2})\b", t)       # LC30/33 (LWAC)
    if m:
        terms.append(f"lc{m.group(1)}/{m.group(2)}")
    else:
        m = re.search(r"\bc\s?(\d{2})\s?[/-]\s?(\d{2})\b", t)    # C30/37, C30-37
        if m:
            terms.append(f"c{m.group(1)}/{m.group(2)}")
    m = re.search(r"\bs\s?([2-4]\d{2})", t)                      # S355(J2+N)
    if m:
        terms.append(f"s{m.group(1)}")
    for g in re.findall(r"\b(?:sa|a)\s?-?(\d{2,3})\b", t):       # A36 A992 SA-312
        terms.append(f"a{g}")
    if re.search(r"\btp\s?-?30[46]l?\b", t) or \
            ("stainless" in t and re.search(r"\b30[46]l?\b", t)):
        terms.append("tp304")            # TP304L / 'Stainless ... 304L' etc.
    m = re.search(r"(\d{3,4})\s*psi\b", t)                       # 4000 psi
    if m:
        terms.append(f"{m.group(1)} psi")
    m = re.search(r"\b(\d)\s*ksi\b", t)                          # 4 ksi
    if m:
        terms.append(f"{m.group(1)} ksi")
    m = re.search(r"\bgr(?:ade|\.)?\s*(\d{2})\b", t)             # Gr. 60
    if m:
        terms.append(f"grade {m.group(1)}")
    if re.search(r"\blight\s?weight\b|\blwc\b|\blwac\b|\blow[- ]density\b", t) \
            or any(x.startswith("lc") for x in terms):
        terms.append("lightweight")      # weight-class qualifier (see
    return tuple(terms)                  # Retriever._apply_qualifiers)


def build_query(ifc_class: str, elem_name: str, mat_name: str, domain: str,
                element_family: str) -> tuple[str, tuple[str, ...]]:
    """Retrieval query for a material context: the descriptor string (with
    the designation-grammar family hint appended when the library carries
    that family — 'C30/37' alone must compete as a *concrete*, not as a
    digit pattern) plus the exact designation terms from the name."""
    family = guess_family(f"{mat_name} {elem_name}")
    descriptor = " ".join(filter(None, [
        ifc_class, elem_name, "material", mat_name,
        "system", domain, "element", element_family,
        family if family in LIBRARY_FAMILIES else None,
    ]))
    return descriptor, designation_terms(mat_name)


def _declared_props(material) -> dict[str, float]:
    """Raw mechanical values from IfcMaterialProperties (no unit conversion)."""
    out: dict[str, float] = {}
    for props in getattr(material, "HasProperties", None) or []:
        for p in getattr(props, "Properties", None) or []:
            if not p.is_a("IfcPropertySingleValue") or p.NominalValue is None:
                continue
            name = (p.Name or "").lower()
            try:
                val = float(p.NominalValue.wrappedValue)
            except (TypeError, ValueError):
                continue                      # string-valued property (e.g. a material label)
            if "young" in name or "elastic" in name:      # YoungModulus / YoungsModulus
                out["E"] = val
            elif "poisson" in name:
                out["nu"] = val
            elif "density" in name:                       # Density / MassDensity
                out["rho"] = val
    return out


class MaterialResolver:
    def __init__(self, config: MaterialConfig, audit: AuditLedger):
        self.config = config
        self.audit = audit
        self._retriever: Retriever | None = None    # lazy: embedding model load is slow
        self._cache: dict[tuple, MaterialDef] = {}  # one decision per material context

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
            self.audit.event("material", f"retrieval backend: {self._retriever.backend}")
        return self._retriever

    def resolve(self, rec, domain: str, element_family: str) -> MaterialDef:
        """rec: ingest ProductRecord; element_family e.g. 'beam', 'shell', 'pipe', 'solid'."""
        import dataclasses

        material0 = _material_of(rec.material)
        cache_key = (material0.Name if material0 is not None else None,
                     rec.ifc_class, domain, element_family)
        if cache_key in self._cache:
            return dataclasses.replace(self._cache[cache_key])
        result = self._resolve_uncached(rec, domain, element_family)
        self._cache[cache_key] = result
        return dataclasses.replace(result)

    def _resolve_uncached(self, rec, domain: str, element_family: str) -> MaterialDef:
        material = _material_of(rec.material)
        mat_name = material.Name if material is not None else ""
        family = guess_family(f"{mat_name} {rec.name}")

        # -- step 1: declared properties -------------------------------------
        if material is not None and self.config.validate_declared:
            props = _declared_props(material)
            if {"E", "nu", "rho"} <= props.keys():
                e, nu, rho = props["E"], props["nu"], props["rho"]
                if plausible(family, e, nu, rho):
                    self.audit.event("material",
                                     f"'{mat_name}' taken from IFC psets (E={e:.3g} Pa)",
                                     guid=rec.guid)
                    return MaterialDef(0, mat_name or "declared", e, nu, rho,
                                       source="ifc-pset")
                if self.config.allow_unit_reinterpretation:
                    e2, rho2 = e * 1e6, rho * 1e12    # MPa -> Pa, t/mm^3 -> kg/m^3
                    if plausible(family, e2, nu, rho2):
                        self.audit.event(
                            "material",
                            f"'{mat_name}' declared values implausible in SI; accepted under "
                            f"mm-tonne-s reinterpretation (E={e2:.4g} Pa, rho={rho2:.4g} kg/m3)",
                            guid=rec.guid, severity="warning")
                        return MaterialDef(0, mat_name or "declared", e2, nu, rho2,
                                           source="ifc-pset-mm-reinterpreted")
                self.audit.event("material",
                                 f"'{mat_name}' declared values fail plausibility "
                                 f"(E={e:.3g}, nu={nu:.3g}, rho={rho:.3g}); falling to retrieval",
                                 guid=rec.guid, severity="warning")

        # -- step 2: dense retrieval -----------------------------------------
        descriptor, terms = build_query(rec.ifc_class, rec.name, mat_name,
                                        domain, element_family)
        entry, score = self.retriever.query(descriptor, terms)
        threshold = (self.config.similarity_threshold
                     if self.retriever.backend == "embedding"
                     else self.config.fallback_threshold)
        out_of_scope = family is not None and family not in LIBRARY_FAMILIES
        no_material = not mat_name.strip()      # no IfcMaterial association at
        # all: structural-context similarity is not material-identity evidence,
        # so acceptance is impossible — flagging is the only correct outcome
        accepted = score >= threshold and not out_of_scope and not no_material
        label = f"{entry.key} [{entry.standard}]"
        pin = f", designation {'/'.join(terms)}" if terms else ""
        if accepted:
            self.audit.event("material",
                             f"retrieval matched '{descriptor}' -> {label} "
                             f"(s={score:.2f}{pin})",
                             guid=rec.guid)
        elif no_material:
            self.audit.event("material",
                             f"no material association on '{rec.name}' "
                             f"({rec.ifc_class}); best candidate {label} "
                             f"(s={score:.2f}) instantiated but FLAGGED for review",
                             guid=rec.guid, severity="warning")
        elif out_of_scope:
            self.audit.event("material",
                             f"'{mat_name}' names material family '{family}' outside the "
                             f"regulatory library scope; best match {label} (s={score:.2f}) "
                             f"NOT accepted; FLAGGED for review",
                             guid=rec.guid, severity="warning")
        else:
            self.audit.event("material",
                             f"retrieval below threshold for '{descriptor}' -> best {label} "
                             f"(s={score:.2f} < {threshold}); FLAGGED for review",
                             guid=rec.guid, severity="warning")
        notes = [f"standard: {entry.standard}", f"query: {descriptor}"]
        if out_of_scope:
            notes.append(f"out-of-scope family: {family}")
        if no_material:
            notes.append("no material association")
        return MaterialDef(
            0, entry.key, entry.elastic_modulus, entry.poisson_ratio, entry.density,
            source="retrieval" if accepted else "flagged",
            confidence=score, flagged=not accepted,
            notes=notes,
        )
