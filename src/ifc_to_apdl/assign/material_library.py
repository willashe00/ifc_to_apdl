"""Curated regulatory material specification library.

Each entry pairs a retrieval descriptor with a complete SI mechanical
property set drawn from European (EN) and American (ASTM / ACI / AISC /
ASME) specifications — the library is scoped to these two regulatory
regimes only (2026-08-24; the earlier GB 50010 containment entry was
replaced by its EN 1992-1-1 C50/60 equivalent, so containment conversions
now instantiate Ecm = 37 GPa, rho = 2500 rather than the Liu-2013 GB
values E = 34.554 GPa, rho = 2549 — override per-run when reproducing
that benchmark). The library is deliberately small and auditable; extend
it as new material families appear.

Concrete moduli: EN 1992-1-1 Table 3.1 secant values Ecm; ACI 318-19
Sec. 19.2.2.1 Ec = 57000*sqrt(f'c [psi]) psi for normal-weight concrete.
Steel: EN 1993-1-1 Sec. 3.2.6 (210 GPa); AISC/ASTM practice (200 GPa,
203 GPa for A106 piping per ASME B31 at ambient).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LibraryEntry:
    key: str
    descriptor: str            # text embedded for dense retrieval
    family: str                # 'steel' | 'concrete' | 'aluminum'
    elastic_modulus: float     # Pa
    poisson_ratio: float
    density: float             # kg/m^3
    standard: str


LIBRARY: list[LibraryEntry] = [
    # -- structural steel, European (EN 10025-2) -----------------------------
    LibraryEntry(
        "steel-s235", "structural steel S235 S235JR mild hot rolled plate section Baustahl European",
        "steel", 210e9, 0.30, 7850, "EN 10025-2"),
    LibraryEntry(
        "steel-s275", "structural steel S275 S275JR hot rolled framing universal beam UB UC European",
        "steel", 210e9, 0.30, 7850, "EN 10025-2"),
    LibraryEntry(
        "steel-s355", "structural steel S355 S355J2 hot rolled framing beam column I-section European",
        "steel", 210e9, 0.30, 7850, "EN 10025-2"),
    # -- structural steel, American ------------------------------------------
    LibraryEntry(
        "steel-a992", "structural steel ASTM A992 grade 50 wide flange W-shape American framing",
        "steel", 200e9, 0.30, 7850, "ASTM A992"),
    LibraryEntry(
        "steel-a36", "structural steel ASTM A36 plate angle channel general carbon American",
        "steel", 200e9, 0.30, 7850, "ASTM A36"),
    LibraryEntry(
        "steel-a615-rebar", "reinforcing steel rebar ASTM A615 grade 60 deformed bar American",
        "steel", 200e9, 0.30, 7850, "ASTM A615"),
    # -- piping steel, American (ASME B31 practice) --------------------------
    LibraryEntry(
        "steel-pipe-a106", "carbon steel pipe ASTM A106 grade B seamless piping segment elbow schedule mild steel",
        "steel", 203e9, 0.30, 7850, "ASTM A106 / ASME B31"),
    LibraryEntry(
        "steel-stainless-304", "stainless steel pipe ASTM A312 TP304 TP304L austenitic piping",
        "steel", 193e9, 0.31, 8000, "ASTM A312"),
    # -- concrete, European (EN 1992-1-1 Table 3.1) --------------------------
    LibraryEntry(
        "concrete-ec2-c25-30", "normal weight concrete class C25/30 Beton cast-in-place Eurocode slab building general",
        "concrete", 31.0e9, 0.20, 2500, "EN 1992-1-1"),
    LibraryEntry(
        "concrete-ec2-c30-37", "reinforced concrete class C30/37 Beton cast-in-place Eurocode structural slab wall building floor",
        "concrete", 33.0e9, 0.20, 2500, "EN 1992-1-1"),
    LibraryEntry(
        "concrete-ec2-c40-50", "high strength concrete class C40/50 cast-in-place Eurocode wall column core building",
        "concrete", 35.0e9, 0.20, 2500, "EN 1992-1-1"),
    LibraryEntry(
        "concrete-ec2-c50-60-containment",
        "reinforced concrete C50/60 nuclear containment structure basemat cylinder wall dome reactor safety-related",
        "concrete", 37.0e9, 0.20, 2500, "EN 1992-1-1"),
    # -- concrete, American (ACI 318-19 Sec. 19.2.2.1, normal weight) --------
    LibraryEntry(
        "concrete-aci-3000", "normal weight concrete f'c 3000 psi 3 ksi cast-in-place ACI American footing slab-on-grade",
        "concrete", 21.5e9, 0.20, 2400, "ACI 318-19"),
    LibraryEntry(
        "concrete-aci-4000", "structural concrete f'c 4000 psi 4 ksi normal weight cast-in-place ACI American slab wall",
        "concrete", 24.9e9, 0.20, 2400, "ACI 318-19"),
    LibraryEntry(
        "concrete-aci-5000", "concrete f'c 5000 psi 5 ksi normal weight ACI American column core high strength",
        "concrete", 27.8e9, 0.20, 2400, "ACI 318-19"),
    # -- lightweight concrete (ACI 318-19 Ec = wc^1.5*33*sqrt(f'c) at
    # -- wc = 115 pcf; EN 1992-1-1 Sec. 11 Elcm = Ecm*(rho/2200)^2) ---------
    LibraryEntry(
        "concrete-aci-lw-3000", "lightweight structural concrete f'c 3000 psi sand-lightweight low density ACI American topping",
        "concrete", 15.4e9, 0.20, 1840, "ACI 318-19"),
    LibraryEntry(
        "concrete-aci-lw-4000", "lightweight structural concrete f'c 4000 psi sand-lightweight low density ACI American slab topping",
        "concrete", 17.7e9, 0.20, 1840, "ACI 318-19"),
    LibraryEntry(
        "concrete-ec2-lc30-33", "lightweight aggregate concrete LC30/33 LWAC low density Eurocode slab topping",
        "concrete", 23.3e9, 0.20, 1850, "EN 1992-1-1 Sec. 11"),
    # -- aluminum, American --------------------------------------------------
    LibraryEntry(
        "aluminum-6061", "aluminum alloy 6061-T6 extrusion",
        "aluminum", 68.9e9, 0.33, 2700, "ASTM B221"),
]

#: physical plausibility windows per family: (E_min, E_max, nu_min, nu_max, rho_min, rho_max)
PLAUSIBILITY = {
    "steel": (1.7e11, 2.3e11, 0.25, 0.35, 7000.0, 9500.0),
    "concrete": (1.2e10, 6.0e10, 0.10, 0.30, 1400.0, 2900.0),  # incl. LWC
    "aluminum": (6.0e10, 8.0e10, 0.28, 0.36, 2500.0, 2900.0),
}

#: families the library can instantiate
LIBRARY_FAMILIES = frozenset(e.family for e in LIBRARY)

#: scope guard: token evidence of common material families the regulatory
#: library deliberately does NOT cover. A name naming one of these must be
#: flagged for review no matter how similar its best retrieval match scores
#: (dense similarity alone cannot rule out e.g. 'PVC Schedule 80' matching a
#: carbon-steel pipe spec through shared piping vocabulary).
OUT_OF_SCOPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "timber": ("timber", "wood", "lumber", "glulam", "plywood", "clt", "lvl",
               "osb", "softwood", "hardwood"),
    "polymer": ("pvc", "hdpe", "pe100", "polyethylene", "polypropylene", "ppr",
                "abs", "frp", "grp", "plastic", "nylon", "ptfe", "pex"),
    "masonry": ("masonry", "brick", "blockwork", "cmu"),
    "glass": ("glass", "glazing"),
    "gypsum": ("gypsum", "plasterboard", "drywall"),
    "insulation": ("insulation", "eps", "xps", "rockwool", "polystyrene",
                   "polyurethane", "pir"),
}


def guess_family(text: str) -> str | None:
    """Material family from a name/context string.

    Grade designations count as family evidence: a material named only
    'C30/37' or 'S355J2+N' must still trigger the right plausibility window
    (a bare-designation concrete carrying copy-pasted steel constants slips
    through any family-blind check).

    Out-of-scope evidence is checked FIRST: a name that explicitly names a
    family the library does not carry (timber, polymer, masonry, ...) must
    not be talked into steel/concrete by secondary vocabulary — flagging is
    the conservative direction.
    """
    import re

    t = text.lower().replace("_", " ")     # underscores are separators in
    tokens = set(re.split(r"[^a-z0-9']+", t))  # authoring conventions (MAT_, M_)
    for fam, kws in OUT_OF_SCOPE_FAMILIES.items():
        if tokens & set(kws):
            return fam
    if re.search(r"\bgl\s?\d{2}[ch]?\b", t):                           # GL24h glulam
        return "timber"
    if "alumin" in t:            # before the generic 'metal' token: Revit's stock
        return "aluminum"        # name is 'Metal - Aluminum' (experiment_set_01)
    if any(k in t for k in ("steel", "metal", "iron", "stahl", "a992", "a36",
                            "a106", "a53", "sa-312", "tp304", "tp316")):
        return "steel"
    if re.search(r"\bs\s?[2-4]\d{2}\s*(j2|jr|j0|k2|\+n|\+m)?\b", t):   # S235..S460
        return "steel"
    if any(k in t for k in ("concrete", "cast-in-place", "cement", "beton", "f'c")):
        return "concrete"
    if re.search(r"\bl?c\s?\d{2}\s*[/-]\s*\d{2}\b", t):                # C30/37, LC30/33
        return "concrete"
    return None


def plausible(family: str | None, e: float, nu: float, rho: float) -> bool:
    if family is None or family not in PLAUSIBILITY:
        # broad structural-material envelope (also for out-of-scope families,
        # where declared IFC properties remain the only usable evidence)
        return 5e9 <= e <= 4e11 and 0.05 <= nu <= 0.45 and 1000 <= rho <= 12000
    lo_e, hi_e, lo_n, hi_n, lo_r, hi_r = PLAUSIBILITY[family]
    return lo_e <= e <= hi_e and lo_n <= nu <= hi_n and lo_r <= rho <= hi_r
