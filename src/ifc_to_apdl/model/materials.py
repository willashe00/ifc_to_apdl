"""Material definition carried by the IR, with the evidence trail that
produced it (declared IFC pset, unit-reinterpreted pset, dense retrieval,
or flagged-for-review placeholder)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialDef:
    id: int
    name: str
    elastic_modulus: float      # Pa
    poisson_ratio: float
    density: float              # kg/m^3
    source: str = ""            # 'ifc-pset' | 'ifc-pset-mm-reinterpreted' | 'retrieval' | 'flagged'
    confidence: float | None = None
    flagged: bool = False
    notes: list[str] = field(default_factory=list)

    def key(self) -> tuple:
        return (round(self.elastic_modulus, 3), round(self.poisson_ratio, 4), round(self.density, 3))
