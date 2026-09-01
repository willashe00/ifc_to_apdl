"""The Analytical Intermediate Representation (IR).

A solver-neutral description of one classified system's analytical model:
a deduplicated node pool, 1-D members / 2-D surfaces / 3-D parametric volumes,
links, supports, sections, and materials — each carrying provenance back to
the source IFC entity and the evidence used for every derived field.

Coordinates in the IR are always the IFC frame: SI metres, Z up. The APDL
writer applies any axis permutation in exactly one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .materials import MaterialDef
from .sections import Section

Vec3 = tuple[float, float, float]


@dataclass
class Provenance:
    guid: str
    ifc_class: str
    name: str = ""
    #: field name -> evidence source string, e.g. {'centerline': 'axis-representation',
    #: 'section': 'IfcIShapeProfileDef', 'material': 'retrieval:ASTM A992 (s=0.83)'}
    evidence: dict[str, str] = field(default_factory=dict)


class NodePool:
    """Global pool of analytical points with tolerance-based identity.

    Points within ``tol`` of an existing point resolve to the same id
    (union-find semantics realized through a quantized spatial hash).
    Ids are 1-based so they can serve directly as APDL keypoint numbers.
    """

    def __init__(self, tol: float = 1e-4):
        self.tol = tol
        self._coords: list[Vec3] = []
        self._grid: dict[tuple[int, int, int], list[int]] = {}

    def __len__(self) -> int:
        return len(self._coords)

    def _cell(self, p: Vec3) -> tuple[int, int, int]:
        q = self.tol if self.tol > 0 else 1e-9
        return (int(math.floor(p[0] / q)), int(math.floor(p[1] / q)), int(math.floor(p[2] / q)))

    def get(self, p: Vec3) -> int:
        """Return the id for point ``p``, creating it if no existing point
        lies within ``tol`` (Chebyshev pre-filter, Euclidean confirm)."""
        cx, cy, cz = self._cell(p)
        best, best_d = None, self.tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for idx in self._grid.get((cx + dx, cy + dy, cz + dz), ()):
                        q = self._coords[idx]
                        d = math.dist(p, q)
                        if d <= best_d:
                            best, best_d = idx, d
        if best is not None:
            return best + 1
        self._coords.append((float(p[0]), float(p[1]), float(p[2])))
        idx = len(self._coords) - 1
        self._grid.setdefault((cx, cy, cz), []).append(idx)
        return idx + 1

    def xyz(self, node_id: int) -> Vec3:
        return self._coords[node_id - 1]

    def items(self) -> Iterable[tuple[int, Vec3]]:
        for i, c in enumerate(self._coords):
            yield i + 1, c


@dataclass
class Arc:
    """Circular-arc geometry for a curved member (elbow)."""
    center: Vec3
    radius: float


@dataclass
class Member1D:
    id: int
    prov: Provenance
    start: int                       # node id
    end: int                         # node id
    etype: str                       # 'BEAM188' | 'PIPE288' | 'ELBOW290'
    section: int                     # section id
    material: int                    # material id
    orient: Optional[Vec3] = None    # direction from member toward orientation point
    offset_y: float = 0.0            # SECOFFSET,USER values in section coords
    offset_z: float = 0.0
    arc: Optional[Arc] = None        # curved member -> LARC emission


@dataclass
class Surface2D:
    id: int
    prov: Provenance
    loop: list[int]                  # ordered boundary node ids (>= 3, coplanar)
    etype: str                       # 'SHELL181'
    section: int
    material: int


@dataclass
class Volume3D:
    """Parametric solid primitive (containment components).

    kinds and params (all SI, IFC frame, axis = global Z unless noted):
      'cylinder':        {r_outer, r_inner (0 for solid), z_min, z_max, cx, cy}
      'spherical_shell': {r_outer, r_inner, cz (center elevation), cx, cy,
                          hemisphere: 'upper'}
    """
    id: int
    prov: Provenance
    kind: str
    params: dict[str, float]
    etype: str                       # 'SOLID187'
    material: int


@dataclass
class SpringLink:
    id: int
    prov: Provenance
    n1: int
    n2: int
    k: float                         # N/m
    dof: str = "UZ"                  # acting DOF (IFC frame)


@dataclass
class PointMass:
    id: int
    prov: Provenance
    node: int
    mass: float                      # kg


@dataclass
class Support:
    """A support applied to a named set of analytical locations.

    ``nodes`` holds pool node/keypoint ids (fixed at those points);
    ``edges`` holds keypoint pairs whose connecting line's meshed nodes are
    all fixed (e.g. a wall base edge). The writer turns each support into a
    CM component + D block, never a coordinate-window NSEL."""
    name: str                        # component name, <= 32 chars alnum/underscore
    nodes: list[int] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    dofs: str = "ALL"
    source: str = ""                 # e.g. 'heuristic:column-base', 'ifc:IfcStructuralPointConnection'


@dataclass
class AnalyticalModel:
    name: str
    domain: str                      # 'building' | 'piping' | 'containment'
    nodes: NodePool = field(default_factory=NodePool)
    members: list[Member1D] = field(default_factory=list)
    surfaces: list[Surface2D] = field(default_factory=list)
    volumes: list[Volume3D] = field(default_factory=list)
    links: list[SpringLink] = field(default_factory=list)
    masses: list[PointMass] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    sections: dict[int, Section] = field(default_factory=dict)
    materials: dict[int, MaterialDef] = field(default_factory=dict)
    #: informational metadata (source file, ifc schema, tolerances used, ...)
    meta: dict[str, object] = field(default_factory=dict)

    # -- registries with dedup ------------------------------------------------

    def add_section(self, section: Section) -> int:
        for sid, s in self.sections.items():
            if s.full_key() == section.full_key():
                return sid
        section.id = len(self.sections) + 1
        self.sections[section.id] = section
        return section.id

    def add_material(self, mat: MaterialDef) -> int:
        for mid, m in self.materials.items():
            if m.key() == mat.key() and m.name == mat.name and \
                    (not (m.flagged or mat.flagged) or m.notes == mat.notes):
                # flagged materials merge only within the same material
                # context (identical retrieval query): every review flag in
                # the deck must stay traceable to its own IFC construct
                return mid
        mat.id = len(self.materials) + 1
        self.materials[mat.id] = mat
        return mat.id

    # -- integrity ------------------------------------------------------------

    def expected_mass(self) -> float:
        """Analytical mass [kg] from IR geometry x density — the reference
        value for the PyMAPDL mass-reconciliation check."""
        total = 0.0
        for m in self.members:
            rho = self.materials[m.material].density
            sec = self.sections[m.section]
            area = _section_area(sec)
            if m.arc is not None:
                p1, p2 = self.nodes.xyz(m.start), self.nodes.xyz(m.end)
                chord = math.dist(p1, p2)
                half = min(1.0, chord / (2.0 * m.arc.radius))
                length = 2.0 * m.arc.radius * math.asin(half)
            else:
                length = math.dist(self.nodes.xyz(m.start), self.nodes.xyz(m.end))
            total += rho * area * length
        for s in self.surfaces:
            rho = self.materials[s.material].density
            sec = self.sections[s.section]
            pts = [self.nodes.xyz(i) for i in s.loop]
            total += rho * getattr(sec, "total_thickness", 0.0) * _polygon_area_3d(pts)
        for v in self.volumes:
            rho = self.materials[v.material].density
            total += rho * _volume_of(v)
        for pm in self.masses:
            total += pm.mass
        return total


def _section_area(sec: Section) -> float:
    from . import sections as S

    if isinstance(sec, S.ISection):
        return 2 * sec.width * sec.tf + (sec.depth - 2 * sec.tf) * sec.tw
    if isinstance(sec, S.TSection):
        return sec.width * sec.tf + (sec.depth - sec.tf) * sec.tw
    if isinstance(sec, S.ChannelSection):
        return 2 * sec.width * sec.tf + (sec.depth - 2 * sec.tf) * sec.tw
    if isinstance(sec, S.LSection):
        return (sec.leg1 + sec.leg2 - sec.t) * sec.t
    if isinstance(sec, S.RectSection):
        return sec.width * sec.depth
    if isinstance(sec, S.RectHollowSection):
        return sec.width * sec.depth - (sec.width - 2 * sec.t) * (sec.depth - 2 * sec.t)
    if isinstance(sec, S.CircSection):
        return math.pi * sec.diameter**2 / 4.0
    if isinstance(sec, (S.PipeSection, S.TubeSection)):
        ro, ri = sec.od / 2.0, sec.od / 2.0 - sec.t
        return math.pi * (ro**2 - ri**2)
    return 0.0


def _polygon_area_3d(pts: list[Vec3]) -> float:
    """Area of a planar polygon in 3-D (Newell's method)."""
    n = [0.0, 0.0, 0.0]
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return 0.5 * math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)


def _volume_of(v: Volume3D) -> float:
    p = v.params
    if v.kind == "cylinder":
        h = p["z_max"] - p["z_min"]
        return math.pi * (p["r_outer"] ** 2 - p.get("r_inner", 0.0) ** 2) * h
    if v.kind == "spherical_shell":
        return 2.0 / 3.0 * math.pi * (p["r_outer"] ** 3 - p["r_inner"] ** 3)
    return 0.0
