"""Typed cross-section definitions (solver-neutral).

Each class stores physical parameters in SI metres and knows how to render its
APDL ``SECTYPE``/``SECDATA`` block; the writer never re-derives numbers, so a
comment and its value can never diverge (legacy weakness W9).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _f(v: float, decimals: int = 6) -> str:
    """Format a section dimension, killing authoring float noise."""
    return f"{round(v, decimals):g}"


@dataclass
class Section:
    id: int = 0
    name: str = ""
    #: eccentricity [m] of the section CENTROID relative to the node line:
    #: offset_y along the section DEPTH axis (positive = toward the
    #: orientation node), offset_z along the WIDTH axis. (APDL ties the
    #: offset to the section definition, so offset variants become distinct
    #: sections.)
    offset_y: float = 0.0
    offset_z: float = 0.0

    def key(self) -> tuple:
        """Deduplication key (rounded dims)."""
        raise NotImplementedError

    def full_key(self) -> tuple:
        return self.key() + (round(self.offset_y, 6), round(self.offset_z, 6))

    def apdl(self) -> list[str]:
        raise NotImplementedError

    def origin_to_centroid(self) -> tuple[float, float] | None:
        """(width, depth) position of the centroid in SECDATA coordinates.

        SECOFFSET,USER places the NODE in the section builder's coordinate
        system, whose origin is NOT the centroid for every type. Verified
        empirically (experiment_03 run_05c + origin checks): first SECDATA
        axis (SECOFFSET arg 1) is the section WIDTH, second the DEPTH;
        RECT/CSOLID/CTUBE build about the centroid, the I-section about the
        bottom-flange midpoint, HREC about a corner. None = unverified
        origin (T/CHAN/L): the offset is refused rather than emitted wrong.
        """
        return (0.0, 0.0)

    def apdl_offset(self) -> list[str]:
        if not (self.offset_y or self.offset_z):
            return []
        oc = self.origin_to_centroid()
        if oc is None:
            return [f"/COM, WARNING: section '{self.name}': centroid offset "
                    f"({_f(self.offset_y)},{_f(self.offset_z)}) NOT emitted "
                    "- SECOFFSET origin unverified for this section type"]
        # node position in section coords = centroid position - eccentricity
        node_w = oc[0] - self.offset_z
        node_d = oc[1] - self.offset_y
        return [f"SECOFFSET,USER,{_f(node_w)},{_f(node_d)}"]


@dataclass
class ISection(Section):
    depth: float = 0.0          # overall depth d
    width: float = 0.0          # flange width bf (doubly symmetric)
    tw: float = 0.0             # web thickness
    tf: float = 0.0             # flange thickness

    def key(self):
        return ("I", round(self.depth, 6), round(self.width, 6), round(self.tw, 6), round(self.tf, 6))

    def apdl(self):
        # SECDATA,W1(bot flange width),W2(top),W3(depth),t1(bot tf),t2(top tf),t3(tw)
        return [
            f"SECTYPE,{self.id},BEAM,I,{self.name[:8]}",
            f"SECDATA,{_f(self.width)},{_f(self.width)},{_f(self.depth)},"
            f"{_f(self.tf)},{_f(self.tf)},{_f(self.tw)}",
        ]

    def origin_to_centroid(self):
        # builder origin at the bottom-flange midpoint; doubly symmetric
        return (0.0, self.depth / 2.0)


@dataclass
class TSection(Section):
    depth: float = 0.0
    width: float = 0.0          # flange width
    tw: float = 0.0             # stem thickness
    tf: float = 0.0             # flange thickness

    def key(self):
        return ("T", round(self.depth, 6), round(self.width, 6), round(self.tw, 6), round(self.tf, 6))

    def apdl(self):
        # SECDATA,W1(flange width),W2(depth),t1(flange),t2(stem)
        return [
            f"SECTYPE,{self.id},BEAM,T,{self.name[:8]}",
            f"SECDATA,{_f(self.width)},{_f(self.depth)},{_f(self.tf)},{_f(self.tw)}",
        ]

    def origin_to_centroid(self):
        return None                     # unverified builder origin


@dataclass
class ChannelSection(Section):
    depth: float = 0.0
    width: float = 0.0          # flange length
    tw: float = 0.0
    tf: float = 0.0

    def key(self):
        return ("C", round(self.depth, 6), round(self.width, 6), round(self.tw, 6), round(self.tf, 6))

    def apdl(self):
        # SECDATA,W1(bot flange),W2(top flange),W3(depth),t1,t2,t3(web)
        return [
            f"SECTYPE,{self.id},BEAM,CHAN,{self.name[:8]}",
            f"SECDATA,{_f(self.width)},{_f(self.width)},{_f(self.depth)},"
            f"{_f(self.tf)},{_f(self.tf)},{_f(self.tw)}",
        ]

    def origin_to_centroid(self):
        return None                     # unverified builder origin


@dataclass
class LSection(Section):
    leg1: float = 0.0
    leg2: float = 0.0
    t: float = 0.0

    def key(self):
        return ("L", round(self.leg1, 6), round(self.leg2, 6), round(self.t, 6))

    def apdl(self):
        return [
            f"SECTYPE,{self.id},BEAM,L,{self.name[:8]}",
            f"SECDATA,{_f(self.leg1)},{_f(self.leg2)},{_f(self.t)},{_f(self.t)}",
        ]

    def origin_to_centroid(self):
        return None                     # unverified builder origin


@dataclass
class RectSection(Section):
    width: float = 0.0          # b (section y)
    depth: float = 0.0          # d (section z)

    def key(self):
        return ("RECT", round(self.width, 6), round(self.depth, 6))

    def apdl(self):
        return [
            f"SECTYPE,{self.id},BEAM,RECT,{self.name[:8]}",
            f"SECDATA,{_f(self.width)},{_f(self.depth)}",
        ]


@dataclass
class RectHollowSection(Section):
    width: float = 0.0
    depth: float = 0.0
    t: float = 0.0

    def key(self):
        return ("HREC", round(self.width, 6), round(self.depth, 6), round(self.t, 6))

    def apdl(self):
        return [
            f"SECTYPE,{self.id},BEAM,HREC,{self.name[:8]}",
            f"SECDATA,{_f(self.width)},{_f(self.depth)},"
            f"{_f(self.t)},{_f(self.t)},{_f(self.t)},{_f(self.t)}",
        ]

    def origin_to_centroid(self):
        # builder origin at the section corner (empirically verified)
        return (self.width / 2.0, self.depth / 2.0)


@dataclass
class CircSection(Section):
    diameter: float = 0.0

    def key(self):
        return ("CSOLID", round(self.diameter, 6))

    def apdl(self):
        return [
            f"SECTYPE,{self.id},BEAM,CSOLID,{self.name[:8]}",
            f"SECDATA,{_f(self.diameter / 2.0)}",
        ]


@dataclass
class PipeSection(Section):
    od: float = 0.0             # outer diameter
    t: float = 0.0              # wall thickness

    def key(self):
        return ("PIPE", round(self.od, 6), round(self.t, 6))

    def apdl(self):
        return [
            f"SECTYPE,{self.id},PIPE,,{self.name[:8]}",
            f"SECDATA,{_f(self.od)},{_f(self.t)}",
        ]


@dataclass
class TubeSection(Section):
    """Circular hollow used by a BEAM188 member (CHS brace/column): the
    PIPE section category is only legal on pipe elements, so framing tubes
    emit the BEAM,CTUBE subtype instead."""
    od: float = 0.0
    t: float = 0.0

    def key(self):
        return ("CTUBE", round(self.od, 6), round(self.t, 6))

    def apdl(self):
        ri = self.od / 2.0 - self.t
        return [
            f"SECTYPE,{self.id},BEAM,CTUBE,{self.name[:8]}",
            f"SECDATA,{_f(ri)},{_f(self.od / 2.0)}",
        ]


@dataclass
class ShellSection(Section):
    #: layers as (thickness [m], material id); single-layer for homogeneous shells
    layers: list[tuple[float, int]] = field(default_factory=list)

    @property
    def total_thickness(self) -> float:
        return sum(t for t, _ in self.layers)

    def key(self):
        return ("SHELL",) + tuple((round(t, 6), m) for t, m in self.layers)

    def apdl(self):
        lines = [f"SECTYPE,{self.id},SHELL,,{self.name[:8]}"]
        for t, mat in self.layers:
            lines.append(f"SECDATA,{_f(t)},{mat}")
        return lines
