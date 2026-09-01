"""Conversion audit ledger.

Every IfcProduct that enters the pipeline must end in exactly one bucket:
CONVERTED, EXCLUDED (by a cited rule), UNHANDLED (reported, never silent),
or FLAGGED (converted but needs engineer review). The ledger also collects
free-form pipeline events (heuristic BC applications, unit reinterpretations,
material cascade decisions) so the emitted model is fully explainable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class AuditStatus(str, Enum):
    CONVERTED = "converted"
    EXCLUDED = "excluded"
    UNHANDLED = "unhandled"
    FLAGGED = "flagged"


@dataclass
class AuditEntry:
    guid: str
    ifc_class: str
    name: str
    status: str
    detail: str = ""
    system: str = ""


@dataclass
class AuditEvent:
    category: str      # 'material' | 'boundary-condition' | 'geometry' | 'units' | 'classification'
    message: str
    guid: str = ""
    severity: str = "info"   # 'info' | 'warning'


@dataclass
class AuditLedger:
    source_file: str = ""
    entries: dict[str, AuditEntry] = field(default_factory=dict)
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, guid: str, ifc_class: str, name: str, status: AuditStatus,
               detail: str = "", system: str = "") -> None:
        self.entries[guid] = AuditEntry(guid, ifc_class, name or "", status.value, detail, system)

    def event(self, category: str, message: str, guid: str = "", severity: str = "info") -> None:
        self.events.append(AuditEvent(category, message, guid, severity))

    # -- integrity ------------------------------------------------------------

    def unaccounted(self, all_guids: set[str]) -> set[str]:
        return all_guids - set(self.entries)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries.values():
            out[e.status] = out.get(e.status, 0) + 1
        return out

    # -- output ---------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        payload = {
            "source_file": self.source_file,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "summary": self.counts(),
            "entries": [asdict(e) for e in self.entries.values()],
            "events": [asdict(e) for e in self.events],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def to_markdown(self, path: str | Path) -> None:
        lines = [f"# Conversion audit — {self.source_file}", ""]
        counts = self.counts()
        lines.append("| status | count |")
        lines.append("|---|---|")
        for k in ("converted", "flagged", "excluded", "unhandled"):
            if k in counts:
                lines.append(f"| {k} | {counts[k]} |")
        lines.append("")
        for status, title in (
            ("unhandled", "Unhandled products (require attention)"),
            ("flagged", "Flagged for engineer review"),
            ("excluded", "Excluded by rule"),
        ):
            rows = [e for e in self.entries.values() if e.status == status]
            if rows:
                lines.append(f"## {title}")
                lines.append("| GUID | class | name | detail |")
                lines.append("|---|---|---|---|")
                for e in rows:
                    lines.append(f"| {e.guid} | {e.ifc_class} | {e.name} | {e.detail} |")
                lines.append("")
        warns = [ev for ev in self.events if ev.severity == "warning"]
        if warns:
            lines.append("## Warnings")
            for ev in warns:
                lines.append(f"- **{ev.category}**: {ev.message}" + (f" ({ev.guid})" if ev.guid else ""))
            lines.append("")
        infos = [ev for ev in self.events if ev.severity == "info"]
        if infos:
            lines.append("## Decisions")
            for ev in infos:
                lines.append(f"- {ev.category}: {ev.message}" + (f" ({ev.guid})" if ev.guid else ""))
        Path(path).write_text("\n".join(lines), encoding="utf-8")
