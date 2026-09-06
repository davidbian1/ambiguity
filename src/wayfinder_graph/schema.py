"""Chart/Ticket JSON schema — see SPEC.md's "Storage" section, which this
must stay in sync with. Plain dataclasses with to_dict/from_dict rather than
pydantic: this is throwaway prototype code, no need for a new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

TicketType = Literal["research", "prototype", "grilling", "task"]
TicketStatus = Literal["open", "resolved"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Decision:
    ticket_id: str
    title: str
    gist: str

    def to_dict(self) -> Dict:
        return {"ticket_id": self.ticket_id, "title": self.title, "gist": self.gist}

    @staticmethod
    def from_dict(d: Dict) -> "Decision":
        return Decision(ticket_id=d["ticket_id"], title=d["title"], gist=d["gist"])


@dataclass
class FogEntry:
    text: str
    added_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict:
        return {"text": self.text, "added_at": self.added_at}

    @staticmethod
    def from_dict(d: Dict) -> "FogEntry":
        return FogEntry(text=d["text"], added_at=d["added_at"])


@dataclass
class OutOfScopeEntry:
    text: str
    ticket_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {"text": self.text, "ticket_id": self.ticket_id}

    @staticmethod
    def from_dict(d: Dict) -> "OutOfScopeEntry":
        return OutOfScopeEntry(text=d["text"], ticket_id=d.get("ticket_id"))


@dataclass
class Chart:
    id: str
    destination: str
    notes: str = ""
    decisions: List[Decision] = field(default_factory=list)
    fog: List[FogEntry] = field(default_factory=list)
    out_of_scope: List[OutOfScopeEntry] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "destination": self.destination,
            "notes": self.notes,
            "decisions": [d.to_dict() for d in self.decisions],
            "fog": [f.to_dict() for f in self.fog],
            "out_of_scope": [o.to_dict() for o in self.out_of_scope],
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Chart":
        return Chart(
            id=d["id"],
            destination=d["destination"],
            notes=d.get("notes", ""),
            decisions=[Decision.from_dict(x) for x in d.get("decisions", [])],
            fog=[FogEntry.from_dict(x) for x in d.get("fog", [])],
            out_of_scope=[OutOfScopeEntry.from_dict(x) for x in d.get("out_of_scope", [])],
            created_at=d.get("created_at", _now()),
        )


@dataclass
class Ticket:
    id: str
    title: str
    type: TicketType
    question: str
    status: TicketStatus = "open"
    hitl: bool = True
    blocked_by: List[str] = field(default_factory=list)
    claimed_by: Optional[str] = None
    resolution: Optional[str] = None
    created_at: str = field(default_factory=_now)
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "question": self.question,
            "status": self.status,
            "hitl": self.hitl,
            "blocked_by": self.blocked_by,
            "claimed_by": self.claimed_by,
            "resolution": self.resolution,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Ticket":
        return Ticket(
            id=d["id"],
            title=d["title"],
            type=d["type"],
            question=d["question"],
            status=d.get("status", "open"),
            hitl=d.get("hitl", True),
            blocked_by=d.get("blocked_by", []),
            claimed_by=d.get("claimed_by"),
            resolution=d.get("resolution"),
            created_at=d.get("created_at", _now()),
            resolved_at=d.get("resolved_at"),
        )
