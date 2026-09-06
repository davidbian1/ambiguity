"""TicketResolver protocol — one implementation per ticket type, each an
independently swappable strategy (see SPEC.md's "Resolvers and tool
loadout"). A real resolver wraps a Claude Agent SDK session with a fixed
tool loadout; these Fake* resolvers stand in for that here, since this is
throwaway prototype code meant to validate resolve_ticket's pause/resume
shape, not to make live API calls.

`step()` is called once per Round: it's handed whatever `resolver_state`
and `human_answer` the previous round returned (both None on a ticket's
first round) and returns either AskHuman (resolve_ticket should pause) or
Resolution (the ticket is done). A real resolver's `resolver_state` would
carry its Agent SDK session_id and pending tool_use_id; here it's just
enough state for the fake resolver to know which round it's on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from .schema import Ticket


@dataclass
class AskHuman:
    prompt: str
    kind: str  # "question" | "choice" | "approval"
    options: Optional[list] = None
    state: Optional[Dict] = None


@dataclass
class Resolution:
    text: str


class TicketResolver(Protocol):
    def step(self, ticket: Ticket, resolver_state: Optional[Dict], human_answer: Optional[Any]) -> "AskHuman | Resolution": ...


class AfkResolver:
    """Research/Task-without-hitl: never asks, resolves in its first round."""

    def step(self, ticket: Ticket, resolver_state: Optional[Dict], human_answer: Optional[Any]):
        return Resolution(text=f"[afk] resolved '{ticket.question}' without human input")


class GrillingResolver:
    """Asks exactly one question, then resolves using whatever answer comes back."""

    def step(self, ticket: Ticket, resolver_state: Optional[Dict], human_answer: Optional[Any]):
        if human_answer is None:
            return AskHuman(
                prompt=f"Regarding '{ticket.question}' — what's the answer?",
                kind="question",
                state={"asked": True},
            )
        return Resolution(text=f"human answered: {human_answer}")
