"""Graph state for the chart-level loop. Deliberately thin: the chart and
its tickets live in the JSON store (see storage.py), not duplicated here —
this only carries what a single round of resolve_ticket needs to survive
an interrupt()/resume cycle. See SPEC.md's "HITL composition".
"""

from typing import Any, Dict, List, Optional, TypedDict


class WayfinderState(TypedDict):
    frontier: List[str]
    current_ticket_id: Optional[str]
    resolver_state: Optional[Dict[str, Any]]
    human_answer: Optional[Any]
    resolution_text: Optional[str]
    round: int
