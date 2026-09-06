"""The four implemented metrics, folded from events.jsonl — see SPEC.md's
"Metrics". Purely observational: nothing here feeds back into the graph.
The fifth (ticket ambiguity score) is a stretch item, deliberately not
implemented — see the map's "Not yet specified".
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def rounds_to_resolve(events: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    result: Dict[str, int] = {}
    for e in events:
        if e["type"] == "round":
            counts[e["ticket_id"]] = e["round"]
        elif e["type"] == "ticket_resolved":
            result[e["ticket_id"]] = counts.get(e["ticket_id"], e["rounds"])
    return result


def frontier_width_series(events: List[Dict]) -> List[Tuple[str, int]]:
    return [(e["at"], e["width"]) for e in events if e["type"] == "frontier_computed"]


def fog_shrinkage_rate(events: List[Dict]) -> float:
    added = sum(1 for e in events if e["type"] == "fog_added")
    graduated = sum(1 for e in events if e["type"] == "fog_graduated")
    if added == 0:
        return 0.0
    return graduated / added


def decision_volatility(events: List[Dict]) -> int:
    return sum(1 for e in events if e["type"] == "ticket_reopened")
