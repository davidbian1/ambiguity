"""Chart-level orchestration graph — see SPEC.md's "Chart-level
orchestration" and "HITL composition" for why this is shaped the way it
is (four phases, `resolve_ticket` looping on itself once per Round).
"""

from datetime import datetime, timezone
from typing import Dict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .resolvers import AskHuman, TicketResolver
from .state import WayfinderState
from .storage import ChartStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_frontier(store: ChartStore) -> list:
    tickets = store.list_tickets()
    resolved_ids = {t.id for t in tickets if t.status == "resolved"}
    frontier = [
        t.id
        for t in tickets
        if t.status == "open"
        and t.claimed_by is None
        and all(b in resolved_ids for b in t.blocked_by)
    ]
    return frontier


def make_recompute_frontier_node(store: ChartStore):
    def recompute_frontier_node(state: WayfinderState) -> Dict:
        frontier = compute_frontier(store)
        store.append_event({"type": "frontier_computed", "width": len(frontier), "ticket_ids": frontier, "at": _now()})
        return {"frontier": frontier}

    return recompute_frontier_node


def route_after_frontier(state: WayfinderState) -> str:
    return END if not state["frontier"] else "select_ticket"


def make_select_ticket_node(store: ChartStore, session_id: str):
    def select_ticket_node(state: WayfinderState) -> Dict:
        ticket_id = state["frontier"][0]
        ticket = store.load_ticket(ticket_id)
        ticket.claimed_by = session_id
        store.save_ticket(ticket)
        store.append_event({"type": "ticket_claimed", "ticket_id": ticket_id, "at": _now()})
        return {"current_ticket_id": ticket_id, "round": 0, "resolver_state": None, "human_answer": None, "resolution_text": None}

    return select_ticket_node


def make_resolve_ticket_node(store: ChartStore, registry: Dict[str, TicketResolver]):
    def resolve_ticket_node(state: WayfinderState) -> Dict:
        ticket = store.load_ticket(state["current_ticket_id"])
        resolver = registry[ticket.type]
        action = resolver.step(ticket, state.get("resolver_state"), state.get("human_answer"))
        store.append_event({"type": "round", "ticket_id": ticket.id, "round": state["round"] + 1, "at": _now()})

        if isinstance(action, AskHuman):
            answer = interrupt({"ticket_id": ticket.id, "prompt": action.prompt, "kind": action.kind, "options": action.options})
            return {"resolver_state": action.state, "human_answer": answer, "round": state["round"] + 1}

        return {"resolution_text": action.text, "resolver_state": None, "human_answer": None, "round": state["round"] + 1}

    return resolve_ticket_node


def route_after_resolve(state: WayfinderState) -> str:
    return "record_resolution" if state["resolution_text"] is not None else "resolve_ticket"


def make_record_resolution_node(store: ChartStore):
    def record_resolution_node(state: WayfinderState) -> Dict:
        ticket = store.load_ticket(state["current_ticket_id"])
        ticket.status = "resolved"
        ticket.resolution = state["resolution_text"]
        ticket.resolved_at = _now()
        store.save_ticket(ticket)
        store.append_event({"type": "ticket_resolved", "ticket_id": ticket.id, "rounds": state["round"], "at": _now()})
        return {"current_ticket_id": None, "resolution_text": None, "resolver_state": None, "human_answer": None}

    return record_resolution_node


def build_wayfinder_graph(store: ChartStore, registry: Dict[str, TicketResolver], session_id: str = "session", checkpointer=None):
    graph = StateGraph(WayfinderState)
    graph.add_node("recompute_frontier", make_recompute_frontier_node(store))
    graph.add_node("select_ticket", make_select_ticket_node(store, session_id))
    graph.add_node("resolve_ticket", make_resolve_ticket_node(store, registry))
    graph.add_node("record_resolution", make_record_resolution_node(store))

    graph.add_edge(START, "recompute_frontier")
    graph.add_conditional_edges("recompute_frontier", route_after_frontier, {END: END, "select_ticket": "select_ticket"})
    graph.add_edge("select_ticket", "resolve_ticket")
    graph.add_conditional_edges(
        "resolve_ticket", route_after_resolve, {"resolve_ticket": "resolve_ticket", "record_resolution": "record_resolution"}
    )
    graph.add_edge("record_resolution", "recompute_frontier")

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
