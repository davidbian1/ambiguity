"""
Drives a compiled clarifier graph to completion, handling the interrupt/
resume cycle for the human-in-the-loop step. The CLI prompt logic lives
here, not in the graph — the graph itself doesn't know or care whether a
human, a test fake, or a different process entirely supplies the answer.
"""

import uuid
from typing import Any, Callable, Dict, Optional

from langgraph.types import Command
from langsmith import traceable

from .prompts import SYSTEM_PROMPT


def prompt_human_cli(payload: Dict) -> Dict:
    """Default human_responder: the only place that calls input()."""
    print(f"\n=== Round {payload['round']} - {payload['topic']} ===")
    print(f"1) {payload['v1']}")
    print(f"2) {payload['v2']}")
    print(
        f"   ambiguity_score={payload['ambiguity_score']:.3f}  "
        f"semantic_volume={payload['semantic_volume']:.3f}  "
        f"spec_projection={payload['spec_projection']:.3f}"
    )
    print("Choose: 1/v1, 2/v2, 3/tolerate, 4/custom, q/stop")

    while True:
        raw = input("> ").strip().lower()
        if raw in ("1", "v1"):
            return {"choice": "v1"}
        if raw in ("2", "v2"):
            return {"choice": "v2"}
        if raw in ("3", "tolerate"):
            return {"choice": "tolerate"}
        if raw in ("4", "custom"):
            custom_text = input("Your own answer: ").strip()
            return {"choice": "custom", "custom_text": custom_text}
        if raw in ("q", "stop"):
            return {"choice": "stop"}
        print("Didn't recognize that - try 1, 2, 3, 4, or q.")


# run_type="chain" groups every graph.invoke() call this makes (one per
# round, plus one per human resume) into a single trace tree, instead of
# each invoke() showing up as its own disconnected top-level run.
@traceable(run_type="chain", name="clarify_session")
def run_clarify(
    graph,
    idea: str,
    thread_id: Optional[str] = None,
    human_responder: Callable[[Dict], Dict] = prompt_human_cli,
) -> Dict:
    """Runs a clarify() session to completion on a compiled graph.

    `thread_id`: pass an existing one to resume a session that was
    interrupted (process crash, closed terminal, etc.) — the checkpointer
    already has its state; a fresh call just picks the interrupt back up.
    """
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "idea": idea,
        "current_spec": idea,
        "round": 1,
        "interpretations": [],
        "analysis": None,
        "descriptions": None,
        "choice": None,
        "custom_text": None,
        "history": [],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    }
    result = graph.invoke(initial_state, config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        response = human_responder(payload)
        result = graph.invoke(Command(resume=response), config)

    return {
        "original_idea": idea,
        "final_spec": result["current_spec"],
        "rounds": result["history"],
        "total_rounds": len(result["history"]),
        "thread_id": thread_id,
    }


# =============================================================================
# Particles-only redesign (ticket 004) — active alongside run_clarify /
# prompt_human_cli above, not replacing them yet.
# =============================================================================


def prompt_axis_cli(payload: Dict) -> Dict:
    """Default human_responder for the particle graph. Presents an axis's
    actual candidate values instead of a v1/v2 pair."""
    print(f"\n=== Round {payload['round']} - {payload['axis']} ===")
    for i, value in enumerate(payload["candidate_values"], start=1):
        print(f"{i}) {value}")
    print(
        f"   ambiguity_score={payload['ambiguity_score']:.3f}  "
        f"semantic_volume={payload['semantic_volume']:.3f}"
    )
    print("Choose a number, type your own answer, or q/stop")

    while True:
        raw = input("> ").strip()
        if raw.lower() in ("q", "stop"):
            return {"choice": "stop"}
        if raw.isdigit() and 1 <= int(raw) <= len(payload["candidate_values"]):
            return {"choice": payload["candidate_values"][int(raw) - 1]}
        if raw:
            return {"choice": raw}  # free-text answer -> score_consistency fallback path
        print("Didn't recognize that - try a number, your own answer, or q.")


@traceable(run_type="chain", name="particle_clarify_session")
def run_particle_clarify(
    graph,
    seed_text: str,
    thread_id: Optional[str] = None,
    human_responder: Callable[[Dict], Dict] = prompt_axis_cli,
) -> Dict:
    """Runs a particle-based clarify() session to completion. Same
    interrupt/resume shape as run_clarify, different result shape — there's
    no rewritten spec prose, just the confirmed axis/value ledger."""
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "seed_text": seed_text,
        "particles": [],
        "weights": [],
        "ledger": {},
        "axis_class": {},
        "pending_axis": None,
        "choice": None,
        "round": 1,
        "history": [],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
    }
    result = graph.invoke(initial_state, config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        response = human_responder(payload)
        result = graph.invoke(Command(resume=response), config)

    # The actually-confirmed value is what the human answered, recorded
    # directly on each history entry — not something to re-derive from
    # particle weights (fragile: ties, and a later round's reweighting
    # could shift which particle looks "heaviest" without changing what
    # was actually confirmed on an earlier axis).
    confirmed = {entry["axis"]: entry["choice"] for entry in result["history"]}
    return {
        "seed_text": seed_text,
        "confirmed": confirmed,
        "rounds": result["history"],
        "total_rounds": len(result["history"]),
        "thread_id": thread_id,
    }
