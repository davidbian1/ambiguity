"""
Builds the clarification loop as a graph: generate -> analyze -> summarize
-> ask_human -> (stop | integrate -> generate | stop).

To insert a step (e.g. a validation pass between generate and analyze):
    graph.add_node("validate", validate_node)
    # replace graph.add_edge("generate", "analyze") with:
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "analyze")

To delegate a step to a separate agent later (e.g. integrate), see the
EXTENSION POINT comment on make_integrate_node in nodes.py — swap that
node's body, nothing here needs to change.

Swap `checkpointer` for a persistent one (e.g. langgraph-checkpoint-sqlite's
SqliteSaver, or a Postgres-backed saver) to make state survive process
restarts — InMemorySaver here is a first-draft default, not a ceiling.
"""

from typing import Any, Callable, Optional

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    ask_axis_node,
    ask_human_node,
    make_analyze_node,
    make_extract_and_generate_node,
    make_generate_node,
    make_integrate_node,
    make_pick_axis_node,
    make_reweight_node,
    make_summarize_node,
)
from .routers import (
    make_route_after_ask,
    make_route_after_human,
    make_route_after_integrate,
    make_route_after_reweight,
)
from .state import ClarifierState, ParticleState


def build_clarifier_graph(
    llm_client: Any,
    embed_fn: Callable[[str], np.ndarray],
    n_interpretations: int = 8,
    max_rounds: int = 5,
    checkpointer: Optional[Any] = None,
):
    graph = StateGraph(ClarifierState)
    graph.add_node("generate", make_generate_node(llm_client, n_interpretations))
    graph.add_node("analyze", make_analyze_node(embed_fn))
    graph.add_node("summarize", make_summarize_node(llm_client))
    graph.add_node("ask_human", ask_human_node)
    graph.add_node("integrate", make_integrate_node(llm_client))

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "analyze")
    graph.add_edge("analyze", "summarize")
    graph.add_edge("summarize", "ask_human")
    graph.add_conditional_edges(
        "ask_human", make_route_after_human(), {END: END, "integrate": "integrate"}
    )
    graph.add_conditional_edges(
        "integrate", make_route_after_integrate(max_rounds), {END: END, "generate": "generate"}
    )

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# =============================================================================
# Particles-only redesign (ticket 004) — active alongside
# build_clarifier_graph above, not replacing it yet. Loop shape:
# extract_and_generate (once) -> pick_axis -> ask_axis -> (stop |
# reweight -> pick_axis | stop).
# =============================================================================


def build_particle_clarifier_graph(
    llm_client: Any,
    n_particles: int = 8,
    max_rounds: int = 8,
    refresh_every: int = 3,
    refresh_top_k: int = 4,
    checkpointer: Optional[Any] = None,
):
    graph = StateGraph(ParticleState)
    graph.add_node("extract_and_generate", make_extract_and_generate_node(llm_client, n_particles))
    graph.add_node("pick_axis", make_pick_axis_node())
    graph.add_node("ask_axis", ask_axis_node)
    graph.add_node("reweight", make_reweight_node(llm_client, refresh_every, refresh_top_k))

    graph.add_edge(START, "extract_and_generate")
    graph.add_edge("extract_and_generate", "pick_axis")
    graph.add_edge("pick_axis", "ask_axis")
    graph.add_conditional_edges("ask_axis", make_route_after_ask(), {END: END, "reweight": "reweight"})
    graph.add_conditional_edges(
        "reweight", make_route_after_reweight(max_rounds), {END: END, "pick_axis": "pick_axis"}
    )

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
