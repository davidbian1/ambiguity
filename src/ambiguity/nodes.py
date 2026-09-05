"""
Graph nodes. Each is a small function: (state) -> partial state update.
Dependencies (llm_client, embed_fn) are closed over via factory functions
rather than stored in state, since state gets checkpointed/serialized and
a live client object shouldn't be.

Node functions are the extension points: to insert a step, add a node here
and rewire two edges in graph.py. To delegate a step to a separate agent
later, replace that node's body with a call into the other agent — the
signature (state in, partial state out) doesn't have to change.
"""

from typing import Dict

from langgraph.types import interrupt

from .analyzer import AmbiguityAnalyzer
from .llm_calls import (
    extract_axes,
    generate_axis_particle,
    generate_interpretations,
    integrate_choice,
    summarize_v1v2,
)
from .particles import compute_metrics, normalize, pick_next_axis, refresh_particles, reweight_particles


def make_generate_node(llm_client, n_interpretations: int):
    def generate_node(state: Dict) -> Dict:
        interpretations, new_messages = generate_interpretations(
            llm_client, state["messages"], state["current_spec"], n_interpretations
        )
        return {"interpretations": interpretations, "messages": new_messages}

    return generate_node


# -----------------------------------------------------------------------
# Particles-only redesign (ticket 004). Runs exactly once (guarded by the
# `axis_class` check) as the graph's entry step — there's no expand_axes
# in this minimal build, so axis extraction never needs to run again.
# -----------------------------------------------------------------------

def make_extract_and_generate_node(llm_client, n_particles: int):
    def extract_and_generate_node(state: Dict) -> Dict:
        if state.get("axis_class"):
            return {}  # already extracted — no-op if the graph loops back here
        new_axes, msgs = extract_axes(llm_client, state["messages"], state["seed_text"])
        axis_class = {a["axis"]: a.get("class", "decision") for a in new_axes}
        decision_axes = [a for a, cls in axis_class.items() if cls == "decision"]

        particles = []
        for _ in range(n_particles):
            particle: Dict[str, str] = {}
            for axis in decision_axes:
                value, particle_msgs = generate_axis_particle(llm_client, state["seed_text"], axis)
                particle[axis] = value
                msgs += particle_msgs
            particles.append(particle)

        weights = normalize([1.0] * len(particles)) if particles else []
        return {"axis_class": axis_class, "particles": particles, "weights": weights, "messages": msgs}

    return extract_and_generate_node


def make_analyze_node(embed_fn):
    analyzer = AmbiguityAnalyzer(embed_fn)

    def analyze_node(state: Dict) -> Dict:
        result = analyzer.analyze(state["current_spec"], state["interpretations"])
        # Reduce to a plain, checkpoint-safe dict here — see state.py's note
        # on why AmbiguityResult itself never enters graph state.
        return {
            "analysis": {
                "v1_text": result.v1.text,
                "v2_text": result.v2.text,
                "ambiguity_score": result.ambiguity_score,
                "semantic_volume": result.semantic_volume,
                "pc1_variance_ratio": result.pc1_variance_ratio,
                "spec_projection": result.spec_projection,
            }
        }

    return analyze_node


# No embed_fn, no PCA — priority comes from counting particle values, not
# geometry (see particles.py's axis_entropy/pick_next_axis).

def make_pick_axis_node():
    def pick_axis_node(state: Dict) -> Dict:
        axis = pick_next_axis(state["particles"], state["weights"], state["axis_class"], state["ledger"])
        return {"pending_axis": axis}

    return pick_axis_node


def make_summarize_node(llm_client):
    def summarize_node(state: Dict) -> Dict:
        result = state["analysis"]
        descriptions, new_messages = summarize_v1v2(
            llm_client, state["messages"], state["current_spec"], result["v1_text"], result["v2_text"]
        )
        return {"descriptions": descriptions, "messages": new_messages}

    return summarize_node


# =============================================================================
# PROPOSED REPLACEMENT for summarize_node (not active — sketch only): THIS
# NODE HAS NO REPLACEMENT, IT JUST GOES AWAY. There's no PCA-picked v1/v2
# text pair to summarize into plain language anymore — the human is shown
# an axis's actual candidate values directly (see ask_human_node's
# replacement below), which need no separate summarization pass.
# =============================================================================


def ask_human_node(state: Dict) -> Dict:
    """Pauses the graph — LangGraph persists this via the checkpointer, so
    the run can be resumed from a different process entirely, not just
    later in the same one."""
    result = state["analysis"]
    descriptions = state["descriptions"]
    response = interrupt(
        {
            "round": state["round"],
            "topic": descriptions["ambiguity_topic"],
            "v1": descriptions["v1_summary"],
            "v2": descriptions["v2_summary"],
            "ambiguity_score": result["ambiguity_score"],
            "semantic_volume": result["semantic_volume"],
            "spec_projection": result["spec_projection"],
        }
    )
    return {"choice": response["choice"], "custom_text": response.get("custom_text")}


# Presents the picked axis's actual candidate values, not a v1/v2 pair.
# No checkpoint/accept-defaults status in this minimal build — that's part
# of the not-yet-built expand_axes slice. Running out of open axes reuses
# the same "stop" exit path a human-initiated stop takes.

def ask_axis_node(state: Dict) -> Dict:
    axis = state["pending_axis"]
    if axis is None:
        return {"choice": "stop"}
    candidate_values = sorted({v for p in state["particles"] for v in [p.get(axis)] if v})
    metrics = compute_metrics(state["particles"], state["weights"], state["axis_class"], state["ledger"])
    response = interrupt(
        {
            "round": state["round"],
            "axis": axis,
            "candidate_values": candidate_values,
            "ambiguity_score": metrics["ambiguity_score"],
            "semantic_volume": metrics["semantic_volume"],
        }
    )
    return {"choice": response["choice"]}


def make_integrate_node(llm_client):
    # EXTENSION POINT: this node currently calls integrate_choice() directly
    # (an LLM call in-process). To delegate this step to a separate agent
    # later (e.g. a dedicated "spec rewriter" agent), swap the body for
    # something like `other_graph.invoke({"decision": ..., "spec": ...})`
    # and adapt its result into the same {"current_spec", "history", ...}
    # shape — nothing else in this graph needs to change.
    def integrate_node(state: Dict) -> Dict:
        result = state["analysis"]
        descriptions = state["descriptions"]
        prior_choices = [
            {"topic": h["topic"], "decision_sentence": h["decision_sentence"]}
            for h in state["history"]
        ]
        integrated, new_messages = integrate_choice(
            llm_client,
            state["messages"],
            state["current_spec"],
            descriptions["ambiguity_topic"],
            descriptions["v1_summary"],
            descriptions["v2_summary"],
            state["choice"],
            prior_choices,
            state.get("custom_text"),
        )
        history_entry = {
            "round": state["round"],
            "topic": descriptions["ambiguity_topic"],
            "v1": descriptions["v1_summary"],
            "v2": descriptions["v2_summary"],
            "choice": state["choice"],
            "custom_text": state.get("custom_text"),
            "decision_sentence": integrated["decision_sentence"],
            "ambiguity_score": result["ambiguity_score"],
            "semantic_volume": result["semantic_volume"],
            "spec_after": integrated["rewritten_spec"],
        }
        return {
            "current_spec": integrated["rewritten_spec"],
            "history": [history_entry],
            "round": state["round"] + 1,
            "messages": new_messages,
        }

    return integrate_node


# Reweights the existing particle population instead of rewriting a whole
# spec document. NO expand_axes here — deliberately deferred per ticket
# 004's stop-gate; axis_class never grows past what extract_and_generate
# found on round 1 in this minimal build.

def make_reweight_node(llm_client, refresh_every: int, top_k: int):
    def reweight_node(state: Dict) -> Dict:
        axis = state["pending_axis"]
        answer = state["choice"]

        weights, msgs = reweight_particles(llm_client, state["particles"], state["weights"], axis, answer)
        ledger = {**state["ledger"], axis: "confirmed"}
        particles = state["particles"]

        if state["round"] % refresh_every == 0:
            particles, weights, refresh_msgs = refresh_particles(
                llm_client, particles, weights, state["axis_class"], ledger, state["seed_text"], top_k
            )
            msgs += refresh_msgs

        metrics = compute_metrics(particles, weights, state["axis_class"], ledger)
        history_entry = {
            "round": state["round"],
            "axis": axis,
            "choice": answer,
            "ambiguity_score": metrics["ambiguity_score"],
            "semantic_volume": metrics["semantic_volume"],
        }
        return {
            "particles": particles,
            "weights": weights,
            "ledger": ledger,
            "history": [history_entry],
            "round": state["round"] + 1,
            "messages": msgs,
        }

    return reweight_node
