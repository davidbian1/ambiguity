"""
Particles-only redesign (ticket 004): pure-ish state-manipulation functions
over a flat {axis: value_or_None} particle population, replacing PCA-based
embedding geometry with counted entropy over actual particle values.

Not wired into the default graph yet — see graph.py's
build_particle_clarifier_graph for the side-by-side path this backs, and
ticket 004 for the explicit test-before-committing-further gate.
"""

from math import log2
from typing import Dict, List, Optional, Tuple

from .llm_calls import generate_axis_particle, score_consistency

Particle = Dict[str, Optional[str]]


def normalize(weights: List[float]) -> List[float]:
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def axis_entropy(particles: List[Particle], weights: List[float], axis: str) -> float:
    """Shannon entropy of this axis's weighted value distribution across
    particles — a real, counted quantity, not an LLM self-judgment."""
    value_weight: Dict[Optional[str], float] = {}
    for particle, w in zip(particles, weights):
        v = particle.get(axis)
        value_weight[v] = value_weight.get(v, 0.0) + w
    total = sum(value_weight.values()) or 1.0
    h = 0.0
    for w in value_weight.values():
        p = w / total
        if p > 0:
            h -= p * log2(p)
    return h


def pick_next_axis(
    particles: List[Particle],
    weights: List[float],
    axis_class: Dict[str, str],
    ledger: Dict[str, str],
) -> Optional[str]:
    """Highest-entropy OPEN decision-class axis, or None if every
    decision-class axis is already in the ledger."""
    open_decision_axes = [a for a, cls in axis_class.items() if cls == "decision" and a not in ledger]
    if not open_decision_axes:
        return None
    return max(open_decision_axes, key=lambda a: axis_entropy(particles, weights, a))


def compute_metrics(
    particles: List[Particle],
    weights: List[float],
    axis_class: Dict[str, str],
    ledger: Dict[str, str],
) -> Dict[str, float]:
    """Maps the original AmbiguityResult.ambiguity_score/semantic_volume
    outputs onto particle state: sum of open-axis entropies, and an
    effective count of distinct surviving value-combinations."""
    open_axes = [a for a in axis_class if a not in ledger]
    ambiguity_score = sum(axis_entropy(particles, weights, a) for a in open_axes)

    combo_weight: Dict[tuple, float] = {}
    for p, w in zip(particles, weights):
        key = tuple(sorted(p.items()))
        combo_weight[key] = combo_weight.get(key, 0.0) + w
    total = sum(combo_weight.values()) or 1.0
    h_combo = 0.0
    for w in combo_weight.values():
        p = w / total
        if p > 0:
            h_combo -= p * log2(p)
    semantic_volume = float(2**h_combo)  # effective number of distinct surviving specs

    return {"ambiguity_score": ambiguity_score, "semantic_volume": semantic_volume}


def reweight_particles(
    llm_client, particles: List[Particle], weights: List[float], axis: str, answer: str
) -> Tuple[List[float], List[Dict]]:
    """Exact-match answers reweight deterministically (no LLM call). Only
    an answer that doesn't match any existing value for this axis falls
    through to score_consistency."""
    existing_values = {p.get(axis) for p in particles}
    exact_match = answer in existing_values
    new_messages: List[Dict] = []
    new_weights = []
    for particle, w in zip(particles, weights):
        if exact_match:
            particle_weight = 1.0 if particle.get(axis) == answer else 1e-6
        else:
            particle_weight, msgs = score_consistency(llm_client, axis, particle.get(axis), answer)
            new_messages += msgs
        new_weights.append(w * particle_weight)
    return normalize(new_weights), new_messages


def refresh_particles(
    llm_client,
    particles: List[Particle],
    weights: List[float],
    axis_class: Dict[str, str],
    ledger: Dict[str, str],
    seed_text: str,
    top_k: int,
) -> Tuple[List[Particle], List[float], List[Dict]]:
    """Small-N heuristic, not ESS/SMC theory: keep top-k by weight, refresh
    the rest via fresh particle generation on still-open axes. Called on a
    fixed round schedule by the caller, not weight-triggered."""
    ranked = sorted(zip(particles, weights), key=lambda pw: -pw[1])
    kept = ranked[:top_k]
    new_particles = [p for p, _ in kept]
    new_weights = [w for _, w in kept]
    open_axes = [a for a in axis_class if a not in ledger]
    new_messages: List[Dict] = []
    while len(new_particles) < len(particles):
        particle: Particle = {}
        for axis in open_axes:
            value, msgs = generate_axis_particle(llm_client, seed_text, axis)
            particle[axis] = value
            new_messages += msgs
        new_particles.append(particle)
        new_weights.append(1.0 / len(particles))
    return new_particles, normalize(new_weights), new_messages
