"""Graph state — the single shape every node reads from and writes to.

`history` and `messages` use an additive reducer (`operator.add`): each node
returns only the *new* items to append, LangGraph merges them into the
running list. Everything else is last-write-wins (LangGraph's default).
"""

import operator
from typing import Annotated, Dict, List, Optional, TypedDict


class ClarifierState(TypedDict):
    idea: str
    current_spec: str
    round: int
    interpretations: List[str]
    # Deliberately a plain dict, not AmbiguityResult — this gets checkpointed,
    # and AmbiguityResult/Interpretation carry raw np.ndarray embeddings that
    # LangGraph's msgpack serializer can't handle without type registration
    # (and won't handle at all in a future version — verified via a live
    # warning during testing). Keys: v1_text, v2_text, ambiguity_score,
    # semantic_volume, pc1_variance_ratio, spec_projection.
    analysis: Optional[Dict[str, object]]
    descriptions: Optional[Dict[str, str]]
    choice: Optional[str]
    custom_text: Optional[str]
    history: Annotated[List[Dict], operator.add]
    # The running conversation — every LLM-backed node appends its prompt +
    # response here instead of only using isolated one-shot calls. By the
    # time `integrate` runs, the model has genuinely seen every earlier
    # round in this same thread, not just whatever we chose to re-inject
    # via prior_choices.
    messages: Annotated[List[Dict], operator.add]


# Particles-only redesign (ticket 004) — active alongside ClarifierState
# above, not replacing it yet (see the test-first gate in particles.py's
# module docstring). `current_spec`/`interpretations`/`analysis`/
# `descriptions` have no equivalent here: there's no evolving spec prose
# and no PCA result to hold, just a population of {axis: value} particles.
#
# `ledger` values are only ever "confirmed" in this minimal build — the
# "accepted-default" status (for a checkpoint-driven stop) is part of the
# not-yet-built expand_axes/checkpoint slice, deliberately deferred.
class ParticleState(TypedDict):
    seed_text: str
    particles: List[Dict[str, Optional[str]]]
    weights: List[float]
    ledger: Dict[str, str]  # axis -> "confirmed"
    axis_class: Dict[str, str]  # axis -> "decision" | "detail"
    pending_axis: Optional[str]
    choice: Optional[str]
    round: int
    history: Annotated[List[Dict], operator.add]
    messages: Annotated[List[Dict], operator.add]
