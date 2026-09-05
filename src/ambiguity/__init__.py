"""
AMBIGUITY CLARIFIER — LangGraph redesign.

The clarification loop is a graph (see graph.py): generate -> analyze ->
summarize -> ask_human -> (stop | integrate -> generate). State is
checkpointed, so a run can be inspected or resumed at any point; the human
step pauses the graph via langgraph.types.interrupt() rather than blocking
on input() inside library code.

See C:\\Users\\dbian\\.claude\\plans\\elegant-prancing-pillow.md for the
design rationale, and .scratch/tickets/ for the earlier geometry/round-loop
decisions this still implements.
"""

from .analyzer import AmbiguityAnalyzer
from .driver import prompt_axis_cli, prompt_human_cli, run_clarify, run_particle_clarify
from .graph import build_clarifier_graph, build_particle_clarifier_graph
from .models import AmbiguityResult, Interpretation
from .state import ClarifierState, ParticleState

__all__ = [
    "AmbiguityAnalyzer",
    "AmbiguityResult",
    "Interpretation",
    "ClarifierState",
    "ParticleState",
    "build_clarifier_graph",
    "run_clarify",
    "prompt_human_cli",
    # Particles-only redesign (ticket 004) — side by side with the above,
    # not replacing it yet.
    "build_particle_clarifier_graph",
    "run_particle_clarify",
    "prompt_axis_cli",
]
