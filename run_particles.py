"""
Entry point: run a real particles-only AMBIGUITY CLARIFIER session (ticket
004) against the live OpenAI API. No embed_fn needed — this path never
embeds text, it tracks discrete {axis: value} particles instead.

Env vars come from a local .env file (see .env.example). Required:
OPENAI_API_KEY. Optional, for LangSmith tracing: LANGSMITH_TRACING,
LANGSMITH_API_KEY, LANGSMITH_PROJECT.

Usage:
    python run_particles.py Build a chess engine
    python run_particles.py                      (prompts interactively)
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from ambiguity import build_particle_clarifier_graph, run_particle_clarify

load_dotenv()

print("OpenAI key:", bool(os.getenv("OPENAI_API_KEY")))
print("LangSmith key:", bool(os.getenv("LANGSMITH_API_KEY")))
print("Tracing:", os.getenv("LANGSMITH_TRACING"))


def _tracing_status() -> str:
    on = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    if not on:
        return "off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)"
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    return f"on (project: {project})"


def main() -> None:
    seed_text = " ".join(sys.argv[1:]).strip() or input("What's the idea? ").strip()
    if not seed_text:
        print("No idea given.")
        sys.exit(1)

    print(f"Tracing: {_tracing_status()}")

    client = OpenAI()
    graph = build_particle_clarifier_graph(client, n_particles=8, max_rounds=8, refresh_every=3, refresh_top_k=4)
    result = run_particle_clarify(graph, seed_text)

    print("\n" + "=" * 80)
    print("CONFIRMED AXES")
    print("=" * 80)
    for axis, value in result["confirmed"].items():
        print(f"  {axis}: {value}")
    print(f"\n({result['total_rounds']} round(s) - thread_id={result['thread_id']})")


if __name__ == "__main__":
    main()
