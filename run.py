"""
Entry point: run a real AMBIGUITY CLARIFIER session against the live
OpenAI API.

Env vars come from a local .env file (see .env.example — copy it to .env
and fill in real values; .env is already gitignored at the repo root).
Required: OPENAI_API_KEY. Optional, for LangSmith tracing:
LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT — omit all
three to run with tracing off (traced calls are harmless no-ops).

Usage:
    python run.py Build a social app where people share recommendations
    python run.py                      (prompts for the idea interactively)
"""

import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable

from ambiguity import build_clarifier_graph, run_clarify

load_dotenv()

print("OpenAI key:", bool(os.getenv("OPENAI_API_KEY")))
print("LangSmith key:", bool(os.getenv("LANGSMITH_API_KEY")))
print("Tracing:", os.getenv("LANGSMITH_TRACING"))


def make_embed_fn(client: OpenAI):
    @traceable(run_type="embedding", name="embed")
    def embed(text: str) -> np.ndarray:
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        vec = np.array(resp.data[0].embedding, dtype=np.float64)
        return vec / np.linalg.norm(vec)

    return embed


def _tracing_status() -> str:
    on = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    if not on:
        return "off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable)"
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    return f"on (project: {project})"


def main() -> None:
    idea = " ".join(sys.argv[1:]).strip() or input("What's the idea? ").strip()
    if not idea:
        print("No idea given.")
        sys.exit(1)

    print(f"Tracing: {_tracing_status()}")

    client = OpenAI()
    graph = build_clarifier_graph(client, make_embed_fn(client))
    result = run_clarify(graph, idea)

    print("\n" + "=" * 80)
    print("FINAL SPEC")
    print("=" * 80)
    print(result["final_spec"])
    print(f"\n({result['total_rounds']} round(s) - thread_id={result['thread_id']})")


if __name__ == "__main__":
    main()
