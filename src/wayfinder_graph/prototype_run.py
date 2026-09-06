"""Throwaway demo — NOT a shipped CLI (see SPEC.md's "Out of scope").
Builds a two-ticket chart (one AFK, one that pauses for a human), runs the
graph to completion using the same interrupt/resume loop shape as
src/ambiguity/driver.py, and prints the computed metrics at the end. Exists
to prove the schema/storage/graph/resolver/metrics pieces actually fit
together, not to be maintained.

Run with: uv run python -m wayfinder_graph.prototype_run
"""

import shutil
import tempfile
from pathlib import Path

from langgraph.types import Command

from .graph import build_wayfinder_graph
from .metrics import decision_volatility, fog_shrinkage_rate, frontier_width_series, rounds_to_resolve
from .resolvers import AfkResolver, GrillingResolver
from .schema import Chart, Ticket
from .storage import JsonChartStore


def seed_chart(chart_dir: Path) -> None:
    store = JsonChartStore(chart_dir)
    store.save_chart(Chart(id="chart-1", destination="Prove the Wayfinder Graph shape holds together."))
    store.save_ticket(Ticket(id="t-research", title="Look something up", type="research", question="What's the answer to everything?"))
    store.save_ticket(
        Ticket(
            id="t-grilling",
            title="Pin a decision",
            type="grilling",
            question="Which color should the button be?",
            blocked_by=["t-research"],
        )
    )


def main() -> None:
    chart_dir = Path(tempfile.mkdtemp(prefix="wayfinder_graph_prototype_"))
    try:
        seed_chart(chart_dir)
        store = JsonChartStore(chart_dir)
        registry = {"research": AfkResolver(), "task": AfkResolver(), "grilling": GrillingResolver()}
        graph = build_wayfinder_graph(store, registry, session_id="prototype-run")
        config = {"configurable": {"thread_id": "prototype-thread"}}

        result = graph.invoke({}, config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"[paused] {payload['prompt']}")
            result = graph.invoke(Command(resume="blue"), config)

        print("\n--- final tickets ---")
        for t in store.list_tickets():
            print(f"{t.id}: status={t.status} resolution={t.resolution!r}")

        events = store.read_events()
        print("\n--- metrics ---")
        print("rounds_to_resolve:", rounds_to_resolve(events))
        print("frontier_width_series:", frontier_width_series(events))
        print("fog_shrinkage_rate:", fog_shrinkage_rate(events))
        print("decision_volatility:", decision_volatility(events))
    finally:
        shutil.rmtree(chart_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
