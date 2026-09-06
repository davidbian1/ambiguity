"""Live smoke test — NOT a shipped CLI (same "throwaway demo driver" status
as prototype_run.py, just backed by real Claude Agent SDK calls instead of
fakes). Proves the resolve_ticket <-> Agent SDK wiring in agent_resolvers.py
actually works against live Claude, including one real defer/resume cycle
for a HITL ticket. Costs real API usage — kept to two small tickets and a
low max_turns budget on purpose.

Run with: uv run python -m wayfinder_graph.real_run
"""

import shutil
import tempfile
from pathlib import Path

from langgraph.types import Command

from .agent_resolvers import grilling_resolver, research_resolver
from .graph import build_wayfinder_graph
from .schema import Chart, Ticket
from .storage import JsonChartStore


def seed_chart(chart_dir: Path) -> None:
    store = JsonChartStore(chart_dir)
    store.save_chart(Chart(id="chart-real-1", destination="Prove the real Agent SDK resolvers work end to end."))
    store.save_ticket(
        Ticket(id="t-research", title="Quick fact lookup", type="research", question="In one short sentence, what year was Python first released?")
    )
    store.save_ticket(
        Ticket(
            id="t-grilling",
            title="Ask a real question",
            type="grilling",
            question="Ask me, the human, a single short yes/no question about whether pineapple belongs on pizza — nothing else.",
            blocked_by=["t-research"],
        )
    )


def main() -> None:
    chart_dir = Path(tempfile.mkdtemp(prefix="wayfinder_graph_real_"))
    try:
        seed_chart(chart_dir)
        store = JsonChartStore(chart_dir)
        registry = {"research": research_resolver(), "grilling": grilling_resolver()}
        graph = build_wayfinder_graph(store, registry, session_id="real-run")
        config = {"configurable": {"thread_id": "real-thread"}}

        result = graph.invoke({}, config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"[paused] agent asked: {payload['prompt']!r} (kind={payload['kind']})")
            result = graph.invoke(Command(resume="yes"), config)

        print("\n--- final tickets ---")
        for t in store.list_tickets():
            print(f"{t.id}: status={t.status}\n  resolution={t.resolution!r}")
    finally:
        shutil.rmtree(chart_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
