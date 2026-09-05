"""Conditional-edge routers for the clarifier graph."""

from langgraph.graph import END


def make_route_after_human():
    def route_after_human(state) -> str:
        return END if state["choice"] == "stop" else "integrate"

    return route_after_human


def make_route_after_integrate(max_rounds: int):
    def route_after_integrate(state) -> str:
        return END if state["round"] > max_rounds else "generate"

    return route_after_integrate


# Particles-only redesign (ticket 004) — active alongside the two routers
# above, not replacing them yet.

def make_route_after_ask():
    def route_after_ask(state) -> str:
        return END if state["choice"] == "stop" else "reweight"

    return route_after_ask


def make_route_after_reweight(max_rounds: int):
    def route_after_reweight(state) -> str:
        return END if state["round"] > max_rounds else "pick_axis"

    return route_after_reweight
