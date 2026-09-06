# Wayfinder Graph — Design Spec

Status: validated with throwaway prototype code, both a fake-resolver version (`schema.py`, `storage.py`, `state.py`, `resolvers.py`, `graph.py`, `metrics.py`, `prototype_run.py`) and a real, live-tested one against the actual Claude Agent SDK (`agent_tools.py`, `agent_resolvers.py`, `real_run.py`) — `uv run python -m wayfinder_graph.real_run` runs a real research ticket and a real HITL grilling ticket (including a genuine pause-and-resume cycle) to completion against live Claude. Still not a shipped, maintained implementation — see `docs/adr/0001-wayfinder-graph-standalone.md` for the standalone-architecture decision and the tracked map ([Wayfinder Graph](https://github.com/davidbian1/ambiguity/issues/1)) for the full decision history this spec compiles. Vocabulary is defined in `CONTEXT.md`; this document assumes it.

## What this is

A LangGraph reimplementation of the `/wayfinder` skill's chart/ticket/frontier/fog workflow. LangGraph orchestrates the chart-level loop; the Claude Agent SDK powers per-ticket resolution via a Resolver bound to each ticket type. Its own JSON-backed state — no interop with the skill's GitHub-Issues-backed maps ([[ADR 0001]]).

## Architecture

### Chart-level orchestration

A LangGraph `StateGraph` over `WayfinderState`, four nodes in a loop:

```
START -> recompute_frontier -> [frontier empty?] -> END
                              -> select_ticket -> resolve_ticket -> [resolved?]
                                                        ^               |
                                                        |  (no: loop)   | (yes)
                                                        +---------------+
                                                                        v
                                                                 record_resolution -> recompute_frontier
```

- **`recompute_frontier`**: reads the chart's tickets from the store, computes the frontier (open, unclaimed, every blocker `resolved`), emits a `frontier_computed` event (feeds the frontier-width metric), stores the ticket-id list in state.
- **`select_ticket`**: takes the first frontier ticket (chart order), claims it (`claimed_by` written to its ticket file), resets the per-ticket round state.
- **`resolve_ticket`**: calls the ticket type's Resolver once per round (once per node visit). If the resolver's next step is "ask the human," it calls `interrupt()` with the question and loops back to itself; if it's "submit resolution," it records the resolution text in state and proceeds onward. See **HITL composition** below for exactly how a round survives a pause.
- **`record_resolution`**: writes the resolution to the ticket's JSON file, marks it `resolved`, appends a `ticket_resolved` event, clears per-ticket state, loops back to `recompute_frontier`.

Each visit to `resolve_ticket` is one **Round** (see `CONTEXT.md`). A ticket may take many rounds; the graph only advances to `record_resolution` once a round ends in a submitted resolution rather than a question.

### HITL composition

Verified against the Claude Agent SDK's actual session-resume mechanics ([Decisions so far](https://github.com/davidbian1/ambiguity/issues/1) on the map, ticket #2): the Agent SDK's own resume is a thin, local-disk-backed convenience — same-machine cross-process resume is built-in, cross-machine is not. It is **not** a durable primitive on its own, so LangGraph's checkpointer stays the actual source of truth for the pause, exactly as `src/ambiguity`'s existing `ask_human_node` already does it:

```python
def resolve_ticket_node(state):
    ticket = store.load_ticket(state["current_ticket_id"])
    resolver = registry[ticket.type]
    action = resolver.step(ticket, state.get("resolver_state"), state.get("human_answer"))
    if isinstance(action, AskHuman):
        answer = interrupt({"prompt": action.prompt, "kind": action.kind, "options": action.options})
        return {"resolver_state": action.state, "human_answer": answer, "round": state["round"] + 1}
    return {"resolution_text": action.text, "resolver_state": None, "human_answer": None}
```

`action.state` — the Resolver's Agent SDK `session_id` plus the pending `tool_use_id` — is recomputed identically on replay and only becomes part of the checkpoint via the node's return value *after* `interrupt()` resolves, matching this repo's established pattern (`nodes.py::ask_human_node`) rather than inventing a new one. Nothing about the pause depends on the Agent SDK's own session storage surviving the gap.

**The actual mechanism, verified live** (see `agent_tools.py`, `agent_resolvers.py`) — simpler than the research ticket's original guess, which assumed a synthetic `tool_result` message had to be injected on resume:

1. `ask_human` is a custom in-process MCP tool. A `PreToolUse` hook returns `{"permissionDecision": "defer"}` for it, which stops the run; the final `ResultMessage.deferred_tool_use` carries `(id, name, input)` — `id` is the pending `tool_use_id`, `input` is the question payload.
2. To resume: rebuild `ClaudeAgentOptions(resume=session_id)` and call `query(prompt="", options=...)` — **no synthetic message needed**. The SDK automatically replays the deferred `ask_human` call against this round's freshly-built tools.
3. This round's `ask_human` handler is built differently than the first: instead of never running (deferred), it now returns the human's answer directly as the tool's result — the closure carries the answer in.
4. The `PreToolUse` hook is rebuilt too: it lets through only the one `tool_use_id` being answered (so the replayed call reaches the handler above) and still defers any *other* `ask_human` call the resolver might make this round (a genuine follow-up question), so a resolver can take several HITL rounds, not just one.
5. `submit_resolution` is a second, never-deferred MCP tool — the resolver's only way to signal it's done. Models will happily end a turn in plain text instead of calling it unless the system prompt says, explicitly, that a plain-text reply does not count as finishing.

One naming gotcha that cost real debugging time: in-process MCP tools are exposed to the model as `mcp__<server-name>__<tool-name>`, not their bare registered name — `allowed_tools` and hook matching both have to use the qualified name (`mcp_tool_name()` in `agent_tools.py`), or the SDK silently denies the call as unauthorized.

### Resolvers and tool loadout

One `TicketResolver` per ticket type, each independently swappable (see **Swap points**). Decided in [ticket #3](https://github.com/davidbian1/ambiguity/issues/3), and — for Research and Grilling — now implemented for real against the Claude Agent SDK in `agent_resolvers.py` (`resolvers.py`'s fakes remain for cheap, offline testing of the graph shape; `agent_resolvers.py`'s versions are the ones that make live API calls). Prototype and Task aren't implemented yet — same tool-loadout table, no resolver code written against it.

| Resolver | Tools | Notes |
|---|---|---|
| Research | `WebSearch`, `WebFetch`, `Read`, `Glob`, `Grep` (SDK built-ins) | AFK by type — never gets `ask_human`. |
| Prototype | `Write`/`Edit` (scoped to a scratch dir only), `ask_human` | Writes never touch the real repo tree. |
| Grilling | `ask_human`, `Read`, `Grep`, `Glob` | Read-only context tools ground its questions in real state. |
| Task | `ask_human` only when the ticket's `hitl: true` | AFK tasks cannot ask by construction, not by resolver discipline. |

Two shared tools, uniform across every resolver that has them:

- **`ask_human(prompt: str, kind: "question" | "choice" | "approval", options: list[str] | None)`** — the only way a resolver reaches a human. Its call is what `resolve_ticket` intercepts and turns into `interrupt()`.
- **`submit_resolution(text: str)`** — the *only* completion signal. `resolve_ticket` never infers "done" from a quiet turn; it ends the loop exactly when this tool is called.

### Storage (Chart/Ticket JSON schema)

One chart directory per effort:

```
<chart_dir>/
  chart.json          # destination, notes, decisions index, fog, out-of-scope
  tickets/
    <ticket-id>.json  # one file per ticket — parallel sessions on different tickets never contend
  events.jsonl         # append-only event log, one JSON object per line
```

`chart.json`:

```json
{
  "id": "chart-...",
  "destination": "...",
  "notes": "...",
  "decisions": [{"ticket_id": "...", "title": "...", "gist": "..."}],
  "fog": [{"text": "...", "added_at": "..."}],
  "out_of_scope": [{"text": "...", "ticket_id": "..." }],
  "created_at": "..."
}
```

`tickets/<id>.json`:

```json
{
  "id": "...",
  "title": "...",
  "type": "research | prototype | grilling | task",
  "question": "...",
  "status": "open | resolved",
  "hitl": true,
  "blocked_by": ["ticket-id", "..."],
  "claimed_by": null,
  "resolution": null,
  "created_at": "...",
  "resolved_at": null
}
```

`hitl` is only consulted by the Task resolver — every other type's HITL-ness is fixed by its type, per `CONTEXT.md`.

A single advisory lock guards `chart.json` writes (claim-time and resolution-time only, per the map's storage decision); ticket files are written only by the session that holds that ticket's claim, so they need no lock beyond atomic write-and-rename.

### Metrics

Computed by folding over `events.jsonl`, not by inspecting current chart/ticket state (state alone can't answer "how many times was this reopened," only "is it currently open"). Purely observational for now — nothing here feeds back into routing:

- **Rounds-to-resolve**: count of `round` events per ticket before its `ticket_resolved` event.
- **Frontier width over time**: the `width` field on each `frontier_computed` event, in order.
- **Fog-shrinkage rate**: `fog_graduated` count vs. `fog_added` count, grouped by session.
- **Decision volatility**: count of `ticket_reopened` events.
- **Ticket ambiguity score** (stretch, not implemented — see `CONTEXT-MAP.md`'s fog entry): would reuse `AmbiguityAnalyzer` against generated candidate resolutions before a ticket resolves; deferred until it's clear how those candidates get generated.

### Swap points

Four, each a `Protocol`:

1. **LLM backend** — whatever the Claude Agent SDK is configured against; not hard-coded into resolver logic.
2. **Tracker backend** — `ChartStore` protocol (`load_chart`, `save_chart`, `load_ticket`, `save_ticket`, `list_tickets`, `append_event`, `read_events`). `JsonChartStore` is the only implementation for now.
3. **Per-ticket-type resolver** — `TicketResolver` protocol (`step(ticket, resolver_state, human_answer) -> AskHuman | Resolution`), one implementation per type, looked up from a plain `dict[str, TicketResolver]` registry.
4. **Metrics module** — pure functions over an event list; swapping in a different metric set means swapping the module, not touching the graph.

## Out of scope

- A polished, multi-command CLI or packaged product — `prototype_run.py` is a throwaway demo driver, not a shipped entrypoint.
- Interoperating with the `/wayfinder` skill's GitHub-Issues-backed maps ([[ADR 0001]]).
- Folding into `src/ambiguity`'s domain model ([[ADR 0001]]).

## Not yet specified

- The ticket-ambiguity-score metric's candidate-generation mechanism (see Metrics above).
