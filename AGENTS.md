# ambiguity

Two independent contexts — see `CONTEXT-MAP.md` for how they relate and where each one's vocabulary lives:

- `src/ambiguity/` — the ambiguity clarifier (spec-interpretation scoring). See `src/ambiguity/CONTEXT.md`.
- `src/wayfinder_graph/` — Wayfinder Graph, documented below.

## Wayfinder Graph (`src/wayfinder_graph/`)

A LangGraph reimplementation of the `/wayfinder` skill's chart/ticket/frontier workflow, with the Claude Agent SDK resolving individual tickets. Full design: `src/wayfinder_graph/SPEC.md`. Vocabulary (Chart, Ticket, Resolver, Claim, Frontier, Fog entry, Resolution, Round): `src/wayfinder_graph/CONTEXT.md`. Why it's a standalone package with its own JSON state — not folded into `src/ambiguity`, not sharing state with the skill's GitHub-Issues maps: `docs/adr/0001-wayfinder-graph-standalone.md`.

### File map

- `schema.py` — `Chart`/`Ticket`/`Decision`/`FogEntry`/`OutOfScopeEntry` dataclasses, `to_dict`/`from_dict` matching SPEC.md's JSON schema exactly. Change the schema here first; SPEC.md is the second edit, same commit.
- `storage.py` — `JsonChartStore`: `chart.json` + one `tickets/<id>.json` per ticket + append-only `events.jsonl`. `_lock` (exclusive-create file lock) guards `chart.json` writes only — ticket files are written only by the session holding that ticket's claim, so they need no lock.
- `state.py` — `WayfinderState`, the LangGraph checkpoint shape. Deliberately thin: chart/ticket data lives in the JSON store, not duplicated here. Holds only what one Round needs to survive an `interrupt()`/resume.
- `graph.py` — the orchestration graph: `recompute_frontier -> select_ticket -> resolve_ticket (loops on itself once per Round) -> record_resolution -> recompute_frontier`, ending when the frontier is empty. `compute_frontier()` is the actual blocking logic: open, unclaimed, every blocker `resolved`.
- `resolvers.py` — the `TicketResolver` protocol: `step(ticket, resolver_state, human_answer) -> AskHuman | Resolution`. Also `AfkResolver`/`GrillingResolver`, fakes for testing the graph shape free and offline (`prototype_run.py`).
- `agent_tools.py` + `agent_resolvers.py` — the real, Claude Agent SDK-backed resolvers (`AgentResolver`, and factories `research_resolver()` / `grilling_resolver()` / `prototype_resolver()` / `task_resolver()` per SPEC.md's tool-loadout table). Costs real API calls; needs `ANTHROPIC_API_KEY`. Research and Grilling are live-tested (`real_run.py`); Prototype and Task are written against the same pattern but have never been run against a real ticket — verify them the same way before trusting them.
- `metrics.py` — folds `events.jsonl` into the four implemented metrics. The fifth (ticket ambiguity score) is unimplemented, deliberately — see SPEC.md's "Not yet specified".
- `prototype_run.py` / `real_run.py` — throwaway demo drivers (fake and real resolvers respectively). Not shipped entrypoints — don't harden these into a CLI; that's explicitly out of scope (SPEC.md).

### Gotchas (Claude Agent SDK) — hard-won live, don't re-derive by trial and error

**MCP tool naming.** An in-process MCP tool registered as `"ask_human"` is exposed to the model as `mcp__<server-name>__<tool-name>` (e.g. `mcp__wayfinder__ask_human`) — never its bare name. `allowed_tools` and PreToolUse hook `tool_name` matching both need the qualified name (`agent_tools.mcp_tool_name()`), or the SDK denies the call with a permission error that reads as unrelated to naming.

**The defer/resume cycle, exactly:**
1. A PreToolUse hook returns `{"permissionDecision": "defer"}` for an `ask_human` call, stopping the run. `ResultMessage.deferred_tool_use` carries `(id, name, input)` — `id` is the `tool_use_id` to remember.
2. LangGraph's `interrupt()` pauses the *graph*, not the SDK, with that question; its resume value becomes the human's answer once `Command(resume=...)` arrives. This is why the SDK's own session persistence is never trusted as the pause's source of truth (SPEC.md's "HITL composition") — LangGraph's checkpointer is.
3. To resume the SDK side: `ClaudeAgentOptions(resume=session_id)` + `query(prompt="", ...)`. **No synthetic `tool_result` message is needed** — resume alone replays the deferred call automatically.
4. The replayed call keeps its *original* `tool_use_id`. That's the whole mechanism for telling "the call being answered" apart from "a new question this round": the hook is rebuilt each round to let exactly that one `tool_use_id` through (deferring any other `ask_human` call, e.g. a genuine follow-up question), and that round's `ask_human` handler is rebuilt to return the human's actual answer as the tool's result — instead of the stub error it returns when no answer is set (`build_resolver_tools(..., human_answer=...)`).

**`submit_resolution` is the only completion signal `resolve_ticket_node` recognizes.** Left unsaid, models end a turn in plain text instead of calling it, and the ticket silently resolves as `"[agent stopped without resolving: end_turn]"` (hit live twice before the system prompt said so explicitly — see `_FINISH_RULE` in `agent_resolvers.py`). Any new resolver's system prompt must state this as a rule, not a hope.

**Each `AgentResolver.step()` call is one Round and one fresh `query()`.** A brand-new MCP server and hook are built every round, closing over that round's `resolution_holder`/`human_answer`/`allow_tool_use_id`. Nothing about the SDK session persists as a live Python object across rounds — only the `session_id` string travels, inside `resolver_state`, inside the LangGraph checkpoint.

### Testing

Fake resolvers (`resolvers.py`, run via `prototype_run.py`) exercise the graph shape — schema, storage, frontier, interrupt/resume — for free and offline. Real resolvers (`agent_resolvers.py`, run via `real_run.py`) make live calls; use these to verify actual Agent SDK behavior, not to test the graph shape (that's what the fakes are for).
