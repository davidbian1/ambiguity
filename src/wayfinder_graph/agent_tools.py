"""Real ask_human / submit_resolution tools, backed by the Claude Agent
SDK's in-process MCP server + PreToolUse hook mechanism (see SPEC.md's
"HITL composition" and "Resolvers and tool loadout" — this is the real
implementation of what resolvers.py's fakes stand in for).

Verified against the installed claude-agent-sdk (0.2.152) source AND a live
run, not just its docs:

- A PreToolUse hook returning ``{"permissionDecision": "defer"}`` stops the
  run; the final ResultMessage carries the call as `deferred_tool_use`
  (id/name/input) — see `claude_agent_sdk.types.DeferredToolUse`'s
  docstring.
- Resuming with `ClaudeAgentOptions(resume=session_id)` automatically
  **replays the deferred tool call** — no synthetic `tool_result` message
  needs to be injected into the prompt. The replayed call keeps the same
  `tool_use_id` it had when deferred, which is exactly how a round tells
  "the call I'm answering" apart from "a new question the resolver is
  asking this round": defer everything EXCEPT that one `tool_use_id`, and
  the ask_human handler for that round returns the human's actual answer
  as the tool's result (rather than never running at all, as it does on
  the round that first defers).
- MCP tools are exposed to the model as `mcp__<server>__<tool>`, not their
  bare registered name — `allowed_tools` and hook matching must use that.
- `submit_resolution` is never deferred; its handler always runs
  immediately and records the text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from claude_agent_sdk import HookMatcher, McpSdkServerConfig, create_sdk_mcp_server, tool

MCP_SERVER_NAME = "wayfinder"
ASK_HUMAN_TOOL_NAME = "ask_human"
SUBMIT_RESOLUTION_TOOL_NAME = "submit_resolution"


def mcp_tool_name(bare_name: str) -> str:
    return f"mcp__{MCP_SERVER_NAME}__{bare_name}"


def build_resolver_tools(resolution_holder: Dict[str, Optional[str]], human_answer: Optional[str] = None) -> McpSdkServerConfig:
    """One fresh MCP server per resolver `step()` call.

    `resolution_holder` is a plain dict the caller inspects afterward —
    `submit_resolution` writes `resolution_holder["text"]` when called.

    `human_answer`: None on the round that first asks (ask_human's call
    will be deferred before this handler ever runs). Set to the human's
    answer on the round resuming after a pause — the PreToolUse hook lets
    exactly that one replayed call through, and this handler returns the
    answer as its result.
    """

    @tool(
        ASK_HUMAN_TOOL_NAME,
        "Ask the human a question and wait for their answer. `kind` must be exactly one of "
        "'question' (open-ended), 'choice' (pick from `options`), or 'approval' (yes/no).",
        {"prompt": str, "kind": str, "options": list},
    )
    async def ask_human(args: Dict[str, Any]) -> Dict[str, Any]:
        if human_answer is not None:
            return {"content": [{"type": "text", "text": human_answer}]}
        # Reached only if the PreToolUse hook failed to defer this call.
        return {"content": [{"type": "text", "text": "error: ask_human should have been deferred, not executed"}]}

    @tool(SUBMIT_RESOLUTION_TOOL_NAME, "Submit the final resolution text for this ticket and end the session.", {"text": str})
    async def submit_resolution(args: Dict[str, Any]) -> Dict[str, Any]:
        resolution_holder["text"] = args["text"]
        return {"content": [{"type": "text", "text": "resolution recorded"}]}

    return create_sdk_mcp_server(name=MCP_SERVER_NAME, tools=[ask_human, submit_resolution])


def ask_human_hook_matchers(allow_tool_use_id: Optional[str]) -> List[HookMatcher]:
    """Defer every ask_human call except `allow_tool_use_id` (the one this
    round is resuming and answering, if any). Pass allow_tool_use_id=None
    on a ticket's first round, when there's nothing to let through yet.
    """

    async def _decide(input_data, tool_use_id, context):
        if input_data.get("tool_name") != mcp_tool_name(ASK_HUMAN_TOOL_NAME):
            return {}
        if tool_use_id == allow_tool_use_id:
            return {}  # let it through to the handler, which has the real answer
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}

    return [HookMatcher(matcher=None, hooks=[_decide])]
