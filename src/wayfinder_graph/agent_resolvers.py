"""Real, Claude Agent SDK-backed resolvers — the live implementation of the
per-resolver tool loadout table in SPEC.md, replacing resolvers.py's fakes.
Bridges the SDK's async `query()` into `TicketResolver.step()`'s sync
interface with `asyncio.run` — each `step()` call is exactly one Round,
matching one `query()` invocation (fresh, or resumed via `resume=`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from .agent_tools import ASK_HUMAN_TOOL_NAME, SUBMIT_RESOLUTION_TOOL_NAME, ask_human_hook_matchers, build_resolver_tools, mcp_tool_name
from .resolvers import AskHuman, Resolution
from .schema import Ticket


async def _run_round(ticket: Ticket, system_prompt: str, built_in_tools: List[str], has_ask_human: bool, cwd: Optional[Path], resolver_state: Optional[Dict], human_answer: Optional[Any]):
    resolution_holder: Dict[str, Optional[str]] = {"text": None}
    answering_tool_use_id = resolver_state["tool_use_id"] if resolver_state else None
    tools_server = build_resolver_tools(resolution_holder, human_answer=human_answer if answering_tool_use_id else None)
    allowed = built_in_tools + [mcp_tool_name(SUBMIT_RESOLUTION_TOOL_NAME)] + ([mcp_tool_name(ASK_HUMAN_TOOL_NAME)] if has_ask_human else [])

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=built_in_tools,
        mcp_servers={"wayfinder": tools_server},
        allowed_tools=allowed,
        hooks={"PreToolUse": ask_human_hook_matchers(answering_tool_use_id)} if has_ask_human else None,
        cwd=str(cwd) if cwd else None,
        resume=resolver_state["session_id"] if resolver_state else None,
    )

    # On resume, `resume=` alone replays the deferred ask_human call — the
    # SDK re-fires it against this round's freshly-built tools/hooks, no
    # synthetic tool_result message needs constructing (verified live).
    prompt: Any = "" if resolver_state else f"Ticket: {ticket.title}\n\nQuestion: {ticket.question}"

    result_message: Optional[ResultMessage] = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result_message = message

    if resolution_holder["text"] is not None:
        return Resolution(text=resolution_holder["text"])

    if result_message is not None and result_message.deferred_tool_use is not None:
        deferred = result_message.deferred_tool_use
        return AskHuman(
            prompt=deferred.input.get("prompt", ticket.question),
            kind=deferred.input.get("kind", "question"),
            options=deferred.input.get("options"),
            state={"session_id": result_message.session_id, "tool_use_id": deferred.id},
        )

    # Neither a resolution nor a deferred question — the resolver ran out of
    # turns or errored without calling either tool. Surface it as a
    # resolution so the ticket doesn't hang forever silently.
    reason = result_message.stop_reason if result_message else "unknown"
    return Resolution(text=f"[agent stopped without resolving: {reason}]")


class AgentResolver:
    def __init__(self, system_prompt: str, built_in_tools: List[str], has_ask_human: bool, cwd: Optional[Path] = None):
        self.system_prompt = system_prompt
        self.built_in_tools = built_in_tools
        self.has_ask_human = has_ask_human
        self.cwd = cwd

    def step(self, ticket: Ticket, resolver_state: Optional[Dict], human_answer: Optional[Any]):
        return asyncio.run(_run_round(ticket, self.system_prompt, self.built_in_tools, self.has_ask_human, self.cwd, resolver_state, human_answer))


_FINISH_RULE = (
    "IMPORTANT: the only way to finish is to call the submit_resolution tool with your final answer as `text`. "
    "A plain-text reply with no tool call does NOT end the ticket — it is treated as a failure. "
    "Never end your turn without having called submit_resolution."
)


def research_resolver() -> AgentResolver:
    return AgentResolver(
        system_prompt="You are a research resolver. Investigate the ticket's question using your read-only tools. "
        f"{_FINISH_RULE} You have no way to ask a human — do not try.",
        built_in_tools=["WebSearch", "WebFetch", "Read", "Glob", "Grep"],
        has_ask_human=False,
    )


def grilling_resolver() -> AgentResolver:
    return AgentResolver(
        system_prompt="You are a grilling resolver. Ground your question in real repo state using your read-only "
        "tools if useful, then ask the human exactly one sharp question via the ask_human tool. Once ask_human "
        f"returns their answer, immediately call submit_resolution with it — do not ask a second question. {_FINISH_RULE}",
        built_in_tools=["Read", "Grep", "Glob"],
        has_ask_human=True,
    )


def prototype_resolver(scratch_dir: Path) -> AgentResolver:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return AgentResolver(
        system_prompt="You are a prototype resolver. Build a small throwaway artifact answering the ticket's "
        "question, using only Write/Edit inside your working directory (never elsewhere), show it to the human "
        f"via ask_human, then call submit_resolution with their reaction. {_FINISH_RULE}",
        built_in_tools=["Write", "Edit"],
        has_ask_human=True,
        cwd=scratch_dir,
    )


def task_resolver(hitl: bool) -> AgentResolver:
    ask_clause = "asking the human via ask_human only if you genuinely need their input." if hitl else "You have no way to ask a human — do not try."
    return AgentResolver(
        system_prompt=f"You are a task resolver. Do the work the ticket describes, {ask_clause} {_FINISH_RULE}",
        built_in_tools=["Read", "Write", "Bash"],
        has_ask_human=hitl,
    )
