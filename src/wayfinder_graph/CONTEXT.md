# Wayfinder Graph

A LangGraph reimplementation of the `/wayfinder` skill's decision-charting workflow: resolves open decisions in a body of work by working a chart of tickets, one at a time, until nothing is left to decide. Its own JSON-backed state, separate from the skill's GitHub-Issues-backed maps — the two never read or write each other's state.

## Language

**Chart**:
The JSON document that is this system's equivalent of the skill's "map": a destination, notes, tickets, decisions, fog, and out-of-scope entries for one effort.
_Avoid_: Map — reserved for the `/wayfinder` skill's own GitHub-Issue artifact; using "map" here invites confusing the two systems' state.

**Destination**:
What reaching the end of a chart looks like — the spec, decision, or change the effort is finding its way to. Same meaning as the skill's Destination.

**Ticket**:
One decision, investigation, or piece of enabling work tracked as a JSON record on a chart: a question, a type, a status, blocking edges, and (once resolved) a resolution. Same concept as the skill's ticket; different storage.

**Ticket type**:
One of `research`, `prototype`, `grilling`, `task` — same four types and HITL/AFK split as the skill.

**Resolver**:
The Claude Agent SDK session and toolset bound to a ticket type — the thing that actually resolves a ticket. One resolver per ticket type (Research Resolver, Prototype Resolver, Grilling Resolver, Task Resolver), each independently swappable.
_Avoid_: "Agent" alone — always say resolver when talking about the per-ticket-type strategy, to keep it distinct from the Claude Agent SDK session it wraps.

**Claim**:
Marking a ticket as being worked, recorded as a `claimed_by` field (a resolver-session id) on the ticket record rather than a tracker assignee. An open ticket with `claimed_by: null` is unclaimed.

**Frontier**:
The set of tickets that are open, unclaimed, and unblocked — every ticket blocking them has status `resolved`. Computed from the chart's blocking edges, not stored.

**Fog entry**:
An item in the chart's fog list: a suspected future question too coarse to phrase as a ticket yet. Graduates into one or more tickets once a resolution sharpens it enough to state precisely.
_Avoid_: "Not yet specified" — that's the skill's section heading for the same concept; fog entry is this system's record for it.

**Resolution**:
The recorded answer to a ticket's question, written by a resolver when a ticket closes. Gisted into the chart's decision index, held in full on the ticket.

**Round**:
One resolver invocation against one ticket. A ticket may take several rounds; the chart's rounds-to-resolve metric counts them per ticket.
