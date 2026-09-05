"""
Thin wrappers around the injected llm_client. Each function takes the
*running* conversation (`messages`), appends its own turn, and returns both
the parsed result and the new messages to fold back into graph state —
this is what gives the whole clarify() run one continuous conversation
instead of isolated one-shot calls.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langsmith import traceable

from .parsing import parse_json_array, parse_json_array_of_objects, parse_json_object
from .prompts import (
    SYSTEM_PROMPT,
    render_decision_context,
    render_extract_axes_prompt,
    render_generate_axis_particle_prompt,
    render_generate_prompt,
    render_integrate_prompt,
    render_score_consistency_prompt,
    render_summarize_prompt,
)

_CONFIG_PATH = Path(__file__).parent / "llm_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _LLM_CONFIG: Dict = json.load(f)


# run_type="llm" is what makes LangSmith render this as an LLM call (prompt/
# completion, token usage) rather than a generic function span. Each caller
# passes langsmith_extra={"name": call_name, ...} so the three distinct call
# sites show up named in traces instead of all appearing as "_call".
@traceable(run_type="llm")
def _call(llm_client, messages: List[Dict], call_name: str) -> str:
    cfg = _LLM_CONFIG[call_name]
    response = llm_client.chat.completions.create(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        messages=messages,
    )
    return response.choices[0].message.content


def _traced_call(llm_client, messages: List[Dict], call_name: str) -> str:
    cfg = _LLM_CONFIG[call_name]
    return _call(
        llm_client,
        messages,
        call_name,
        langsmith_extra={"name": call_name, "metadata": {"model": cfg["model"], "temperature": cfg["temperature"]}},
    )


def generate_interpretations(llm_client, messages: List[Dict], spec: str, n_interpretations: int) -> Tuple[List[str], List[Dict]]:
    prompt = render_generate_prompt(spec, n_interpretations)
    content = _traced_call(llm_client, messages + [{"role": "user", "content": prompt}], "generate_interpretations")
    interpretations = parse_json_array(content)
    new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
    return interpretations, new_messages

def summarize_v1v2(llm_client, messages: List[Dict], spec: str, v1_text: str, v2_text: str) -> Tuple[Dict[str, str], List[Dict]]:
    prompt = render_summarize_prompt(spec, v1_text, v2_text)
    content = _traced_call(llm_client, messages + [{"role": "user", "content": prompt}], "summarize_v1v2")
    parsed = parse_json_object(content)
    descriptions = (
        {
            "v1_summary": parsed.get("v1_summary", v1_text),
            "v2_summary": parsed.get("v2_summary", v2_text),
            "ambiguity_topic": parsed.get("ambiguity_topic", "unnamed ambiguity"),
        }
        if parsed is not None
        else {"v1_summary": v1_text, "v2_summary": v2_text, "ambiguity_topic": "unnamed ambiguity"}
    )
    new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
    return descriptions, new_messages

def integrate_choice(
    llm_client,
    messages: List[Dict],
    spec: str,
    ambiguity_topic: str,
    v1_summary: str,
    v2_summary: str,
    choice: str,
    prior_choices: List[Dict[str, str]],
    custom_text: Optional[str] = None,
) -> Tuple[Dict[str, str], List[Dict]]:
    decision_context = render_decision_context(choice, ambiguity_topic, v1_summary, v2_summary, custom_text)
    prompt = render_integrate_prompt(spec, ambiguity_topic, decision_context, prior_choices)
    content = _traced_call(llm_client, messages + [{"role": "user", "content": prompt}], "integrate_choice")
    parsed = parse_json_object(content)
    if parsed is None:
        fallback_sentence = f"{choice}: {custom_text or v1_summary or v2_summary}"
        integrated = {"rewritten_spec": content.strip(), "decision_sentence": fallback_sentence}
    else:
        integrated = {
            "rewritten_spec": parsed.get("rewritten_spec", spec),
            "decision_sentence": parsed.get("decision_sentence", f"{choice}: {ambiguity_topic}"),
        }
    new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
    return integrated, new_messages


# ---------------------------------------------------------------------------
# Particles-only redesign (ticket 004) — active alongside the functions
# above, not replacing them yet. See prompts.yaml's comment on why.
# ---------------------------------------------------------------------------

def extract_axes(llm_client, messages: List[Dict], seed_text: str) -> Tuple[List[Dict[str, str]], List[Dict]]:
    prompt = render_extract_axes_prompt(seed_text)
    content = _traced_call(llm_client, messages + [{"role": "user", "content": prompt}], "extract_axes")
    # Array of OBJECTS ({"axis":..., "class":...}), not strings — using
    # parse_json_array here mangled each object into its Python repr string
    # (found via a real test run: axis names came back as
    # "{'axis': 'platform', 'class': 'decision'}" instead of "platform").
    axes = parse_json_array_of_objects(content)
    normalized = [
        {"axis": a.get("axis", "unnamed"), "class": a.get("class", "decision")} for a in axes
    ]
    new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
    return normalized, new_messages


def generate_axis_particle(llm_client, seed_text: str, axis_name: str) -> Tuple[str, List[Dict]]:
    prompt = render_generate_axis_particle_prompt(seed_text, axis_name)
    # Deliberately NOT threaded through the running `messages` conversation:
    # each particle-generation call should reason fresh from seed_text alone,
    # not see other particles' values or prior rounds — that's exactly the
    # cross-contamination the "force everything else unconstrained" design
    # is trying to avoid.
    content = _traced_call(llm_client, [{"role": "user", "content": prompt}], "generate_axis_particle")
    parsed = parse_json_object(content)
    value = parsed.get("value") if parsed else content.strip()
    new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
    return value, new_messages


def score_consistency(
    llm_client, axis_name: str, particle_value: Optional[str], human_answer: str
) -> Tuple[float, List[Dict]]:
    """FALLBACK PATH ONLY — callers should exact-match first (see
    particles.py's reweight_particles) and only call this for hedged/
    free-text answers that don't cleanly match an existing value."""
    prompt = render_score_consistency_prompt(axis_name, particle_value, human_answer)
    content = _traced_call(llm_client, [{"role": "user", "content": prompt}], "score_consistency")
    parsed = parse_json_object(content)
    try:
        weight = float(parsed["weight"]) if parsed else 0.5
    except (KeyError, TypeError, ValueError):
        weight = 0.5  # neutral fallback on an unparseable response, not a crash
    weight = max(0.0, min(1.0, weight))
    new_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
    return weight, new_messages
