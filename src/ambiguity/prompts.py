"""Prompt rendering — every prompt template lives in prompts.yaml, one file.

YAML over JSON here specifically for its block scalars (`|`): a prompt body
is real multi-line text with no \\n escaping. This also matches the closest
thing to an ecosystem convention for externalized prompts — LangChain's own
prompt serialization (`langchain_core.prompts.load_prompt`) supports YAML
for the same reason.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

_TEMPLATES_PATH = Path(__file__).parent / "prompts.yaml"
with open(_TEMPLATES_PATH, "r", encoding="utf-8") as f:
    _TEMPLATES: Dict = yaml.safe_load(f)

SYSTEM_PROMPT: str = _TEMPLATES["system"].strip()

_GENERATE_TEMPLATE = _TEMPLATES["generate_interpretations"].strip()
_SUMMARIZE_TEMPLATE = _TEMPLATES["summarize_v1v2"].strip()
_INTEGRATE_TEMPLATE = _TEMPLATES["integrate_choice"].strip()
_DECISION_CONTEXT_TEMPLATES = {
    choice: text.strip() for choice, text in _TEMPLATES["decision_context"].items()
}

_EXTRACT_AXES_TEMPLATE = _TEMPLATES["extract_axes"].strip()
_GENERATE_AXIS_PARTICLE_TEMPLATE = _TEMPLATES["generate_axis_particle"].strip()
_SCORE_CONSISTENCY_TEMPLATE = _TEMPLATES["score_consistency"].strip()


def render_generate_prompt(spec: str, n_interpretations: int) -> str:
    return _GENERATE_TEMPLATE.format(spec=spec, n_interpretations=n_interpretations)


def render_summarize_prompt(spec: str, v1_text: str, v2_text: str) -> str:
    return _SUMMARIZE_TEMPLATE.format(spec=spec, v1_text=v1_text, v2_text=v2_text)


def render_decision_context(
    choice: str,
    ambiguity_topic: str,
    v1_summary: str,
    v2_summary: str,
    custom_text: Optional[str] = None,
) -> str:
    try:
        template = _DECISION_CONTEXT_TEMPLATES[choice]
    except KeyError:
        raise ValueError(f"unknown choice: {choice!r}") from None
    return template.format(
        ambiguity_topic=ambiguity_topic,
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        custom_text=custom_text,
    )


def render_integrate_prompt(
    spec: str,
    ambiguity_topic: str,
    decision_context: str,
    prior_choices: List[Dict[str, str]],
) -> str:
    prior_block = "\n".join(
        f'- {p["topic"]}: {p["decision_sentence"]}' for p in prior_choices
    ) or "(none yet)"
    return _INTEGRATE_TEMPLATE.format(
        spec=spec,
        ambiguity_topic=ambiguity_topic,
        decision_context=decision_context,
        prior_block=prior_block,
    )


def render_extract_axes_prompt(seed_text: str) -> str:
    return _EXTRACT_AXES_TEMPLATE.format(seed_text=seed_text)


def render_generate_axis_particle_prompt(seed_text: str, axis_name: str) -> str:
    return _GENERATE_AXIS_PARTICLE_TEMPLATE.format(seed_text=seed_text, axis_name=axis_name)


def render_score_consistency_prompt(axis_name: str, particle_value: Optional[str], human_answer: str) -> str:
    return _SCORE_CONSISTENCY_TEMPLATE.format(
        axis_name=axis_name, particle_value=particle_value, human_answer=human_answer
    )
