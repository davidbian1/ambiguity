"""LLM-response parsing: markdown-fence stripping with graceful fallbacks."""

import json
from typing import Any, Dict, List, Optional

# Note: no text should contain the value "json" 
# outside of a code fence, or else the parsing will break.

def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "")
    return text.strip()

def parse_json_array(text: str) -> List[str]:
    """Parse a JSON array of strings; fall back to non-empty lines."""
    cleaned = strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in cleaned.splitlines() if line.strip()]

def parse_json_array_of_objects(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array of objects (not strings — see parse_json_array
    for that). Falls back to an empty list, not a str()-coerced mangling
    of each object, since dict items have no sane single-line text form."""
    cleaned = strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    return []


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None
