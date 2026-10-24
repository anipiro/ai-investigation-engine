"""Prompt-injection heuristics for untrusted document text.

Retrieved text is always passed to the LLM as quoted data inside <doc> tags, and the system
prompt tells the model never to follow instructions found there. On top of that, chunks matching
these patterns are flagged at ingestion time; flagged chunks are shown with a warning in the UI
and the agent is told explicitly that the chunk contains suspected injected instructions.
"""

from __future__ import annotations

import re

_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) ",
    r"you are now (a|an|in) ",
    r"new instructions?:",
    r"system prompt",
    r"</?\s*(system|instructions?|assistant)\s*>",
    r"(reveal|print|output) (your|the) (system )?(prompt|instructions|api key)",
    r"respond (only )?with ['\"]",
    r"do not (cite|mention) (any )?(sources|documents)",
]
_RX = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def injection_signals(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _RX.finditer(text)})


def is_suspicious(text: str) -> bool:
    return _RX.search(text) is not None


def neutralize_tags(text: str) -> str:
    """Prevent document text from closing our <doc> wrapper or opening fake role tags."""
    return re.sub(r"<\s*/?\s*(doc|system|user|assistant|instructions?)\b[^>]*>", "[tag removed]", text, flags=re.IGNORECASE)
