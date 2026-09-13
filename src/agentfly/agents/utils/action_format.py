"""Pure message-text and action-format helpers shared by rollout and conversion."""

from typing import Dict


def message_text(msg: Dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
    return str(content) if content is not None else ""

def action_format_valid(text: str) -> float:
    """verl-agent's ``alfworld_projection`` validity — a pure FORMAT check on the model's
    raw output, NOT an env-semantic one: the response must contain a ``<action>...</action>``
    block AND a ``<think>...</think>`` block AND no Chinese characters. Returns 1.0/0.0.

    verl-agent finds ``<action>`` on the lowercased text and ``<think>`` on the original;
    a no-tool-call turn (no ``<action>`` body) is naturally 0.0 here too. Both rollout
    strategies use this so the invalid-action penalty means the same thing in each.
    """
    import re

    if not text:
        return 0.0
    low = text.lower()
    has_action = "<action>" in low and "</action>" in low
    has_think = "<think>" in text and "</think>" in text
    no_chinese = re.search("[一-鿿]", text) is None
    return 1.0 if (has_action and has_think and no_chinese) else 0.0
