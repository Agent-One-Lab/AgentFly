"""Pure message-text and action-format helpers shared by rollout and conversion."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def message_text(msg: Dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
    return str(content) if content is not None else ""


def generation_opens_think(agent) -> bool:
    """Whether the agent's chat template ends the generation prompt inside an open
    ``<think>`` block (native thinking, e.g. Qwen3.5-4B). The model's output then carries
    only the closing ``</think>``, so the format check must count the prompt's opening tag.

    Rendered once with the same chat-bricks template and tokenizer the backend samples
    with, and cached on the agent. False when the prompt cannot be rendered (logged).
    """
    cached = getattr(agent, "_generation_opens_think", None)
    if cached is not None:
        return cached
    try:
        from chat_bricks import Chat

        chat = Chat(
            template=agent.template,
            messages=[{"role": "user", "content": "x"}],
            tokenizer=agent.tokenizer,
        )
        opened = chat.prompt(add_generation_prompt=True).rstrip().endswith("<think>")
    except Exception as e:  # noqa: BLE001 — a diagnostic render must not break conversion
        logger.warning("Could not render the generation prompt to detect native thinking: %s", e)
        opened = False
    agent._generation_opens_think = opened
    return opened


def action_format_valid(text: str, think_opened: bool = False) -> float:
    """verl-agent's ``alfworld_projection`` validity — a pure FORMAT check on the model's
    raw output, NOT an env-semantic one: the response must contain a ``<action>...</action>``
    block AND a ``<think>...</think>`` block AND no Chinese characters. Returns 1.0/0.0.

    verl-agent finds ``<action>`` on the lowercased text and ``<think>`` on the original;
    a no-tool-call turn (no ``<action>`` body) is naturally 0.0 here too. Both rollout
    strategies use this so the invalid-action penalty means the same thing in each.

    ``think_opened``: the generation prompt already opened the ``<think>`` block (see
    :func:`generation_opens_think`), so the output's own ``</think>`` closes it.
    """
    import re

    if not text:
        return 0.0
    if think_opened:
        text = "<think>" + text
    low = text.lower()
    has_action = "<action>" in low and "</action>" in low
    has_think = "<think>" in text and "</think>" in text
    no_chinese = re.search("[一-鿿]", text) is None
    return 1.0 if (has_action and has_think and no_chinese) else 0.0
