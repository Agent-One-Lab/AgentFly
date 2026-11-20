"""Optional vLLM reasoning-parser integration: split a model's raw output into reasoning and answer.

Chat templates differ in how they take an assistant turn's reasoning. Some (Qwen3, Qwen3.5) accept the whole
generated text as ``content`` and find the reasoning in it themselves. Others (K2-Horizon) never look inside
``content``: they require a separate reasoning field and write the reasoning block themselves, so a turn stored
as one text either fails to render or renders the block twice. An agent given ``reasoning_parser_name`` stores
each turn as ``reasoning_content`` + ``content``, split by vLLM's parser for that model (the same registry
``vllm serve --reasoning-parser`` uses), as ``tool_parser_name`` does for tool calls.
"""

import logging
from typing import Any, Optional, Tuple

from . import tool_parser

logger = logging.getLogger("agentfly")

# Message keys a backend may already carry the reasoning in (an OpenAI-compatible server with a reasoning parser).
REASONING_KEYS = ("reasoning_content", "reasoning", "think")
REASONING_SENTINEL = "agentfly_reasoning_sentinel"
CONTENT_SENTINEL = "agentfly_content_sentinel"
# (template, parser name, raw-text backend) -> None, or the error to raise again
_CHECKED: dict = {}


def create_reasoning_parser(reasoning_parser_name: str, tokenizer: Any) -> Optional[Any]:
    """A vLLM reasoning parser instance for ``reasoning_parser_name`` (None without a tokenizer).

    Raises:
        ImportError: if vLLM's reasoning parsers are not importable.
    """
    if tokenizer is None:
        return None
    try:
        from vllm.reasoning import ReasoningParserManager
    except ImportError:
        try:
            from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
        except ImportError as e:
            raise ImportError(
                "vLLM reasoning parsers are not available. Install vllm to use reasoning_parser_name."
            ) from e
    return ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(tokenizer)


def _request() -> Any:
    """A minimal request for parsers that read one (most ignore it)."""
    if not tool_parser.ensure_vllm_tool_parser():
        return None
    return tool_parser.ChatCompletionRequest(messages=[{"role": "user", "content": "dummy"}])


def split_reasoning(parser: Any, text: str) -> Tuple[str, str]:
    """``(reasoning, content)`` of one raw model output; either may be empty."""
    extract = getattr(parser, "extract_reasoning", None) or parser.extract_reasoning_content
    reasoning, content = extract(text, _request())
    return reasoning or "", content or ""


def _render(template: str, tokenizer: Any, messages: list, add_generation_prompt: bool) -> str:
    from chat_bricks import Chat

    return Chat(template=template, messages=messages, tokenizer=tokenizer).prompt(
        add_generation_prompt=add_generation_prompt
    )


def check_reasoning_template(
    template: Optional[str], tokenizer: Any, parser: Any, parser_name: Optional[str], raw_text_backend: bool
) -> None:
    """Check once per (template, parser) that the two agree, so a mismatch stops the run at start-up.

    Without a parser (and with a backend that returns raw text): the template must accept an assistant turn that
    has ``content`` only. With a parser: the template must render ``reasoning_content``, and the parser must split
    the text the template renders for a turn (which is what the model generates after the generation prompt) back
    into the same two fields; a template that drops the field, or reformats the text around the reasoning, fails.

    Raises:
        ValueError: on a mismatch.
    """
    if template is None or tokenizer is None:
        return
    key = (template, parser_name or (type(parser).__name__ if parser is not None else None), raw_text_backend)
    if key in _CHECKED:
        if _CHECKED[key] is not None:
            raise ValueError(_CHECKED[key])
        return
    _CHECKED[key] = _mismatch(template, tokenizer, parser, raw_text_backend)
    if _CHECKED[key] is not None:
        raise ValueError(_CHECKED[key])


def _mismatch(template: str, tokenizer: Any, parser: Any, raw_text_backend: bool) -> Optional[str]:
    user = {"role": "user", "content": "probe"}
    if parser is None:
        if not raw_text_backend:
            return None  # a server that parses reasoning returns the field itself
        try:
            _render(template, tokenizer, [user, {"role": "assistant", "content": CONTENT_SENTINEL}], False)
            return None
        except Exception as plain_error:  # noqa: BLE001
            try:
                both = {"role": "assistant", "content": CONTENT_SENTINEL, "reasoning_content": ""}
                _render(template, tokenizer, [user, both], False)
            except Exception:  # noqa: BLE001
                return None  # the template fails for another reason; not a reasoning-field problem
            return (
                f"Template {template!r} does not accept an assistant turn without a reasoning field "
                f"({plain_error}). Set reasoning_parser_name (vLLM's reasoning parser for this model) so turns "
                "are stored as reasoning_content + content."
            )
    assistant = {"role": "assistant", "content": CONTENT_SENTINEL, "reasoning_content": REASONING_SENTINEL}
    generation_prompt = _render(template, tokenizer, [user], True)
    full = _render(template, tokenizer, [user, assistant], False)
    if REASONING_SENTINEL not in full or CONTENT_SENTINEL not in full:
        return (
            f"Template {template!r} does not render reasoning_content, so with a reasoning parser the reasoning "
            "would be dropped from the history. Use the model's own chat template (template=<model id>), or "
            "leave reasoning_parser_name unset."
        )
    common = next(
        (i for i, (a, b) in enumerate(zip(generation_prompt, full)) if a != b), min(len(generation_prompt), len(full))
    )
    turn = full[common:]
    generated = turn[: turn.index(CONTENT_SENTINEL) + len(CONTENT_SENTINEL)]
    got = split_reasoning(parser, generated)
    if got != (REASONING_SENTINEL, CONTENT_SENTINEL):
        return (
            f"The reasoning parser and template {template!r} do not round-trip: the template renders a turn as "
            f"{generated!r} after the generation prompt, and the parser splits that into reasoning {got[0]!r} "
            f"and content {got[1]!r}. The history would differ from the text the model generated."
        )
    return None
