"""Token accounting for the rollout loop.

Self-contained token math that only needs the host's tokenizer/template/processor. Pulled
out of ``ChainRollout`` so the loop isn't interleaved with tokenization arithmetic. Both
functions take a :class:`~agentfly.agents.rollout.host.RolloutHost` (the rollout/agent) and
read its ``tokenizer`` / ``template`` / ``processor`` / ``max_model_len``.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from chat_bricks import Chat

if TYPE_CHECKING:
    from .host import RolloutHost
    from .structures import Node


def observation_token_length(host: "RolloutHost", observation: Any) -> int:
    """Token length of an observation's text. Uses ``host.tokenizer``; 0 if unavailable."""
    tokenizer = getattr(host, "tokenizer", None)
    if tokenizer is None:
        return 0
    text = observation if isinstance(observation, str) else str(observation)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:
        return 0


def estimate_chat_prompt_tokens(
    host: "RolloutHost", current_node: "Node", tools: Optional[List] = None,
    cap: bool = True,
) -> int:
    """Prompt token count via chat_bricks ``Chat`` + ``tokenize`` (same path as training).

    ``cap`` clamps the result to ``max_model_len`` (default, historical behavior).
    Pass ``cap=False`` to get the true, uncapped rendered length — needed to
    detect a prompt that actually *exceeds* the context window (the capped value
    would mask it).
    """
    tokenizer = getattr(host, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Tokenizer is required when using max_model_len to set max_tokens."
        )
    template = getattr(host, "template", None)
    if not template:
        raise ValueError(
            "template is required for the first prompt length estimate."
        )
    messages = current_node.messages.messages
    processor = getattr(host, "processor", None)
    # Backend-aware tool-call rendering. The estimate must render the tool call
    # exactly ONCE, matching what generation/training tokenize:
    #   * parser-based backends (a local tool_parser is set, e.g. qwen3_coder):
    #     the raw generated text is kept in `content`, so the call is ALREADY in
    #     content. Rendering the structured `tool_calls` too would double-count,
    #     so ignore them (ignore_tool_calls=True).
    #   * structured-only backends (no parser, e.g. an OpenAI/GPT API): the call
    #     is ONLY in the structured `tool_calls` field (content has none), so we
    #     MUST render them (ignore_tool_calls=False) or the call vanishes.
    # Hardcoding either value is wrong for the other backend; derive it from
    # whether a local parser produced the (in-content) tool call.
    ignore_tool_calls = getattr(host, "tool_parser", None) is not None
    chat = Chat(
        template,
        messages,
        tokenizer=tokenizer,
        ignore_tool_calls=ignore_tool_calls,
    )
    inputs = chat.tokenize(
        tokenizer,
        add_generation_prompt=True,
        tools=tools,
        processor=processor,
    )
    am = inputs["attention_mask"]
    if isinstance(am, list):
        n = sum(am)
    else:
        n = int(am.sum().item())
    max_model_len = getattr(host, "max_model_len", None)
    if max_model_len is not None and cap:
        n = min(n, max_model_len)
    return n
