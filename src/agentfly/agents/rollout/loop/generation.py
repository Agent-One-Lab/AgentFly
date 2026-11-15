"""Shared response generation and backend-response normalization for rollouts.

Chain and step rollout strategies share these helpers for backend response-dict
handling (``response_dict`` / ``choices`` / ``total_lengths``) and per-call
generation-config budgeting.

The orchestrating functions (:func:`generate_response`, :func:`prepare_generation_config`)
take the ``agent`` (a :class:`~agentfly.agents.rollout.agent.RolloutAgent`) because they call
agent hooks (``generate_async`` / ``parse`` / ``skills_payload``).
Token estimation lives in :mod:`.tokens`. The pure helpers take only their data.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .tokens import estimate_chat_prompt_tokens

if TYPE_CHECKING:
    from ....core import Context
    from ..agent import RolloutAgent
    from ..structures import Step


# Fields agentfly constructs itself; everything else the server returned on the
# assistant message is carried through as-is.
OWNED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "loss", "status"})


FORCED = "forced"  # message key: this assistant turn was replayed, not sampled


def normalize_generate_response(responses: Any) -> Dict[str, Any]:
    """Unwrap a backend response to a single dict. ``ClientBackend`` returns a list of dicts."""
    if isinstance(responses, list) and len(responses) == 1:
        return responses[0]
    if isinstance(responses, dict):
        return responses
    return {}


def overlay_raw_message_fields(messages, responses) -> None:
    """Preserve EVERY extra field the server returned on each assistant message
    (e.g. ``reasoning`` / ``reasoning_content``) by copying it onto the message agentfly
    built, so multi-turn requests echo it back.

    General by design — we keep every non-null key from the raw response message except
    the ones agentfly owns (content/tool_calls/role/...), so a new provider field never
    needs to be added to an allowlist. Some reasoning servers (e.g. k2moe) even REJECT
    assistant history that lacks a thinking field; preserving the real
    ``reasoning_content`` satisfies them.
    """
    rd = responses.get("response_dict") if isinstance(responses, dict) else None
    choices = (rd or {}).get("choices") or []
    for i, msg in enumerate(messages or []):
        if i >= len(choices) or not isinstance(msg, dict):
            continue
        raw = (choices[i] or {}).get("message") or {}
        for k, v in raw.items():
            if k in OWNED_MSG_KEYS or v is None:
                continue
            msg.setdefault(k, v)


def attach_token_ids(messages, responses) -> None:
    """Carry the backend's generated token ids onto each assistant message as
    ``token_ids`` (one list per response, same order as ``response_texts``).

    Training tokenization (chat-bricks) splices these ids verbatim for the
    assistant span instead of re-tokenizing the decoded text, which removes
    retokenization "token drift" (the sampled sequence and the re-encoded text
    can tokenize differently, e.g. ``<think>``) and keeps native tool-call turns
    exactly as sampled. The ids are the full generation; when the parser kept only a
    prefix as ``content``, :func:`trim_token_ids_to_content` then cuts them to match.

    Only backends that expose ids set ``response_ids`` (the verl training
    backend); when absent nothing is attached and tokenization re-encodes text
    as before.
    """
    response_ids = responses.get("response_ids") if isinstance(responses, dict) else None
    if not response_ids:
        return
    for msg, ids in zip(messages or [], response_ids):
        if isinstance(msg, dict) and ids is not None:
            msg["token_ids"] = list(ids)


def _content_text(msg: Dict[str, Any]) -> Optional[str]:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
        return content[0].get("text")
    return None


CONTENT_IS_PREFIX = "content_is_prefix"


def trim_token_ids_to_content(messages, response_texts, tokenizer) -> int:
    """Cut a message's sampled ``token_ids`` when its parser kept only a prefix of the response as
    ``content`` on purpose and marked it with ``CONTENT_IS_PREFIX`` (ActionAgent's
    ``truncate_history_at_action``), so the next prompt and the training row hold exactly that prefix,
    not the full generation. The marker is removed here.

    The ids are cut at the shortest prefix whose decoded text covers the kept text, and the content
    becomes that decoded text, so text and ids agree (a token at the boundary may add a few
    characters). Unmarked messages are left alone: other parsers' content can be a prefix of the
    response too (native tool calls move the call text out of ``content``) while their ids must stay
    whole. Returns the number of ids removed, over all messages."""
    removed = 0
    for msg, full in zip(messages or [], response_texts or []):
        if not isinstance(msg, dict) or not msg.pop(CONTENT_IS_PREFIX, False):
            continue
        ids = msg.get("token_ids")
        kept = _content_text(msg)
        if tokenizer is None or not ids or not isinstance(full, str) or kept is None:
            continue
        if len(kept) >= len(full) or not full.startswith(kept):
            continue

        def decoded(k):
            return tokenizer.decode(ids[:k], skip_special_tokens=True)

        lo, hi = 1, len(ids)  # smallest k whose text is at least as long as the kept text
        while lo < hi:
            mid = (lo + hi) // 2
            if len(decoded(mid)) >= len(kept):
                hi = mid
            else:
                lo = mid + 1
        k = lo
        while k < len(ids) and not decoded(k).startswith(kept):
            k += 1  # decoded length is not strictly monotone around multi-byte characters
        if k >= len(ids) or not decoded(k).startswith(kept):
            continue
        text = decoded(k)
        removed += len(ids) - k
        msg["token_ids"] = list(ids[:k])
        msg["content"] = text if isinstance(msg.get("content"), str) else [{"type": "text", "text": text}]
    return removed


def extract_total_length(responses: dict) -> Optional[int]:
    """Extract total token length (after chat template) from a ``generate_async`` response dict."""
    if "total_lengths" not in responses:
        return None
    tl = responses["total_lengths"]
    if tl is None:
        return None
    if hasattr(tl, "tolist"):  # e.g. torch tensor
        tl = tl.tolist()
    if isinstance(tl, list):
        return int(tl[0]) if tl else None
    return int(tl)


def prepare_generation_config(
    agent: "RolloutAgent",
    generation_config: Optional[Dict[str, Any]],
    current_step: "Step",
    tools: Optional[List] = None,
) -> Dict[str, Any]:
    """Merge max_tokens / max_new_tokens; if ``max_model_len`` is set, cap completion budget.

    Only the first generation in a chain (``total_token_length == 0`` on the step) uses
    chat_bricks ``Chat.tokenize``; later turns use ``max_model_len - total_token_length``.
    """
    config = dict(generation_config) if generation_config else {}
    max_model_len = getattr(agent, "max_model_len", None)

    if config.get("max_tokens") is not None:
        config.pop("max_new_tokens", None)
    else:
        max_new = config.pop("max_new_tokens", None)
        if max_new is not None:
            config["max_tokens"] = max_new

    if max_model_len is not None:
        if current_step.total_token_length == 0:
            prompt_tok = estimate_chat_prompt_tokens(agent, current_step, tools=tools)
        else:
            prompt_tok = current_step.total_token_length
        budget = max_model_len - prompt_tok
        budget = max(1, budget)
        explicit = config.get("max_tokens")
        if explicit is None:
            config["max_tokens"] = budget
        else:
            try:
                explicit_int = int(explicit)
            except (TypeError, ValueError):
                explicit_int = budget
            config["max_tokens"] = max(1, min(explicit_int, budget))

    return config


async def generate_response(
    agent: "RolloutAgent",
    chain,
    current_step,
    tools,
    depth,
    episode_id,
    generation_config,
    context: "Context",
    use_token_ids: bool = True,
) -> Tuple[Any, Optional[int]]:
    """Generate one assistant message. Returns ``(message, total_token_length)``.

    ``use_token_ids``: attach the backend's sampled ids to the message (see
    :func:`attach_token_ids`). False leaves them off, so every later render re-encodes
    the message from text.
    """
    forced = getattr(chain, "forced_responses", None)
    forced_turn = bool(forced)
    if forced_turn:
        # A replayed turn (a branch re-enacts the turn it branches from): no sampling, no token ids.
        raw = {"response_texts": [forced.pop(0)]}
        use_token_ids = False
    else:
        effective_config = prepare_generation_config(agent, generation_config, current_step, tools=tools)
        skills_payload = agent.skills_payload()
        extra_kwargs = {"skills": skills_payload} if skills_payload else {}
        extra_kwargs["tool_call_source"] = getattr(agent, "default_tool_call_source", "parser")
        raw = await agent.generate_async(
            [current_step.messages.messages],
            return_dict=True,
            tools=tools,
            **extra_kwargs,
            **effective_config,
        )
    responses = normalize_generate_response(raw)
    response_texts = responses.get("response_texts")
    # The prompt each response was sampled from goes to the parser directly, not through
    # ``context.metadata``, which every conversation of the episode shares.
    new_msg = agent.parse(
        response_texts,
        tool_calls=responses.get("tool_calls"),
        context=context,
        trajectory_segments=[current_step.messages.messages],
    )
    overlay_raw_message_fields(new_msg, responses)
    removed = 0
    if use_token_ids:
        attach_token_ids(new_msg, responses)
        removed = trim_token_ids_to_content(new_msg, response_texts, getattr(agent, "tokenizer", None))
    else:
        # Also drop ids a server may have returned on the raw message (overlaid above).
        for msg in new_msg or []:
            if isinstance(msg, dict):
                msg.pop("token_ids", None)
                msg.pop(CONTENT_IS_PREFIX, None)
    total_length = extract_total_length(responses)
    if total_length is not None and removed:
        total_length -= removed
    if forced_turn and isinstance(new_msg[0], dict):
        new_msg[0][FORCED] = True  # a replayed turn: conversion treats it as prefix, not as a policy target
    return (new_msg[0], total_length)
