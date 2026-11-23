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

from ...utils.reasoning_parser import REASONING_KEYS, split_reasoning
from .routing import attach_rollout_signals, drop_rollout_signals, sync_rollout_signals
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
            if k in OWNED_MSG_KEYS:
                continue
            if v is None:
                # A turn without reasoning comes back with a null reasoning field. A server that rejects assistant
                # history lacking a thinking field (K2) needs it echoed as an empty one.
                if k in REASONING_KEYS:
                    msg.setdefault(k, "")
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


REASONING_UNCLOSED = "reasoning_unclosed"  # message key: the turn ended (token limit) before its reasoning closed


def split_reasoning_responses(agent, response_texts, responses) -> Tuple[Any, Optional[List[Optional[Dict[str, Any]]]]]:
    """With a reasoning parser on the agent, split each raw response into reasoning and answer.

    Returns ``(texts to parse, splits)``. ``splits`` is None without a parser; otherwise one entry per response:
    None when the response is left whole (the backend already returned a reasoning field for it), or
    ``{"reasoning", "head", "unclosed"}``. ``head`` is the part of the raw response before the answer (reasoning
    and its closing tag), or None when the answer is not the tail of the response; the sampled ids cover
    ``head`` + answer.
    """
    parser = getattr(agent, "reasoning_parser", None)
    if parser is None or not response_texts:
        return response_texts, None
    rd = responses.get("response_dict") if isinstance(responses, dict) else None
    choices = (rd or {}).get("choices") or []
    end_token = getattr(parser, "end_token", None)
    texts, splits = [], []
    for i, text in enumerate(response_texts):
        raw = ((choices[i] or {}).get("message") or {}) if i < len(choices) else {}
        if not isinstance(text, str) or any(raw.get(k) is not None for k in REASONING_KEYS):
            texts.append(text)
            splits.append(None)
            continue
        reasoning, content = split_reasoning(parser, text)
        texts.append(content)
        splits.append(
            {
                "reasoning": reasoning,
                "head": text[: len(text) - len(content)] if text.endswith(content) else None,
                "unclosed": isinstance(end_token, str) and end_token not in text,
            }
        )
    return texts, splits


def attach_reasoning(messages, splits) -> None:
    """Put each split response's reasoning on its assistant message as ``reasoning_content``."""
    for msg, split in zip(messages or [], splits or []):
        if not isinstance(msg, dict) or split is None:
            continue
        msg["reasoning_content"] = split["reasoning"]
        if split["unclosed"]:
            msg[REASONING_UNCLOSED] = True


def _content_text(msg: Dict[str, Any]) -> Optional[str]:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
        return content[0].get("text")
    return None


CONTENT_IS_PREFIX = "content_is_prefix"
# Set by a parser whose content leaves out part of the response that is not a suffix (ActionAgent's
# ``strip_reasoning`` removes the native reasoning before the answer): the sampled ids no longer match the content,
# so they are dropped and the turn is re-rendered from its text.
REASONING_STRIPPED = "reasoning_stripped"


def trim_token_ids_to_content(messages, response_texts, tokenizer, splits=None) -> int:
    """Cut a message's sampled ``token_ids`` when its parser kept only a prefix of the response as
    ``content`` on purpose and marked it with ``CONTENT_IS_PREFIX`` (ActionAgent's
    ``truncate_history_at_action``), so the next prompt and the training row hold exactly that prefix,
    not the full generation. The marker is removed here.

    The ids are cut at the shortest prefix whose decoded text covers the kept text, and the content
    becomes that decoded text, so text and ids agree (a token at the boundary may add a few
    characters). Unmarked messages are left alone: other parsers' content can be a prefix of the
    response too (native tool calls move the call text out of ``content``) while their ids must stay
    whole. Returns the number of ids removed, over all messages.

    ``splits`` (see :func:`split_reasoning_responses`): for a response split into reasoning and answer, the
    ids also cover the reasoning before the answer (``head``), so the kept text is ``head`` + content. When
    the answer is not the tail of the response the ids cannot be cut to it, and they are dropped (the turn
    is then re-encoded from its text)."""
    removed = 0
    splits = splits or []
    for i, (msg, full) in enumerate(zip(messages or [], response_texts or [])):
        if not isinstance(msg, dict) or not msg.pop(CONTENT_IS_PREFIX, False):
            continue
        ids = msg.get("token_ids")
        kept = _content_text(msg)
        split = splits[i] if i < len(splits) else None
        head = ""
        if split is not None:
            if split["head"] is None:
                msg.pop("token_ids", None)
                continue
            head = split["head"]
        if tokenizer is None or not ids or not isinstance(full, str) or kept is None:
            continue
        kept = head + kept
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
        if not text.startswith(head):
            continue
        text = text[len(head):]
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
        if context is not None and getattr(getattr(agent, "llm_engine", None), "supports_routing_key", False):
            # One key per agent session (a subagent has its own), so its turns reach the same engine.
            running = getattr(context, "running", None) or [0]
            extra_kwargs["routing_key"] = f"{context.rollout_id}:{running[-1]}"
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
    # With a reasoning parser, the answer alone is parsed (tool calls are not read out of the reasoning) and
    # the reasoning is stored beside it; the sampled ids stay whole.
    parse_texts, splits = split_reasoning_responses(agent, response_texts, responses)
    new_msg = agent.parse(
        parse_texts,
        tool_calls=responses.get("tool_calls"),
        context=context,
        trajectory_segments=[current_step.messages.messages],
    )
    overlay_raw_message_fields(new_msg, responses)
    attach_reasoning(new_msg, splits)
    removed = 0
    if use_token_ids:
        attach_token_ids(new_msg, responses)
        removed = trim_token_ids_to_content(new_msg, response_texts, getattr(agent, "tokenizer", None), splits)
        for msg in new_msg or []:
            if isinstance(msg, dict) and msg.pop(REASONING_STRIPPED, False):
                msg.pop("token_ids", None)
        # The engine's log-probabilities and routed-expert rows describe the sampled ids one for one.
        attach_rollout_signals(new_msg, responses)
        sync_rollout_signals(new_msg)
    else:
        drop_rollout_signals(new_msg)
        # Also drop ids a server may have returned on the raw message (overlaid above).
        for msg in new_msg or []:
            if isinstance(msg, dict):
                msg.pop("token_ids", None)
                msg.pop(CONTENT_IS_PREFIX, None)
                msg.pop(REASONING_STRIPPED, None)
    total_length = extract_total_length(responses)
    if total_length is not None and removed:
        total_length -= removed
    if forced_turn and isinstance(new_msg[0], dict):
        new_msg[0][FORCED] = True  # a replayed turn: conversion treats it as prefix, not as a policy target
    return (new_msg[0], total_length)
