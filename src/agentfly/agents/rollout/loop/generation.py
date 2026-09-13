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
    exactly as sampled. The ids are the FULL generation even though the parsed
    ``content`` may be truncated (e.g. at ``</action>``) — that matches training
    on the sampled response like verl-agent's ``batch['responses']``.

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
    chain_id,
    generation_config,
    context: "Context",
) -> Tuple[Any, Optional[int]]:
    """Generate one assistant message. Returns ``(message, total_token_length)``."""
    effective_config = prepare_generation_config(
        agent, generation_config, current_step, tools=tools
    )
    skills_payload = agent.skills_payload()
    extra_kwargs = {"skills": skills_payload} if skills_payload else {}
    extra_kwargs["tool_call_source"] = getattr(
        agent, "default_tool_call_source", "parser"
    )
    raw = await agent.generate_async(
        [current_step.messages.messages],
        return_dict=True,
        tools=tools,
        **extra_kwargs,
        **effective_config,
    )
    responses = normalize_generate_response(raw)
    response_texts = responses.get("response_texts")
    context.metadata = {
        **context.metadata,
        "trajectory_segments": [current_step.messages.messages],
    }
    new_msg = agent.parse(
        response_texts,
        tool_calls=responses.get("tool_calls"),
        context=context,
    )
    overlay_raw_message_fields(new_msg, responses)
    attach_token_ids(new_msg, responses)
    total_length = extract_total_length(responses)
    return (new_msg[0], total_length)
