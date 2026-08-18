"""Response generation + backend-response normalization for the rollout loop.

These were methods on ``ChainRollout`` that encode *backend response-dict shape*
knowledge (``response_dict`` / ``choices`` / ``total_lengths``) and the per-call
generation-config budgeting. They are pulled out as free functions so the loop class
(:mod:`.chain`) is not responsible for "how we talk to a backend and read its response."

The orchestrating functions (:func:`generate_response`, :func:`prepare_generation_config`)
take the ``rollout`` (a :class:`~agentfly.agents.rollout.host.RolloutHost` that is also the
loop) because they call host hooks (``generate_async`` / ``parse``) and ``_skills_payload``.
Token estimation lives in :mod:`.tokens`. The pure helpers take only their data.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .tokens import estimate_chat_prompt_tokens

if TYPE_CHECKING:
    from ...core import Context
    from .chain import ChainRollout
    from .structures import Node


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
    rollout: "ChainRollout",
    generation_config: Optional[Dict[str, Any]],
    current_node: "Node",
    tools: Optional[List] = None,
) -> Dict[str, Any]:
    """Merge max_tokens / max_new_tokens; if ``max_model_len`` is set, cap completion budget.

    Only the first generation in a chain (``total_token_length == 0`` on the node) uses
    chat_bricks ``Chat.tokenize``; later turns use ``max_model_len - total_token_length``.
    """
    config = dict(generation_config) if generation_config else {}
    max_model_len = getattr(rollout, "max_model_len", None)

    if config.get("max_tokens") is not None:
        config.pop("max_new_tokens", None)
    else:
        max_new = config.pop("max_new_tokens", None)
        if max_new is not None:
            config["max_tokens"] = max_new

    if max_model_len is not None:
        if current_node.total_token_length == 0:
            prompt_tok = estimate_chat_prompt_tokens(rollout, current_node, tools=tools)
        else:
            prompt_tok = current_node.total_token_length
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
    rollout: "ChainRollout",
    chain,
    current_node,
    tools,
    depth,
    chain_id,
    generation_config,
    context: "Context",
) -> Tuple[Any, Optional[int]]:
    """Generate one assistant message. Returns ``(message, total_token_length)``."""
    effective_config = prepare_generation_config(
        rollout, generation_config, current_node, tools=tools
    )
    skills_payload = rollout._skills_payload()
    extra_kwargs = {"skills": skills_payload} if skills_payload else {}
    extra_kwargs["tool_call_source"] = getattr(
        rollout, "default_tool_call_source", "parser"
    )
    raw = await rollout.generate_async(
        [current_node.messages.messages],
        return_dict=True,
        tools=tools,
        **extra_kwargs,
        **effective_config,
    )
    responses = normalize_generate_response(raw)
    response_texts = responses.get("response_texts")
    context.metadata = {
        **context.metadata,
        "trajectory_segments": [current_node.messages.messages],
    }
    new_msg = rollout.parse(
        response_texts,
        tool_calls=responses.get("tool_calls"),
        context=context,
    )
    overlay_raw_message_fields(new_msg, responses)
    total_length = extract_total_length(responses)
    return (new_msg[0], total_length)
