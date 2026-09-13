from typing import Optional

from chat_bricks import get_template, tokenize_conversations

# NOTE: ``AutoProcessor`` / ``AutoTokenizer`` (transformers -> torch, ~5s) are
# imported lazily inside the factory functions. This module is imported by
# agent_base, so a bare ``import agentfly.agents`` must not pull transformers.


def create_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer  # lazy: heavy import

    # Pass trust_remote_code EXPLICITLY so a repo shipping custom code (e.g.
    # MiniMaxAI/MiniMax-M3-MXFP8) never drops into an interactive y/N PROMPT that
    # hangs a headless run — leaving it unset (None) makes transformers prompt on
    # a TTY. True matches the vLLM backend's own tokenizer load. The broad except
    # still degrades to None for an API model id that isn't an HF repo (e.g.
    # "deepseek-v4-pro") — the tokenizer is optional for client/API backends.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True)
    except Exception:
        tokenizer = None

    return tokenizer


def create_processor(model_name_or_path: str):
    from transformers import AutoProcessor  # lazy: heavy import

    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True)
    except Exception:
        processor = None

    return processor


def get_jinja_template(template_name: str) -> Optional[str]:
    """Return the Jinja chat template string for ``template_name``, or ``None`` if unavailable.

    Closed-source or non-HF model ids (e.g. GPT/Claude API names) typically cannot be resolved
    by chat-bricks; those cases return ``None`` instead of raising.
    """
    try:
        return get_template(template_name).jinja_template()
    except Exception:
        return None


def tokenize_trajectories(
    agent,
    messages_list,
    template=None,
    tokenizer=None,
    processor=None,
    return_reward_mask: bool = False,
    concatenate_mm_inputs: bool = True,
    train_on_last_turn: bool = False,
):
    import torch  # lazy: heavy import, only needed when building tensors

    # TODO: we will remove this argument in the future
    train_on_last_turn = False

    inputs = tokenize_conversations(
        messages_list,
        tokenizer=tokenizer,
        template=template or agent.template,
        processor=processor or agent.processor,
        max_length=agent.max_model_len,
        return_reward_mask=return_reward_mask,
        add_generation_prompt=True,
        # Render exactly the tools generation rendered (agent.prompt_tools is
        # the single source for both), so training tokens match the sampled prompt.
        tools=agent.prompt_tools() or None,
        concatenate_mm_inputs=concatenate_mm_inputs,
        ignore_tool_calls=True,
        train_on_last_turn_only=train_on_last_turn,
    )
    position_ids = torch.clip(
        torch.cumsum(inputs["attention_mask"], dim=-1) - 1, min=0, max=None
    )
    inputs["position_ids"] = position_ids

    return inputs
