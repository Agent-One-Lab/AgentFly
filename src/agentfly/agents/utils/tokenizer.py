from typing import Optional

from chat_bricks import get_template
from transformers import AutoProcessor, AutoTokenizer


def create_tokenizer(model_name_or_path: str):
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
