import copy
from enum import Enum
import os
import warnings
import torch
import transformers
from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info
import re
import logging
from .templates import Chat, get_template
from ... import AGENT_DATA_DIR

# Set up logging that won't be overridden by other modules
LOGGER = logging.getLogger(__name__)

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')   # matches any ANSI color/style code

def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return ANSI_RE.sub('', s)

def is_vlm_template(template: str) -> bool:
    return template in ["qwen2.5-vl"]


def convert_messages_to_openai_format(messages: list) -> list:
    """
    Convert messages to OpenAI format.
    TODO: add more processing for other types of content
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        # if "tool_calls" in message:
        #     del message["tool_calls"]
        # if "tool_call_id" in message:
        #     del message["tool_call_id"]
        if "tool_choice" in message:
            del message["tool_choice"]
    return messages


def convert_messages_to_hf_format(messages: list) -> list:
    """
    Convert messages to Hugging Face format.
    """
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            for item in content:
                if 'type' in item:
                    if item['type'] == 'image_url':
                        item['type'] = 'image'
                        item['image'] = item['image_url']['url']
                        del item['image_url']
                    else:
                        # TODO: handle other types of content
                        pass
        message['content'] = content
    return messages

def transform_multi_turn_reward_mask(action_mask):
    """
    Given a binary action_mask of shape (batch_size, sequence_length),
    returns a tensor of the same shape with 1 only at the position where the action_mask is 1 and the next position is 0,
    """
    # action_mask: shape (batch_size, sequence_length)
    batch_size, seq_length = action_mask.shape
    
    # Create a shifted version of the attention mask by shifting left.
    # For the last column, we append a column of zeros.
    shifted = torch.cat([
        action_mask[:, 1:], 
        torch.zeros(batch_size, 1, dtype=action_mask.dtype, device=action_mask.device)
    ], dim=1)
    
    # Identify positions where the attention_mask is 1 and the shifted mask is 0.
    # This means either the next position is 0 or we're at the last element.
    last_ones_mask = (action_mask == 1) & (shifted == 0)
    
    # Optionally, convert boolean mask to integers (0s and 1s).
    return last_ones_mask.int()


def transform_reward_mask(action_mask):
    """
    Given a binary attention_mask of shape (batch_size, sequence_length),
    returns a tensor of the same shape with 1 only at the rightmost (last) 1 per row,
    and 0 everywhere else.
    """
    batch_size, seq_length = action_mask.shape

    # Check for rows that contain at least one 1.
    has_one = action_mask.sum(dim=1) > 0

    # Reverse each row so that the first occurrence of 1 corresponds to the last 1 in the original.
    reversed_mask = action_mask.flip(dims=[1])

    # For each row, find the index of the first occurrence of 1 in the reversed row.
    # Note: torch.argmax returns 0 if no element is 1, so we will handle rows with no ones separately.
    first_one_idx_reversed = torch.argmax(reversed_mask, dim=1)

    # Convert to the original index position.
    last_indices = seq_length - 1 - first_one_idx_reversed

    # Create an output tensor initialized with zeros.
    output = torch.zeros_like(action_mask)

    # For rows that have at least one 1, set the found last index to 1.
    # We use advanced indexing to assign 1 to the appropriate positions.
    row_indices = torch.arange(batch_size)
    output[row_indices[has_one], last_indices[has_one]] = 1

    return output


def tokenize_conversation(
    messages,
    tokenizer,
    template,
    max_length,
    tools=None,
    processor=None,
    return_tensors="pt",
    add_generation_prompt=False,
):
    """
    We want to tokenize the whole conversation. But we can't just simply
    use get_prompt to get string prompt and tokenize it. Because the loss
    can only be computed on model's response. We want:
        input_ids
        attention_mask
        labels: should be -100 for user prompt and input id for model's response
        action_mask: should be 0 for user prompt and 1 for model's response
    :param messages:
    :param tokenizer:
    :param conv_template:
    :param max_length:
    :return: input_ids, attention_mask, labels, action_mask
    """
    # Check if tokenizer is our interface or a HuggingFace tokenizer
    if hasattr(tokenizer, 'tokenizer'):  # Our interface
        # Use the underlying HuggingFace tokenizer for Chat template
        hf_tokenizer = tokenizer.tokenizer
    else:  # Direct HuggingFace tokenizer
        hf_tokenizer = tokenizer
    
    chat = Chat(template=template, messages=messages, tokenizer=hf_tokenizer)
    inputs = chat.tokenize(hf_tokenizer, add_generation_prompt=add_generation_prompt, tools=tools)
    
    if max_length is not None:
        inputs['input_ids'] = inputs['input_ids'][:, :max_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]
        inputs['labels'] = inputs['labels'][:, :max_length]
        inputs['action_mask'] = inputs['action_mask'][:, :max_length]

    return inputs

def convert_inputs_to_vision_inputs(template: str,
                                    inputs: dict,
                                    processor,          # AutoProcessor (not bare tokenizer)
                                    messages: list):
    """
    Expand `inputs` (built from a chat template that contains ONE
    <|image_pad|> / <|video_pad|> placeholder per asset) into the real
    processor outputs and stretch `action_mask` / `labels` so they stay
    aligned after the pad tokens are repeated.

    Returns
    -------
    dict   -- processor(...) result + expanded masks
    """
    assert template == "qwen2.5-vl", "Only qwen2.5-vl is supported"


    # ------------------------------------------------------------------
    # 1. special-token ids
    # ------------------------------------------------------------------
    tk = processor.tokenizer
    conv = get_conv_template(template)
    image_pad_id  = tk.encode(conv.image_token,  add_special_tokens=False)[0]
    video_pad_id  = tk.encode(conv.video_token,  add_special_tokens=False)[0]
    vis_start_id  = tk.encode(conv.vision_start, add_special_tokens=False)[0]
    vis_end_id    = tk.encode(conv.vision_end,   add_special_tokens=False)[0]

    repeat_ids = torch.tensor([image_pad_id, video_pad_id], dtype=torch.long)
    vision_ids = torch.tensor([image_pad_id, video_pad_id,
                               vis_start_id, vis_end_id], dtype=torch.long)

    # ------------------------------------------------------------------
    # 2. run the processor (adds patch-level vision tokens)
    # ------------------------------------------------------------------
    LOGGER.debug(f"[Template::convert_inputs_to_vision_inputs] messages: {messages}")
    imgs, vids = process_vision_info(messages)
    text = format_conversation(messages, template).get_prompt()
    proc_out = processor(
        text=text,
        images=imgs,
        videos=vids,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    new_ids = proc_out["input_ids"][0]            # (new_len,)
    new_attention_mask = proc_out["attention_mask"][0]
    device  = new_ids.device

    # ------------------------------------------------------------------
    # 3. original (pre-processor) tensors
    # ------------------------------------------------------------------
    old_ids    = inputs["input_ids"][0].to(device)
    old_action = inputs.get("action_mask", None)
    old_labels = inputs.get("labels", None)
    old_attention_mask = inputs.get("attention_mask", None)

    # ------------------------------------------------------------------
    # 4. build boolean masks that mark NON-repeated positions
    # ------------------------------------------------------------------
    
    new_keep = ~torch.isin(new_ids, repeat_ids)   # True where we copy from old
    old_keep = ~torch.isin(old_ids, repeat_ids)

    # sanity check
    assert new_keep.sum() == old_keep.sum(), f"Mismatch after dropping repeated vision tokens: {new_ids} {old_ids}"

    # ------------------------------------------------------------------
    # 5. allocate expanded tensors and copy en masse
    # ------------------------------------------------------------------
    if old_action is not None:
        exp_action = torch.zeros_like(new_ids, dtype=old_action.dtype)
        exp_action[new_keep] = old_action[0][old_keep].to(device)
        proc_out["action_mask"] = exp_action.unsqueeze(0)

    if old_labels is not None:
        exp_labels = torch.full_like(new_ids, -100, dtype=old_labels.dtype)
        exp_labels[new_keep] = old_labels[0][old_keep].to(device)
        proc_out["labels"] = exp_labels.unsqueeze(0)

    return proc_out


def tokenize_conversations(messages_list, tokenizer, conv_template, max_length, processor=None, return_tensors="pt", return_reward_mask=False):
    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    batch_action_masks = []
    # TODO: add multiprocessing
    for messages in messages_list:
        inputs = tokenize_conversation(messages, tokenizer, conv_template, max_length, processor=processor)
        batch_input_ids.append(inputs['input_ids'].squeeze(0))
        batch_attention_masks.append(inputs['attention_mask'].squeeze(0))
        batch_labels.append(inputs['labels'].squeeze(0))
        batch_action_masks.append(inputs['action_mask'].squeeze(0))
    
    if return_tensors == "pt":
        # Use pad_token_id from the tokenizer interface
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
        batch_attention_masks = torch.nn.utils.rnn.pad_sequence(batch_attention_masks, batch_first=True, padding_value=0)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        batch_action_masks = torch.nn.utils.rnn.pad_sequence(batch_action_masks, batch_first=True, padding_value=0)

    inputs = dict(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_masks,
        labels=batch_labels,
        action_mask=batch_action_masks
    )

    if return_reward_mask:
        inputs['reward_mask'] = transform_reward_mask(batch_action_masks)

    return inputs

def visualize_template(template, messages=None, tools=None, **kwargs):
    if not messages:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am fine, thank you."},
            {"role": "user", "content": "Want to play a game?"},
            {"role": "assistant", "content": "Sure, what game?"},
            {"role": "user", "content": "Guess the number."},
        ]

    chat = Chat(template=template, messages=messages)
    print(chat.prompt(tools=tools))
    print(chat.prompt_with_mask(tools=tools))


def visualize_jinja_template(tokenizer, messages=None, tools=None, **kwargs):
    if not messages:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am fine, thank you."},
            {"role": "user", "content": "Want to play a game?"},
            {"role": "assistant", "content": "Sure, what game?"},
            {"role": "user", "content": "Guess the number."},
        ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, **kwargs)
    print(prompt)

def compare_hf_template(tokenizer, template_name, messages=None, tools=None, add_generation_prompt=False):
    official_prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt)
    chat = Chat(template_name, messages, tokenizer)
    implemented_prompt = chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools)
    is_equal = official_prompt == implemented_prompt
    highlighted_prompt = chat.prompt_with_mask(add_generation_prompt=add_generation_prompt, tools=tools)
    plain_highlighted_prompt = strip_ansi(highlighted_prompt)
    is_equal_between_implemented_prompts = implemented_prompt == plain_highlighted_prompt
    jinja_template = chat.template.jinja_template()
    tokenizer.chat_template = jinja_template
    implemented_jinja_prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt)
    is_equal_between_jinja_prompts = implemented_jinja_prompt == implemented_prompt
    return is_equal, is_equal_between_implemented_prompts, is_equal_between_jinja_prompts, official_prompt, implemented_prompt, implemented_jinja_prompt, highlighted_prompt


def vllm_serve(model_name_or_path, template, tp, pp, dp):
    port = 8000
    jinja_template = get_template(template).jinja_template()
    if not os.path.exists(f"{AGENT_DATA_DIR}/cache"):
        os.makedirs(f"{AGENT_DATA_DIR}/cache")
    with open(f"{AGENT_DATA_DIR}/cache/jinja_template.jinja", "w") as f:
        f.write(jinja_template)
    # command = f"vllm serve {model_name_or_path} --chat-template {AGENT_DATA_DIR}/cache/jinja_template.jinja --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --data-parallel-size {dp} --port {port} --enable-auto-tool-choice --tool-call-parser hermes --expand-tools-even-if-tool-choice-none"
    command = f"vllm serve {model_name_or_path} --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --data-parallel-size {dp} --port {port} --enable-auto-tool-choice --tool-call-parser hermes --expand-tools-even-if-tool-choice-none"

    print(command)
    os.system(command)


if __name__=="__main__":
    "python -m agents.agents.templates.utils"
    # model = "/mnt/sharefs/users/haonan.li/models/Qwen2.5-7B-instruct-am_think_v1_distilled"
    model = "Qwen/Qwen2.5-7B-Instruct"
    # vllm_serve(model, "qwen2.5-think", 2, 1, 4)
    vllm_serve(model, "qwen2.5", 1, 1, 1)

