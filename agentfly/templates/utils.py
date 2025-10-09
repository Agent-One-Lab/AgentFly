from collections import defaultdict
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
from .. import AGENT_DATA_DIR
from typing import Any
from .vision_processor import get_processor

LOGGER = logging.getLogger(__name__)

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')   # matches any ANSI color/style code

def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return ANSI_RE.sub('', s)


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
    **kwargs, # Additional kwargs for the chat template, e.g. enable_thinking
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
    chat = Chat(template=template, messages=messages, tokenizer=tokenizer)
    inputs = chat.tokenize(tokenizer, add_generation_prompt=add_generation_prompt, tools=tools, processor=processor, **kwargs)
    
    if max_length is not None:
        inputs['input_ids'] = inputs['input_ids'][:, :max_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]
        if 'labels' in inputs:
            inputs['labels'] = inputs['labels'][:, :max_length]
        if 'action_mask' in inputs:
            inputs['action_mask'] = inputs['action_mask'][:, :max_length]

    return inputs

def convert_inputs_to_vision_inputs(template: str,
                                    inputs: dict,
                                    processor,          # AutoProcessor (not bare tokenizer)
                                    messages: list):
    """
    NEW PIPELINE: Template processes messages → Human-readable prompt → Vision processor → LLM-ready inputs
    
    The correct pipeline is:
    1. Template processes messages to get human-readable prompt with single multi-modal tokens
    2. Vision processor handles image/video processing and token expansion
    3. Final result is directly usable by LLMs with model(**inputs)
    """
    # Get the vision processor for this template
    vision_processor = get_processor(template)
    if vision_processor is None:
        raise ValueError(f"No vision processor registered for template: {template}")
    
    # Step 1: Template processes messages to get human-readable prompt
    from .templates import Chat
    chat = Chat(template=template, messages=messages, tokenizer=processor.tokenizer)
    prompt = chat.prompt()  # This gives us human-readable prompt with single multi-modal tokens
    
    # Step 2: Extract vision inputs from messages
    images, videos = extract_vision_inputs_from_messages(messages)
    
    # Step 3: Vision processor handles the complete pipeline
    # This expands tokens and generates LLM-ready inputs
    final_inputs = vision_processor.process_for_llm(
        prompt=prompt,
        images=images,
        videos=videos,
        processor=processor,
        tokenizer=processor.tokenizer
    )
    
    return final_inputs

def extract_vision_inputs_from_messages(messages: list) -> tuple[list, list]:
    """Extract images and videos from messages"""
    images, videos = [], []
    
    for message in messages:
        if isinstance(message.get('content'), list):
            for item in message['content']:
                if item.get('type') in ['image', 'image_url']:
                    if 'image' in item:
                        images.append(item['image'])
                    elif 'image_url' in item:
                        images.append(item['image_url']['url'])
                elif item.get('type') in ['video', 'video_url']:
                    if 'video' in item:
                        videos.append(item['video'])
                    elif 'video_url' in item:
                        videos.append(item['video_url']['url'])
    
    return images, videos

def process_prompt_with_vision(
    prompt: str,
    template: str,
    processor: Any,
    images: list = None,
    videos: list = None,
) -> dict:
    """Process a prompt with vision support"""
    vision_processor = get_processor(template)
    if vision_processor is None:
        # If no vision processor, just return tokenized prompt
        return processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
    
    # Use vision processor to handle the complete pipeline
    return vision_processor.process_for_llm(
        prompt=prompt,
        images=images or [],
        videos=videos or [],
        processor=processor,
        tokenizer=processor.tokenizer
    )


def tokenize_conversations(
    messages_list,
    tokenizer,
    template,
    max_length,
    processor=None,
    return_tensors="pt",
    return_reward_mask=False,
    add_generation_prompt=False,
    padding_side="right",
    concatenate_mm_inputs=False
):
    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    batch_action_masks = []
    batch_mm_inputs = []
    # TODO: add multiprocessing
    for messages in messages_list:
        inputs = tokenize_conversation(messages, tokenizer, template, max_length, processor=processor, add_generation_prompt=add_generation_prompt)
        batch_input_ids.append(inputs['input_ids'].squeeze(0))
        batch_attention_masks.append(inputs['attention_mask'].squeeze(0))
        batch_labels.append(inputs['labels'].squeeze(0))
        batch_action_masks.append(inputs['action_mask'].squeeze(0))
        mm_inputs = {}
        if "pixel_values" in inputs:
            mm_inputs["pixel_values"] = inputs["pixel_values"]
        else:
            mm_inputs["pixel_values"] = None
        if "image_grid_thw" in inputs:
            mm_inputs["image_grid_thw"] = inputs["image_grid_thw"]
        else:
            mm_inputs["image_grid_thw"] = None

        batch_mm_inputs.append(mm_inputs)

    if return_tensors == "pt":
        # Use pad_token_id from the tokenizer interface
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id, padding_side=padding_side)
        batch_attention_masks = torch.nn.utils.rnn.pad_sequence(batch_attention_masks, batch_first=True, padding_value=0, padding_side=padding_side)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100, padding_side=padding_side)
        batch_action_masks = torch.nn.utils.rnn.pad_sequence(batch_action_masks, batch_first=True, padding_value=0, padding_side=padding_side)

    # convert [{"pixel_values": tensor, "image_grid_thw": tensor}, ...] to {"key1":  concat_tensor, "key2": concat_tensor, ...}
    concatenated_mm_inputs = {}
    if concatenate_mm_inputs:
        for key in batch_mm_inputs[0].keys():
            if isinstance(mm_inputs[key], torch.Tensor):
                concatenated_mm_inputs[key] = torch.cat([mm_inputs[key] for mm_inputs in batch_mm_inputs if mm_inputs[key] is not None], dim=0)

    inputs = dict(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_masks,
        labels=batch_labels,
        action_mask=batch_action_masks
    )

    if return_reward_mask:
        inputs['reward_mask'] = transform_reward_mask(batch_action_masks)

    # Check if we need mm_inputs
    mm_keys = list(batch_mm_inputs[0].keys())
    return_mm_inputs = False
    for key in mm_keys:
        if any(mm_inputs[key] is not None for mm_inputs in batch_mm_inputs):
            return_mm_inputs = True
            break

    if return_mm_inputs:
        if concatenate_mm_inputs:
            inputs.update(concatenated_mm_inputs)
        else:
            inputs["mm_inputs"] = batch_mm_inputs

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

def compare_hf_template(tokenizer, template_name, messages=None, tools=None, add_generation_prompt=False, **kwargs):
    official_prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
    chat = Chat(template_name, messages=messages, tokenizer=tokenizer)
    implemented_prompt = chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools, **kwargs)
    is_equal = official_prompt == implemented_prompt
    highlighted_prompt = chat.prompt_with_mask(add_generation_prompt=add_generation_prompt, tools=tools, **kwargs)
    plain_highlighted_prompt = strip_ansi(highlighted_prompt)
    is_equal_between_implemented_prompts = implemented_prompt == plain_highlighted_prompt
    jinja_template = chat.template.jinja_template()
    
    official_jinja_prompt = tokenizer.chat_template
    tokenizer.chat_template = jinja_template
    implemented_jinja_prompt = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
    is_equal_between_jinja_prompts = implemented_jinja_prompt == implemented_prompt
    tokenizer.chat_template = official_jinja_prompt
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
    model = "Qwen/Qwen2.5-3B-Instruct"
    # vllm_serve(model, "qwen2.5-think", 2, 1, 4)
    vllm_serve(model, "qwen2.5", 1, 1, 1)

