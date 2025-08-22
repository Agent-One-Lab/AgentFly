"""
This file is a modified version of the original fastchat/conversation.py file.

The original file can be found at:
https://github.com/LM-SYS/fastchat/blob/main/fastchat/conversation.py

"""

from collections import defaultdict
from copy import copy
import dataclasses
from enum import Enum, auto, IntEnum
import json
from typing import List, Any, Dict, Union, Tuple
import warnings

import torch
from .preprocess import open_image_from_any
from transformers import PreTrainedTokenizer
import re

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ASSISTANT_PREFIX = "assistant_prefix"
    

class SeparatorStyle(IntEnum):
    """Separator styles."""
    NO_COLON_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    CHATGLM = auto()
    CHATML = auto()
    ADD_SPACE_TWO = auto()
    GENERAL = auto() # No sep, we take all sep as part of the role
    GENERAL_STOP_ALL = auto() # No sep, and add stop_str at each turn



@dataclasses.dataclass
class Template:
    """A class that manages prompt templates and keeps all conversation history."""
    # Properties
    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The template of the system prompt with tool usage
    system_template_with_tools: str = None
    # The system message
    system_message: str = ""
    # Stop criteria (the default one is EOS token)
    stop_words: Union[str, List[str]] = None
    # Behaviors
    # The tool template
    tool_template: str = None
    # The user template
    user_template: str = None
    # The assistant template
    assistant_template: str = None

    ## vision part
    vision_start: str = None
    vision_end: str = None
    image_token: str = None
    video_token: str = None

    def render(self, messages: List[Dict], tools=None, add_generation_prompt: bool = False) -> str:
        """Render the template with the given messages and kwargs.
        messages: [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello, how are you?"
                    }
                ]
            },
            {
                "role": "assistant",
        ]
        """
        elements = []
        roles = []
        if tools:
            tools = self._encode_system_tools(tools)

        for i, message in enumerate(messages):

            if i == 0 and self._detect_role(message["role"]) == Role.SYSTEM:
                system_message = self._encode_system_message(message["content"], tools=tools)
                elements.append(system_message)
                roles.append(Role.SYSTEM)
                # This message is done
                continue
            elif i == 0 and self._detect_role(message["role"]) != Role.SYSTEM:
                system_message = self._encode_system_message_default(tools=tools)
                elements.append(system_message)
                roles.append(Role.SYSTEM)
                # This message is not done, we need to handle other roles

            if self._detect_role(message["role"]) == Role.USER:
                user_message = self._encode_user_message(message["content"])
                elements.append(user_message)
                roles.append(Role.USER)
            elif self._detect_role(message["role"]) == Role.ASSISTANT:
                assistant_message = self._encode_assistant_message(message["content"])
                elements.append(assistant_message)
                roles.append(Role.ASSISTANT)
            elif self._detect_role(message["role"]) == Role.TOOL:
                tool_message = self._encode_tool_message(message["content"])
                elements.append(tool_message)
                roles.append(Role.TOOL)
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        
        if add_generation_prompt:
            generation_prefix = self._encode_generation_prompt()
            elements.append(generation_prefix)
            roles.append(Role.ASSISTANT_PREFIX)
        
        prompt = "".join(elements)
        return prompt, elements, roles

    def _detect_role(self, role: str) -> Role:
        if role == "system":
            return Role.SYSTEM
        elif role == "user":
            return Role.USER
        elif role == "assistant":
            return Role.ASSISTANT
        elif role == "tool":
            return Role.TOOL
        else:
            raise ValueError(f"Invalid role: {role}")
        
    def _encode_system_tools(self, tools: List[Dict]) -> str:
        return "\n".join([json.dumps(tool) for tool in tools])

    def _encode_system_message_default(self, tools=None) -> str:
        if tools is None:
            return self.system_template.format(system_message=self.system_message)
        else:
            if self.system_template_with_tools:
                return self.system_template_with_tools.format(system_message=self.system_message, tools=tools)
            else:
                return self.system_template.format(system_message=self.system_message)

    def _encode_system_message(self, content, tools=None) -> str:
        if tools is None:
            system_message = content[0]['text']
            return self.system_template.format(system_message=system_message)
        else:
            system_message = content[0]['text']
            if self.system_template_with_tools is None:
                return self.system_template.format(system_message=system_message)
            else:
                return self.system_template_with_tools.format(system_message=system_message, tools=tools)
    
    def _encode_user_message(self, content: List[Dict]) -> str:
        text = ""
        for item in content:
            if item["type"] == "text":
                text += item["text"]
            elif item["type"] == "image":
                text += self.vision_start + self.image_token + self.vision_end
            elif item["type"] == "video":
                text += self.vision_start + self.video_token + self.vision_end
            else:
                raise ValueError(f"Invalid message type: {item['type']}")
        user_message = self.user_template.format(content=text)
        return user_message
    
    def _encode_assistant_message(self, content: List[Dict]) -> str:
        assert len(content) == 1, "Assistant message must be a single message"
        text = content[0]["text"]
        assistant_message = self.assistant_template.format(content=text)
        return assistant_message
    
    def _encode_tool_message(self, content: List[Dict]) -> str:
        assert len(content) == 1, "Tool message must be a single message"
        text = content[0]["text"]
        tool_message = self.tool_template.format(observation=text)
        return tool_message
    
    def _encode_generation_prompt(self) -> str:
        if "{content}" in self.assistant_template:
            prefix = self.assistant_template.split("{content}")[0]
            return prefix
        else:
            raise ValueError(f"Assistant template {self.assistant_template} does not contain {{content}}")

    def _split_assistant_message(self, assistant_message: str) -> List[str]:
        # Split the assistant message into generation prefix, content, and generation suffix
        generation_prefix = self._encode_generation_prompt()
        assert assistant_message.startswith(generation_prefix), f"Assistant message {assistant_message} does not start with {generation_prefix}"
        content_suffix = assistant_message[len(generation_prefix):]
        for stop_word in self.stop_words:
            if stop_word in content_suffix:
                stop_word_index = content_suffix.index(stop_word)
                content = content_suffix[:stop_word_index+len(stop_word)]
                suffix = content_suffix[stop_word_index+len(stop_word):]
                break
        return generation_prefix, content, suffix


    def encode(self, messages: List[Dict], tokenizer: PreTrainedTokenizer, return_tensors: str = None, tools=None) -> str:
        prompt, elements, roles = self.render(messages, tools=tools)
        elements, mask_flags = self._postprocess_elements(elements, roles, messages=messages)
        input_ids = []
        attention_mask = []
        labels = []
        action_mask = []
        for element, mask_flag in zip(elements, mask_flags):
            cur_input_ids = tokenizer.encode(element, add_special_tokens=False)
            input_ids.extend(cur_input_ids)
            attention_mask.extend([1] * len(cur_input_ids))
            if mask_flag:
                labels.extend([-100] * len(cur_input_ids))
                action_mask.extend([0] * len(cur_input_ids))
            else:
                labels.extend(cur_input_ids)
                action_mask.extend([1] * len(cur_input_ids))
        inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            action_mask=action_mask
        )
        if return_tensors == "pt":
            inputs = {k: torch.tensor([v]) for k, v in inputs.items()}
        return inputs
        

    def _postprocess_elements(self, elements: List[str], roles , messages) -> List[str]:
        # Flag non-assistant messages
        new_elements = []
        mask_flags = []
        for i, element in enumerate(elements):
            if roles[i] == Role.ASSISTANT:
                new_elements.append(element)
                mask_flags.append(False)
            else:
                new_elements.append(element)
                mask_flags.append(True)

        # return new_elements, mask_flags
        loss_allowed_elem_set = set()
        has_explicit_loss = messages is not None and any(isinstance(m, dict) and ("loss" in m) for m in messages)
        if has_explicit_loss:
            # Build a mapping from element index -> message index
            role_to_msg_idx = [None] * len(roles)
            msg_idx = 0
            # Handle first role possibly being an auto-inserted default system
            if len(roles) > 0 and roles[0] == Role.SYSTEM:
                if messages and messages[0]["role"] == "system":
                    role_to_msg_idx[0] = 0
                    msg_idx = 1
                else:
                    role_to_msg_idx[0] = None  # inserted default system

        # Map remaining roles 1:1 to messages
        for i in range(1, len(roles)):
            if roles[i] in (Role.USER, Role.ASSISTANT, Role.TOOL):
                if msg_idx < len(messages):
                    role_to_msg_idx[i] = msg_idx
                    msg_idx += 1

        # Collect assistant element indices whose backing message has loss=True
        for i, r in enumerate(roles):
            if r == Role.ASSISTANT:
                mi = role_to_msg_idx[i]
                if mi is not None and messages[mi].get("loss", True):
                    loss_allowed_elem_set.add(i)
     
        # merge non-assistant messages and handle the generation prefix and suffixes
        merged_elements = []
        merged_mask_flags = []
        prev_element = ""
        prev_mask_flag = False
        for i, (element, mask_flag) in enumerate(zip(new_elements, mask_flags)):
            if i == 0:
                prev_element = element
                prev_mask_flag = mask_flag
                continue
            else:
                if prev_mask_flag == mask_flag:
                    # Both previous and current elements are assistant messages
                    if not mask_flag:
                        prefix, content, suffix = self._split_assistant_message(element)
                        merged_elements.append(prefix)
                        merged_mask_flags.append(True)
                        merged_elements.append(content)
                        merged_mask_flags.append(False if (not has_explicit_loss or i in loss_allowed_elem_set) else True)                        
                        prev_element = suffix
                        prev_mask_flag = True # We need to mask the suffix
                    # Both previous and current elements are non-assistant messages
                    else:
                        prev_element += element
                        prev_mask_flag = True
                else:
                    # Previous element is not assistant message, but the current one is
                    if not mask_flag:
                        prefix, content, suffix = self._split_assistant_message(element)
                        prev_element += prefix
                        prev_mask_flag = True
                        merged_elements.append(prev_element)
                        merged_mask_flags.append(prev_mask_flag)
                        merged_elements.append(content)
                        merged_mask_flags.append(False if (not has_explicit_loss or i in loss_allowed_elem_set) else True)
                        prev_element = suffix
                        prev_mask_flag = True
                    # Previous element is assistant message, but the current one is not
                    else:
                        prev_element += element
                        prev_mask_flag = True
        if prev_element != "":
            merged_elements.append(prev_element)
            merged_mask_flags.append(prev_mask_flag)
        return merged_elements, merged_mask_flags


    def get_vision_inputs(self):
        vision_inputs = defaultdict(list)
        for role, message, _ in self.messages:
            if isinstance(message, list):
                for item in message:
                    if item['type'] == 'text':
                        continue
                    elif item['type'] == 'image':
                        vision_inputs['image'].append(open_image_from_any(item['image']))
                    elif item['type'] == 'video':
                        raise NotImplementedError("Video is not supported for chat template.")
                    else:
                        raise ValueError(f"Invalid message type: {item['type']}")
        return vision_inputs

        # if self.name == "qwen2.5-vl":
        # jinja_template = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['role'] == 'tool'%}<tool_response>\n{% endif %}{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% if message['role'] == 'tool' %}\n\n</tool_response>{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""

    def jinja_template(self) -> str:
        """
        Build a Hugging-Face-style chat-template (Jinja-mini dialect) that mimics
        `self.render`.  The template expects three variables in its context:

            • messages               – list[dict]  (same format you pass to .render)
            • add_generation_prompt  – bool        (default False)
            • tools                  – list[dict]  (optional, for tool-enabled templates)

        No other Python state is referenced, so the string can be cached in the
        tokenizer and shipped to a different process.
        """
        # ------------------------------------------------------------------
        # 1.  Pre-compute constant strings so the inner template stays tiny
        # ------------------------------------------------------------------
        default_system = self.system_template.format(system_message=self.system_message)
        
        # Don't pre-format system_template_with_tools - handle it in Jinja
        system_template_with_tools_raw = self.system_template_with_tools if self.system_template_with_tools else None

        try:
            u_pref, u_suff = self.user_template.split("{content}")
            a_pref, a_suff = self.assistant_template.split("{content}")
        except ValueError as e:   # missing {content}
            raise ValueError("`user_template` / `assistant_template` must contain "
                            "`{content}` placeholder") from e

        if self.tool_template:
            t_pref, t_suff = self.tool_template.split("{observation}")
        else:                     # tools optional
            t_pref, t_suff = "", ""

        # tokens for images / videos
        img_tok = (self.vision_start or "") + (self.image_token or "") + (self.vision_end or "")
        vid_tok = (self.vision_start or "") + (self.video_token or "") + (self.vision_end or "")

        # ------------------------------------------------------------------
        # 2.  Assemble the Jinja text; everything in triple-quotes is copied
        #     verbatim into the tokenizer.  We splice in the constants that
        #     never change for this Template instance.
        # ------------------------------------------------------------------
        template_parts = [
            f"{{% set _u_pref  = {u_pref!r} %}}",
            f"{{% set _u_suff  = {u_suff!r} %}}",
            f"{{% set _a_pref  = {a_pref!r} %}}",
            f"{{% set _a_suff  = {a_suff!r} %}}",
            f"{{% set _t_pref  = {t_pref!r} %}}",
            f"{{% set _t_suff  = {t_suff!r} %}}",
            f"{{% set _img_tok = {img_tok!r} %}}",
            f"{{% set _vid_tok = {vid_tok!r} %}}",
            f"{{% set _default_system = {default_system!r} %}}",
            f"{{% set _system_message = {self.system_message!r} %}}",
            f"{{% set _system_template = {self.system_template!r} %}}",
        ]
        
        # Add system template with tools if available
        if system_template_with_tools_raw:
            template_parts.append(f"{{% set _system_template_with_tools = {system_template_with_tools_raw!r} %}}")
        
        template_parts.extend([
            # Handle system message first (matching render logic)
            "{% if messages and messages[0]['role'] == 'system' %}",
            "{% if tools and _system_template_with_tools %}",
            "{% if messages[0]['content'] is string %}",
            "{{ _system_template_with_tools.format(system_message=messages[0]['content'], tools=tools | map('tojson') | join('\\n')) }}",
            "{% else %}",
            "{{ _system_template_with_tools.format(system_message=messages[0]['content'][0]['text'], tools=tools | map('tojson') | join('\\n')) }}",
            "{% endif %}",
            "{% else %}",
            "{% if messages[0]['content'] is string %}",
            "{% set formatted_system = _system_template | replace('{system_message}', messages[0]['content']) %}{{ formatted_system }}",
            "{% else %}",
            "{% set formatted_system = _system_template | replace('{system_message}', messages[0]['content'][0]['text']) %}{{ formatted_system }}",
            "{% endif %}",
            "{% endif %}",
            "{% else %}",
            "{% if tools and _system_template_with_tools %}",
            "{{ _system_template_with_tools.format(system_message=_system_message, tools=tools | map('tojson') | join('\\n')) }}",
            "{% else %}",
            "{{ _default_system }}",
            "{% endif %}",
            "{% endif %}",
            # Process remaining messages (skip first if it was system)
            "{% for m in messages %}",
            "{% if not (loop.first and m['role'] == 'system') %}",
            "{% if m['role'] == 'user' %}",
            "{% set ns = namespace(txt='') %}",
            "{% if m['content'] is string %}",
            "{% set ns.txt = m['content'] %}",
            "{% else %}",
            "{% for item in m['content'] %}",
            "{% if item['type'] == 'text'  %}",
            "{% set ns.txt = ns.txt + item['text'] %}",
            "{% elif item['type'] == 'image' %}",
            "{% set ns.txt = ns.txt + _img_tok %}",
            "{% elif item['type'] == 'video' %}",
            "{% set ns.txt = ns.txt + _vid_tok %}",
            "{% endif %}",
            "{% endfor %}",
            "{% endif %}",
            "{{ _u_pref }}{{ ns.txt }}{{ _u_suff }}",
            "{% elif m['role'] == 'assistant' %}",
            "{% if m['content'] is string %}",
            "{{ _a_pref }}{{ m['content'] }}{{ _a_suff }}",
            "{% else %}",
            "{{ _a_pref }}{{ m['content'][0]['text'] }}{{ _a_suff }}",
            "{% endif %}",
            "{% elif m['role'] == 'tool' %}",
            "{% if m['content'] is string %}",
            "{{ _t_pref }}{{ m['content'] }}{{ _t_suff }}",
            "{% else %}",
            "{{ _t_pref }}{{ m['content'][0]['text'] }}{{ _t_suff }}",
            "{% endif %}",
            "{% endif %}",
            "{% endif %}",
            "{% endfor %}",
            "{% if add_generation_prompt %}",
            "{{ _a_pref }}",
            "{% endif %}"
        ])
        
        return "".join(template_parts)


    def render_with_mask(self, messages: List[Dict], add_generation_prompt: bool = False, tools=None):
        from termcolor import colored
        prompt, elements, roles = self.render(messages, add_generation_prompt=add_generation_prompt, tools=tools)
        elements, mask_flags = self._postprocess_elements(elements, roles)


        prompt = ""
        for element, mask_flag in zip(elements, mask_flags):
            if mask_flag:
                prompt += colored(element, "red")
            else:
                prompt += element
        return prompt, elements, mask_flags

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def set_tools(self, tools: List[Dict]):
        """Set the tools."""
        if self.tool_aggregator == "DEFAULT":
            self.tools = json.dumps(tools)
        elif self.tool_aggregator == "STACKED":
            self.tools = "\n".join([json.dumps(tool) for tool in tools])
        else:
            raise ValueError(f"Invalid tool aggregator: {self.tool_aggregator}")

    def copy(self):
        return Template(
            name=self.name,
            system_template=self.system_template,
            system_template_with_tools=self.system_template_with_tools,
            system_message=self.system_message,
            user_template=self.user_template,
            assistant_template=self.assistant_template,
            tool_template=self.tool_template,
            stop_words=self.stop_words,
            vision_start=self.vision_start,
            vision_end=self.vision_end,
            image_token=self.image_token,
            video_token=self.video_token,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "system_template_with_tools": self.system_template_with_tools,
            "stop_words": self.stop_words,
            "vision_start": self.vision_start,
            "vision_end": self.vision_end,
            "image_token": self.image_token,
            "video_token": self.video_token,
        }

class Chat:
    def __init__(self, template: str, messages: List[List[str]]=None, tools=None, tokenizer: PreTrainedTokenizer = None):
        self.template = get_template(template)
        self.messages = self.convert_to_hf_format_messages(messages)
        self.tokenizer = tokenizer
        self.tools = tools
        self.flags = {}

    def _detect_labels(self, messages):
        message = messages[0]
        if 'role' in message and "content" in message:
            return 'role', 'content'
        elif 'from' in message and "value" in message:
            return 'from', 'value'
        else:
            raise ValueError(f"Cannot find role label and content label in the data.")

    
    def _convert_single_message_to_hf_format(self, message: Dict) -> Dict:
        if isinstance(message['content'], str):
            message['content'] = [{"type": "text", "text": message['content']}]
        elif isinstance(message['content'], list):
            for item in message['content']:
                if item['type'] == 'text':
                    continue
                else:
                    # Not sure what to do with other types of content
                    pass

    def convert_to_hf_format_messages(self, messages: List[Dict]) -> List[Dict]:
        if messages is None:
            return None
        role_label, content_label = self._detect_labels(messages)
        hf_messages = []
        for message in messages:
            d = {"role": message[role_label], "content": message[content_label]}
            if "loss" in message:
                d["loss"] = message["loss"]
            hf_messages.append(d)
        for message in hf_messages:
            self._convert_single_message_to_hf_format(message)

        return hf_messages

    def set_messages(self, messages: List[Dict]):
        self.messages = self.convert_to_hf_format_messages(messages)

    def prompt(self, add_generation_prompt=False, tools=None) -> str:
        self.flags['add_generation_prompt'] = add_generation_prompt
        prompt, _, _ = self.template.render(messages=self.messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return prompt

    def prompt_with_mask(self, add_generation_prompt=False, tools=None) -> str:
        prompt_with_mask, _, _ = self.template.render_with_mask(messages=self.messages, add_generation_prompt=add_generation_prompt, tools=tools)
        return prompt_with_mask

    def tokenize(self, tokenizer: PreTrainedTokenizer = None, add_generation_prompt=False, tools=None) -> List[int]:
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tools is None:
            tools = self.tools
        return self.template.encode(messages=self.messages, tokenizer=tokenizer, return_tensors="pt", tools=tools)

    def append(self, message: Union[Dict, List[Dict]]):
        self._convert_single_message_to_hf_format(message)
        self.messages.append(message)


# A global registry for all conversation templates
TEMPLATES: Dict[str, Template] = {}


def register_template(template: Template, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in TEMPLATES
        ), f"{template.name} has been registered."

    TEMPLATES[template.name] = template


def get_template(name: str) -> Template:
    """Get a conversation template."""
    return TEMPLATES[name].copy()


register_template(
    Template(
        name="qwen2.5-no-system-tool",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
    )
)

register_template(
    Template(
        name="qwen2.5",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_message="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        system_template_with_tools="""<|im_start|>system\n{system_message}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call><|im_end|>\n""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
    )
)


register_template(
    Template(
        name="qwen2.5-think",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_message="You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
        # system_template_with_tools="""<|im_start|>You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object inside <answer> and </answer> tags with function name and arguments within <tool_call></tool_call> XML tags:\n<answer>\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>\n</answer><|im_end|>\n""",
        system_template_with_tools="""<|im_start|>You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then call tools or provide the answer. The thinking process is enclosed within <think> </think> tags, i.e., <think> [reasoning process here] </think> [response here].\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<think> [reasoning process here] </think>\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>\nYou must think first before calling any tool.<|im_end|>\n""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n<think>{content}<|im_end|>\n",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
        vision_start="<|vision_start|>",
        vision_end="<|vision_end|>",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
    )
)

register_template(
    Template(
        name="deepseek-prover",
        system_template="{system_message}\n",
        system_message="You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
        user_template="### Instruction:\n{content}\n",
        assistant_template="### Response:\n{content}\n<|EOT|>\n",
        stop_words=["<|EOT|>"],
    )
)

# register_conv_template(
#     Template(
#         name="qwen3",
#         system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
#         # system_message="",
#         system_template_tool="""<|im_start|>system\n{system_message}# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call><|im_end|>\n""",
#         roles=("<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_start|>user\n<tool_response>\n{observation}\n</tool_response>"),
#         tool_aggregator="STACKED",
#         sep_style=SeparatorStyle.GENERAL_STOP_ALL,
#         sep="",
#         stop_token_ids=[
#             151643,
#             151644,
#             151645,
#         ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
#         stop_str="<|im_end|>\n",
#     )
# )

# register_conv_template(
#     Template(
#         name="qwen2.5-vl",
#         system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
#         system_message="You are a helpful assistant.",
#         roles=("<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_start|>tool\n<tool_response>\n{observation}\n</tool_response>"),
#         tool_aggregator="STACKED",
#         sep_style=SeparatorStyle.GENERAL_STOP_ALL,
#         sep="",
#         vision_start="<|vision_start|>",
#         vision_end="<|vision_end|>",
#         image_token="<|image_pad|>",
#         video_token="<|video_pad|>",
#         stop_token_ids=[
#             151643,
#             151644,
#             151645,
#         ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
#         stop_str="<|im_end|>\n",
#     )
# )



if __name__ == "__main__":
    pass
