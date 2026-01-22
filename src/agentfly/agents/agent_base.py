import inspect
import json
import logging
import os
from abc import ABC
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from termcolor import colored

from ..templates import tokenize_conversations
from ..templates.templates import get_template
from ..tools.tool_base import BaseTool
from ..utils.monitor import JsonlSink, Monitor, WandbSink
from .chain.chain_base import ChainRollout
from .chain.streaming_observer import ConsoleStreamObserver, StreamingManager
from .llm_backends import (AsyncVerlBackend, AsyncVLLMBackend, ClientBackend,
                           TransformersBackend)
from .llm_backends.backend_configs import BACKEND_CONFIGS
from .utils.messages import MessagesList
from .utils.tokenizer import create_processor, create_tokenizer

try:
    from ..verl.protocol import DataProto
except ImportError:
    print("verl can not be imported.")
    pass

# Try to import vLLM tool parser components
try:
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager

    VLLM_TOOL_PARSER_AVAILABLE = True

    def silence_tool_parsers():
        # vLLM has used both namespaces across versions
        prefixes = [
            "vllm.tool_parsers",
            "vllm.entrypoints.openai.tool_parsers",
        ]
        for p in prefixes:
            lg = logging.getLogger(p)
            lg.setLevel(logging.CRITICAL + 1)
            lg.propagate = False  # don't bubble to root handlers

    silence_tool_parsers()

except ImportError:
    VLLM_TOOL_PARSER_AVAILABLE = False
    AutoTokenizer = None
    ChatCompletionRequest = None
    ToolParserManager = None

Logger = logging.getLogger(__name__)


class BaseAgent(ChainRollout, ABC):
    """
    Base class for all agents. All agent should subclass this class. A customized agent can implement the following methods:

    - generate_async: generate responses asynchronously.

    - parse: parse the tool call from the generated response.

    """

    def __init__(
        self,
        model_name_or_path,
        template: str = None,
        system_prompt: str = None,
        tools: List = None,
        max_model_len: int = None,
        backend: str = "async_vllm",
        backend_config: Any = None,
        reward_fn: Callable = None,
        streaming: str = "console",
        debug: bool = False,
        monitors: List[str] = ["wandb"],
        wandb_project_name: str = None,
        wandb_run_name: str = None,
        local_cache_dir: str = None,
        tool_parser: Optional[Any] = None,
        tool_parser_name: Optional[str] = None,
        **kwargs,  # To pass other unused arguments
    ):
        """
        Args:
            model_name_or_path: The name of the model to use.
            template: The template to use for the agent.
            system_prompt: The system prompt to use for the agent.
            tools: The tools to use for the agent.
            debug: Whether to enable debug mode.
            backend: The backend to use for the agent.
            tool_parser: Optional tool parser instance from vLLM. If provided, will be used for parsing tool calls.
            tool_parser_name: Optional name of the tool parser to use (e.g., "hermes", "pythonic").
                             If provided and tool_parser is None, will create a parser using this name.
        """
        self._validate_init_args(
            model_name_or_path,
            template,
            system_prompt,
            tools,
            backend,
            backend_config,
            reward_fn,
            streaming,
            debug,
            monitors,
            wandb_project_name,
            wandb_run_name,
            local_cache_dir,
            tool_parser,
            tool_parser_name,
        )

        self.debug = debug
        self.backend = backend
        self.template = template
        self.tools = tools
        self.max_model_len = max_model_len

        self.tool_names = [tool.name for tool in tools]
        if isinstance(system_prompt, str):
            system_prompt = system_prompt.replace("\\n", "\n")
        self.system_prompt = system_prompt
        self.model_name_or_path = model_name_or_path

        # Handle backend configuration
        if backend_config is None:
            # Use default configuration for the backend
            config_class = BACKEND_CONFIGS.get(backend)
            if config_class:
                self.backend_config = config_class()
            else:
                self.backend_config = None
        else:
            self.backend_config = backend_config

        self.llm_engine = self._init_llm_engine(model_name_or_path, backend)

        # Create appropriate tokenizer for trajectory processing
        self.tokenizer = create_tokenizer(model_name_or_path)
        self.processor = create_processor(model_name_or_path)

        self._reward_fn = reward_fn

        if self.template is None:
            self.jinja_template = None
        else:
            self.jinja_template = get_template(self.template).jinja_template()

        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.local_cache_dir = local_cache_dir
        self.local_run_cache_dir = None
        self._initialize_monitor(monitors)

        self.streaming_manager = StreamingManager()
        if streaming == "console":
            self.streaming_manager.add_observer(ConsoleStreamObserver())
        else:
            # TODO: Support other streaming modes
            raise ValueError(f"Streaming mode {streaming} is not supported.")

        # Initialize tool parser
        self.tool_parser = tool_parser
        if self.tool_parser is None and tool_parser_name is not None:
            if not VLLM_TOOL_PARSER_AVAILABLE:
                raise ImportError(
                    "vLLM tool parser is not available. Please install vllm to use tool_parser_name."
                )
            ParserCls = ToolParserManager.get_tool_parser(tool_parser_name)
            self.tool_parser = ParserCls(self.tokenizer)

        super().__init__()
        if kwargs:
            raise ValueError(f"Unused arguments for agent: {kwargs}")

    def _validate_init_args(
        self,
        model_name_or_path,
        template,
        system_prompt,
        tools,
        backend,
        backend_config,
        reward_fn,
        streaming,
        debug,
        monitors,
        wandb_project_name,
        wandb_run_name,
        local_cache_dir,
        tool_parser,
        tool_parser_name,
    ):
        if backend == "client":
            assert template is None, (
                "For client backend, we do not support chat template. Set the template when deploying the model."
            )
        if backend == "async_vllm":
            assert template is not None, (
                "For async vllm backend, chat template is required."
            )
        if tool_parser is not None and tool_parser_name is not None:
            raise ValueError(
                "Cannot specify both tool_parser and tool_parser_name. Use only one."
            )

    def _bind_method_tools(self):
        tool_methods = []
        for name, method in inspect.getmembers(self):
            if isinstance(method, BaseTool):
                tool_methods.append(method)
        for tool_method in tool_methods:
            if hasattr(tool_method, "is_method") and tool_method.is_method:
                tool_method.instance = self

    def _init_llm_engine(self, model_name_or_path: str, backend: str):
        assert not (self.template and backend == "client"), (
            "For client backend, we do not support template. Set the template when deploying the model."
        )
        if isinstance(model_name_or_path, str):
            # Extract backend-specific configuration
            config_kwargs = {}
            if self.backend_config:
                config_kwargs = {
                    k: v
                    for k, v in self.backend_config.__dict__.items()
                    if not k.startswith("_")
                }

            if backend == "transformers":
                llm_engine = TransformersBackend(
                    model_name_or_path, self.template, **config_kwargs
                )
            elif backend == "async_vllm":
                llm_engine = AsyncVLLMBackend(
                    model_name_or_path, self.template, **config_kwargs
                )
            elif backend == "async_verl":
                llm_engine = AsyncVerlBackend(
                    llm_engine=None,
                    model_name_or_path=model_name_or_path,
                    template=self.template,
                    **config_kwargs,
                )
            elif backend == "client":
                llm_engine = ClientBackend(
                    model_name_or_path, self.template, **config_kwargs
                )
            else:
                raise ValueError(f"Backend {backend} is not supported.")
        else:
            raise ValueError("model_name_or_path must be a string.")

        return llm_engine

    def _preprocess_messages(self, messages: List[Dict]):
        """
        Do some necessary preprocessings to the messages, such as adding the sytem prompt
        Args:
            messages: List of messages to preprocess.

        Returns:
            List of preprocessed messages.
        """
        messages_list = MessagesList.from_data(messages)
        for messages in messages_list:
            if self.system_prompt:
                messages.set_system_prompt(self.system_prompt, enforce=False)

        return messages_list.to_list()

    def _preprocess_backends(self):
        self.llm_engine.preprocess()

    def _postprocess_backends(self):
        self.llm_engine.postprocess()

    def _initialize_monitor(self, monitors: List[str]) -> None:
        for monitor in monitors:
            if monitor == "local":
                assert self.local_cache_dir is not None, (
                    "local_cache_dir must be set when using local monitor."
                )
                self.local_run_cache_dir = f"{os.path.join(self.local_cache_dir, os.path.basename(self.model_name_or_path), datetime.now().strftime('%Y%m%d_%H%M%S'))}"
                Monitor.add_sink("jsonl", JsonlSink(f"{self.local_run_cache_dir}/"))
            elif monitor == "wandb":
                Monitor.add_sink(
                    "wandb",
                    WandbSink(
                        project=self.wandb_project_name, run_name=self.wandb_run_name
                    ),
                )
            else:
                raise ValueError(f"Monitor {monitor} is not supported.")

    async def run(
        self,
        messages: Union[List[dict], np.ndarray, Dict],
        max_turns: int,
        generation_config: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        """
        This is the main interface for running the agent. It is a wrapper of different
        rollout methods, which must be asynchronous. Currently, we only support chain-based rollout.
        Args:
            messages: List of messages to generate responses for.
            max_turns: The maximum number of turns to generate.
            generation_config: The generation configuration.
            **kwargs: Additional keyword arguments for generation.

        """
        processed_messages = self._preprocess_messages(messages)
        self._preprocess_backends()

        await self.run_async(
            processed_messages,
            max_turns=max_turns,
            generation_config=generation_config,
            **kwargs,
        )

        self._postprocess_backends()

    def set_llm_engine(self, llm_engine: Any, tokenizer: Any, processor: Any):
        assert self.backend == "async_verl", (
            "Only async verl backend is supported for now"
        )

        self.llm_engine.llm_engine = llm_engine
        self.tokenizer = tokenizer
        self.processor = processor

    def generate(self, messages_list_or_inputs: List[List[Dict]], **kwargs):
        return self.llm_engine.generate(messages_list_or_inputs, **kwargs)

    async def generate_async(self, messages_list_or_inputs: List[List[Dict]], **kwargs):
        """
        Generate responses asynchronously. This method is used to generate responses for a list of messages. In a customized agent, this method can be overridden to implement more complex generation logic. For example, retrieve some relevant context from the database.

        Args:
            messages_list_or_inputs: List of messages to generate responses for.
            **args: Additional arguments for generation.

        Returns:
            List of responses.
        """
        return await self.llm_engine.generate_async(messages_list_or_inputs, **kwargs)

    async def generate_streaming(
        self, messages_list_or_inputs: List[List[Dict]], **kwargs
    ):
        """
        Generate responses with streaming support. This method yields response chunks as they are generated.

        Args:
            messages_list_or_inputs: List of messages to generate responses for.
            **args: Additional arguments for generation.

        Yields:
            str: Response chunks as they are generated.
        """
        if hasattr(self.llm_engine, "generate_streaming"):
            async for chunk in self.llm_engine.generate_streaming(
                messages_list_or_inputs, **kwargs
            ):
                yield chunk
        else:
            # Fallback to non-streaming generation
            responses = await self.generate_async(messages_list_or_inputs, **kwargs)
            for response in responses:
                yield response

    @property
    def timing_data(self):
        return self.timer.timing_data

    @property
    def trajectories(self):
        trajectories = self.get_messages()

        return trajectories

    def tokenize_trajectories(
        self,
        template=None,
        tokenizer=None,
        return_reward_mask: bool = False,
        concatenate_mm_inputs: bool = True,
    ):
        if tokenizer is None:
            tokenizer = self.tokenizer

        trajectories = self.trajectories
        messages_list = []
        other_info_list = []
        for trajectory in trajectories:
            messages = trajectory["messages"]
            messages_list.append(messages)
            have_called_tool = False
            for message in messages:
                if message["role"] == "tool":
                    have_called_tool = True
                    break
            info = {}
            for key, value in trajectory.items():
                if key != "messages":
                    info[key] = value
            info["have_called_tool"] = have_called_tool

            last_response = None

            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                if message["role"] == "assistant":
                    last_message = message
                    last_response = last_message["content"][0]["text"]
                    break

            info["last_response"] = last_response
            other_info_list.append(info)

        inputs = tokenize_conversations(
            messages_list,
            tokenizer=tokenizer,
            template=template or self.template,
            processor=self.processor,
            max_length=self.max_model_len,
            return_reward_mask=return_reward_mask,
            add_generation_prompt=True,
            concatenate_mm_inputs=concatenate_mm_inputs,
        )
        position_ids = torch.clip(
            torch.cumsum(inputs["attention_mask"], dim=-1) - 1, min=0, max=None
        )
        inputs["position_ids"] = position_ids

        assert inputs["input_ids"].shape[0] == len(other_info_list)

        return inputs, other_info_list

    def extract_final_response(self, messages: List[Dict[str, Any]]) -> str:
        last_message_content = messages[-1]["content"][0]["text"]
        last_message_role = messages[-1]["role"]
        # First try extracting the response if it is returned from a tool
        if last_message_role == "assistant":
            return last_message_content
        elif last_message_role == "tool":
            return last_message_content
        else:
            raise ValueError(
                f"The last message role must be assistant or tool, but got {last_message_role}"
            )

    def parse(self, responses: List[str]) -> List[Dict]:
        """
        This method is used to define the interaction logic of the agent. It can be used to parse the tool call from the response.
        If tool_parser is provided, it will use the vLLM tool parser by default. Otherwise, subclasses should override this method.

        Args:
            responses: List of responses to parse.
            **args: Additional arguments for parsing.

        Returns:
            messages: Assistant messages in the following format:

        ```python
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "..."
                    },
                ],
                "tool_calls": [
                    {
                        "id": "...",
                        "type": "function",
                        "function": {
                            "name": "...",
                            "arguments": "..."
                        }
                    }
                ]
            }
        ]
        ```
        """
        # If tool_parser is available, use it
        if self.tool_parser is not None:
            return self._parse_with_tool_parser(responses)
        else:
            # If no tool_parser, raise NotImplementedError to force subclasses to implement
            raise NotImplementedError(
                "parse method must be implemented by subclass or tool_parser must be provided. "
                "Either override this method or provide tool_parser/tool_parser_name in __init__."
            )

    def _parse_with_tool_parser(self, responses: List[str]) -> List[Dict]:
        """
        Parse responses using vLLM tool parser.

        Args:
            responses: List of response strings to parse.
            tools: List of tool objects.
            **args: Additional arguments.

        Returns:
            List of assistant messages with tool_calls.
        """
        if not VLLM_TOOL_PARSER_AVAILABLE:
            raise ImportError("vLLM tool parser is not available. Please install vllm.")

        # Convert tools to vLLM format (tool.schema is already in OpenAI format)
        tool_schemas = []
        if self.tools:
            for tool in self.tools:
                if tool is None:
                    continue
                schema = tool.schema
                # tool.schema is already in the format: {"type": "function", "function": {...}}
                if isinstance(schema, dict):
                    tool_schemas.append(schema)
                else:
                    Logger.warning(
                        f"Tool {getattr(tool, 'name', 'unknown')} has invalid schema format: {type(schema)}"
                    )
                    continue

        new_messages_list = []
        for response in responses:
            # Create a ChatCompletionRequest for the parser
            # We use a minimal request structure
            req_dict = {
                "messages": [
                    {"role": "user", "content": "dummy"}
                ],  # Dummy message, not used for parsing
                "tool_choice": "auto",
            }
            if tool_schemas:
                req_dict["tools"] = tool_schemas

            req = ChatCompletionRequest(**req_dict)

            # Adjust request (some parsers may modify it)
            req = self.tool_parser.adjust_request(req)

            # Extract tool calls from the response
            info = self.tool_parser.extract_tool_calls(response, req)

            # Format tool calls to match our expected format
            formatted_tool_calls = []
            if info.tool_calls:
                for tool_call in info.tool_calls:
                    # tool_call is a vLLM ToolCall object with attributes: id, type, function
                    # function is a FunctionCall object with attributes: name, arguments
                    if hasattr(tool_call, "function") and hasattr(
                        tool_call.function, "name"
                    ):
                        # Handle ToolCall object from vLLM
                        arguments_str = tool_call.function.arguments
                        # Validate that arguments is a valid JSON string
                        try:
                            json.loads(arguments_str)
                            # If valid JSON, append the tool call
                            formatted_tool_calls.append(
                                {
                                    "id": getattr(tool_call, "id", None),
                                    "type": getattr(tool_call, "type", "function"),
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": arguments_str,  # Already a JSON string
                                    },
                                }
                            )
                        except (json.JSONDecodeError, TypeError):
                            # Invalid JSON, skip this tool call
                            # Logger.warning(f"Invalid JSON in tool call arguments for {tool_call.function.name}: {arguments_str}")
                            continue
                    elif isinstance(tool_call, dict):
                        # Fallback: handle dictionary format (for compatibility)
                        if "function" in tool_call:
                            func_info = tool_call["function"]
                            arguments_str = (
                                func_info.get("arguments", "")
                                if isinstance(func_info, dict)
                                else getattr(func_info, "arguments", "")
                            )
                            # Validate that arguments is a valid JSON string
                            try:
                                json.loads(arguments_str)
                                # If valid JSON, append the tool call
                                formatted_tool_calls.append(
                                    {
                                        "id": tool_call.get("id", None),
                                        "type": "function",
                                        "function": {
                                            "name": func_info.get("name", "")
                                            if isinstance(func_info, dict)
                                            else getattr(func_info, "name", ""),
                                            "arguments": arguments_str,
                                        },
                                    }
                                )
                            except (json.JSONDecodeError, TypeError):
                                # Invalid JSON, skip this tool call
                                tool_name = (
                                    func_info.get("name", "")
                                    if isinstance(func_info, dict)
                                    else getattr(func_info, "name", "unknown")
                                )
                                Logger.warning(
                                    f"Invalid JSON in tool call arguments for {tool_name}: {arguments_str}"
                                )
                                continue

            # Use the full response text (not the text after removing tool calls)
            content_text = response

            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                "tool_calls": formatted_tool_calls,
                "loss": True,
            }

            # Add status if available
            if hasattr(info, "status"):
                message["status"] = info.status
            elif len(formatted_tool_calls) > 0:
                message["status"] = "continue"
            else:
                message["status"] = "terminal"

            new_messages_list.append(message)

        return new_messages_list

    @property
    def rewards(self):
        messages_list = []
        # answers = []
        reward_values = []
        other_values = defaultdict(list)
        for trajectory in self.trajectories:
            messages = trajectory["messages"]
            messages_list.append(messages)
            reward_value_or_dict = trajectory["reward"]

            if isinstance(reward_value_or_dict, dict):
                reward_values.append(reward_value_or_dict["reward"])
                for key, value in reward_value_or_dict.items():
                    if key != "reward":
                        other_values[key].append(value)
            else:
                reward_values.append(reward_value_or_dict)

        return reward_values, other_values

    def print_messages(self, index: int = 0):
        messages = self.get_messages()
        for message in messages[index]["messages"]:
            role = message["role"]
            text = f"{role}: "
            if "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    text += content
                elif isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            text += item["text"]
                        elif item["type"] == "image":
                            text += colored("ImagePlaceholder", "red")
                elif content is None:
                    assert role == "assistant", (
                        f"Invalid content type: {type(content)} for role {role}"
                    )
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                        for tool_call in tool_calls:
                            text += f"Tool call: {tool_call['name']} Arguments: {tool_call['arguments']}"
                    else:
                        raise ValueError(
                            f"Invalid message: {message} must have content or tool_calls."
                        )
            print(text)

    def get_verl_data_proto(self):
        inputs, other_info_list = self.tokenize_trajectories(
            return_reward_mask=True, concatenate_mm_inputs=False
        )
        group_ids_list = [info["group_id"] for info in other_info_list]
        group_ids = np.array(group_ids_list, dtype=object)
        batch_size = len(group_ids_list)
        unique_group_ids = []
        seen_group_ids = set()
        for group_id in group_ids_list:
            if group_id not in seen_group_ids:
                unique_group_ids.append(group_id)
                seen_group_ids.add(group_id)
        # Do evaluation here
        reward_values, other_values = self.rewards
        inputs["rm_scores"] = inputs["reward_mask"] * torch.tensor(
            reward_values, dtype=torch.float32
        ).unsqueeze(dim=-1)  # BS x L
        # Handle other values as np.array
        for key, values in other_values.items():
            aligned_values = list(values)
            if len(aligned_values) == len(unique_group_ids) and unique_group_ids:
                group_to_value = {
                    group_id: aligned_values[idx]
                    for idx, group_id in enumerate(unique_group_ids)
                }
                aligned_values = [
                    group_to_value[group_id] for group_id in group_ids_list
                ]
            elif len(aligned_values) == 1 and batch_size > 1:
                aligned_values = aligned_values * batch_size
            if len(aligned_values) != batch_size:
                self.logger.warning(
                    f"Adjusting rm_{key} length from {len(aligned_values)} to {batch_size} to match batch size."
                )
                if len(aligned_values) < batch_size:
                    aligned_values = aligned_values + [0.0] * (
                        batch_size - len(aligned_values)
                    )
                else:
                    aligned_values = aligned_values[:batch_size]
            inputs[f"rm_{key}"] = np.array(aligned_values)
        # We handle the group id in the agent side, to be compatible with GRPO
        inputs["uid"] = group_ids

        if "mm_inputs" in inputs:
            mm_inputs = inputs.pop("mm_inputs")
            inputs["multi_modal_inputs"] = np.array(mm_inputs, dtype=object)
        batch = DataProto.from_single_dict(inputs, meta_info={"use_agent": True})

        return batch
