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
from chat_bricks import (
    get_template,
    split_messages_with_assistant,
    tokenize_conversations,
)
from termcolor import colored

from ..templates import *  # noqa: F403
from ..tools.tool_base import BaseTool
from ..utils.monitor import JsonlSink, Monitor, WandbSink
from ..utils.verl import pad_tensor_to_rank_size
from .chain.chain_base import ChainRollout
from .chain.streaming_observer import ConsoleStreamObserver, StreamingManager
from .llm_backends import AsyncVerlBackend, AsyncVLLMBackend, ClientBackend
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

logger = logging.getLogger(__name__)


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
        tools: List = [],
        max_model_len: int = None,
        backend_config: Optional[Dict[str, Any]] = None,
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
            backend_config: Dict specifying the backend and its parameters. Must include "backend" (e.g. "async_vllm", "client").
                Other keys are passed as kwargs to that backend (e.g. "gpu_memory_utilization" for async_vllm).
                Defaults to {"backend": "async_vllm"}.
            tool_parser: Optional tool parser instance from vLLM. If provided, will be used for parsing tool calls.
            tool_parser_name: Optional name of the tool parser to use (e.g., "hermes", "pythonic"). If provided and tool_parser is None, will create a parser using this name.

        """
        if backend_config is None:
            backend_config = {"backend": "async_vllm"}
        self._validate_init_args(
            model_name_or_path,
            template,
            system_prompt,
            tools,
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
        self.backend_config = backend_config
        self.backend = backend_config["backend"]
        self.tools = tools
        self.max_model_len = max_model_len
        self.tool_names = [tool.name for tool in tools]

        if isinstance(system_prompt, str):
            system_prompt = system_prompt.replace("\\n", "\n")
        self.system_prompt = system_prompt
        self.model_name_or_path = model_name_or_path

        # Create appropriate tokenizer for trajectory processing
        self.tokenizer = create_tokenizer(model_name_or_path)
        self.processor = create_processor(model_name_or_path)

        self._reward_fn = reward_fn

        # We use model name as template if no template is provided
        # For a model name, chat-bricks will use HF's template by default
        if template:
            self.template = template
        else:
            self.template = self.model_name_or_path

        if self.template is None:
            self.jinja_template = None
        else:
            self.jinja_template = get_template(self.template).jinja_template()

        self.llm_engine = self._init_llm_engine(model_name_or_path, self.backend)

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
        if not isinstance(backend_config, dict) or "backend" not in backend_config:
            raise ValueError(
                "backend_config must be a dict with at least a 'backend' key (e.g. {'backend': 'async_vllm'})."
            )
        backend = backend_config["backend"]
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
        if isinstance(model_name_or_path, str):
            # Backend params: all keys in backend_config except "backend". Optionally merge with default config.
            config_kwargs = {k: v for k, v in self.backend_config.items() if k != "backend"}
            config_class = BACKEND_CONFIGS.get(backend)
            if config_class:
                try:
                    default_instance = config_class()
                    default_dict = {
                        k: v
                        for k, v in default_instance.__dict__.items()
                        if not k.startswith("_")
                    }
                    default_dict.update(config_kwargs)
                    config_kwargs = default_dict
                except Exception:
                    pass

            if backend == "async_vllm":
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
                # ClientBackend(model_name_or_path, base_url=..., ); no template as 2nd arg
                llm_engine = ClientBackend(model_name_or_path, **config_kwargs)
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
        tools = [tool.schema for tool in self.tools]
        if self.system_prompt and "{tools}" in self.system_prompt:
            system_prompt = self.system_prompt.replace(
                "{tools}", json.dumps(tools, indent=4)
            )
        else:
            system_prompt = self.system_prompt

        for messages in messages_list:
            if system_prompt:
                messages.set_system_prompt(system_prompt, enforce=False)

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

    def postprocess_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Preprocess the trajectories of the agent.
        """
        return trajectories

    @property
    def trajectories(self):
        """Get the trajectories of the agent."""

        trajectories = self.get_trajectories()
        trajectories = self.postprocess_trajectories(trajectories)
        return trajectories

    def tokenize_trajectories(
        self,
        messages_list,
        template=None,
        tokenizer=None,
        processor=None,
        return_reward_mask: bool = False,
        concatenate_mm_inputs: bool = True,
        train_on_last_turn: bool = False,
    ):

        # TODO: we will remove this argument in the future
        train_on_last_turn = False

        inputs = tokenize_conversations(
            messages_list,
            tokenizer=tokenizer,
            template=template or self.template,
            processor=processor or self.processor,
            max_length=self.max_model_len,
            return_reward_mask=return_reward_mask,
            add_generation_prompt=True,
            concatenate_mm_inputs=concatenate_mm_inputs,
            ignore_tool_calls=True,
            train_on_last_turn_only=train_on_last_turn,
        )
        position_ids = torch.clip(
            torch.cumsum(inputs["attention_mask"], dim=-1) - 1, min=0, max=None
        )
        inputs["position_ids"] = position_ids

        return inputs

    def extract_final_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract the final response text from a trajectory.

        We scan messages in reverse order and take the last assistant/tool
        message as the final response.
        """
        for msg in reversed(messages):
            last_message_role = msg.get("role")
            if last_message_role not in ("assistant", "tool"):
                continue
            content = msg.get("content") or []
            if (
                isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and content[0].get("type") == "text"
            ):
                return content[0].get("text", "")
            # Fallback: stringify first content part if structure is unexpected
            if isinstance(content, list) and content:
                return str(content[0])

        raise ValueError(
            "No assistant or tool message found in trajectory when extracting final response."
        )

    def parse(self, responses: List[str], current_segments: List[List[Dict]]) -> List[Dict]:
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
                    logger.warning(
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
                            # logger.warning(f"Invalid JSON in tool call arguments for {tool_call.function.name}: {arguments_str}")
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
                                logger.warning(
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
        reward_values = []
        other_values = defaultdict(list)
        for trajectory in self.trajectories:
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

    def get_verl_data_proto(
        self,
        train_on_last_turn: bool = False,
        world_size: int = 1,
        reward_on_last_segment_only: bool = False,
    ):
        trajectories = self.trajectories
        segments_list = []
        other_info_list = []
        for batch_idx, trajectory in enumerate(trajectories):
            trajectory_segments = trajectory["trajectory_segments"]
            for segment_idx, segment in enumerate(trajectory_segments):
                segments_list.append(segment)
                info = {
                    key: value for key, value in trajectory.items() if key != "trajectory_segments"
                }
                info["batch_idx"] = batch_idx
                info["segment_idx"] = segment_idx
                other_info_list.append(info)

        inputs = self.tokenize_trajectories(
            messages_list=segments_list,
            tokenizer=self.tokenizer,
            processor=self.processor,
            return_reward_mask=True,
            concatenate_mm_inputs=False,
            train_on_last_turn=train_on_last_turn,
        )

        reward_values, other_values = self.rewards

        # Expand to segment level:
        # - Default: trajectory i has n_i segments -> repeat reward i for all n_i segments.
        # - If reward_on_last_segment_only=True: use 0.0 for first n_i-1 segments, reward i only on the last.
        reward_on_last_segment_only = True
        reward_values_segment: List[float] = []
        for trajectory in trajectories:
            n = len(trajectory["trajectory_segments"])
            r = trajectory["reward"]
            if isinstance(r, dict):
                r = r["reward"]
            if reward_on_last_segment_only:
                if n <= 0:
                    continue
                reward_values_segment.extend([0.0] * (n - 1))
                reward_values_segment.append(r)
            else:
                reward_values_segment.extend([r] * n)
        num_trajectories = len(trajectories)
        other_values_segment = {}
        for key, values in other_values.items():
            # When reward is scalar (not dict), that trajectory never appends to other_values[key],
            # so values may be shorter than num_trajectories. Pad to match.
            if len(values) < num_trajectories:
                values = list(values) + [0.0] * (num_trajectories - len(values))
            other_values_segment[key] = []
            for traj_idx, trajectory in enumerate(trajectories):
                n = len(trajectory["trajectory_segments"])
                val = values[traj_idx]
                other_values_segment[key].extend([val] * n)
        reward_values = reward_values_segment
        other_values = other_values_segment

        # Number of segments per trajectory (one integer per trajectory).
        repeat_times = [len(t["trajectory_segments"]) for t in trajectories]

        if world_size > 1:
            pad_size = (
                world_size - inputs["input_ids"].shape[0] % world_size
            ) % world_size
            for k, v in inputs.items():
                inputs[k] = pad_tensor_to_rank_size(v, world_size)
            if pad_size > 0:
                # Pad other_info_list with copies of the last element (matches last-row repeat in pad_tensor_to_rank_size)
                other_info_list = other_info_list + [other_info_list[-1]] * pad_size
                # Pad reward_values and other_values to match the padded inputs
                reward_values = reward_values + [reward_values[-1]] * pad_size
                other_values = {
                    k: v + [v[-1]] * pad_size for k, v in other_values.items()
                }
                # Add pad_size to the last trajectory's segment count
                repeat_times[-1] += pad_size

        group_ids_list = [info["group_id"] for info in other_info_list]
        segment_index_list = [info["segment_idx"] for info in other_info_list]
        batch_index_list = [info["batch_idx"] for info in other_info_list]
        discarded_segment_list = [bool(info.get("discarded", False)) for info in other_info_list]
        group_ids = np.array(group_ids_list, dtype=object)
        segment_index = np.array(segment_index_list, dtype=np.int32)
        batch_index = np.array(batch_index_list, dtype=np.int32)


        batch_size = len(group_ids_list)
        unique_group_ids = []
        seen_group_ids = set()
        for group_id in group_ids_list:
            if group_id not in seen_group_ids:
                unique_group_ids.append(group_id)
                seen_group_ids.add(group_id)

        # For discarded trajectories, mask out all response tokens for every segment row.
        if discarded_segment_list:
            discarded_tensor = torch.tensor(
                discarded_segment_list, dtype=torch.bool, device=inputs["attention_mask"].device
            ).unsqueeze(dim=-1)
            if "action_mask" in inputs:
                inputs["action_mask"] = inputs["action_mask"] * (~discarded_tensor).to(
                    dtype=inputs["action_mask"].dtype
                )
            if "reward_mask" in inputs:
                inputs["reward_mask"] = inputs["reward_mask"] * (~discarded_tensor).to(
                    dtype=inputs["reward_mask"].dtype
                )

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
                logger.warning(
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
        inputs["segment_idx"] = segment_index
        inputs["batch_idx"] = batch_index
        
        if "mm_inputs" in inputs:
            mm_inputs = inputs.pop("mm_inputs")
            inputs["multi_modal_inputs"] = np.array(mm_inputs, dtype=object)
        batch = DataProto.from_single_dict(
            inputs, meta_info={"use_agent": True, "repeat_times": repeat_times}
        )

        return batch
