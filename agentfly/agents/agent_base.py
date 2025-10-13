from abc import ABC, abstractmethod
from collections import defaultdict
import inspect
from datetime import datetime
import json
from ..tools.tool_base import Tool
from .utils.messages import MessagesList
from ..templates.templates import get_template
from .. import AGENT_DATA_DIR
from .llm_backends import (
    AsyncVLLMBackend,
    AsyncVerlBackend,
    ClientBackend,
    TransformersBackend,
)
from termcolor import colored
from .llm_backends.backend_configs import BACKEND_CONFIGS
from ..utils.logging import get_logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..templates import tokenize_conversations
from .chain.chain_base import ChainRollout
import os
import transformers
import warnings
import logging
from .chain.streaming_observer import ConsoleStreamObserver, StreamingManager
from .utils.tokenizer import create_processor, create_tokenizer
from ..utils.monitor import JsonlSink, Monitor, WandbSink
try:
    from verl.protocol import DataProto
except ImportError:
    print("verl can not be imported.")
    pass

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
        template: str=None,
        system_prompt: str = None,
        tools: List = None,
        max_length: int=None,
        backend: str = "async_vllm",
        backend_config: Any = None,
        reward_fn: Callable = None,
        log_file: str = "agent",
        streaming: str = "console",
        debug: bool = False,
        monitors: List[str] = [],
        wandb_project_name: str = None,
        wandb_run_name: str = None,
        local_cache_dir: str = None,
        **kwargs # To pass other unused arguments
    ):
        """
        Args:
            model_name_or_path: The name of the model to use.
            template: The template to use for the agent.
            system_prompt: The system prompt to use for the agent.
            tools: The tools to use for the agent.
            max_length: The maximum length of the response.
            debug: Whether to enable debug mode.
            backend: The backend to use for the agent.
        """
        torch.set_printoptions(threshold=10_000)
        self.logger = get_logger(directory=os.path.join(AGENT_DATA_DIR, "debug"), filename=log_file, level="DEBUG" if debug else "INFO")
        self.debug = debug
        self.backend = backend
        self.template = template
        # TODO: Make max_length aligned with training
        self.max_length = max_length
        self.tools = tools
        
        tool_methods = []
        for name, method in inspect.getmembers(self):
            if isinstance(method, Tool):
                tool_methods.append(method)
        for tool_method in tool_methods:
            if hasattr(tool_method, 'is_method') and tool_method.is_method:
                tool_method.instance = self
        
        self.tool_names = [tool.name for tool in tools]
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
        super().__init__()
        if kwargs:
            warnings.warn(f"Unused arguments for agent initialization: {kwargs}")
    
    def _init_llm_engine(self, model_name_or_path: str, backend: str):
        assert not (self.template and backend == "client"), "For client backend, we do not support template. Set the template when deploying the model."
        if isinstance(model_name_or_path, str):
            # Extract backend-specific configuration
            config_kwargs = {}
            if self.backend_config:
                config_kwargs = {k: v for k, v in self.backend_config.__dict__.items() 
                               if not k.startswith('_')}
            
            if backend == "transformers":
                llm_engine = TransformersBackend(
                    model_name_or_path, 
                    self.template, 
                    max_length=self.max_length,
                    **config_kwargs
                )
            elif backend == "async_vllm":
                llm_engine = AsyncVLLMBackend(
                    model_name_or_path, 
                    self.template, 
                    max_length=self.max_length,
                    **config_kwargs
                )
            elif backend == "async_verl":
                llm_engine = AsyncVerlBackend(
                    llm_engine=None, 
                    model_name_or_path=model_name_or_path, 
                    template=self.template, 
                    max_length=self.max_length,
                    **config_kwargs
                )
            elif backend == "client":
                print(f"config_kwargs: {config_kwargs}")
                llm_engine = ClientBackend(
                    model_name_or_path, 
                    self.template, 
                    max_length=self.max_length,
                    **config_kwargs
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

    def _initialize_monitor(self, monitors: List[str]) -> None:
        for monitor in monitors:
            if monitor == "local":
                assert self.local_cache_dir is not None, "local_cache_dir must be set when using local monitor."
                self.local_run_cache_dir = f"{os.path.join(self.local_cache_dir, os.path.basename(self.model_name_or_path), datetime.now().strftime('%Y%m%d_%H%M%S'))}"
                Monitor.add_sink("jsonl", JsonlSink(f"{self.local_run_cache_dir}/"))
            elif monitor == "wandb":
                Monitor.add_sink("wandb", WandbSink(project=self.wandb_project_name, run_name=self.wandb_run_name))
            else:
                raise ValueError(f"Monitor {monitor} is not supported.")

    async def run(self,
        messages: Union[List[dict], np.ndarray, Dict],
        max_turns: int,
        generation_config: Optional[Dict[str, Any]] = None,
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

        return await self.run_async(
            processed_messages,
            max_turns=max_turns,
            generation_config=generation_config,
            **kwargs,
        )

    def set_llm_engine(self, llm_engine: Any, tokenizer: Any, processor: Any):
        assert self.backend == "async_verl", "Only async verl backend is supported for now"

        self.llm_engine.llm_engine = llm_engine
        self.tokenizer = tokenizer
        self.processor = processor
        
    def generate(self, messages_list_or_inputs: List[List[Dict]], **args):
        return self.llm_engine.generate(messages_list_or_inputs, **args)

    async def generate_async(self, messages_list_or_inputs: List[List[Dict]], **args):
        """
        Generate responses asynchronously. This method is used to generate responses for a list of messages. In a customized agent, this method can be overridden to implement more complex generation logic. For example, retrieve some relevant context from the database.

        Args:
            messages_list_or_inputs: List of messages to generate responses for.
            **args: Additional arguments for generation.

        Returns:
            List of responses.
        """
        return await self.llm_engine.generate_async(messages_list_or_inputs, **args)
    
    async def generate_streaming(self, messages_list_or_inputs: List[List[Dict]], **kwargs):
        Logger.debug(f"[BaseAgent] generate_streaming kwargs: {kwargs}")
        """
        Generate responses with streaming support. This method yields response chunks as they are generated.

        Args:
            messages_list_or_inputs: List of messages to generate responses for.
            **args: Additional arguments for generation.

        Yields:
            str: Response chunks as they are generated.
        """
        if hasattr(self.llm_engine, 'generate_streaming'):
            async for chunk in self.llm_engine.generate_streaming(messages_list_or_inputs, **kwargs):
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

    def tokenize_trajectories(self, template = None, tokenizer = None, return_reward_mask: bool = False, concatenate_mm_inputs: bool = True):
        if tokenizer is None:
            tokenizer = self.tokenizer
            
        trajectories = self.trajectories
        self.logger.info("================ Trajectory ================")
        self.logger.info(trajectories[0])
        messages_list = []
        other_info_list = []
        for trajectory in trajectories:
            messages = trajectory["messages"]
            messages_list.append(messages)
            have_called_tool = False
            for message in messages:
                if message['role'] == 'tool':
                    have_called_tool = True
                    break
            info = {}
            for key, value in trajectory.items():
                if key != "messages":
                    info[key] = value
            info['have_called_tool'] = have_called_tool
            last_message = trajectory["messages"][-1]
            if last_message['role'] != 'assistant':
                last_message = trajectory["messages"][-2]
            assert last_message['role'] == 'assistant', f"The last message must be an assistant message, but got trajectory: {trajectory}"
            last_response = last_message['content'][0]['text']
            info['last_response'] = last_response
            other_info_list.append(info)

        inputs = tokenize_conversations(
            messages_list,
            tokenizer=tokenizer,
            template=template or self.template,
            processor=self.processor,
            max_length=self.max_length,
            return_reward_mask=return_reward_mask,
            add_generation_prompt=True,
            concatenate_mm_inputs=concatenate_mm_inputs,
        )
        position_ids = torch.clip(torch.cumsum(inputs['attention_mask'], dim=-1) - 1, min=0, max=None)
        inputs['position_ids'] = position_ids

        assert inputs['input_ids'].shape[0] == len(other_info_list)

        return inputs, other_info_list
    

    def extract_final_response(self, messages: List[Dict[str, Any]]) -> str:
        last_message_content = messages[-1]["content"][0]['text']
        last_message_role = messages[-1]["role"]
        # First try extracting the response if it is returned from a tool
        if last_message_role == "assistant":
            return last_message_content
        elif last_message_role == "tool":
            return last_message_content
        else:
            raise ValueError(f"The last message role must be assistant or tool, but got {last_message_role}")

    @abstractmethod
    def parse(self, responses: List[str], tools: List[Any], **args) -> Tuple[dict, int, int]:
        """
        This method is used to define the interaction logic of the agent. It can be used to parse the tool call from the response. In a customized agent, more complex interaction logic can be defined. For example, take a specific token as the tool call token.

        Args:
            responses: List of responses to parse.
            tools: List of tools to use.
            **args: Additional arguments for parsing.

        Returns:
            messages: Assistant messages in the following format:
            
            .. code-block:: python

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
                                "name": "...",
                                "arguments": "..."
                            }
                        ]
                    }
                ]
        """
        raise NotImplementedError
    
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
            if 'content' in message:
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
                    assert role == "assistant", f"Invalid content type: {type(content)} for role {role}"
                    if "tool_calls" in message:
                        tool_calls = message["tool_calls"]
                        for tool_call in tool_calls:
                            text += f"Tool call: {tool_call['name']} Arguments: {tool_call['arguments']}"
                    else:
                        raise ValueError(f"Invalid message: {message} must have content or tool_calls.")
            print(text)
    
    def get_verl_data_proto(self):
        inputs, other_info_list = self.tokenize_trajectories(return_reward_mask=True, concatenate_mm_inputs=False)
        group_ids = np.array([info["group_id"] for info in other_info_list], dtype=object)
        # Do evaluation here
        reward_values, other_values = self.rewards
        inputs["rm_scores"] = inputs["reward_mask"] * torch.tensor(reward_values).unsqueeze(dim=-1) # BS x L
        self.logger.info(f"reward_values: {reward_values}")
        # Handle other values as np.array
        for key, values in other_values.items():
            inputs[f"rm_{key}"] = np.array(values)
        # We handle the group id in the agent side, to be compatible with GRPO
        inputs["uid"] = group_ids
        
        if "mm_inputs" in inputs:
            mm_inputs = inputs.pop("mm_inputs")
            inputs["multi_modal_inputs"] = np.array(mm_inputs, dtype=object)
        batch = DataProto.from_single_dict(inputs, meta_info={"use_agent": True})

        return batch
 
