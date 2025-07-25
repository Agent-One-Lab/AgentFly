from abc import ABC, abstractmethod
from collections import defaultdict
import json

from .templates.templates import get_template
from ..__init__ import AGENT_DATA_DIR
from .llm_backend import AsyncVLLMBackend, AsyncVerlBackend, ClientBackend, TransformersBackend, VLLMBackend, VerlBackend
from ..utils.logging import get_logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from .templates.utils import is_vlm_template, tokenize_conversations
from .chain.chain_base import ChainGeneration
import os
import transformers
import warnings
try:
    from verl.protocol import DataProto
except ImportError:
    print("verl can not be imported.")
    pass


class BaseAgent(ChainGeneration, ABC):
    """
    Base class for all agents. All agent should subclass this class. A customized agent can implement the following methods:
    
    - generate_async: generate responses asynchronously.

    - parse: parse the tool call from the generated response.

    """
    def __init__(
        self,
        model_name_or_path, 
        template: str,
        system_prompt: str = None,
        tools: List = None,
        max_length: int=8192,
        debug: bool = False,
        backend: str = "transformers",
        reward_fn: Callable = None,
        log_file: str = "agent",
        project_name: str = None,
        run_name: str = None,
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
        self.backend = backend
        self.template = template
        self.max_length = max_length
        self.tools = tools
        self.system_prompt = system_prompt
        self.model_name_or_path = model_name_or_path
        self.llm_engine, self.tokenizer, self.processor = self._init_llm_engine(model_name_or_path, backend)
        self._reward_fn = reward_fn
        self.jinja_template = get_template(self.template).jinja_template()
        self.project_name = project_name
        self.run_name = run_name
        super().__init__()
        if kwargs:
            warnings.warn(f"Unused arguments for agent initialization: {kwargs}")
    
    def _init_llm_engine(self, model_name_or_path: str, backend: str):
        if isinstance(model_name_or_path, str):
            if backend == "transformers":
                llm_engine = TransformersBackend(model_name_or_path, self.template, max_length=self.max_length)
            elif backend == "vllm":
                llm_engine = VLLMBackend(model_name_or_path, self.template, max_length=self.max_length)
            elif backend == "async_vllm":
                llm_engine = AsyncVLLMBackend(model_name_or_path, self.template, max_length=self.max_length)
            elif backend == "verl":
                llm_engine = VerlBackend(llm_engine=None, model_name_or_path=model_name_or_path, template=self.template, max_length=self.max_length)
            elif backend == "async_verl":
                llm_engine = AsyncVerlBackend(llm_engine=None, model_name_or_path=model_name_or_path, template=self.template, max_length=self.max_length)
            elif backend == "client":
                llm_engine = ClientBackend(model_name_or_path, self.template, max_length=self.max_length)
            else:
                raise ValueError(f"Backend {backend} is not supported.")
        else:
            raise ValueError("model_name_or_path must be a string.")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        if is_vlm_template(self.template):
            processor = transformers.AutoProcessor.from_pretrained(model_name_or_path)
        else:
            processor = None
        return llm_engine, tokenizer, processor

    def set_llm_engine(self, llm_engine: Any, tokenizer: Any):
        assert self.backend == "async_verl", "Only async verl backend is supported for now"
        self.llm_engine.llm_engine = llm_engine
        self.tokenizer = tokenizer
        
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

    @property
    def timing_data(self):
        return self.timer.timing_data
    
    def forward(self, messages_list_or_inputs: List[List[Dict]], **args):
        if isinstance(messages_list_or_inputs, List):
            inputs = tokenize_conversations(messages_list_or_inputs, tokenizer=self.tokenizer, conv_template=self.template, max_length=self.max_length, processor=self.processor)
        else:
            raise ValueError("messages_list_or_inputs must be a list of messages or a dictionary of padded inputs.")
        
        if isinstance(self.llm_engine, transformers.PreTrainedModel):
            return self.llm_engine.forward(**inputs, **args) # Only support transformers models for now.
        else:
            raise ValueError("llm_engine must be a transformers.PretrainedModel.")

    @property
    def trajectories(self):
        trajectories = self.get_messages()

        return trajectories

    def tokenize_trajectories(self, return_action_mask: bool = False, return_reward_mask: bool = False):
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

        inputs = tokenize_conversations(messages_list, tokenizer=self.tokenizer, conv_template=self.template, processor=self.processor, max_length=self.max_length, return_reward_mask=return_reward_mask)
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
            try:
                response = json.loads(last_message_content)
                if "content" in response:
                    return response["content"]
            except json.JSONDecodeError:
                return last_message_content

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
    

    def get_verl_data_proto(self):
        inputs, other_info_list = self.tokenize_trajectories(return_action_mask=True, return_reward_mask=True)
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
        batch = DataProto.from_single_dict(inputs, meta_info={"use_agent": True})

        return batch
 