"""
LLM Backend module for reward functions.
This module provides a unified interface to different LLM implementations.
"""
import asyncio
from collections import deque
import copy
from functools import partial
import json
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
import uuid
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ... import AGENT_DATA_DIR
from ...utils.vision import image_to_data_uri
from ...utils.verl import pad_tensor_to_rank_size
from vllm import LLM, AsyncLLMEngine, SamplingParams, AsyncEngineArgs
import openai
from ...templates import Chat
import logging
import PIL


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

try:
    from verl.protocol import DataProto
    from verl.single_controller.ray.base import RayWorkerGroup
except ImportError:
    print("verl can not be imported.")
    pass

class LLMBackend:
    """Base class for LLM backends.
    
    This abstract base class provides a unified interface for different LLM implementations.
    All backend implementations must inherit from this class and implement the required methods.
    
    Attributes:
        config: Configuration dictionary containing backend-specific parameters.
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs

    def apply_chat_template(self, messages_list: List[List[Dict]], template: str, add_generation_prompt: bool=True, tools: List[Dict]=None) -> List[str]:
        """Apply chat template to messages list"""
        prompts = []
        vision_inputs = []
        for messages in messages_list:
            chat = Chat(template, messages)
            prompts.append(chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools))
            # We only support image inputs for now
            vision_inputs.append(chat.vision_inputs())

        return prompts, vision_inputs
    
    def generate(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt"""
        raise NotImplementedError("Subclasses must implement generate()")
    
    async def generate_streaming(self, messages_list: List[List[Dict]], streaming_callback: Optional[Callable] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming support"""
        raise NotImplementedError("Subclasses must implement generate_streaming()")

class TransformersBackend(LLMBackend):
    """HuggingFace Transformers implementation for local model inference.
    
    This backend uses the Hugging Face Transformers library to load and run models locally.
    It supports both synchronous and asynchronous text generation with streaming capabilities.
    """
    
    def __init__(self, model_name_or_path: str, template: str, max_length: int=None, temperature: float=1.0, max_new_tokens: int=1024, **kwargs):
        """Initialize TransformersBackend.
        
        Args:
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            max_length (int): Maximum sequence length for input/output. Defaults to 8192.
            temperature (float): Sampling temperature for text generation. Defaults to 1.0.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name_or_path
        self.max_length = max_length
        self.temperature = temperature
        self.template = template
        self.max_new_tokens = max_new_tokens
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
        )
        self.llm_engine = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True
        )
    
    def generate(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using Transformers"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        kwargs.update({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        })

        prompts, _ = self.apply_chat_template(messages_list, self.template)
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.llm_engine.device)
        input_length = inputs['input_ids'].shape[1]
        outputs = self.llm_engine.generate(
            **inputs,
            **kwargs
        )[:, input_length:]
        
        response_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return response_texts
    
    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Async wrapper for generate"""
        return self.generate(messages_list, **kwargs)
    
    async def generate_streaming(self, messages_list: List[List[Dict]], streaming_callback: Optional[Callable] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming support using Transformers"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        prompts, _ = self.apply_chat_template(messages_list, self.template)
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.llm_engine.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Use streaming generation
        generated_tokens = []
        for i in range(max_new_tokens):
            outputs = self.llm_engine.generate(
                **inputs,
                max_new_tokens=1,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            new_token = outputs[0][-1].unsqueeze(0)
            generated_tokens.append(new_token)
            
            # Decode the new token
            new_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
            
            if streaming_callback:
                await streaming_callback(new_text)
            
            yield new_text
            
            # Check for EOS
            if new_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Update input for next iteration
            inputs['input_ids'] = torch.cat([inputs['input_ids'], new_token.unsqueeze(0)], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(1, 1, device=inputs['attention_mask'].device)], dim=1)

class VLLMBackend(LLMBackend):
    """vLLM implementation for high-performance model inference.
    
    This backend uses the vLLM library for optimized inference of large language models.
    vLLM provides efficient memory management and high throughput for model serving.
    """
    
    def __init__(self, model_name_or_path: str, template: str, max_length: int=None, temperature: float=1.0, max_new_tokens: int=1024, **kwargs):
        """Initialize VLLMBackend.
        
        Args:
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            max_length (int): Maximum sequence length for input/output. Defaults to 8192.
            temperature (float): Sampling temperature for text generation. Defaults to 1.0.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.model_name = model_name_or_path
        self.max_length = max_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.template = template
        # Load model
        self.llm_engine = LLM(model=self.model_name)
    
    def _process_inputs(self, prompts: List[str], vision_inputs: Dict[str, List[PIL.Image.Image]]):
        inputs = []
        for prompt, vision_input in zip(prompts, vision_inputs):
            mixed_inputs = {
                "prompt": prompt,
            }
            if vision_input:
                mixed_inputs['multi_modal_data'] = vision_input
            inputs.append(mixed_inputs)
        return inputs

    
    def generate(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using vLLM"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        n = kwargs.get("num_return_sequences", 1)
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        prompts, vision_inputs = self.apply_chat_template(messages_list, self.template)
        inputs = self._process_inputs(prompts, vision_inputs)
        print(f"inputs: {inputs}")
        outputs = self.llm_engine.generate(
            inputs,
            sampling_params=sampling_params,
        )
        response_texts = []
        for output in outputs:
            for sequence in output.outputs:
                response_texts.append(sequence.text)
        return response_texts
    
    def generate_async(self, messages_list: str, **kwargs) -> str:
        raise NotImplementedError("VLLM backend does not support async generation")

    async def generate_streaming(self, messages_list: List[List[Dict]], streaming_callback: Optional[Callable] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming support using vLLM"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        tools = kwargs.get("tools", None)
        prompts, vision_inputs = self.apply_chat_template(messages_list, self.template, tools=tools)
        inputs = self._process_inputs(prompts, vision_inputs)
        
        # For streaming, we process one input at a time
        for input_data in inputs:
            outputs_gen = self.llm_engine.generate(
                input_data,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
            )
            
            async for output in outputs_gen:
                for sequence in output.outputs:
                    # Stream each token
                    if hasattr(sequence, 'text'):
                        if streaming_callback:
                            await streaming_callback(sequence.text)
                        yield sequence.text

class AsyncVLLMBackend(LLMBackend):
    """Asynchronous vLLM implementation for high-performance model inference.
    
    This backend uses the vLLM AsyncLLMEngine for asynchronous inference, providing
    better resource utilization and scalability for concurrent requests.
    """
    
    def __init__(self, model_name_or_path: str, template: str, max_length: int=None, temperature: float=1.0, max_new_tokens: int=1024, **kwargs):
        """Initialize AsyncVLLMBackend.
        
        Args:
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            max_length (int): Maximum sequence length for input/output. Defaults to 8192.
            temperature (float): Sampling temperature for text generation. Defaults to 1.0.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters that will be passed to AsyncEngineArgs.
        """
        super().__init__(**kwargs)

        self.model_name = model_name_or_path
        self.max_length = max_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.template = template
        
        if 'engine_args' in kwargs:
            engine_args = kwargs.pop('engine_args')
            engine_args.model = self.model_name
        else:
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                **kwargs,
            )
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            engine_args
        )
        
    def _process_inputs(self, prompts: List[str], vision_inputs: Dict[str, List[PIL.Image.Image]]):
        inputs = []
        for prompt, vision_input in zip(prompts, vision_inputs):
            mixed_inputs = {
                "prompt": prompt,
            }
            if vision_input:
                mixed_inputs['multi_modal_data'] = vision_input
            inputs.append(mixed_inputs)
        return inputs

    async def _generate_single(self, prompt: str, sampling_params: SamplingParams) -> str:
        outputs_gen = self.llm_engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
        )
        async for output in outputs_gen:
            final_output = output
        return final_output.outputs
        
    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using vLLM"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        n = kwargs.get("num_return_sequences", 1)
        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        tools = kwargs.get("tools", None)
        prompts, vision_inputs = self.apply_chat_template(messages_list, self.template, tools=tools)
        inputs = self._process_inputs(prompts, vision_inputs)
        if n > 1:
            inputs = [_input for _input in inputs for _ in range(n)]
        # LOGGER.debug(f"[AsyncVLLMBackend] inputs: {inputs}")
        tasks = [self._generate_single(_input, sampling_params) for _input in inputs]
        outputs = await asyncio.gather(*tasks)
        # Flatten the outputs
        outputs = [output for output_list in outputs for output in output_list]
        response_texts = [output.text for output in outputs]
        # LOGGER.debug(f"[AsyncVLLMBackend] response_texts: {response_texts}")

        return response_texts
    
    async def generate_streaming(self, messages_list: List[List[Dict]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming support using Async vLLM"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        tools = kwargs.get("tools", None)
        prompts, vision_inputs = self.apply_chat_template(messages_list, self.template, tools=tools)
        inputs = self._process_inputs(prompts, vision_inputs)
        
        # For streaming, we process one input at a time
        for input_data in inputs:
            outputs_gen = self.llm_engine.generate(
                input_data,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
            )
            
            async for output in outputs_gen:
                for sequence in output.outputs:
                    # Stream each token
                    if hasattr(sequence, 'text'):
                        yield sequence.text

class AsyncVerlBackend(LLMBackend):
    """Asynchronous Verl implementation for distributed model inference.
    
    This backend uses the Verl framework for distributed and asynchronous model inference.
    Verl provides capabilities for running models across multiple workers and handling
    complex inference pipelines.
    """
    
    def __init__(self, llm_engine, model_name_or_path: str, template: str, max_length: int=None, **kwargs):
        """Initialize AsyncVerlBackend.
        
        Args:
            llm_engine: Verl engine instance for distributed inference.
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            max_length (int): Maximum sequence length for input/output. Defaults to 8192.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.model_name = model_name_or_path
        self.max_length = max_length
        self.template = template
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
        )
        self.llm_engine = llm_engine
    
    def _process_inputs(self, prompts: List[str], vision_inputs: Dict[str, List[PIL.Image.Image]]):
        inputs = []
        for prompt, vision_input in zip(prompts, vision_inputs):
            mixed_inputs = {
                "prompt": prompt,
            }
            if vision_input:
                mixed_inputs['multi_modal_data'] = vision_input
            inputs.append(mixed_inputs)
        return inputs
    
    def generate(self, messages_list: str, **kwargs) -> str:
        raise NotImplementedError("Async Verl backend does not support sync generation")

    def _convert_to_openai_chat_without_tool_call_processing(self, messages: list) -> list:
        """
        We use the pure generated content as the history. So we don't want any tool call to be part of the history.
        This is used when models are not openai's official models like GPT-4o.
        """
        messages = copy.deepcopy(messages)
        for message in messages:
            if "tool_calls" in message:
                del message["tool_calls"]
            if "tool_call_id" in message:
                del message["tool_call_id"]
            if "tool_choice" in message:
                del message["tool_choice"]
        return messages
    
    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using Verl"""
        # We need to build a DataProto from the prompts

        generation_config = {}
        tensors = torch.ones(len(messages_list), dtype=torch.int64)
        messages_list = [self._convert_to_openai_chat_without_tool_call_processing(messages) for messages in messages_list]
        tools = kwargs.get("tools", None)
        tools_list = np.array([tools] * len(messages_list))
        data = {"input_ids": tensors, "raw_prompt": np.array(messages_list), "tools": tools_list}
        
        n = kwargs.get("num_return_sequences", 1)
        temperature = kwargs.get("temperature", 1.0)
        generation_config["temperature"] = temperature
        generation_config["n"] = n
        # Only for compatibility with Verl DataProto

        batch = DataProto.from_single_dict(data, meta_info={"n": n, "temperature": temperature})

        gen_batch_output = await self.llm_engine.generate_sequences_async(batch, **generation_config)
        response_texts = gen_batch_output.batch['responses'].tolist() # np.array of strings with length BS
        return response_texts


class ClientBackend(LLMBackend):
    """OpenAI-compatible client backend for remote API inference.
    
    This backend provides a thin wrapper around OpenAI-compatible chat APIs,
    supporting both synchronous and asynchronous operations. It includes built-in
    rate limiting and retry mechanisms for reliable API communication.
    """

    def __init__(
        self,
        model_name_or_path: str,
        template: str,
        base_url: str = "http://localhost:8000/v1",
        max_requests_per_minute: int = 100,
        timeout: int = 600,
        api_key: str = "EMPTY",
        max_length: int = None,
        max_new_tokens: int = 1024,
        **kwargs,
    ):
        """Initialize ClientBackend.
        
        Args:
            model_name_or_path (str): Name of the model to use for inference.
            template (str): Chat template to use for formatting messages.
            base_url (str): Base URL for the API endpoint. Defaults to localhost:8000.
            max_requests_per_minute (int): Rate limiting for API requests. Defaults to 100.
            timeout (int): Request timeout in seconds. Defaults to 600.
            api_key (str): API key for authentication. Defaults to "EMPTY" for local servers.
            max_length (int): Maximum sequence length for input/output. Defaults to 8192.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        # --- connection
        self.model_name = model_name_or_path
        self.base_url = base_url
        self.template = template
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        # --- rate limiting (token bucket, 1 r/s = 60 r/m)
        self._tokens = asyncio.Semaphore(max_requests_per_minute)
        self._max_tokens = max_requests_per_minute
        self._refill_task = None  # started lazily

        # --- misc
        self.timeout = timeout

    # --------------------------------------------------------------------- #
    # Low‑level single request (runs in threadpool so it doesn't block loop)
    # --------------------------------------------------------------------- #
    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _blocking_call(self, messages: List[List[Dict]], **kwargs) -> str:
        # LOGGER.debug(f"[ClientBackend] _blocking_call kwargs: {kwargs}")
        if "num_return_sequences" in kwargs:
            n = kwargs.pop("num_return_sequences")
        else:
            n = 1

        LOGGER.debug(f"[ClientBackend] messages: {messages}")

        with open(f"{AGENT_DATA_DIR}/debug/messages_{len(messages)}.json", "w") as f:
            json.dump(messages, f, indent=4)
        with open(f"{AGENT_DATA_DIR}/debug/args.json", 'w') as f:
            json.dump(kwargs, f, indent=4)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=self.timeout,
            max_completion_tokens=self.max_new_tokens,
            n=n,
            **kwargs,
        )
        resp_json = resp.dict()
        LOGGER.debug(f"[ClientBackend] resp_json: {resp_json}")
        response_texts = [choice["message"]["content"] for choice in resp_json["choices"]]
        tool_calls = [choice["message"]["tool_calls"] for choice in resp_json["choices"]]

        if kwargs.get("tool_choice", "none") == "none":
            return response_texts
        else:
            return {
                "response_texts": response_texts,
                "tool_calls": tool_calls,
            }
    
    async def generate_streaming(self, messages: List[List[Dict]], **kwargs) -> AsyncGenerator[str, None]:
        """
        This is actually the not streaming. We simply return the generated text.
        """
        LOGGER.debug(f"[ClientBackend] generate_streaming kwargs: {kwargs}")
        response_texts_dicts = await self.async_generate(messages, **kwargs)
        for response in response_texts_dicts:
            yield response

    async def _call(self, messages: List[List[Dict]], **kwargs) -> str:
        # acquire a rate‑limit token
        async with self._tokens:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, partial(self._blocking_call, messages, **kwargs))

    def _convert_to_openai_chat_without_tool_call_processing(self, messages: list, is_openai_model: bool = False) -> list:
        """
        We use the pure generated content as the history. So we don't want any tool call to be part of the history.
        This is used when models are not openai's official models like GPT-4o.
        TODO: we need to add support for openai models
        """
        messages = copy.deepcopy(messages)
        
        for message in messages:
            if is_openai_model:
                if message['role'] == 'assistant':
                    if 'tool_calls' in message:
                        if 'content' in message and message['content'][0]['text'] is None:
                            del message['content']
            else:
                if "tool_calls" in message:
                    del message["tool_calls"]
                if "tool_call_id" in message:
                    del message["tool_call_id"]
                if "tool_choice" in message:
                    del message["tool_choice"]

            if 'content' in message and isinstance(message["content"], list):
                new_content = []
                for item in message["content"]:
                    if item["type"] in ["image"]:
                        # OpenAI chat completion API only supports image_url
                        # And we keep all images to be base64 for compatibility
                        image = image_to_data_uri(item["image"])
                        new_content.append({"type": "image_url", "image_url": {"url": image}})
                    else:
                        new_content.append(item)
                message["content"] = new_content

        return messages

    def _preprocess_messages_and_args(self, messages_list, **kwargs):
        is_openai_model = False
        if "gpt" in self.model_name.lower() or "api.openai.com" in self.base_url:
            is_openai_model = True

        messages_list = [self._convert_to_openai_chat_without_tool_call_processing(messages, is_openai_model) for messages in messages_list]
        
        if is_openai_model:
            kwargs['tool_choice'] = "auto"
        else:
            # For self-deployed models, we will use the response to extract tool calls
            kwargs['tool_choice'] = "none"
        
        return messages_list, kwargs


    # Public API ‑‑ sync or async depending on caller's context
    def async_generate(
        self,
        messages: List[List[Dict]] | List[Dict],
        **kwargs,
    ) -> List[str] | asyncio.Task:
        """
        • Pass a *list of messages* → single completion.
        • Pass a *list of list of messages* → batch completions (max parallelism).

        Returns:
          • In an *async* context → **awaitable Task** (so caller writes `await backend.generate(...)`).
          • In a *sync* context  → real list of strings (blocks until done).
        """
        # normalise argument
        if messages and isinstance(messages[0], dict):
            messages_list = [messages]  # single
        else:
            messages_list = messages     # batch

        messages_list, kwargs = self._preprocess_messages_and_args(messages_list, **kwargs)
        # LOGGER.debug(f"[ClientBackend] messages_list: {messages_list}")

        async def _runner():
            tasks = [asyncio.create_task(self._call(_input, **kwargs)) for _input in messages_list]
            # Flatten the response list
            response_texts_list_or_dict = await asyncio.gather(*tasks)
            # return is a dict if tool_choice is not none, otherwise a list of strings
            if isinstance(response_texts_list_or_dict[0], dict):
                # Flatten the response list
                response_texts = [text for response in response_texts_list_or_dict for text in response["response_texts"]]
                tool_calls = [tool_call for response in response_texts_list_or_dict for tool_call in response["tool_calls"]]
                return [
                    {"response_text": response_text, "tool_calls": _tool_calls} for response_text, _tool_calls in zip(response_texts, tool_calls)
                ]
            else:
                response_texts = [text for response in response_texts_list_or_dict for text in response]
                return response_texts

        try:
            loop = asyncio.get_running_loop()  # ➊ already inside a loop?
        except RuntimeError:
            # --- synchronous caller: spin a loop just for this call
            return asyncio.run(_runner())

        # --- asynchronous caller: schedule task & hand it back
        # (don't block the caller's event loop)
        return loop.create_task(_runner())
    

    async def generate_async(self,
            messages: List[List[Dict]] | List[Dict],
            **kwargs) -> List[str]:
        return await self.async_generate(messages, **kwargs)

    # Background token‑bucket refill (one token each 60/max_rpm seconds)
    async def _refill_tokens(self):
        interval = 60 / self._max_tokens
        while True:
            await asyncio.sleep(interval)
            if self._tokens._value < self._max_tokens:
                self._tokens.release()

    def _ensure_refiller_running(self):
        if self._refill_task is None or self._refill_task.done():
            loop = asyncio.get_event_loop()
            self._refill_task = loop.create_task(self._refill_tokens())

    # Automatically start the refiller at first public use
    def __getattribute__(self, name):
        if name == "generate":
            self._ensure_refiller_running()
        return super().__getattribute__(name)
