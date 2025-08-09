"""
LLM Backend module for reward functions.
This module provides a unified interface to different LLM implementations.
"""
import asyncio
from asyncore import loop
from collections import deque
from functools import partial
import time
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
import uuid
from .templates.utils import convert_messages_to_openai_format
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.verl import pad_tensor_to_rank_size
import os
os.environ["VLLM_USE_V1"] = "1"
from vllm import LLM, AsyncLLMEngine, SamplingParams, AsyncEngineArgs
import openai
from .templates.templates import Chat
from .templates.vision_processor import get_processor
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
    """Base class for LLM backends"""
    
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
    """HuggingFace Transformers implementation"""
    
    def __init__(self, model_name_or_path: str, template: str, max_length: int=8192, temperature: float=1.0, max_new_tokens: int=1024, **kwargs):
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
    """vLLM implementation"""
    
    def __init__(self, model_name_or_path: str, template: str, max_length: int=8192, temperature: float=1.0, max_new_tokens: int=1024, **kwargs):
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
    """Async vLLM implementation"""
    
    def __init__(self, model_name_or_path: str, template: str, max_length: int=8192, temperature: float=1.0, max_new_tokens: int=1024, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name_or_path
        self.max_length = max_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.template = template
        # Load model
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=self.model_name,
            )
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
        LOGGER.debug(f"[AsyncVLLMBackend] inputs: {inputs}")
        tasks = [self._generate_single(_input, sampling_params) for _input in inputs]
        outputs = await asyncio.gather(*tasks)
        # Flatten the outputs
        outputs = [output for output_list in outputs for output in output_list]
        response_texts = [output.text for output in outputs]
        LOGGER.debug(f"[AsyncVLLMBackend] response_texts: {response_texts}")

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

class VerlBackend(LLMBackend):
    """Verl implementation"""
    
    def __init__(self, llm_engine, model_name_or_path: str, template: str, max_length: int=8192, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name_or_path
        self.max_length = max_length
        self.template = template
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
        )
        self.llm_engine = llm_engine
    
    def generate(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using Verl"""
        # We need to build a DataProto from the prompts
        prompts = self.apply_chat_template(messages_list, self.template)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        # We need to do padding for compatibility with the verl DataProto, which 
        # assumes that the batch size must be divisible by the dp size
        world_size = self.llm_engine.world_size
        inputs['input_ids'] = pad_tensor_to_rank_size(inputs['input_ids'], world_size)
        inputs['attention_mask'] = pad_tensor_to_rank_size(inputs['attention_mask'], world_size)

        position_ids = torch.clip(torch.cumsum(inputs.attention_mask, dim=-1) - 1, min=0, max=None)
        inputs['position_ids'] = position_ids

        n = kwargs.get("num_return_sequences", 1)
        temperature = kwargs.get("temperature", 1.0)
        use_agent = True
        batch = DataProto.from_single_dict(inputs, meta_info={"n": n, "use_agent": use_agent, "temperature": temperature})

        gen_batch_output = self.llm_engine.generate_sequences(batch)
        responses = gen_batch_output.batch['responses'] # BS x L
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True) # List of string with length BS
        response_texts = response_texts[:len(prompts)*n]
        return response_texts
    
    def generate_async(self, messages_list: str, **kwargs) -> str:
        raise NotImplementedError("Verl backend does not support async generation")
    

class AsyncVerlBackend(LLMBackend):
    """Verl implementation"""
    
    def __init__(self, llm_engine, model_name_or_path: str, template: str, max_length: int=8192, **kwargs):
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
    
    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using Verl"""
        # We need to build a DataProto from the prompts

        generation_config = {}
        tensors = torch.ones(len(messages_list), dtype=torch.int64)
        messages_list = [convert_messages_to_openai_format(messages) for messages in messages_list]
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
    """
    Thin async/sync wrapper around OpenAI-compatible chat API.
    Call `generate(...)` with *one* or *many* message lists.
    """

    def __init__(
        self,
        model_name_or_path: str,
        template: str,
        base_url: str = "http://localhost:8000/v1",
        max_requests_per_minute: int = 100,
        timeout: int = 600,
        api_key: str = "EMPTY",
        max_length: int = 8192,
        max_new_tokens: int = 1024,
        **kwargs,
    ):
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
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _blocking_call(self, messages: List[List[Dict]], **kw) -> str:
        if "num_return_sequences" in kw:
            n = kw.pop("num_return_sequences")
        else:
            n = 1
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=self.timeout,
            max_tokens=self.max_new_tokens,
            n=n,
            tool_choice="none",
            **kw,
        )
        response_texts = [choice.message.content for choice in resp.choices]

        return response_texts

    async def _call(self, messages: List[List[Dict]], **kw) -> str:
        # acquire a rate‑limit token
        async with self._tokens:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, partial(self._blocking_call, messages, **kw))


    # Public API ‑‑ sync or async depending on caller's context
    def async_generate(
        self,
        messages: List[List[Dict]] | List[Dict],
        **kw,
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

        messages_list = [convert_messages_to_openai_format(messages) for messages in messages_list]

        async def _runner():
            tasks = [asyncio.create_task(self._call(_input, **kw)) for _input in messages_list]
            texts_list = await asyncio.gather(*tasks)
            response_texts = [text for texts in texts_list for text in texts]
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
            **kw) -> List[str]:
        return await self.async_generate(messages, **kw)

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
