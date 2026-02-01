"""
LLM Backend module for reward functions.
This module provides a unified interface to different LLM implementations.
"""

import asyncio
import copy
import logging
import uuid
from functools import partial
from typing import AsyncGenerator, Callable, Dict, List, Optional

import numpy as np
import openai
import PIL
import torch
from google import genai
from google.genai import types
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from chat_bricks import Chat
from ...utils.vision import image_to_data_uri

logger = logging.getLogger(__name__)

try:
    from ...verl.protocol import DataProto
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

    def apply_chat_template(
        self,
        messages_list: List[List[Dict]],
        template: str,
        add_generation_prompt: bool = True,
        tools: List[Dict] = None,
    ) -> List[str]:
        """Apply chat template to messages list"""
        prompts = []
        vision_inputs = []
        for messages in messages_list:
            chat = Chat(template, messages)
            prompts.append(
                chat.prompt(add_generation_prompt=add_generation_prompt, tools=tools)
            )
            # We only support image inputs for now
            vision_inputs.append(chat.vision_inputs())

        return prompts, vision_inputs

    def prepare(self):
        """Prepare the backend"""
        pass

    def generate(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt"""
        raise NotImplementedError("Subclasses must implement generate()")

    async def generate_streaming(
        self,
        messages_list: List[List[Dict]],
        streaming_callback: Optional[Callable] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming support"""
        raise NotImplementedError("Subclasses must implement generate_streaming()")

    def preprocess(self):
        """Preprocess the backend"""
        pass

    def postprocess(self):
        """Postprocess the backend"""
        pass


class TransformersBackend(LLMBackend):
    """HuggingFace Transformers implementation for local model inference.

    This backend uses the Hugging Face Transformers library to load and run models locally.
    It supports both synchronous and asynchronous text generation with streaming capabilities.
    """

    def __init__(
        self,
        model_name_or_path: str,
        template: str,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        **kwargs,
    ):
        """Initialize TransformersBackend.

        Args:
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            temperature (float): Sampling temperature for text generation. Defaults to 1.0.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.model_name = model_name_or_path
        self.temperature = temperature
        self.template = template
        self.max_new_tokens = max_new_tokens
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.llm_engine = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", trust_remote_code=True
        )

    def generate(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using Transformers"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        kwargs.update(
            {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
            }
        )

        prompts, _ = self.apply_chat_template(messages_list, self.template)

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.llm_engine.device)
        input_length = inputs["input_ids"].shape[1]
        outputs = self.llm_engine.generate(**inputs, **kwargs)[:, input_length:]

        response_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return response_texts

    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Async wrapper for generate"""
        return self.generate(messages_list, **kwargs)

    async def generate_streaming(
        self,
        messages_list: List[List[Dict]],
        streaming_callback: Optional[Callable] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming support using Transformers"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        prompts, _ = self.apply_chat_template(messages_list, self.template)

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.llm_engine.device)

        # Use streaming generation
        generated_tokens = []
        for i in range(max_new_tokens):
            outputs = self.llm_engine.generate(
                **inputs,
                max_new_tokens=1,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
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
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], new_token.unsqueeze(0)], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones(1, 1, device=inputs["attention_mask"].device),
                ],
                dim=1,
            )


class AsyncVLLMBackend(LLMBackend):
    """Asynchronous vLLM implementation for high-performance model inference.

    This backend uses the vLLM AsyncLLMEngine for asynchronous inference, providing
    better resource utilization and scalability for concurrent requests.
    """

    def __init__(self, model_name_or_path: str, template: str, **kwargs):
        """Initialize AsyncVLLMBackend.

        Args:
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            temperature (float): Sampling temperature for text generation. Defaults to 1.0.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters that will be passed to AsyncEngineArgs.
        """
        super().__init__(**kwargs)

        self.model_name = model_name_or_path
        self.template = template

        if "engine_args" in kwargs:
            engine_args = kwargs.pop("engine_args")
            engine_args.model = self.model_name
        else:
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                **kwargs,
            )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _process_inputs(
        self, prompts: List[str], vision_inputs: Dict[str, List[PIL.Image.Image]]
    ):
        inputs = []
        for prompt, vision_input in zip(prompts, vision_inputs):
            mixed_inputs = {
                "prompt": prompt,
            }
            if vision_input:
                mixed_inputs["multi_modal_data"] = vision_input
            inputs.append(mixed_inputs)
        return inputs

    async def _generate_single(
        self, prompt: str, sampling_params: SamplingParams
    ) -> str:
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
        sampling_params = {}
        if "temperature" in kwargs:
            sampling_params["temperature"] = kwargs["temperature"]  
        if "n" in kwargs:
            sampling_params["n"] = kwargs["n"]
        if "max_tokens" in kwargs:
            sampling_params["max_tokens"] = kwargs.get("max_tokens")
        sampling_params = SamplingParams(**sampling_params)
        n = kwargs.get("n", 1)

        tools = kwargs.get("tools", None)
        prompts, vision_inputs = self.apply_chat_template(
            messages_list, self.template, tools=tools
        )
        inputs = self._process_inputs(prompts, vision_inputs)
        if n > 1:
            inputs = [_input for _input in inputs for _ in range(n)]
        logger.debug(f"[AsyncVLLMBackend] inputs: {inputs}")
        tasks = [self._generate_single(_input, sampling_params) for _input in inputs]
        outputs = await asyncio.gather(*tasks)
        # Flatten the outputs
        outputs = [output for output_list in outputs for output in output_list]
        response_texts = [output.text for output in outputs]
        logger.debug(f"[AsyncVLLMBackend] response_texts: {response_texts}")

        return response_texts

    async def generate_streaming(
        self, messages_list: List[List[Dict]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming support using Async vLLM"""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        n = kwargs.get("n", 1)
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        tools = kwargs.get("tools", None)
        prompts, vision_inputs = self.apply_chat_template(
            messages_list, self.template, tools=tools
        )
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
                    if hasattr(sequence, "text"):
                        yield sequence.text


class AsyncVerlBackend(LLMBackend):
    """Asynchronous Verl implementation for distributed model inference.

    This backend uses the Verl framework for distributed and asynchronous model inference.
    Verl provides capabilities for running models across multiple workers and handling
    complex inference pipelines.
    """

    def __init__(self, llm_engine, model_name_or_path: str, template: str, **kwargs):
        """Initialize AsyncVerlBackend.

        Args:
            llm_engine: Verl engine instance for distributed inference.
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.llm_engine = llm_engine

    def preprocess(self):
        """Preprocess the backend"""
        self.llm_engine.wake_up()
        if self.llm_engine.reward_model_manager:
            self.llm_engine.reward_model_manager.wake_up()

    def postprocess(self):
        """Postprocess the backend"""
        self.llm_engine.sleep()
        if self.llm_engine.reward_model_manager:
            self.llm_engine.reward_model_manager.sleep()

    def _process_inputs(
        self, prompts: List[str], vision_inputs: Dict[str, List[PIL.Image.Image]]
    ):
        inputs = []
        for prompt, vision_input in zip(prompts, vision_inputs):
            mixed_inputs = {
                "prompt": prompt,
            }
            if vision_input:
                mixed_inputs["multi_modal_data"] = vision_input
            inputs.append(mixed_inputs)
        return inputs

    def generate(self, messages_list: str, **kwargs) -> str:
        raise NotImplementedError("Async Verl backend does not support sync generation")

    def _convert_to_openai_chat_without_tool_call_processing(
        self, messages: list
    ) -> list:
        """
        We use the pure generated content as the history. So we don't want any tool call to be part of the history.
        This is used when models are not openai's official models like GPT-4o.
        """
        # messages = copy.deepcopy(messages)
        # for message in messages:
        #     if "tool_calls" in message:
        #         del message["tool_calls"]
        #     if "tool_call_id" in message:
        #         del message["tool_call_id"]
        #     if "tool_choice" in message:
        #         del message["tool_choice"]
        # return messages

        processed_messages = []
        for message in messages:
            processed_message = {}
            for k, v in message.items():
                if k not in ["tool_calls", "tool_call_id", "tool_choice"]:
                    processed_message[k] = v
            processed_messages.append(processed_message)
        return processed_messages

    def _process_messages(self, messages: List[Dict]):
        new_messages = []
        for message in messages:
            new_message = {}
            new_message.update(message)
            if isinstance(message["content"], list):
                if len(message["content"]) == 1:
                    assert message["content"][0]["type"] == "text"
                    new_message["content"] = message["content"][0]["text"]
                else:
                    new_message["content"] = message["content"]

            new_messages.append(new_message)
        return new_messages

    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using Verl"""
        # We need to build a DataProto from the prompts

        generation_config = {}
        tensors = torch.ones(len(messages_list), dtype=torch.int64)
        messages_list = [self._process_messages(messages) for messages in messages_list]
        messages_list = [
            self._convert_to_openai_chat_without_tool_call_processing(messages)
            for messages in messages_list
        ]
        tools = kwargs.get("tools", None)
        tools_list = np.array([tools] * len(messages_list))
        data = {
            "input_ids": tensors,
            "raw_prompt": np.array(messages_list),
            "tools": tools_list,
        }

        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "n" in kwargs:
            generation_config["n"] = kwargs["n"]
        if "max_tokens" in kwargs:
            generation_config["max_tokens"] = kwargs["max_tokens"]

        logger.debug(f"[AsyncVerlBackend] generation_config: {generation_config}")

        batch = DataProto.from_single_dict(
            data, meta_info={"generation_config": generation_config}
        )

        gen_batch_output = await self.llm_engine.generate_sequences_async(batch)
        response_ids = gen_batch_output.batch[
            "responses"
        ].tolist()  # np.array of strings with length BS
        assert len(response_ids) == len(messages_list)
        response_texts = [
            self.tokenizer.decode(response_id, skip_special_tokens=True)
            for response_id in response_ids
        ]

        return response_texts


class ClientBackend(LLMBackend):
    """OpenAI-compatible and Google Gemini client backend for remote API inference.

    This backend provides a thin wrapper around OpenAI-compatible chat APIs and Google Gemini API,
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
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 1024.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        # --- connection
        self.model_name = model_name_or_path
        self.base_url = base_url
        self.template = template

        # Detect if it's a Gemini model
        self.is_gemini = self._is_gemini_model(model_name_or_path, base_url)

        if self.is_gemini:
            # Initialize once to avoid overhead and connection leaks
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

        # --- rate limiting (token bucket, 1 r/s = 60 r/m)
        self._tokens = asyncio.Semaphore(max_requests_per_minute)
        self._max_tokens = max_requests_per_minute
        self._refill_task = None  # started lazily

        # --- misc
        self.timeout = timeout

    def _is_gemini_model(self, model_name: str, base_url: str) -> bool:
        """Check if the model is a Google Gemini model."""
        gemini_indicators = ["gemini", "generativelanguage.googleapis.com"]
        model_lower = model_name.lower()
        base_url_lower = base_url.lower()
        return any(
            indicator in model_lower or indicator in base_url_lower
            for indicator in gemini_indicators
        )

    def _prepare_gemini_payload(self, messages: List[Dict]):
        """Separates system instructions from chat history and converts to Gemini format."""
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
                continue

            # Convert roles: user -> user, assistant -> model
            gemini_role = "model" if role == "assistant" else "user"

            # Handle parts (Text or Image)
            parts = []
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(types.Part.from_text(text=item["text"]))
                    elif item.get("type") in ["image_url", "image"]:
                        # Assuming helper converts to PIL or Bytes
                        img = self._process_image(item)
                        parts.append(types.Part.from_image(image=img))

            contents.append(types.Content(role=gemini_role, parts=parts))

        return system_instruction, contents

    def _blocking_call_gemini(self, messages: List[Dict], **kwargs) -> Dict:
        """Make a blocking call to Gemini API with full response preservation."""
        import json

        from google.genai import types

        system_instruction, contents = self._prepare_gemini_payload(messages)

        # 1. Prepare all configuration parameters in one place
        config_kwargs = {}

        # Standard parameters
        if "temperature" in kwargs:
            config_kwargs["temperature"] = kwargs["temperature"]

        # Map 'n' to 'candidate_count'
        if "n" in kwargs:
            config_kwargs["candidate_count"] = kwargs["n"]
        elif "candidate_count" in kwargs:
            config_kwargs["candidate_count"] = kwargs["candidate_count"]

        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_tokens"]
        elif "max_new_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_new_tokens"]

        # FIX: Move tools into the config dictionary
        if "tools" in kwargs:
            config_kwargs["tools"] = kwargs["tools"]

        # Safety settings
        config_kwargs["safety_settings"] = [
            types.SafetySetting(category=cat, threshold="BLOCK_NONE")
            for cat in [
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]

        # 2. Create the unified config object
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            **config_kwargs,
        )

        try:
            # 3. Call without the 'tools' keyword argument
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )

            # Convert to dictionary using pydantic's model_dump
            raw_response_dict = response.model_dump(mode="json")

            response_texts = []
            all_tool_calls = []

            if response.candidates:
                for candidate in response.candidates:
                    cand_text = ""
                    cand_tool_calls = []

                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                cand_text += part.text
                            if part.function_call:
                                func = part.function_call
                                cand_tool_calls.append(
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {
                                            "name": func.name,
                                            "arguments": json.dumps(func.args)
                                            if func.args
                                            else "{}",
                                        },
                                    }
                                )

                    if not cand_text and not cand_tool_calls:
                        cand_text = f"[Empty Response: {candidate.finish_reason}]"

                    response_texts.append(cand_text)
                    all_tool_calls.append(cand_tool_calls if cand_tool_calls else None)

            return {
                "response_texts": response_texts,
                "tool_calls": all_tool_calls,
                "response_dict": raw_response_dict,
            }

        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            return {
                "response_texts": [""],
                "tool_calls": [None],
                "response_dict": {"error": str(e)},
            }

    def _blocking_call_openai(self, messages: List[Dict], **kwargs) -> Dict:
        """Make a blocking call to OpenAI API."""
        logger.debug(f"[ClientBackend] OpenAI model_name: {self.model_name}")
        logger.debug(f"[ClientBackend] OpenAI messages: {len(messages)}")
        logger.debug(f"[ClientBackend] OpenAI kwargs: {kwargs}")

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=self.timeout,
            **kwargs,
        )
        resp_json = resp.dict()
        logger.debug(f"[ClientBackend] resp_json: {resp_json}")
        response_texts = [
            choice["message"]["content"] for choice in resp_json["choices"]
        ]
        tool_calls = [
            choice["message"].get("tool_calls") for choice in resp_json["choices"]
        ]

        return {
            "response_texts": response_texts,
            "tool_calls": tool_calls,
            "response_dict": resp_json,
        }

    # --------------------------------------------------------------------- #
    # Low‑level single request (runs in threadpool so it doesn't block loop)
    # --------------------------------------------------------------------- #
    # @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _blocking_call(self, messages: List[Dict], **kwargs) -> Dict:
        """Route to appropriate blocking call based on model type."""
        if self.is_gemini:
            return self._blocking_call_gemini(messages, **kwargs)
        else:
            return self._blocking_call_openai(messages, **kwargs)

    async def generate_streaming(
        self, messages: List[List[Dict]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        This is actually the not streaming. We simply return the generated text.
        """
        logger.debug(f"[ClientBackend] generate_streaming kwargs: {kwargs}")
        response_texts_dicts = await self.generate(messages, **kwargs)
        for response in response_texts_dicts:
            yield response

    async def _call(self, messages: List[Dict], **kwargs) -> Dict:
        # acquire a rate‑limit token
        async with self._tokens:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, partial(self._blocking_call, messages, **kwargs)
            )

    def _convert_to_openai_chat_without_tool_call_processing(
        self, messages: list, is_openai_model: bool = False
    ) -> list:
        """
        We use the pure generated content as the history. So we don't want any tool call to be part of the history.
        This is used when models are not openai's official models like GPT-4o.
        TODO: we need to add support for openai models
        """
        messages = copy.deepcopy(messages)

        for message in messages:
            if is_openai_model:
                if message["role"] == "assistant":
                    if "tool_calls" in message:
                        if (
                            "content" in message
                            and message["content"][0]["text"] is None
                        ):
                            del message["content"]
            else:
                if "tool_calls" in message:
                    del message["tool_calls"]
                if "tool_call_id" in message:
                    del message["tool_call_id"]
                if "tool_choice" in message:
                    del message["tool_choice"]

            if "content" in message and isinstance(message["content"], list):
                new_content = []
                for item in message["content"]:
                    if item["type"] in ["image"]:
                        # OpenAI chat completion API only supports image_url
                        # And we keep all images to be base64 for compatibility
                        image = image_to_data_uri(item["image"])
                        new_content.append(
                            {"type": "image_url", "image_url": {"url": image}}
                        )
                    else:
                        new_content.append(item)
                message["content"] = new_content

        return messages

    def _preprocess_messages_and_args(self, messages_list, **kwargs):
        is_openai_model = False
        if not self.is_gemini and (
            "gpt" in self.model_name.lower() or "api.openai.com" in self.base_url
        ):
            is_openai_model = True

        if not self.is_gemini:
            messages_list = [
                self._convert_to_openai_chat_without_tool_call_processing(
                    messages, is_openai_model
                )
                for messages in messages_list
            ]

        if "tools" in kwargs:
            if self.is_gemini:
                # Gemini handles tools differently - convert to function declarations
                # This will be handled in the Gemini call if needed
                pass
            elif is_openai_model:
                kwargs["tool_choice"] = "auto"
            else:
                # For self-deployed models, we will use the response to extract tool calls
                kwargs["tool_choice"] = "none"

        return messages_list, kwargs

    # Public API ‑‑ sync or async depending on caller's context
    def generate(
        self,
        messages: List[List[Dict]] | List[Dict],
        return_full_response_dict: bool = False,
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
            messages_list = messages  # batch
        logger.debug(f"[ClientBackend] messages_list: {messages_list}")
        # messages_list = [self._convert_to_openai_chat_without_tool_call_processing(messages) for messages in messages_list]
        messages_list, kwargs = self._preprocess_messages_and_args(
            messages_list, **kwargs
        )

        async def _runner():
            # Ensure refiller is running in this event loop
            self._ensure_refiller_running()
            tasks = [
                asyncio.create_task(self._call(_input, **kwargs))
                for _input in messages_list
            ]
            # Flatten the response list
            response_dicts = await asyncio.gather(*tasks)

            # Build response structure
            final_response_dicts = []
            all_response_texts = []

            for response_dict in response_dicts:
                response_texts = response_dict["response_texts"]
                tool_calls = response_dict["tool_calls"]
                full_response_dict = response_dict["response_dict"]

                # Collect all response texts for non-full-response mode
                all_response_texts.extend(response_texts)

                # Build the structured response dict
                final_response_dicts.append(
                    {
                        "response_texts": response_texts,
                        "tool_calls": tool_calls,
                        "response_dict": full_response_dict,
                    }
                )

            if return_full_response_dict:
                return final_response_dicts
            else:
                return all_response_texts

        try:
            loop = asyncio.get_running_loop()  # ➊ already inside a loop?
        except RuntimeError:
            # --- synchronous caller: spin a loop just for this call
            return asyncio.run(_runner())

        # --- asynchronous caller: schedule task & hand it back
        # (don't block the caller's event loop)
        return loop.create_task(_runner())

    async def generate_async(
        self, messages: List[List[Dict]] | List[Dict], **kwargs
    ) -> List[str]:
        return await self.generate(messages, **kwargs)

    # Background token‑bucket refill (one token each 60/max_rpm seconds)
    async def _refill_tokens(self):
        interval = 60 / self._max_tokens
        while True:
            await asyncio.sleep(interval)
            if self._tokens._value < self._max_tokens:
                self._tokens.release()

    def _ensure_refiller_running(self):
        if self._refill_task is None or self._refill_task.done():
            try:
                # Try to get running loop first
                loop = asyncio.get_running_loop()
                self._refill_task = loop.create_task(self._refill_tokens())
            except RuntimeError:
                # No event loop running, this will be handled by the caller
                # The refiller will be started when we're in an event loop
                pass
