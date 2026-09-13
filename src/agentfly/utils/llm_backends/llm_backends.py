"""
LLM Backend module for reward functions.
This module provides a unified interface to different LLM implementations.
"""
# Defer annotation evaluation so lazily-imported types (e.g. vLLM ``SamplingParams``
# in a method signature) don't force those heavy modules at import time.
from __future__ import annotations

import asyncio
import copy
import logging
import os
import random
import time
import uuid
from functools import partial
from typing import Dict, List, Optional, Union

import httpx
import numpy as np
import PIL
from chat_bricks import Chat

# NOTE: the heavy backends are imported lazily inside the methods that use them,
# so importing this module — which agent_base and the resources/rewards packages
# pull in transitively — costs nothing up front:
#   * ``vllm`` (~4s), ``google.genai`` (~1s)  — VLLMBackend / Gemini path
#   * ``openai`` (~1s)                         — ClientBackend methods
#   * ``torch`` (~3s), ``transformers``        — AsyncVerlBackend only
# (``AutoModelForCausalLM`` was imported but unused, so it's dropped.)

from ...utils.vision import image_to_data_uri

logger = logging.getLogger(__name__)

# NOTE: ``DataProto`` (from the verl trainer) is imported lazily inside
# AsyncVerlBackend where it's used — it pulls the whole verl/torch/ray stack,
# which importing this module (agent_base pulls it in) must not load up front.


class LLMBackend:
    """Base class for LLM backends.

    This abstract base class provides a unified interface for different LLM implementations.
    All backend implementations must inherit from this class and implement the required methods.

    Attributes:
        config: Configuration dictionary containing backend-specific parameters.
    """
    def apply_chat_template(
        self,
        messages_list: List[List[Dict]],
        template: str,
        add_generation_prompt: bool = True,
        tools: List[Dict] = None,
        skills: List[Dict] = None,
    ) -> List[str]:
        """Apply chat template to messages list"""
        prompts = []
        vision_inputs = []
        for messages in messages_list:
            chat = Chat(template, messages)
            prompts.append(
                chat.prompt(
                    add_generation_prompt=add_generation_prompt,
                    tools=tools,
                    skills=skills,
                )
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

    def _check_splice_continuity(self, sent_prompt_ids, messages_list, tools) -> None:
        """Guard: a sampled turn must reappear in later prompts as exactly the prompt it was
        sampled from followed by its sampled ids (token-prefix continuity).

        For a checked row, let ``k`` be the last assistant message carrying ``token_ids``.
        The ids this backend renders for the current call must start with
        ``render(messages[:k], add_generation_prompt=True) + token_ids_k``. Anything the
        renderer inserts between the generation prompt and the ids (template glue, a
        re-rendered think block) breaks this and silently makes every update off-policy
        -- the prompt guard above cannot see it because sampling and training share the
        render. Same cadence and reporting as :meth:`_check_prompt_consistency`;
        ``check_prompt_consistency="strict"`` raises.
        """
        if not self.check_prompt_consistency:
            return
        # _check_prompt_consistency already advanced the call counter for this call.
        if self._prompt_check_calls != 1 and self._prompt_check_calls % self._PROMPT_CHECK_EVERY != 0:
            return
        try:
            n = min(self._PROMPT_CHECK_ROWS, len(messages_list))
            checked, broken, first_diff = 0, 0, None
            for i in range(n):
                messages = list(messages_list[i])
                k = next(
                    (j for j in range(len(messages) - 1, -1, -1)
                     if messages[j].get("role") == "assistant" and messages[j].get("token_ids")),
                    None,
                )
                if k is None:
                    continue
                if sent_prompt_ids is not None:
                    current = list(sent_prompt_ids[i])
                else:
                    current = self._render_prompt_ids([messages], tools, allow_vision=True)[0]
                turn_prompt = self._render_prompt_ids([messages[:k]], tools, allow_vision=True)[0]
                expected = list(turn_prompt) + [int(t) for t in messages[k]["token_ids"]]
                checked += 1
                if current[: len(expected)] != expected:
                    broken += 1
                    if first_diff is None:
                        first_diff = (i, k, current, expected)
            self.prompt_check_stats["continuity_checked"] += checked
            self.prompt_check_stats["continuity_broken"] += broken
            if broken:
                i, k, cur, exp = first_diff
                d = next((j for j in range(min(len(cur), len(exp))) if cur[j] != exp[j]), min(len(cur), len(exp)))
                msg = (
                    f"[AsyncVerlBackend] SPLICE DISCONTINUITY: {broken}/{checked} rows: the prompt "
                    f"does not start with (turn prompt + sampled ids) of assistant message {k}; "
                    f"first diff at {d}: prompt={self.tokenizer.decode(cur[d : d + 24])!r} "
                    f"expected={self.tokenizer.decode(exp[d : d + 24])!r}"
                )
                print(msg, flush=True)
                logger.warning(msg)
                if self.check_prompt_consistency == "strict":
                    raise RuntimeError(msg)
            elif checked:
                print(
                    f"[AsyncVerlBackend] splice_check: {checked}/{checked} rows continuous "
                    "(prompt == turn prompt + sampled ids)",
                    flush=True,
                )
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 — a diagnostic must never break generation
            print(f"[AsyncVerlBackend] splice continuity check failed: {e}", flush=True)
            logger.warning("[AsyncVerlBackend] splice continuity check failed: %s", e)

    def preprocess(self):
        """Preprocess the backend"""
        pass

    def postprocess(self):
        """Postprocess the backend"""
        pass



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
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # lazy: heavy import

        super().__init__()
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

    def _convert_to_chat_without_tool_call_processing(
        self, messages: List[Dict]
    ) -> List[Dict]:
        """
        Keep pure assistant text history for local chat-template rendering.
        Remove tool-call artifacts that may confuse template/tool parsing.
        """
        processed_messages = []
        for message in messages:
            processed_message = {}
            for k, v in message.items():
                if k not in ["tool_calls", "tool_call_id", "tool_choice"]:
                    processed_message[k] = v
            processed_messages.append(processed_message)
        return processed_messages

    async def _generate_single(
        self, prompt: str, sampling_params: SamplingParams
    ):
        outputs_gen = self.llm_engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=str(uuid.uuid4()),
        )
        async for output in outputs_gen:
            final_output = output
        return final_output

    async def _apply_chat_template_async(
        self,
        messages_list: List[List[Dict]],
        tools: List[Dict] = None,
        skills: List[Dict] = None,
    ):
        """
        Render chat templates off the event loop.
        Chat.prompt()/vision_inputs() are synchronous and can block under load.
        """
        return await asyncio.to_thread(
            self.apply_chat_template,
            messages_list,
            self.template,
            True,
            tools,
            skills,
        )

    async def generate_async(self, messages_list: str, **kwargs) -> str:
        """Generate text from prompt using vLLM"""
        return_dict = kwargs.pop("return_dict", False)
        sampling_params = {}
        if "temperature" in kwargs:
            sampling_params["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs or "max_new_tokens" in kwargs:
            if "max_tokens" in kwargs and "max_new_tokens" in kwargs:
                raise ValueError("max_tokens and max_new_tokens cannot be used together")

            if "max_tokens" in kwargs:
                sampling_params["max_tokens"] = kwargs.get("max_tokens")
            else:
                sampling_params["max_tokens"] = kwargs.get("max_new_tokens")

        # Force SamplingParams.n=1 and instead duplicate inputs to get n
        # completions per prompt. vLLM v1's AsyncLLM with n>1 has a generator-
        # exhaustion bug that deadlocks the `async for output in outputs_gen`
        # loop in _generate_single, so we route around it.
        from vllm import SamplingParams  # lazy: heavy import

        sampling_params = SamplingParams(**sampling_params)
        n = kwargs.get("n", 1)

        tools = kwargs.get("tools", None)
        skills = kwargs.get("skills", None)
        messages_list = [
            self._convert_to_chat_without_tool_call_processing(messages)
            for messages in messages_list
        ]
        print(f"[AsyncVLLMBackend.generate_async] applying chat template")
        prompts, vision_inputs = await self._apply_chat_template_async(
            messages_list, tools=tools, skills=skills
        )
        inputs = self._process_inputs(prompts, vision_inputs)
        if n > 1:
            inputs = [_input for _input in inputs for _ in range(n)]
        logger.debug("[AsyncVLLMBackend] prepared %d input(s)", len(inputs))
        tasks = [self._generate_single(_input, sampling_params) for _input in inputs]

        print(f"[AsyncVLLMBackend.generate_async] generating ...")
        request_outputs = await asyncio.gather(*tasks)
        # Flatten candidate texts across requests.
        response_texts = [
            out.text
            for req_out in request_outputs
            for out in getattr(req_out, "outputs", [])
        ]
        # Keep one total length per request to align with rollout.strategies.chain_rollout._extract_total_length().
        total_lengths = []
        for req_out in request_outputs:
            prompt_len = len(getattr(req_out, "prompt_token_ids", []) or [])
            completion_lens = [
                len(getattr(out, "token_ids", []) or [])
                for out in getattr(req_out, "outputs", [])
            ]
            completion_len = max(completion_lens) if completion_lens else 0
            total_lengths.append(prompt_len + completion_len)
        logger.debug(f"[AsyncVLLMBackend] response_texts: {response_texts}")

        if return_dict:
            return {
                "response_texts": response_texts,
                "tool_calls": None,
                "response_dict": {"backend": "async_vllm"},
                "total_lengths": total_lengths,
            }
        return response_texts


def extract_response_ids(batch) -> List[List[int]]:
    """Per-row generated token ids from a verl generation batch, padding stripped.

    verl right-pads ``responses`` to ``response_length``. The valid length of a
    row is the non-pad count of the response slice of ``attention_mask`` (the
    same rule verl uses), with the prompt width taken from ``prompts``. The
    result is exactly the sampled sequence — eos included when the model
    stopped on it — suitable for splicing into training tokenization verbatim.
    """
    responses = batch["responses"]
    attention_mask = batch["attention_mask"]
    if "prompts" in batch.keys():
        prompt_len = batch["prompts"].shape[1]
    else:
        prompt_len = attention_mask.shape[1] - responses.shape[1]
    resp_lens = attention_mask[:, prompt_len:].sum(dim=1).tolist()
    return [responses[i, : int(n)].tolist() for i, n in enumerate(resp_lens)]


class AsyncVerlBackend(LLMBackend):
    """Asynchronous Verl implementation for distributed model inference.

    This backend uses the Verl framework for distributed and asynchronous model inference.
    Verl provides capabilities for running models across multiple workers and handling
    complex inference pipelines.
    """

    def __init__(
        self,
        llm_engine,
        model_name_or_path: str,
        template: str,
        check_prompt_consistency: Union[bool, str] = True,
    ):
        """Initialize AsyncVerlBackend.

        Args:
            llm_engine: Verl engine instance for distributed inference.
            model_name_or_path (str): Name or path of the pre-trained model to load.
            template (str): Chat template to use for formatting messages.
            check_prompt_consistency (bool | "strict"): Periodically verify that the prompt
                verl sampled from equals the ids this backend rendered with the training
                tokenizer call (see :meth:`_check_prompt_consistency`). ``True`` (default)
                reports a mismatch on stdout; ``"strict"`` raises on it; ``False`` disables
                it. Set via ``agent.init_config.backend_config.check_prompt_consistency``.
        """
        from transformers import AutoTokenizer  # lazy: heavy import

        super().__init__()
        self.model_name = model_name_or_path
        self.template = template
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.llm_engine = llm_engine
        # Sampling-vs-training prompt guard bookkeeping (see _check_prompt_consistency).
        self.check_prompt_consistency = check_prompt_consistency
        self._prompt_check_calls = 0
        self.prompt_check_stats = {
            "checked": 0, "mismatched": 0,
            # Splice continuity (see _check_splice_continuity): rows whose last
            # sampled turn is exactly "its prompt + its sampled ids" inside the
            # prompt rendered for this call.
            "continuity_checked": 0, "continuity_broken": 0,
        }
        self._warned_vision_fallback = False

    # Check on the first call, then periodically; a few rows each time.
    _PROMPT_CHECK_EVERY = 200
    _PROMPT_CHECK_ROWS = 4

    @staticmethod
    def _has_vision(messages) -> bool:
        """True if any message carries an image/video content part (no image is loaded)."""
        for m in messages:
            content = m.get("content")
            if isinstance(content, list) and any(
                isinstance(part, dict)
                and part.get("type") in ("image", "image_url", "image_base64", "video")
                for part in content
            ):
                return True
        return False

    def _render_prompt_ids(self, messages_list, tools, *, allow_vision: bool = False):
        """Render each row's prompt ids with chat-bricks — the call ``tokenize_trajectories``
        makes (same template, tokenizer, ``tools``, ``ignore_tool_calls``, generation
        prompt). Earlier assistant turns are spliced from their sampled ``token_ids``, so
        the prompt vLLM samples from is, by construction, the prefix of the training row.

        Returns ``None`` when a row carries vision inputs (unless ``allow_vision``): ids and
        images are not threaded through verl together yet, so those rows keep verl's own
        text render of ``raw_prompt``.
        """
        from chat_bricks import Chat  # lazy: keep import cost off the hot path

        template = self.template or self.model_name
        rendered = []
        for messages in messages_list:
            if not allow_vision and self._has_vision(messages):
                return None
            chat = Chat(
                template=template,
                messages=list(messages),
                tokenizer=self.tokenizer,
                ignore_tool_calls=True,
            )
            ids = chat.tokenize(self.tokenizer, add_generation_prompt=True, tools=tools or None)
            rendered.append(ids["input_ids"][0].tolist())
        return rendered

    def _check_prompt_consistency(self, sent_prompt_ids, messages_list, tools, gb) -> None:
        """Guard: the prompt verl actually sampled from must equal what training renders.

        ``sent_prompt_ids`` is the render this backend handed verl (see
        :meth:`_render_prompt_ids`), so the check is an exact comparison against the
        ``prompts`` verl returns — it catches verl ignoring or re-rendering the ids. Rows
        that were not pre-tokenized (vision fallback) are re-rendered with chat-bricks
        instead. Runs on the first call and every ``_PROMPT_CHECK_EVERY`` calls, a few rows
        each time, and reports through ``print`` (visible in the Ray log; a plain
        ``logger.warning`` is not). ``check_prompt_consistency="strict"`` raises on a
        mismatch; ``False`` disables the check.
        """
        if not self.check_prompt_consistency:
            return
        self._prompt_check_calls += 1
        if self._prompt_check_calls != 1 and self._prompt_check_calls % self._PROMPT_CHECK_EVERY != 0:
            return
        try:
            prompts, attention_mask = gb["prompts"], gb["attention_mask"]
            prompt_len = prompts.shape[1]
            n = min(self._PROMPT_CHECK_ROWS, len(messages_list))
            depths = [
                sum(1 for m in messages_list[i] if m.get("role") == "assistant") for i in range(n)
            ]
            mismatched, first_diff = 0, None
            for i in range(n):
                sampled = prompts[i][attention_mask[i, :prompt_len].bool()].tolist()
                if sent_prompt_ids is not None:
                    rendered = list(sent_prompt_ids[i])
                else:
                    rendered = self._render_prompt_ids([messages_list[i]], tools, allow_vision=True)[0]
                if sampled != rendered:
                    mismatched += 1
                    if first_diff is None:
                        first_diff = (sampled, rendered)
            self.prompt_check_stats["checked"] += n
            self.prompt_check_stats["mismatched"] += mismatched
            if mismatched:
                s, r = first_diff
                k = next((j for j in range(min(len(s), len(r))) if s[j] != r[j]), min(len(s), len(r)))
                msg = (
                    f"[AsyncVerlBackend] PROMPT MISMATCH (sampling vs training render): "
                    f"{mismatched}/{n} rows differ (assistant turns in checked rows={depths}); "
                    f"sampled {len(s)} tok vs rendered {len(r)} tok, first diff at {k}: "
                    f"sampled={self.tokenizer.decode(s[k : k + 40])!r} "
                    f"rendered={self.tokenizer.decode(r[k : k + 40])!r}"
                )
                print(msg, flush=True)
                logger.warning(msg)
                if self.check_prompt_consistency == "strict":
                    raise RuntimeError(msg)
            else:
                # print (not logger.info) so it is greppable next to the [StepRollout] batch line
                print(
                    f"[AsyncVerlBackend] prompt_check: {n}/{n} rows identical "
                    f"(sampling == training render; assistant turns={depths})",
                    flush=True,
                )
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 — a diagnostic must never break generation
            print(f"[AsyncVerlBackend] prompt consistency check failed: {e}", flush=True)
            logger.warning("[AsyncVerlBackend] prompt consistency check failed: %s", e)

    def preprocess(self):
        """Preprocess the backend"""
        # Don't do anything for now
        pass

    def postprocess(self):
        """Postprocess the backend"""
        pass

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
        """

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

    async def generate_async(self, messages_list: str, return_dict: bool = False, **kwargs) -> str:
        """Generate text from prompt using Verl"""
        # We need to build a DataProto from the prompts

        import torch  # lazy: heavy import
        from ...verl.protocol import DataProto  # lazy: pulls the verl/torch/ray stack

        generation_config = {}
        tensors = torch.ones(len(messages_list), dtype=torch.int64)
        tools = kwargs.get("tools", None)
        # Render the prompt ids from the trajectory messages as-is (list content,
        # tool_calls, sampled ``token_ids``) — exactly what ``tokenize_trajectories``
        # sees — BEFORE the raw-prompt cleanup below, so the sampled prompt is the
        # training row's prefix. verl executes these ids instead of re-rendering the
        # messages from text (which re-tokenizes earlier assistant turns).
        original_messages_list = messages_list
        prompt_ids = self._render_prompt_ids(messages_list, tools)
        if prompt_ids is None and not self._warned_vision_fallback:
            self._warned_vision_fallback = True
            print(
                "[AsyncVerlBackend] prompt carries vision inputs: verl renders the prompt "
                "itself (pre-tokenized ids are text-only for now)",
                flush=True,
            )
        messages_list = [self._process_messages(messages) for messages in messages_list]
        messages_list = [
            self._convert_to_openai_chat_without_tool_call_processing(messages)
            for messages in messages_list
        ]
        data_source = kwargs.get("data_source", None)
        tools_list = np.array([tools] * len(messages_list))
        data_source_list = np.array([data_source] * len(messages_list))
        data = {
            "input_ids": tensors,
            "raw_prompt": np.array(messages_list),
            "tools": tools_list,
            "data_source": data_source_list,
        }
        if prompt_ids is not None:
            # One object per row (rows differ in length; a plain np.array would try to
            # build a 2-D int array).
            prompt_ids_arr = np.empty(len(prompt_ids), dtype=object)
            for i, ids in enumerate(prompt_ids):
                prompt_ids_arr[i] = ids
            data["prompt_ids"] = prompt_ids_arr

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
        gb = gen_batch_output.batch
        assert len(gb["responses"]) == len(messages_list)
        self._check_prompt_consistency(prompt_ids, original_messages_list, tools, gb)
        self._check_splice_continuity(prompt_ids, original_messages_list, tools)
        # Padding-stripped sampled ids (eos included when the model stopped on it).
        response_ids = extract_response_ids(gb)
        # Same text as before: decoding the padded row with skip_special_tokens
        # already dropped the (special) pad tokens, so stripping first is a no-op
        # for the text and makes the ids trustworthy.
        response_texts = [
            self.tokenizer.decode(response_id, skip_special_tokens=True)
            for response_id in response_ids
        ]

        if return_dict:
            total_lengths = gb["attention_mask"].sum(dim=1)

            return {
                "response_texts": response_texts,
                # Generated token ids, verbatim: consumed downstream as the
                # assistant message's ``token_ids`` so training splices the
                # sampled tokens instead of re-tokenizing decoded text.
                "response_ids": response_ids,
                "total_lengths": total_lengths,
            }

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
        base_url: str = "http://localhost:8000/v1",
        max_requests_per_minute: int = 100,
        timeout: int = 3600,
        api_key: str = "EMPTY",
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
        super().__init__()

        # --- connection
        self.model_name = model_name_or_path
        self.base_url = base_url

        # Detect if it's a Gemini model
        self.is_gemini = self._is_gemini_model(model_name_or_path, base_url)

        if self.is_gemini:
            from google import genai  # lazy: heavy import

            # Initialize once to avoid overhead and connection leaks
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            import openai  # lazy: heavy import

            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

        # --- rate limiting (token bucket, 1 r/s = 60 r/m)
        self._tokens = asyncio.Semaphore(max_requests_per_minute)
        self._max_tokens = max_requests_per_minute
        self._refill_task = None  # started lazily

        # --- misc
        # AF_REQUEST_TIMEOUT bounds ONE model call (seconds). The 3600s default
        # means a degraded endpoint stalls a rollout instead of failing it, so
        # callers that retry (e.g. the trajectory sweep) want a much smaller
        # value — a stuck call then fails fast and is retried rather than
        # holding the slot for the whole agent budget.
        self.timeout = int(os.environ.get("AF_REQUEST_TIMEOUT") or timeout)


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
        from google.genai import types  # lazy: heavy import

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
        elif self.temperature:
            config_kwargs["temperature"] = self.temperature

        # Map 'n' to 'candidate_count'
        if "n" in kwargs:
            config_kwargs["candidate_count"] = kwargs["n"]
        elif "candidate_count" in kwargs:
            config_kwargs["candidate_count"] = kwargs["candidate_count"]
        elif self.n:
            config_kwargs["candidate_count"] = self.n

        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_tokens"]
        elif "max_new_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_new_tokens"]
        elif self.max_tokens:
            config_kwargs["max_output_tokens"] = self.max_tokens

        # FIX: Move tools into the config dictionary. Only when non-empty: agents that
        # don't render tools pass ``[]`` (see BaseAgent.prompt_tools), and an empty
        # tools list is not the same as "no tools" to every API.
        if kwargs.get("tools"):
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

            total_lengths = None
            if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
                total_lengths = getattr(
                    response.usage_metadata, "total_token_count", None
                )

            return {
                "response_texts": response_texts,
                "tool_calls": all_tool_calls,
                "response_dict": raw_response_dict,
                "total_lengths": total_lengths,
            }

        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            return {
                "response_texts": [""],
                "tool_calls": [None],
                "response_dict": {"error": str(e)},
                "total_lengths": None,
            }

    def _blocking_call_openai(self, messages: List[Dict], **kwargs) -> Dict:
        """Make a blocking call to OpenAI API.

        Uses STREAMING by default (set AF_NO_STREAM=1 to disable): some
        gateway paths (e.g. comet smg, 2026-07-29) close idle non-streaming
        connections after ~300s, which kills any generation slower than 5 min;
        SSE keeps bytes flowing so long generations survive. Deltas are
        re-accumulated into the exact non-streaming response shape.
        """
        import openai  # lazy: heavy import (used in the except clauses below)

        logger.debug(f"[ClientBackend] OpenAI model_name: {self.model_name}")
        logger.debug(f"[ClientBackend] OpenAI messages: {len(messages)}")
        logger.debug(f"[ClientBackend] OpenAI kwargs: {kwargs}")

        # Connection drops (server resets, mid-stream cuts under load) are
        # retried here at the single-request level — losing a whole builder
        # trajectory to one reset is far more expensive than re-issuing a turn.
        attempts = int(os.environ.get("AF_CONN_RETRIES", "8"))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                if os.environ.get("AF_NO_STREAM"):
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        timeout=self.timeout,
                        **kwargs,
                    )
                    resp_json = resp.dict()
                else:
                    resp_json = self._streaming_chat_call(messages, **kwargs)
                break
            except (openai.APIConnectionError, openai.NotFoundError,
                    openai.RateLimitError, openai.InternalServerError,
                    openai.APITimeoutError,
                    httpx.RemoteProtocolError, httpx.ReadError) as e:
                # NotFoundError: comet's registry transiently drops workers
                # (model_not_found for a served model); Remote/ReadError: the
                # server can cut a stream mid-flight.
                # RateLimit/InternalServer/Timeout added 2026-07-31: under load
                # the endpoint answers 429 and 502 `upstream unreachable`, and
                # those were NOT retried — one of them ended the whole rollout
                # (795 agentfly failures in 45min at -j 500). Retrying the
                # REQUEST is far cheaper than losing the trajectory.
                last_exc = e
                logger.warning(
                    f"[ClientBackend] retryable error attempt {attempt + 1}/{attempts}: "
                    f"{type(e).__name__}: {e}; retrying"
                )
                # exponential with jitter — a fixed ramp makes every concurrent
                # agent retry in lockstep, re-creating the spike that failed.
                time.sleep(min(90, (2 ** attempt) * 5) * (0.5 + random.random()))
        else:
            raise last_exc
        logger.debug(f"[ClientBackend] resp_json: {resp_json}")
        # A tool-call turn returns content=null; coerce to "" so downstream
        # string handling (content blocks, joins, templating) never sees None.
        response_texts = [
            (choice["message"].get("content") or "") for choice in resp_json["choices"]
        ]
        tool_calls = [
            choice["message"].get("tool_calls") for choice in resp_json["choices"]
        ]
        usage = resp_json.get("usage") or {}
        total_lengths = usage.get("total_tokens") if usage else None

        return {
            "response_texts": response_texts,
            "tool_calls": tool_calls,
            "response_dict": resp_json,
            "total_lengths": total_lengths,
        }

    def _streaming_chat_call(self, messages: List[Dict], **kwargs) -> Dict:
        """chat.completions with stream=True, re-accumulated to the plain
        (non-streaming) response-dict shape so downstream code is unchanged."""
        import openai  # lazy: heavy import (used in the except clause below)

        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                stream_options={"include_usage": True},
                **stream_kwargs,
            )
        except openai.BadRequestError:
            # some servers reject stream_options — retry without usage chunk
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                **stream_kwargs,
            )

        choices: Dict[int, Dict] = {}
        usage = None
        resp_id = resp_model = None
        for chunk in stream:
            cj = chunk.dict() if hasattr(chunk, "dict") else chunk
            resp_id = resp_id or cj.get("id")
            resp_model = resp_model or cj.get("model")
            if cj.get("usage"):
                usage = cj["usage"]
            for ch in cj.get("choices") or []:
                idx = ch.get("index", 0)
                acc = choices.setdefault(idx, {
                    "index": idx,
                    "message": {"role": "assistant", "content": "",
                                "reasoning_content": "", "tool_calls": {}},
                    "finish_reason": None,
                })
                if ch.get("finish_reason"):
                    acc["finish_reason"] = ch["finish_reason"]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    acc["message"]["content"] += delta["content"]
                if delta.get("reasoning_content"):
                    acc["message"]["reasoning_content"] += delta["reasoning_content"]
                for tc in delta.get("tool_calls") or []:
                    ti = tc.get("index", 0)
                    tacc = acc["message"]["tool_calls"].setdefault(ti, {
                        "id": None, "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if tc.get("id"):
                        tacc["id"] = tc["id"]
                    if tc.get("type"):
                        tacc["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        tacc["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tacc["function"]["arguments"] += fn["arguments"]

        out_choices = []
        for idx in sorted(choices):
            acc = choices[idx]
            msg = acc["message"]
            tcs = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"])]
            msg["tool_calls"] = tcs if tcs else None
            if not msg["reasoning_content"]:
                msg.pop("reasoning_content")
            out_choices.append(acc)
        return {
            "id": resp_id,
            "model": resp_model,
            "object": "chat.completion",
            "choices": out_choices,
            "usage": usage,
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

    async def _call(self, messages: List[Dict], **kwargs) -> Dict:
        # acquire a rate‑limit token
        async with self._tokens:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, partial(self._blocking_call, messages, **kwargs)
            )

    # ------------------------------------------------------------------ #
    # Completions endpoint (/v1/completions) — raw prompt path
    # ------------------------------------------------------------------ #

    def _blocking_complete_openai(self, prompt: str, **kwargs) -> Dict:
        """Single blocking call to the OpenAI-compatible /v1/completions endpoint.

        Mirrors `_blocking_call_openai` but for the raw-prompt API. Useful when
        the caller has already pre-rendered the chat template and wants the
        server to consume the prompt verbatim (e.g., evaluating models on
        custom prompt formats without going through chat.completions's
        server-side templating).
        """
        resp = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            timeout=self.timeout,
            **kwargs,
        )
        resp_json = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        choices = resp_json.get("choices") or []
        return {
            "response_texts": [c.get("text", "") for c in choices],
            "tool_calls": [None] * len(choices),
            "response_dict": resp_json,
            "total_lengths": (resp_json.get("usage") or {}).get("total_tokens"),
        }

    async def _call_complete(self, prompt: str, **kwargs) -> Dict:
        async with self._tokens:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, partial(self._blocking_complete_openai, prompt, **kwargs)
            )

    def complete(
        self,
        prompts: "List[str] | str",
        return_dict: bool = False,
        **kwargs,
    ):
        """Public API for the /v1/completions endpoint.

        • Pass a single prompt string → single completion.
        • Pass a list of prompts       → batch completions (parallel via asyncio).

        Returns:
          • In an *async* context → awaitable Task (caller awaits).
          • In a *sync* context   → list of completion strings (or list of
            response dicts when `return_dict=True`).

        This is the prompt-completion counterpart to ``generate(messages, ...)``.
        Gemini does not have a completions endpoint, so this is OpenAI/vLLM
        only.
        """
        if self.is_gemini:
            raise NotImplementedError(
                "Gemini does not expose a /v1/completions endpoint. "
                "Use generate(messages=...) instead."
            )

        if isinstance(prompts, str):
            prompts_list = [prompts]
        else:
            prompts_list = list(prompts)

        async def _runner():
            self._ensure_refiller_running()
            tasks = [
                asyncio.create_task(self._call_complete(p, **kwargs))
                for p in prompts_list
            ]
            response_dicts = await asyncio.gather(*tasks)
            if return_dict:
                return response_dicts
            return [t for rd in response_dicts for t in rd["response_texts"]]

        try:
            loop = asyncio.get_running_loop()  # already inside a loop?
        except RuntimeError:
            return asyncio.run(_runner())
        return loop.create_task(_runner())

    async def complete_async(
        self,
        prompts: "List[str] | str",
        return_dict: bool = False,
        **kwargs,
    ):
        return await self.complete(prompts, return_dict, **kwargs)

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
        # ``tool_call_source`` (from the agent's default_tool_call_source) lets the
        # caller force NATIVE server-side tool calling even on a self-deployed
        # OpenAI-compatible endpoint that isn't on the allowlist below. "backend"
        # => request tool_choice="auto" and keep the structured tool_calls the
        # server returns; "parser" (default) => the legacy self-deployed path
        # (tool_choice="none" + local parsing from content).
        tool_call_source = kwargs.pop("tool_call_source", "parser")

        # Hosted OpenAI-compatible endpoints parse tool calls server-side and
        # return them in the structured `tool_calls` field. Self-deployed
        # vLLM/sglang do not, so we have to fall back to local parsing. Keep
        # this list explicit: false negatives clobber `tool_choice` to "none"
        # and silently break function calling on the affected provider.
        is_openai_model = False
        if not self.is_gemini and (
            "gpt" in self.model_name.lower()
            or "api.openai.com" in self.base_url
            or "api.deepseek.com" in self.base_url
            or "api.openai-next.com" in self.base_url
            or "api.anthropic.com" in self.base_url
            or "openrouter.ai" in self.base_url
        ):
            is_openai_model = True

        # Native server-side tool calling: allowlisted hosted models, OR any
        # endpoint where the agent explicitly opted into backend tool calls.
        native_tools = is_openai_model or tool_call_source == "backend"

        if not self.is_gemini:
            messages_list = [
                self._convert_to_openai_chat_without_tool_call_processing(
                    messages, native_tools
                )
                for messages in messages_list
            ]

        if "tools" in kwargs:
            if self.is_gemini:
                # Gemini handles tools differently - convert to function declarations
                # This will be handled in the Gemini call if needed
                pass
            elif native_tools:
                kwargs["tool_choice"] = "auto"
            else:
                # For self-deployed models, we will use the response to extract tool calls
                kwargs["tool_choice"] = "none"

        # Skills are a non-standard OpenAI field. Self-deployed servers (vLLM)
        # consume them via the chat template by way of ``chat_template_kwargs``
        # in ``extra_body``. Skills are unsupported on Gemini / hosted OpenAI.
        if "skills" in kwargs:
            skills = kwargs.pop("skills")
            if skills and not self.is_gemini and not is_openai_model:
                extra_body = kwargs.setdefault("extra_body", {})
                chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
                chat_template_kwargs["skills"] = skills

        return messages_list, kwargs

    # Public API ‑‑ sync or async depending on caller's context
    def generate(
        self,
        messages: List[List[Dict]] | List[Dict],
        return_dict: bool = False,
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
                total_lengths = response_dict.get("total_lengths")

                # Collect all response texts for non-full-response mode
                all_response_texts.extend(response_texts)

                # Build the structured response dict
                final_response_dicts.append(
                    {
                        "response_texts": response_texts,
                        "tool_calls": tool_calls,
                        "response_dict": full_response_dict,
                        "total_lengths": total_lengths,
                    }
                )

            if return_dict:
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
        self, messages: List[List[Dict]] | List[Dict], return_dict: bool = False, **kwargs
    ) -> List[str]:
        return await self.generate(messages, return_dict, **kwargs)

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
