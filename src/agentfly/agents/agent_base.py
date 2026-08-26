import inspect
import json
import logging
import os
from abc import ABC
from math import lcm
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from chat_bricks import (
    split_messages_with_assistant,
    tokenize_conversations,
)
from termcolor import colored

from ..templates import *  # noqa: F403
from ..rewards.types import RewardResult
from ..tools.tool_base import BaseTool
from ..tools.src.skills import Skill
from ..core.context import Context
from ..core.context_config import ContextConfig
from ..utils.monitor import JsonlSink, Monitor, WandbSink
from ..utils.verl import pad_tensor_batch_dim_with_zeros, pad_tensor_to_rank_size
from .rollout.chain import ChainRollout
from .types import RunResult, Trajectory
from ..utils.llm_backends import AsyncVerlBackend, AsyncVLLMBackend, ClientBackend
from ..utils.llm_backends.backend_configs import BACKEND_CONFIGS
from .utils.messages import MessagesList
from .utils.tokenizer import create_processor, create_tokenizer, get_jinja_template
from .utils.tool_parser import (
    ChatCompletionRequest,
    VLLM_TOOL_PARSER_AVAILABLE,
    create_tool_parser,
)

try:
    from ..verl.protocol import DataProto
except ImportError:
    print("verl can not be imported.")
    pass

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


REWARD_DECOMPOSITION_MODES = ("last_only", "broadcast", "uniform", "geometric")


def decompose_trajectory_reward(
    reward: float,
    num_segments: int,
    mode: str = "last_only",
    gamma: float = 0.9,
) -> List[float]:
    """Distribute a trajectory's outcome reward across its segments.

    Modes:
        - "last_only": [0, 0, ..., 0, R]
              Sparse outcome — only the final segment receives the reward.
        - "broadcast": [R, R, ..., R]
              Each segment sees the full R. Total trajectory reward = n * R
              (NOT conservative — inflates magnitude with segment count).
        - "uniform":   [R/n, R/n, ..., R/n]
              Sum-to-R, equal weight per segment. Conservative.
        - "geometric": [R · γ^{n-1-k} · (1 − γ) / (1 − γ^n)] for k = 0..n-1
              Sum-to-R, more weight on segments closer to the outcome.
              γ → 1 collapses to "uniform". Conservative.

    Conservative modes preserve total trajectory reward equal to R, so PPO
    clipping ranges, KL budgets, and learning rates tuned against sparse
    rewards continue to apply without retuning.

    Args:
        reward: trajectory outcome reward R.
        num_segments: number of segments n in the trajectory (>= 1).
        mode: one of REWARD_DECOMPOSITION_MODES.
        gamma: decay rate for "geometric" mode. Ignored otherwise.

    Returns:
        List of per-segment rewards of length num_segments.

    Raises:
        ValueError: if mode is unknown or num_segments < 1.
    """
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}")
    if mode not in REWARD_DECOMPOSITION_MODES:
        raise ValueError(
            f"mode must be one of {REWARD_DECOMPOSITION_MODES}, got: {mode!r}"
        )

    n = num_segments
    if mode == "last_only":
        return [0.0] * (n - 1) + [reward]
    if mode == "broadcast":
        return [reward] * n
    if mode == "uniform":
        per = reward / n
        return [per] * n
    # geometric
    if n == 1 or gamma >= 1.0:
        per = reward / n
        return [per] * n
    raw = [gamma ** (n - 1 - k) for k in range(n)]
    total = sum(raw)
    return [reward * (w / total) for w in raw]


def _resolve_reward_decomposition_config() -> tuple:
    """Resolve (mode, gamma) for trajectory reward decomposition from env vars.

    Env vars:
        REWARD_DECOMPOSITION:       one of REWARD_DECOMPOSITION_MODES.
                                    Default: "last_only".
        REWARD_DECOMPOSITION_GAMMA: float decay rate for "geometric" mode.
                                    Default: 0.9.
    """
    env_decomp = os.getenv("REWARD_DECOMPOSITION")
    if env_decomp is not None and env_decomp.strip() != "":
        mode = env_decomp.strip().lower()
        if mode not in REWARD_DECOMPOSITION_MODES:
            raise ValueError(
                f"REWARD_DECOMPOSITION must be one of {list(REWARD_DECOMPOSITION_MODES)}, "
                f"got: {env_decomp!r}"
            )
    else:
        mode = "last_only"

    env_gamma = os.getenv("REWARD_DECOMPOSITION_GAMMA", "0.9")
    try:
        gamma = float(env_gamma)
    except ValueError as exc:
        raise ValueError(
            f"REWARD_DECOMPOSITION_GAMMA must be a float, got: {env_gamma!r}"
        ) from exc

    return mode, gamma


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
        skills: List = [],
        max_model_len: int = None,
        backend_config: Optional[Dict[str, Any]] = None,
        reward_fn: Callable = None,
        debug: bool = False,
        monitors: List[str] = ["wandb"],
        wandb_project_name: str = None,
        wandb_run_name: str = None,
        local_cache_dir: str = None,
        tool_parser: Optional[Any] = None,
        tool_parser_name: Optional[str] = None,
        default_tool_call_source: str = "parser",
        invalid_tool_call_as_observation: bool = True,
        **kwargs,  # To pass other unused arguments
    ):
        """
        Args:
            model_name_or_path: The name of the model to use.
            template: The template to use for the agent.
            system_prompt: The system prompt to use for the agent.
            tools: The tools to use for the agent.
            skills: Optional list of ``Skill`` objects (or skill names) to advertise to the model in the system
                prompt's ``<available_skills>`` section. The skill content (SKILL.md body, scripts, references)
                is loaded on demand via the ``load_skill`` tool — only ``name`` + ``description`` are surfaced here.
            debug: Whether to enable debug mode.
            backend_config: Dict specifying the backend and its parameters. Must include "backend" (e.g. "async_vllm", "client").
                Other keys are passed as kwargs to that backend (e.g. "gpu_memory_utilization" for async_vllm).
                Defaults to {"backend": "async_vllm"}.
            tool_parser: Optional tool parser instance from vLLM. If provided, will be used for parsing tool calls.
            tool_parser_name: Optional name of the tool parser to use (e.g., "hermes", "pythonic"). If provided and tool_parser is None, will create a parser using this name.
            default_tool_call_source: Which tool calls to use when both are available. ``"parser"`` (default) re-parses
                the response text with the agent's own tool parser (falling back to backend-provided tool calls only
                when no parser is set); ``"backend"`` prefers tool calls already parsed by the inference backend
                (e.g. an OpenAI-compatible server with ``--enable-auto-tool-choice``), falling back to the local
                parser per response when the backend provided none.
            invalid_tool_call_as_observation: Policy for an *invalid tool call* — one the model formed with the
                wrong shape: an unknown/hallucinated tool name, arguments that are not a JSON object, or unknown
                argument names. ``True`` (default) feeds the model a ``status="error"`` observation with an
                informative hint (which tools are available / which arguments are expected) so the chain continues
                and the model can recover on the next turn; ``False`` raises ``InvalidToolCallError`` (fail-fast),
                aborting the rollout on the first invalid call. This is deliberately distinct from an error raised
                *inside* a tool's body while it runs — that is a runtime tool error, governed separately by the
                ``TOOL_ERROR_AS_OBSERVATION`` env policy.

        """
        if default_tool_call_source not in ("parser", "backend"):
            raise ValueError(
                f"default_tool_call_source must be 'parser' or 'backend', got: {default_tool_call_source!r}"
            )
        if backend_config is None:
            backend_config = {"backend": "async_vllm"}
        self._validate_init_args(
            model_name_or_path,
            template,
            system_prompt,
            tools,
            backend_config,
            reward_fn,
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
        self.default_tool_call_source = default_tool_call_source
        self.invalid_tool_call_as_observation = invalid_tool_call_as_observation
        self.tool_names = [tool.name for tool in tools]
        self.skills = self._normalize_skills(skills)
        self.skill_names = [s.name for s in self.skills]

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
            self.jinja_template = get_jinja_template(self.template)

        self.llm_engine = self._init_llm_engine(model_name_or_path, self.backend)

        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.local_cache_dir = local_cache_dir
        self.local_run_cache_dir = None
        self._initialize_monitor(monitors)

        # Initialize tool parser
        self.tool_parser = tool_parser
        if self.tool_parser is None and tool_parser_name is not None:
            self.tool_parser = create_tool_parser(tool_parser_name, self.tokenizer)

        super().__init__()

        # Result of the most recent agent.run(...) call. Populated by run();
        # consumed by get_verl_data_proto(). None until the first run.
        self._last_run_result: Optional["RunResult"] = None

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

            # For async_vllm, prefer explicit kwargs like tensor/data parallel sizes over
            # a default config's pre-built engine_args placeholder.
            if backend == "async_vllm":
                explicit_async_vllm_keys = {
                    "tensor_parallel_size",
                    "data_parallel_size",
                    "pipeline_parallel_size",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "dtype",
                    "quantization",
                    "trust_remote_code",
                }
                if any(k in self.backend_config for k in explicit_async_vllm_keys):
                    config_kwargs.pop("engine_args", None)

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
        system_prompt = self.system_prompt
        if system_prompt:
            if "{tools}" in system_prompt:
                system_prompt = system_prompt.replace(
                    "{tools}", json.dumps(tools, indent=4)
                )
            # Skills ride in the system message content (not the chat template's
            # ``chat_template_kwargs``) so hosted models (OpenAI/Gemini) advertise
            # them the same way local vLLM does. ``{skills}`` expands to the full
            # block or "" — no dangling header when the agent has no skills.
            if "{skills}" in system_prompt:
                system_prompt = system_prompt.replace(
                    "{skills}", self._render_skills_block()
                )

        for messages in messages_list:
            if system_prompt:
                messages.set_system_prompt(system_prompt, enforce=False)

        return messages_list.to_list()

    @staticmethod
    def _normalize_skills(skills) -> List["Skill"]:
        """Accept ``Skill`` objects or skill names (str); resolve names to ``Skill``."""
        if not skills:
            return []
        normalized: List[Skill] = []
        for item in skills:
            if isinstance(item, Skill):
                normalized.append(item)
            elif isinstance(item, str):
                from ..tools.src.skills import load_skills
                normalized.extend(load_skills([item]))
            else:
                raise TypeError(
                    f"skills entries must be Skill or str, got {type(item).__name__}"
                )
        return normalized

    def _render_skills_block(self) -> str:
        """Render the agent's skills for the system-prompt ``{skills}`` slot.

        Returns the full ``<available_skills>`` block (lead-in + one entry per
        skill) when the agent has skills, or an empty string when it has none —
        so the ``{skills}`` slot leaves no dangling header for a skill-less agent.
        Only ``name`` + ``description`` are surfaced here; the SKILL.md body,
        scripts, and references are loaded on demand via the ``load_skill`` tool.
        """
        if not self.skills:
            return ""
        entries = "\n".join(
            f"<skill>\n<name>{s.name}</name>\n"
            f"<description>{s.description}</description>\n</skill>"
            for s in self.skills
        )
        return (
            "You have the following available skills:\n"
            f"<available_skills>\n{entries}\n</available_skills>"
        )

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
        context_config: Optional[ContextConfig] = None,
        **kwargs,
    ) -> RunResult:
        """Run the agent on a batch of messages and return the rollout result.

        This is the main interface for running the agent. It is a wrapper of
        different rollout methods, which must be asynchronous. Currently we
        only support chain-based rollout.

        Args:
            messages: List of messages to generate responses for.
            max_turns: The maximum number of turns to generate.
            generation_config: The generation configuration.
            context_config: Optional settings for :class:`~agentfly.core.context.Context` (resource backend).
            **kwargs: Additional keyword arguments for generation (passed to ``run_async``).

        Returns:
            :class:`~agentfly.agents.types.RunResult` containing the trajectories
            produced by this run. The same result is also cached on the agent so
            that :meth:`get_verl_data_proto` can be called afterwards without
            re-running.
        """
        processed_messages = self._preprocess_messages(messages)
        self._preprocess_backends()

        await self.run_async(
            processed_messages,
            max_turns=max_turns,
            generation_config=generation_config,
            context_config=context_config,
            **kwargs,
        )

        self._postprocess_backends()

        trajectories = self.postprocess_trajectories(self.get_trajectories())
        self._last_run_result = RunResult(trajectories=trajectories)
        return self._last_run_result

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

    @property
    def timing_data(self):
        return self.timer.timing_data

    def postprocess_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Hook for subclasses to post-process trajectories before they're
        wrapped in a :class:`RunResult`. Default implementation is identity.
        """
        return trajectories

    def _require_last_run(self) -> RunResult:
        """Return the cached :class:`RunResult` from the most recent
        :meth:`run` call, raising a clear error if ``run`` hasn't been called yet."""
        if self._last_run_result is None:
            raise RuntimeError(
                "agent.run(...) has not been called yet; no RunResult is available. "
                "Call `result = await agent.run(...)` first."
            )
        return self._last_run_result

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

    def parse(
        self,
        responses: List[str],
        context: Optional[Context] = None,
        tool_calls: Optional[List] = None,
        **kwargs,
    ) -> List[Dict]:
        """
        This method is used to define the interaction logic of the agent. It can be used to parse the tool call from the response.
        If tool_parser is provided, it will use the vLLM tool parser by default. Otherwise, subclasses should override this method.

        Args:
            responses: List of responses to parse.
            context: Optional rollout context carrying trajectory and metadata.
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
        if tool_calls is not None and not any(tool_calls):
            tool_calls = None

        # ``default_tool_call_source`` decides whether the local parser or the
        # backend-provided tool calls take precedence when both are available.
        prefer_backend = getattr(self, "default_tool_call_source", "parser") == "backend"

        # If a local tool parser is available, use it.
        if self.tool_parser is not None:
            if self.tools is None or len(self.tools) == 0:
                return [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                        "tool_calls": [],
                        "loss": True,
                        "status": "terminal",
                    }
                    for response in responses
                ]
            # When the parser is preferred (default), ignore backend tool calls and
            # re-parse the text; when the backend is preferred, pass them through so
            # ``_parse_with_tool_parser`` uses them (falling back to the parser per response).
            return self._parse_with_tool_parser(
                responses,
                backend_tool_calls=tool_calls if prefer_backend else None,
            )
        elif tool_calls is not None or self.backend == "client":
            # No local parser: tool calls (if any) come from the backend itself, e.g. an
            # OpenAI-compatible server with --enable-auto-tool-choice. A response with no
            # tool call is a valid terminal text turn, not a parsing failure — so the client
            # backend never needs a local parser, even when this turn produced no tool call.
            def _calls_for(i: int):
                if tool_calls is None or i >= len(tool_calls):
                    return None
                return tool_calls[i]

            return [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "tool_calls": self._format_tool_calls(_calls_for(i)),
                    "loss": True,
                    "status": "continue" if _calls_for(i) else "terminal",
                }
                for i, response in enumerate(responses)
            ]
        else:
            # No parser, and the backend does not parse tool calls for us (e.g. async_vllm /
            # verl return raw text only). The subclass must implement parsing or provide one.
            raise NotImplementedError(
                "parse method must be implemented by subclass or tool_parser must be provided. "
                "Either override this method or provide tool_parser/tool_parser_name in __init__."
            )

    def _format_tool_calls(self, raw_tool_calls: Optional[List]) -> List[Dict]:
        """Normalize tool calls (vLLM ToolCall objects or OpenAI-style dicts) to our format.

        Accepts either vLLM ``ToolCall`` objects (from the local tool parser) or plain
        dicts (from an OpenAI-compatible backend). Tool calls whose ``arguments`` are not
        valid JSON are dropped.
        """
        formatted: List[Dict] = []
        for tool_call in raw_tool_calls or []:
            # vLLM ToolCall object with .function.name / .function.arguments
            if hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                call_id = getattr(tool_call, "id", None)
                call_type = getattr(tool_call, "type", "function")
            elif isinstance(tool_call, dict) and "function" in tool_call:
                func_info = tool_call["function"]
                if isinstance(func_info, dict):
                    name = func_info.get("name", "")
                    arguments_str = func_info.get("arguments", "")
                else:
                    name = getattr(func_info, "name", "")
                    arguments_str = getattr(func_info, "arguments", "")
                call_id = tool_call.get("id", None)
                call_type = tool_call.get("type", "function")
            else:
                continue

            # Validate that arguments is a valid JSON string before accepting the call.
            try:
                json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Invalid JSON in tool call arguments for {name}: {arguments_str}"
                )
                continue

            formatted.append(
                {
                    "id": call_id,
                    "type": call_type,
                    "function": {"name": name, "arguments": arguments_str},
                }
            )
        return formatted

    def _parse_with_tool_parser(
        self,
        responses: List[str],
        backend_tool_calls: Optional[List] = None,
    ) -> List[Dict]:
        """
        Build assistant messages, preferring tool calls already parsed by the backend.

        When the inference backend (e.g. an OpenAI-compatible server started with
        ``--enable-auto-tool-choice``) parses tool calls itself, it strips the tool-call
        markup out of ``content`` and returns structured ``tool_calls``. In that case we
        must use those directly: re-running the local vLLM parser on the stripped text
        would find nothing. Only when the backend did not provide tool calls for a given
        response do we fall back to parsing the raw text with ``self.tool_parser``.

        Args:
            responses: List of response strings to parse.
            backend_tool_calls: Optional per-response tool calls from the backend, aligned
                to ``responses``. Each entry is a list of OpenAI-style tool-call dicts, or
                ``None`` when the backend did not parse any for that response.

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
        for i, response in enumerate(responses):
            backend_calls = (
                backend_tool_calls[i]
                if backend_tool_calls is not None and i < len(backend_tool_calls)
                else None
            )

            if backend_calls:
                # Backend already parsed tool calls and stripped them from the text;
                # trust those instead of re-parsing the stripped response.
                formatted_tool_calls = self._format_tool_calls(backend_calls)
                info_status = None
            else:
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
                formatted_tool_calls = self._format_tool_calls(info.tool_calls)
                info_status = getattr(info, "status", None)

            # Use the full response text (not the text after removing tool calls)
            content_text = response

            message = {
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                "tool_calls": formatted_tool_calls,
                "loss": True,
            }

            # Add status if available
            if info_status is not None:
                message["status"] = info_status
            elif len(formatted_tool_calls) > 0:
                message["status"] = "continue"
            else:
                message["status"] = "terminal"

            new_messages_list.append(message)

        return new_messages_list


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
        pad_to_multiple_of: Optional[int] = None,
    ):
        """Convert the last ``run``'s trajectories into a verl ``DataProto``.

        This is the agent→trainer boundary. It flattens trajectories into rows
        (one row per **segment**; for the common non-folded rollout that is one
        row per trajectory), tokenizes them, and packs the fields the trainer
        consumes. The returned ``DataProto`` carries the following contract.

        ``batch`` (tensors, shape ``[B, L]`` unless noted):

        - ``input_ids`` / ``attention_mask`` / ``position_ids`` — the sequence.
        - ``action_mask`` — 1 on assistant (policy) tokens, 0 elsewhere; the
          per-token loss mask. Contiguous runs of 1s delimit turns.
        - ``reward_mask`` — 1 on the single token that carries the outcome
          reward (the trajectory's last assistant token).
        - ``rm_scores`` — the scalar outcome reward placed on the ``reward_mask``
          token (``reward_mask * reward``); ``rm_scores.sum(-1)`` per row is the
          episode outcome.
        - ``multi_modal_inputs`` — present only for vision-language models.

        ``non_tensor_batch`` (arrays, shape ``[B]``):

        - ``uid`` — prompt/group id shared by a prompt's rollouts; the group a
          group-relative estimator (GRPO, GiGPO episode level) normalizes within.
        - ``batch_idx`` — per-trajectory index; the de-facto trajectory id (all
          rows of one trajectory share it).
        - ``segment_idx`` — segment order within a trajectory (0 when not folded).
        - ``rm_<key>`` — one array per extra key a reward dict returned
          (e.g. ``rm_f1``), broadcast to each of the trajectory's rows.
        - ``step_observations`` / ``step_rewards`` — per-turn lists (turn order,
          projected from the row's ``tool_results``): the observation the agent
          acted on (grouping anchor) and the per-step reward. Consumed by
          step-level estimators (GiGPO), which map them to token spans via a
          ``turn_ids`` derived from ``action_mask``; ignored otherwise.

        ``meta_info``:

        - ``use_agent`` — marks this as an agent-produced batch.
        - ``repeat_times`` — number of segments per trajectory.

        Args:
            train_on_last_turn: (forced ``False``) restrict the loss to the last turn.
            world_size: data-parallel world size.
            pad_to_multiple_of: pad the batch dimension to a multiple of this
                (extra rows repeat the last row) so it shards evenly.
        """
        run_result = self._require_last_run()
        trajectories = run_result.trajectories
        segments_list = []
        other_info_list = []
        for batch_idx, trajectory in enumerate(trajectories):
            trajectory_segments = trajectory.segments
            for segment_idx, segment in enumerate(trajectory_segments):
                segments_list.append(segment)
                # Per-segment metadata: everything about the trajectory
                # except segments themselves, plus segment indexing.
                info = trajectory.model_dump(exclude={"segments"})
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

        reward_values = run_result.rewards
        other_values = defaultdict(list)
        for k, v in run_result.reward_extras.items():
            other_values[k] = list(v)

        # Expand trajectory reward to segment level. See docstring of
        # `decompose_trajectory_reward` for the available modes.
        decomposition_mode, decomposition_gamma = _resolve_reward_decomposition_config()
        reward_values_segment: List[float] = []
        for trajectory in trajectories:
            n = trajectory.num_segments
            if n <= 0:
                continue
            reward_values_segment.extend(
                decompose_trajectory_reward(
                    reward=trajectory.reward,
                    num_segments=n,
                    mode=decomposition_mode,
                    gamma=decomposition_gamma,
                )
            )
        num_trajectories = len(trajectories)
        other_values_segment = {}
        for key, values in other_values.items():
            # When reward is scalar (not dict), that trajectory never appends to other_values[key],
            # so values may be shorter than num_trajectories. Pad to match.
            if len(values) < num_trajectories:
                values = list(values) + [0.0] * (num_trajectories - len(values))
            other_values_segment[key] = []
            for traj_idx, trajectory in enumerate(trajectories):
                n = trajectory.num_segments
                val = values[traj_idx]
                other_values_segment[key].extend([val] * n)
        reward_values = reward_values_segment
        other_values = other_values_segment

        # Number of segments per trajectory (one integer per trajectory).
        repeat_times = [t.num_segments for t in trajectories]

        align = pad_to_multiple_of
        if align and align > 1:
            n = inputs["input_ids"].shape[0]
            pad_size = (align - n % align) % align
            for k, v in inputs.items():
                if k == "action_mask" and isinstance(v, torch.Tensor) and not v.is_nested:
                    inputs[k] = pad_tensor_batch_dim_with_zeros(v, align)
                else:
                    inputs[k] = pad_tensor_to_rank_size(v, align)
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

        # Per-turn rollout signals projected from each row's trajectory
        # ``tool_results`` (turn order). Only these clean arrays cross the
        # boundary; the raw ``tool_results`` stay agent-side. Consumed by
        # step-level estimators (GiGPO); ignored otherwise.
        #
        # ANCHOR = the state the agent acted FROM (pre-action s_t). GiGPO groups
        # turns by the state the action was taken from; ``tool_results[t].observation``
        # is the observation returned AFTER action t (post-action s_{t+1}), so we shift
        # the observations right by one — anchor[t] = the previous turn's observation,
        # with a shared marker for the first turn (all chains of a prompt share s_0, and
        # grouping is scoped within uid). ``step_reward[t]`` is the reward from action t
        # (``tool_results[t]``), which is already correctly aligned.
        step_observations_list = []
        step_rewards_list = []
        for info in other_info_list:
            tr = info.get("tool_results") or []
            # Prefer the raw ``anchor`` (undecorated state key) over the LLM-facing
            # ``observation`` (which may carry an admissible-action menu that fragments
            # exact-hash grouping); fall back to ``observation`` when no anchor is set.
            post_obs = [
                r.get("anchor") if r.get("anchor") is not None else r.get("observation")
                for r in tr
            ]
            step_observations_list.append((["__init__"] + post_obs[:-1]) if post_obs else [])
            step_rewards_list.append([r.get("step_reward") for r in tr])


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
        # 1D object arrays of per-turn lists (np.empty avoids equal-length rows
        # collapsing into a 2D array).
        step_observations = np.empty(len(step_observations_list), dtype=object)
        step_observations[:] = step_observations_list
        step_rewards = np.empty(len(step_rewards_list), dtype=object)
        step_rewards[:] = step_rewards_list
        inputs["step_observations"] = step_observations
        inputs["step_rewards"] = step_rewards
        
        if "mm_inputs" in inputs:
            mm_inputs = inputs.pop("mm_inputs")
            inputs["multi_modal_inputs"] = np.array(mm_inputs, dtype=object)
        batch = DataProto.from_single_dict(
            inputs, meta_info={"use_agent": True, "repeat_times": repeat_times}
        )

        return batch
