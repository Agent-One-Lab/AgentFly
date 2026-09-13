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
# NOTE: ``torch`` is imported lazily inside the two methods that use it
# (``tokenize_trajectories`` / ``to_verl_dataproto``) so a bare
# ``import agentfly.agents`` doesn't pull torch (~3s). It's only needed when
# actually building tensors for the trainer.
from chat_bricks import (
    split_messages_with_assistant,
    tokenize_conversations,
)

from ..templates import *  # noqa: F403
from ..rewards.types import RewardResult
from ..tools.tool_base import BaseTool, submit_tool_call
from ..tools.src.skills import Skill
from ..core.context import Context
from ..core.context_config import ContextConfig
from ..utils.monitor import JsonlSink, Monitor, WandbSink
from ..utils.verl import pad_tensor_batch_dim_with_zeros, pad_tensor_to_rank_size
from .rollout.registry import resolve_rollout
from .rollout.conversion import result_to_dataproto
from .types import RunResult, Trajectory
from ..utils.llm_backends import AsyncVerlBackend, AsyncVLLMBackend, ClientBackend
from ..utils.llm_backends.backend_configs import BACKEND_CONFIGS
from .utils.messages import MessagesList
from .utils.tokenizer import create_processor, create_tokenizer, get_jinja_template
# Import the module (not the vLLM-derived values) so the vLLM stack loads lazily:
# read ``tool_parser.ChatCompletionRequest`` / ``VLLM_TOOL_PARSER_AVAILABLE`` at
# runtime, after ``tool_parser.ensure_vllm_tool_parser()``, rather than binding
# them here at import time.
from .utils import tool_parser
from .utils.tool_parser import create_tool_parser

# NOTE: ``DataProto`` (from the verl trainer) is imported lazily inside
# ``to_verl_dataproto`` — it pulls the whole verl/torch/ray stack, which a bare
# ``import agentfly.agents`` must not load. It's only needed when actually
# converting trajectories to a verl batch (i.e. under the trainer).


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
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
        render_tools_in_prompt: bool = True,
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
            render_tools_in_prompt: Whether the tools' schemas are rendered into the model prompt
                (the chat template's tools block). ``True`` for native tool-calling agents. ``False``
                for agents whose prompt already carries the environment interface directly (e.g. the
                ``<action>`` format), where the tools exist only to execute the parsed action. Read
                exclusively through :meth:`prompt_tools`, which both generation and training
                tokenization consult, so the two always render the same prompt.

        Note:
            Loop control — whether a malformed tool call, a tool-body error, or a no-tool-call
            turn continues / ends / raises — is NOT an agent setting. It's the rollout's, via the
            run-config policies ``on_no_tool_call`` / ``on_invalid_tool_call`` / ``on_tool_error``
            (see :class:`~agentfly.agents.rollout.base.Rollout`).

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
        self.tool_names = [tool.name for tool in tools]
        self.render_tools_in_prompt = render_tools_in_prompt
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
        rollout: Union[str, "Rollout"] = "chain",
        rollout_config: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = {},
        context_config: Optional[ContextConfig] = None,
        **kwargs,
    ) -> RunResult:
        """Run the agent on a batch of messages and return the rollout result.

        This is the main interface for running the agent. The *rollout* is how the
        agent completes the task — a per-call :class:`~agentfly.agents.rollout.base.Rollout`
        strategy (``"chain"`` by default). The agent passes itself to the strategy as the
        agent; the strategy drives the loop and reaches agent-specific behavior back
        through the agent.

        Args:
            messages: List of messages to generate responses for.
            max_turns: The maximum number of turns to generate.
            rollout: The rollout strategy — a registered name (``"chain"``), an import
                reference (``"pkg.mod:MyRollout"``), or a ``Rollout`` instance.
            generation_config: The generation configuration.
            context_config: Optional settings for :class:`~agentfly.core.context.Context` (resource backend).
            **kwargs: Additional keyword arguments forwarded to the rollout strategy
                (e.g. ``num_chains``, ``max_concurrent_chains`` for the chain rollout).

        Returns:
            :class:`~agentfly.agents.types.RunResult` containing the trajectories
            produced by this run, including its rollout identifier. Pass this result
            explicitly to :meth:`to_verl_dataproto` for training conversion.
            Latest-run caches are retained only for inspection and timing helpers.
        """
        processed_messages = self._preprocess_messages(messages)
        self._preprocess_backends()

        # ``rollout_config`` holds the strategy's constructor kwargs (e.g.
        # ``{"prompt_builder": "alfworld_flat", "history_length": 2}`` for StepRollout),
        # so a training config can select AND configure the rollout. Coerce an OmegaConf
        # mapping to a plain dict; ignored when ``rollout`` is already a Rollout instance.
        rc = dict(rollout_config) if rollout_config else {}
        resolved_rollout = resolve_rollout(rollout, **rc)
        run_result = await resolved_rollout.run(
            agent=self,
            messages=processed_messages,
            max_turns=max_turns,
            generation_config=generation_config,
            context_config=context_config,
            **kwargs,
        )

        self._postprocess_backends()

        trajectories = self.postprocess_trajectories(run_result.trajectories)
        run_result = RunResult(trajectories=trajectories, rollout=run_result.rollout)
        return run_result

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

    def postprocess_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Hook for subclasses to post-process trajectories before they're
        wrapped in a :class:`RunResult`. Default implementation is identity.
        Training-row eligibility is handled by conversion, not this rollout hook.
        """
        return trajectories

    # ---- L1 rollout hooks (defaults; specialized agents override) ----
    # These are reached by a rollout strategy through the agent (the agent). The
    # defaults live here so `agent.<hook>` resolves to the agent's (possibly
    # overridden) behavior under composition.

    def maybe_append_context_trigger_user_message(self, current_step) -> None:
        """Hook for subclasses to append a user turn before the next LLM call. No-op by default."""
        return

    def skills_payload(self) -> Optional[List[Dict[str, str]]]:
        """Skill list as JSON-serializable dicts for ``chat_template_kwargs``.

        This is the *template* path: chat-bricks renders the skills slot from
        these dicts. It is mutually exclusive with the agent-layer ``{skills}``
        substitution in ``_preprocess_messages`` — when the system prompt carries
        a ``{skills}`` slot the agent already renders the block into the system
        message, so we return None here to avoid injecting skills twice.

        Returns None when no skills are configured (so the backend can skip the
        ``extra_body`` field entirely) or when the agent renders skills itself.
        """
        skills = getattr(self, "skills", None)
        if not skills:
            return None
        system_prompt = getattr(self, "system_prompt", None)
        if system_prompt and "{skills}" in system_prompt:
            return None
        return [{"name": s.name, "description": s.description} for s in skills]

    def prompt_tools(self) -> List[Dict]:
        """Tool schemas rendered into the model prompt (``[]`` for none).

        This is the ONLY place prompt tools come from: the rollout passes the
        result to the backend for generation AND to ``tokenize_trajectories``
        for training, so sampling and training are conditioned on the same
        prompt by construction. Returns ``[]`` (not ``None``) so it can flow
        through numpy/DataProto columns unchanged.
        """
        if not self.render_tools_in_prompt or not self.tools:
            return []
        return [tool.schema for tool in self.tools]

    def validate_tool_call(self, tool_call):
        tool_name = tool_call["function"]["name"]
        # TODO: validate tool input
        tool_input = tool_call["function"]["arguments"]  # noqa: F841
        if tool_name not in self.tool_names:
            return False
        return True

    async def execute_tool_call(
        self,
        context,
        tool_call,
        newest_messages,
        chain,
        chain_id,
        depth,
        have_set_resources,
    ):
        """Execute a tool call."""
        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]

        # Set up tools if needed (reset resources that may have been acquired at chain start)
        if not have_set_resources:
            env_args = {
                k: context.metadata[k]
                for k in ("task_name", "variation_idx")
                if k in context.metadata
            }
            # We have moved reset to the tool call
            # if env_args:
            #     await context.reset_resource(scope="rollout", env_args=env_args)
            #     await context.reset_resource(scope="global", env_args=env_args)
            # else:
            #     await context.reset_resource(scope="rollout")
            #     await context.reset_resource(scope="global")
            # have_set_resources = True

        # Execute tool call. A malformed call comes back as a ``status="invalid"`` result
        # (not raised); the rollout applies its ``on_invalid_tool_call`` policy.
        result = await submit_tool_call(
            tool_name,
            tool_input,
            context=context,
            allowed_tool_names=self.tool_names,
        )

        return result

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
        if not tool_parser.ensure_vllm_tool_parser():
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

                req = tool_parser.ChatCompletionRequest(**req_dict)

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

    def to_verl_dataproto(
        self,
        run_result: RunResult,
        *,
        train_on_last_turn: bool = False,
        world_size: int = 1,
        pad_to_multiple_of: Optional[int] = None,
        log_token_drift: bool = True,
    ):
        """Convert an explicit ``RunResult`` into a verl ``DataProto``.

        This is the agent→trainer boundary. It flattens trajectories into rows
        (one row per trainable **segment**), tokenizes them, and packs trainer fields.
        ``run_result.rollout`` selects the stateless converter; latest-run caches
        are never consulted. Built-ins are ``"chain"`` and ``"step"``; missing
        or unsupported identifiers raise ``ValueError``.

        Empty/context-only segments are excluded from the batch, as are rows
        with no action-mask tokens after tokenization. No trajectory, segment,
        or runtime step is removed from the supplied result. An entirely
        untrainable batch raises ``ValueError``.

        Use the in-memory result: JSON serialization excludes internal steps and
        is not a complete training snapshot. The agent supplies its current
        tokenizer, processor, template, and prompt tools.

        The returned ``DataProto`` carries the following contract.

        ``batch`` (tensors, shape ``[B, L]`` unless noted):

        - ``input_ids`` / ``attention_mask`` / ``position_ids`` — the sequence.
        - ``action_mask`` — 1 on assistant (policy) tokens, 0 elsewhere; the
          per-token loss mask. Contiguous runs of 1s delimit turns.
        - ``reward_mask`` — 1 on the single token that carries the outcome
          reward (each segment's last assistant token).
        - ``rm_scores`` — the scalar outcome reward placed on the ``reward_mask``
          token (``reward_mask * reward``). The full episode outcome is broadcast
          to every retained segment, including earlier context-folded views;
          it is not divided by the number of segments.
        - ``multi_modal_inputs`` — present only for vision-language models.

        ``non_tensor_batch`` (arrays, shape ``[B]``):

        - ``uid`` — prompt/group id shared by a prompt's rollouts; the group a
          group-relative estimator (GRPO, GiGPO episode level) normalizes within.

        Chain-specific arrays:

        - ``batch_idx`` — per-trajectory index; the de-facto trajectory id (all
          rows of one trajectory share it).
        - ``segment_idx`` — original segment index within a trajectory; skipping
          untrainable segments does not renumber it.
        - ``step_observations`` / ``step_rewards`` / ``step_invalids`` — per-turn
          lists (turn order, projected from the trajectory's ``steps``): the
          observation the agent acted on (grouping anchor), the per-step **raw**
          env reward, and a 0/1 invalid-action flag per turn. Consumed by step-level
          estimators (GiGPO), which map them to token spans via a ``turn_ids``
          derived from ``action_mask`` and apply the invalid-action penalty as a
          local post-discount deduction; ignored otherwise.

        Step-specific arrays: ``traj_uid``, ``anchor_obs``, ``step_env_reward``,
        ``is_action_valid``, and ``active_masks`` (0 for padding). These are scalar
        signals for each segment's generation step, rather than per-turn lists.
        Both converters broadcast ``rm_<key>`` reward metrics to segment rows.

        ``meta_info``:

        - ``use_agent`` — marks this as an agent-produced batch.
        - ``repeat_times`` — training-row count per original trajectory, including
          padding copies; zero for trajectories with no retained rows.
        - ``layout`` — ``"per_segment"`` (chain) or ``"per_step"`` (step); the trainer
          uses it to pick the estimator implementation in :mod:`agentfly.algorithms`.

        Args:
            run_result: The result to convert, even if another run has completed.
            train_on_last_turn: (forced ``False``) restrict the loss to the last turn.
            world_size: data-parallel world size.
            pad_to_multiple_of: pad the batch dimension to a multiple of this
                (extra rows repeat the last row) so it shards evenly.
            log_token_drift: Include the token-drift diagnostic (spliced vs re-tokenized ids) in the batch log and meta_info.
        """
        return result_to_dataproto(
            agent=self,
            run_result=run_result,
            train_on_last_turn=train_on_last_turn,
            world_size=world_size,
            pad_to_multiple_of=pad_to_multiple_of,
            log_token_drift=log_token_drift,
        )
