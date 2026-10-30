""":class:`AutoAgent`: create an agent from a config dict or a model name.

Resolves the agent type (a registered name or an import reference), the tools and the reward,
then constructs the agent; also used by the trainer to build the agent from Hydra config.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from omegaconf import DictConfig, OmegaConf

from ..rewards.reward_base import get_reward_from_name
from ..tools import get_tools_from_names
from .agent_base import BaseAgent
from .specialized.action_agent import ActionAgent
from .specialized.code_agent import CodeAgent
from .specialized.gui_agent import GUIAgent
from .specialized.hf_agent import HFAgent, SearchR1Agent
from .specialized.miniswe import MinisweAgent
from .specialized.react_agent import ReactAgent
from .specialized.swe_agents import BashSWEAgent, Qwen3CoderSWEAgent


class AutoAgent:
    """
    AutoAgent is a class that automatically handles agent initialization based on configuration.

    Built-in agent types:
    - 'react': ReactAgent for ReAct-style reasoning and tool use
    - 'code': CodeAgent for code generation and execution

    These agents are registered automatically. Additional custom agents can be
    registered using the register_agent method.
    """

    AGENT_MAPPING = {}

    @classmethod
    def register_agent(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type in the AGENT_MAPPING.

        Args:
            agent_type: The name identifier for the agent type (e.g., 'react', 'code')
            agent_class: The agent class to instantiate for this type
        """
        cls.AGENT_MAPPING[agent_type.lower()] = agent_class

    @classmethod
    def _get_agent_class(cls, agent_type: str) -> Type[BaseAgent]:
        """
        Get the agent class for a given agent type.

        Args:
            agent_type: Type of agent ('react', 'code', etc.)

        Returns:
            The agent class

        Raises:
            ValueError: If the agent type is not registered
        """
        from ..utils.references import resolve_reference

        # Accepts a registered type OR an import reference
        # ("module:AgentClass", "/path.py:AgentClass").
        try:
            return resolve_reference(agent_type, cls.AGENT_MAPPING, "agent")
        except KeyError as e:
            raise ValueError(str(e)) from e

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent from a configuration dictionary.

        Args:
            config: A dictionary containing the agent configuration.
                Required keys:
                    - agent_type: A registered agent type ('react', 'hf', 'code', ...) or an
                      import reference to an agent class ("module:Class", "/path.py:Class")
                    - model_name_or_path: Model name or path
                    - backend_config: Dict with a "backend" key (e.g. {"backend": "async_vllm"})
                Optional keys:
                    - tools: Registered names, import references, or tool objects (default: none)
                    - reward_fn: The reward function (e.g. a ``@reward``)
                    - reward_name: A registered reward name or import reference (instead of reward_fn)
                    - subagents: ``{name: settings}`` for subagents the agent can create; tools in
                      each ``settings`` may be names or import references too
                    - template, system_prompt, max_model_len, ...: other agent arguments

        Returns:
            An initialized agent instance.
        """
        # Extract and validate required parameters
        if config is None:
            raise ValueError("Config could not be None")

        # construct a copy for agent_kwargs
        if isinstance(config, dict):
            agent_kwargs = {}
            for k, v in config.items():
                agent_kwargs[k] = v
        elif isinstance(config, DictConfig):
            agent_kwargs = OmegaConf.to_container(config)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        required_params = ["agent_type", "backend_config"]
        missing_params = [param for param in required_params if not config.get(param)]

        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        backend_config = agent_kwargs["backend_config"]
        if not isinstance(backend_config, dict) or "backend" not in backend_config:
            raise ValueError(
                f"backend_config must be a dict with at least a 'backend' key (e.g. {{'backend': 'async_vllm'}}), got {backend_config}"
            )

        agent_type = config["agent_type"]
        agent_kwargs.pop("agent_type")
        # Tools may be registered names, import references, or tool objects, mixed. The same
        # holds for the tools in each subagent declaration.
        def resolve_tools(tool_list):
            return [get_tools_from_names([t])[0] if isinstance(t, str) else t for t in (tool_list or [])]

        tools = resolve_tools(agent_kwargs.get("tools"))
        subagents = agent_kwargs.get("subagents")
        if isinstance(subagents, DictConfig):
            subagents = OmegaConf.to_container(subagents)
        if subagents:
            resolved = {}
            for name, declared in subagents.items():
                declared = OmegaConf.to_container(declared) if isinstance(declared, DictConfig) else dict(declared)
                if "tools" in declared:
                    declared["tools"] = resolve_tools(declared["tools"])
                resolved[name] = declared
            agent_kwargs["subagents"] = resolved
        agent_class = cls._get_agent_class(agent_type)
        reward_fn = agent_kwargs.get("reward_fn")
        reward_name = agent_kwargs.pop("reward_name", None)
        if reward_name is not None:
            if reward_fn is not None:
                raise ValueError("Pass either reward_fn or reward_name, not both.")
            reward_fn = get_reward_from_name(reward_name)

        agent_kwargs["tools"] = tools
        agent_kwargs["reward_fn"] = reward_fn

        if "use_agent" in agent_kwargs:
            agent_kwargs.pop("use_agent")
        # BaseAgent only accepts backend_config, not backend
        agent_kwargs.pop("backend", None)

        agent = agent_class(**agent_kwargs)

        return agent

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        agent_type: str,
        template: str,
        tools: Optional[List] = None,
        debug: bool = False,
        reward_fn: Optional[Callable] = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent directly from a model name/path and agent type.

        Args:
            model_name_or_path: Pretrained model name or path
            agent_type: A registered agent type ('react', 'hf', 'code', ...) or an import
                reference to an agent class ("module:Class", "/path.py:Class")
            template: Conversation template name
            tools: Tools: registered names, import references, or tool objects
            debug: Whether to enable debug logging
            reward_fn: The reward function (e.g. a ``@reward``); or pass ``reward_name``
                (a registered name or import reference) in ``kwargs`` instead
            **kwargs: Other agent arguments, e.g. ``backend_config`` (required)

        Returns:
            An initialized agent instance.
        """
        # Create config dictionary and reuse from_config logic
        config = {
            "agent_type": agent_type,
            "model_name_or_path": model_name_or_path,
            "template": template,
            "tools": tools or [],
            "debug": debug,
            "reward_fn": reward_fn,
            **kwargs,
        }

        return cls.from_config(config)


# Auto-register built-in agent types
AutoAgent.register_agent("react", ReactAgent)
AutoAgent.register_agent("code", CodeAgent)
AutoAgent.register_agent("gui", GUIAgent)
AutoAgent.register_agent("hf", HFAgent)
AutoAgent.register_agent("searchr1", SearchR1Agent)
AutoAgent.register_agent("action", ActionAgent)
AutoAgent.register_agent("miniswe", MinisweAgent)
AutoAgent.register_agent("bash_swe", BashSWEAgent)
AutoAgent.register_agent("qwen3coder_swe", Qwen3CoderSWEAgent)
