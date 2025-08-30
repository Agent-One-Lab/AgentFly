from typing import Any, Callable, Dict, List, Optional, Type, Union

from .specialized.think_agent import ThinkAgent
from .specialized.openai_agent import OpenAIAgent
from ..tools import get_tools_from_names
from .agent_base import BaseAgent
from .react.react_agent import ReactAgent
from .specialized.code_agent import CodeAgent
from .specialized.gui_agent import GUIAgent
from .specialized.hf_agent import HFAgent
from ..rewards.reward_base import get_reward_from_name



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
        agent_type = agent_type.lower()
        
        if agent_type not in cls.AGENT_MAPPING:
            available_types = list(cls.AGENT_MAPPING.keys())
            raise ValueError(f"Unknown agent type: '{agent_type}'. Available types: {available_types}")
            
        return cls.AGENT_MAPPING[agent_type]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent from a configuration dictionary.
        
        Args:
            config: A dictionary containing the agent configuration.
                Required keys:
                    - agent_type: Type of agent ('react', 'code', etc.)
                    - model_name_or_path: Model name or path
                    - template: Conversation template
                Optional keys:
                    - tools: List of tool objects
                    - vllm: Whether to use vLLM for inference (default: False)
                    - debug: Whether to enable debug logging (default: False)
                    - log_file: Log file name (default: "agent")
                    - task_info: Task-specific information (for ReactAgent)
                    - reward_function: Reward function to use (default: None)
                    - reward_name: Name of registered reward function to use
                    - reward_args: Arguments to pass to the reward function
                    
        Returns:
            An initialized agent instance.
        """
        # Extract and validate required parameters
        if config is None:
            raise ValueError("Config could not be None")

        # construct a copy for agent_kwargs
        agent_kwargs = {}
        for k, v in config.items():
            agent_kwargs[k] = v
        
        required_params = ["agent_type", "template", "tools", "backend"]
        missing_params = [param for param in required_params if not config.get(param)]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        agent_type = config["agent_type"]
        agent_kwargs.pop("agent_type")
        tools = get_tools_from_names(config["tools"])
        agent_class = cls._get_agent_class(agent_type)
        reward_name = config.get("reward_name")
        if reward_name is not None:
            reward_fn = get_reward_from_name(reward_name)
            agent_kwargs.pop("reward_name")
        else:
            reward_fn = None
        
        agent_kwargs['tools'] = tools
        agent_kwargs['reward_fn'] = reward_fn

        if "use_agent" in agent_kwargs:
            agent_kwargs.pop("use_agent")
        
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
        log_file: str = "agent",
        reward_fn: Optional[Callable] = None,
        **kwargs
    ) -> BaseAgent:
        """
        Create an agent directly from a model name/path and agent type.
        
        Args:
            model_name_or_path: Pretrained model name or path
            agent_type: Type of agent ('react', 'code', etc.)
            template: Conversation template name
            tools: List of tool objects
            vllm: Whether to use vLLM for inference
            debug: Whether to enable debug logging
            log_file: Log file name
            wrapper: Whether to use the agent as a wrapper
            reward_function: Reward function instance to use (takes precedence)
            reward_name: Name of registered reward function to use
            reward_args: Arguments to pass to the reward function constructor
            **kwargs: Additional arguments specific to the agent type
            
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
            "log_file": log_file,
            "reward_fn": reward_fn,
            **kwargs
        }
            
        return cls.from_config(config)

# Auto-register built-in agent types
AutoAgent.register_agent("react", ReactAgent)
AutoAgent.register_agent("code", CodeAgent)
AutoAgent.register_agent("openai", OpenAIAgent)
AutoAgent.register_agent("think", ThinkAgent)
AutoAgent.register_agent("gui", GUIAgent)
AutoAgent.register_agent("hf", HFAgent)