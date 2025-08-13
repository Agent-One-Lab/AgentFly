from typing import Any, Dict, List, Optional, Type, Union

from .specialized.think_agent import ThinkAgent
from .specialized.openai_agent import OpenAIAgent
from ..tools import get_tools_from_names
from .agent_base import BaseAgent
from .react.react_agent import ReactAgent
from .specialized.code_agent import CodeAgent
from .specialized.gui_agent import GUIAgent
from ..rewards.reward_base import get_reward_from_name

# Registry for agent types - will be populated dynamically
AGENT_MAPPING = {}

class AutoAgent:
    """
    AutoAgent is a class that automatically handles agent initialization based on configuration.
    
    Built-in agent types:
    - 'react': ReactAgent for ReAct-style reasoning and tool use
    - 'code': CodeAgent for code generation and execution
    
    These agents are registered automatically. Additional custom agents can be
    registered using the register_agent method.
    """
    
    @classmethod
    def register_agent(cls, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent type in the AGENT_MAPPING.
        
        Args:
            agent_type: The name identifier for the agent type (e.g., 'react', 'code')
            agent_class: The agent class to instantiate for this type
        """
        AGENT_MAPPING[agent_type.lower()] = agent_class
    
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
        
        if agent_type not in AGENT_MAPPING:
            available_types = list(AGENT_MAPPING.keys())
            raise ValueError(f"Unknown agent type: '{agent_type}'. Available types: {available_types}")
            
        return AGENT_MAPPING[agent_type]
    
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
        required_params = ["agent_type", "template", "tools", "backend"]
        missing_params = [param for param in required_params if not config.get(param)]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        agent_type = config["agent_type"]
        tools = get_tools_from_names(config["tools"])
        agent_class = cls._get_agent_class(agent_type)
        
        # construct a copy for agent_kwargs
        agent_kwargs = {}
        for k, v in config.items():
            agent_kwargs[k] = v
        
        agent_kwargs.pop("agent_type")
        agent_kwargs['tools'] = tools
        if "reward_name" in config and config["reward_name"] is not None:
            agent_kwargs.pop("reward_name")
            reward_fn = get_reward_from_name(config["reward_name"])
            agent_kwargs["reward_fn"] = reward_fn
        
        agent = agent_class(**agent_kwargs)

        return agent
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        agent_type: str,
        template: str,
        tools: Optional[List] = None,
        vllm: bool = False,
        debug: bool = False,
        log_file: str = "agent",
        wrapper: bool = False,
        reward_name: Optional[str] = None,
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
            "vllm": vllm,
            "debug": debug,
            "log_file": log_file,
            "wrapper": wrapper,
            "reward_name": reward_name,
            **kwargs
        }
            
        return cls.from_config(config)

# Auto-register built-in agent types
AutoAgent.register_agent("react", ReactAgent)
AutoAgent.register_agent("code", CodeAgent)
AutoAgent.register_agent("openai", OpenAIAgent)
AutoAgent.register_agent("think", ThinkAgent)
AutoAgent.register_agent("gui", GUIAgent)