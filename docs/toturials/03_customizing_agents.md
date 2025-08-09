# Tutorial 3: Customizing Agents

This tutorial covers how to customize agents in AgentFly by creating new agent types or modifying existing ones. Agents are the core components that orchestrate tool usage, reasoning, and response generation.

## Table of Contents

1. [Understanding Agent Architecture](#understanding-agent-architecture)
2. [Using Predefined Agents](#using-predefined-agents)
3. [Creating Custom Agents](#creating-custom-agents)
4. [Advanced Agent Patterns](#advanced-agent-patterns)
5. [Agent Configuration](#agent-configuration)
6. [Best Practices](#best-practices)

## Understanding Agent Architecture

AgentFly agents inherit from `BaseAgent` and implement two key methods:
- `generate_async()`: Generate responses using the LLM
- `parse()`: Parse LLM responses to extract tool calls

The agent lifecycle involves:
1. Receiving input messages
2. Generating responses via LLM
3. Parsing tool calls from responses
4. Executing tools
5. Continuing the conversation until completion

## Using Predefined Agents

### ReactAgent

The ReAct (Reasoning + Acting) agent follows the thought-action-observation pattern:

```python
from agents.agents.react.react_agent import ReactAgent
from agents.tools.src.code.tools import code_interpreter
from agents.tools.src.search.google_search import google_search_serper
from agents.tools.src.react.tools import answer

# Create a ReAct agent
react_agent = ReactAgent(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    tools=[code_interpreter, google_search_serper, answer],
    template="qwen2.5-no-tool",
    system_prompt="You are a helpful assistant that thinks step by step.",
    backend="async_vllm"
)

# Use the agent
messages = [{
    "messages": [
        {"role": "user", "content": "Solve: What is 15% of 240?"}
    ],
    "question": "What is 15% of 240?"
}]

await react_agent.run_async(
    max_steps=5,
    start_messages=messages,
    num_chains=3
)

# Get results
trajectories = react_agent.trajectories
rewards = react_agent.rewards
```

### CodeAgent

Specialized agent for code generation and execution:

```python
from agents.agents.specialized.code_agent import CodeAgent

code_agent = CodeAgent(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    tools=[code_interpreter],
    template="qwen2.5-no-tool",
    max_steps=8
)
```

### Using AutoAgent

For convenient agent creation based on type:

```python
from agents.agents.auto import AutoAgent

# Create agent by type
agent = AutoAgent.create(
    agent_type="react",  # or "code"
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    tools=["code_interpreter", "google_search_serper"],
    template="qwen2.5-no-tool"
)
```

## Creating Custom Agents

### Basic Custom Agent

```python
from agents.agents.agent_base import BaseAgent
from typing import List, Dict
import json
import re

class CustomReasoningAgent(BaseAgent):
    """
    Custom agent that implements structured reasoning.
    """
    
    def __init__(self, reasoning_steps: int = 3, **kwargs):
        """
        Initialize the custom agent.
        
        Args:
            reasoning_steps (int): Number of reasoning steps to enforce
            **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(**kwargs)
        self.reasoning_steps = reasoning_steps
        
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """
        Generate responses using the LLM.
        
        Args:
            messages_list: List of message conversations
            **args: Additional generation arguments
        
        Returns:
            List of generated responses
        """
        # Add custom prompt modification if needed
        modified_messages = self._add_reasoning_prompt(messages_list)
        
        # Call parent's generation method
        return await self.llm_engine.generate_async(modified_messages, **args)
    
    def parse(self, responses: List[str], tools) -> List[Dict]:
        """
        Parse responses to extract tool calls and reasoning steps.
        
        Args:
            responses: List of LLM responses
            tools: Available tools
            
        Returns:
            List of parsed actions
        """
        parsed_actions = []
        
        for response in responses:
            # Extract reasoning steps
            reasoning_steps = self._extract_reasoning_steps(response)
            
            # Extract tool calls
            tool_calls = self._extract_tool_calls(response, tools)
            
            # Validate reasoning quality
            reasoning_quality = self._validate_reasoning(reasoning_steps)
            
            action = {
                "response": response,
                "reasoning_steps": reasoning_steps,
                "tool_calls": tool_calls,
                "reasoning_quality": reasoning_quality,
                "should_continue": len(tool_calls) > 0
            }
            
            parsed_actions.append(action)
        
        return parsed_actions
    
    def _add_reasoning_prompt(self, messages_list: List[List[Dict]]) -> List[List[Dict]]:
        """Add reasoning instructions to messages."""
        reasoning_instruction = f"""
Before taking any action, think through the problem in exactly {self.reasoning_steps} steps:
1. Understand the problem
2. Plan your approach  
3. Execute your plan

Format your response as:
REASONING:
Step 1: [analysis]
Step 2: [planning]
Step 3: [execution plan]

ACTION:
[your action or tool call]
"""
        
        modified_messages = []
        for messages in messages_list:
            # Add reasoning instruction to system prompt or first message
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += reasoning_instruction
                modified_messages.append(messages)
            else:
                system_msg = {"role": "system", "content": reasoning_instruction}
                modified_messages.append([system_msg] + messages)
        
        return modified_messages
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response."""
        reasoning_pattern = r"REASONING:\s*(.*?)(?=ACTION:|$)"
        reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
        
        if not reasoning_match:
            return []
        
        reasoning_text = reasoning_match.group(1)
        
        # Extract numbered steps
        step_pattern = r"Step \d+:\s*([^\\n]+)"
        steps = re.findall(step_pattern, reasoning_text)
        
        return steps
    
    def _extract_tool_calls(self, response: str, tools) -> List[Dict]:
        """Extract tool calls from response."""
        action_pattern = r"ACTION:\s*(.*?)$"
        action_match = re.search(action_pattern, response, re.DOTALL)
        
        if not action_match:
            return []
        
        action_text = action_match.group(1).strip()
        
        # Try to parse as JSON tool call
        try:
            if action_text.startswith('{') and action_text.endswith('}'):
                tool_call = json.loads(action_text)
                if "name" in tool_call and "arguments" in tool_call:
                    return [tool_call]
        except json.JSONDecodeError:
            pass
        
        # Try to parse as function call format
        func_pattern = r"(\w+)\((.*)\)"
        func_match = re.search(func_pattern, action_text)
        if func_match:
            func_name = func_match.group(1)
            func_args = func_match.group(2)
            
            # Check if it's a valid tool
            tool_names = [tool.name for tool in tools] if tools else []
            if func_name in tool_names:
                return [{
                    "name": func_name,
                    "arguments": {"input": func_args}  # Simplified argument parsing
                }]
        
        return []
    
    def _validate_reasoning(self, reasoning_steps: List[str]) -> float:
        """Validate the quality of reasoning steps."""
        if len(reasoning_steps) < self.reasoning_steps:
            return 0.5  # Incomplete reasoning
        
        # Check for quality indicators
        quality_score = 0.0
        quality_keywords = ["because", "therefore", "however", "given", "since"]
        
        for step in reasoning_steps:
            step_lower = step.lower()
            if any(keyword in step_lower for keyword in quality_keywords):
                quality_score += 0.2
            if len(step.split()) >= 5:  # Sufficient detail
                quality_score += 0.1
        
        return min(quality_score, 1.0)
```

### Multi-Modal Agent

```python
from agents.agents.agent_base import BaseAgent
from PIL import Image
import base64
import io

class MultiModalAgent(BaseAgent):
    """
    Agent that can handle both text and image inputs.
    """
    
    def __init__(self, image_understanding_prompt: str = None, **kwargs):
        """
        Initialize multi-modal agent.
        
        Args:
            image_understanding_prompt: Custom prompt for image understanding
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(**kwargs)
        self.image_prompt = image_understanding_prompt or self._default_image_prompt()
        
    def _default_image_prompt(self) -> str:
        """Default prompt for image understanding."""
        return """
When you receive an image, analyze it carefully and describe:
1. What objects or people you see
2. The setting or environment
3. Any text visible in the image
4. Relevant details for answering the user's question
"""
    
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """Generate responses with image understanding."""
        # Process messages to handle images
        processed_messages = self._process_multimodal_messages(messages_list)
        
        # Generate responses
        return await self.llm_engine.generate_async(processed_messages, **args)
    
    def _process_multimodal_messages(self, messages_list: List[List[Dict]]) -> List[List[Dict]]:
        """Process messages to handle image content."""
        processed = []
        
        for messages in messages_list:
            processed_conv = []
            
            for message in messages:
                if self._contains_image(message):
                    # Add image understanding prompt
                    processed_message = self._process_image_message(message)
                else:
                    processed_message = message
                
                processed_conv.append(processed_message)
            
            processed.append(processed_conv)
        
        return processed
    
    def _contains_image(self, message: Dict) -> bool:
        """Check if message contains image content."""
        content = message.get("content", [])
        if isinstance(content, list):
            return any(item.get("type") == "image" for item in content)
        return False
    
    def _process_image_message(self, message: Dict) -> Dict:
        """Process a message containing images."""
        content = message.get("content", [])
        processed_content = []
        
        for item in content:
            if item.get("type") == "image":
                # Add image analysis prompt
                image_analysis = {
                    "type": "text",
                    "text": f"{self.image_prompt}\n\nNow analyze this image:"
                }
                processed_content.append(image_analysis)
                processed_content.append(item)
            else:
                processed_content.append(item)
        
        return {
            **message,
            "content": processed_content
        }
    
    def parse(self, responses: List[str], tools) -> List[Dict]:
        """Parse responses with image context awareness."""
        parsed_actions = []
        
        for response in responses:
            # Standard tool call extraction
            tool_calls = self._extract_standard_tool_calls(response, tools)
            
            # Check for image-specific actions
            image_actions = self._extract_image_actions(response)
            
            action = {
                "response": response,
                "tool_calls": tool_calls,
                "image_actions": image_actions,
                "should_continue": len(tool_calls) > 0 or len(image_actions) > 0
            }
            
            parsed_actions.append(action)
        
        return parsed_actions
    
    def _extract_image_actions(self, response: str) -> List[Dict]:
        """Extract image-specific actions from response."""
        image_actions = []
        
        # Look for image generation requests
        if "generate image" in response.lower() or "create image" in response.lower():
            image_actions.append({
                "type": "generate_image",
                "description": response
            })
        
        # Look for image analysis requests
        if "analyze image" in response.lower() or "describe image" in response.lower():
            image_actions.append({
                "type": "analyze_image",
                "description": response
            })
        
        return image_actions
```

### Planning Agent

```python
class PlanningAgent(BaseAgent):
    """
    Agent that creates explicit plans before execution.
    """
    
    def __init__(self, max_plan_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_plan_steps = max_plan_steps
        self.current_plan = None
        self.plan_step = 0
        
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """Generate with planning phase."""
        planning_messages = self._add_planning_phase(messages_list)
        return await self.llm_engine.generate_async(planning_messages, **args)
    
    def _add_planning_phase(self, messages_list: List[List[Dict]]) -> List[List[Dict]]:
        """Add planning phase to conversation."""
        planning_prompt = f"""
Before solving this problem, create a detailed plan with up to {self.max_plan_steps} steps.

Format your response as:
PLAN:
1. [first step]
2. [second step]
...

EXECUTION:
[Execute first step]
"""
        
        modified_messages = []
        for messages in messages_list:
            if self.current_plan is None:
                # First turn - create plan
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] += planning_prompt
                else:
                    system_msg = {"role": "system", "content": planning_prompt}
                    messages = [system_msg] + messages
            else:
                # Subsequent turns - follow plan
                step_prompt = f"""
Continue with your plan. You are on step {self.plan_step + 1} of your plan:
{self.current_plan[self.plan_step] if self.plan_step < len(self.current_plan) else 'Plan completed'}

Execute the next step.
"""
                messages.append({"role": "system", "content": step_prompt})
            
            modified_messages.append(messages)
        
        return modified_messages
    
    def parse(self, responses: List[str], tools) -> List[Dict]:
        """Parse with plan tracking."""
        parsed_actions = []
        
        for response in responses:
            # Extract plan if present
            plan = self._extract_plan(response)
            if plan:
                self.current_plan = plan
                self.plan_step = 0
            
            # Extract current execution
            execution = self._extract_execution(response)
            tool_calls = self._extract_tool_calls_from_execution(execution, tools)
            
            # Update plan progress
            if tool_calls and self.current_plan:
                self.plan_step += 1
            
            action = {
                "response": response,
                "plan": self.current_plan,
                "current_step": self.plan_step,
                "execution": execution,
                "tool_calls": tool_calls,
                "should_continue": (
                    len(tool_calls) > 0 and 
                    self.plan_step < len(self.current_plan) if self.current_plan else True
                )
            }
            
            parsed_actions.append(action)
        
        return parsed_actions
    
    def _extract_plan(self, response: str) -> List[str]:
        """Extract plan steps from response."""
        plan_pattern = r"PLAN:\s*(.*?)(?=EXECUTION:|$)"
        plan_match = re.search(plan_pattern, response, re.DOTALL)
        
        if not plan_match:
            return None
        
        plan_text = plan_match.group(1)
        steps = re.findall(r"\d+\.\s*([^\\n]+)", plan_text)
        return steps[:self.max_plan_steps]
    
    def _extract_execution(self, response: str) -> str:
        """Extract execution section from response."""
        execution_pattern = r"EXECUTION:\s*(.*?)$"
        execution_match = re.search(execution_pattern, response, re.DOTALL)
        
        if execution_match:
            return execution_match.group(1).strip()
        
        return response  # Fallback to full response
```

## Advanced Agent Patterns

### Agent with Memory

```python
class MemoryAgent(BaseAgent):
    """
    Agent that maintains conversation memory across interactions.
    """
    
    def __init__(self, memory_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.conversation_memory = []
        self.fact_memory = {}
        
    def add_to_memory(self, interaction: Dict):
        """Add interaction to memory."""
        self.conversation_memory.append(interaction)
        
        # Keep only recent interactions
        if len(self.conversation_memory) > self.memory_size:
            self.conversation_memory = self.conversation_memory[-self.memory_size:]
        
        # Extract facts for fact memory
        facts = self._extract_facts(interaction)
        self.fact_memory.update(facts)
    
    def _extract_facts(self, interaction: Dict) -> Dict:
        """Extract factual information from interaction."""
        facts = {}
        
        # Simple fact extraction (can be enhanced with NLP)
        response = interaction.get("response", "")
        
        # Look for factual statements
        fact_patterns = [
            r"(.+) is (.+)",
            r"(.+) has (.+)",
            r"(.+) contains (.+)"
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if len(match) == 2:
                    subject, predicate = match
                    facts[subject.strip()] = predicate.strip()
        
        return facts
    
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """Generate with memory context."""
        memory_enhanced_messages = self._add_memory_context(messages_list)
        return await self.llm_engine.generate_async(memory_enhanced_messages, **args)
    
    def _add_memory_context(self, messages_list: List[List[Dict]]) -> List[List[Dict]]:
        """Add relevant memory to messages."""
        enhanced_messages = []
        
        for messages in messages_list:
            # Find relevant memories
            query = " ".join([msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)])
            relevant_memories = self._find_relevant_memories(query)
            
            if relevant_memories:
                memory_context = "RELEVANT CONTEXT:\n" + "\n".join(relevant_memories)
                
                # Add to system message or create new one
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] += f"\n\n{memory_context}"
                else:
                    system_msg = {"role": "system", "content": memory_context}
                    messages = [system_msg] + messages
            
            enhanced_messages.append(messages)
        
        return enhanced_messages
    
    def _find_relevant_memories(self, query: str, max_memories: int = 5) -> List[str]:
        """Find memories relevant to the current query."""
        relevant = []
        
        # Simple keyword-based relevance (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        for memory in self.conversation_memory[-20:]:  # Recent memories
            memory_text = memory.get("response", "")
            memory_words = set(memory_text.lower().split())
            
            # Calculate overlap
            overlap = len(query_words & memory_words)
            if overlap > 2:  # Threshold for relevance
                relevant.append(memory_text[:200] + "...")  # Truncate
        
        # Add relevant facts
        for subject, predicate in self.fact_memory.items():
            if any(word in subject.lower() or word in predicate.lower() for word in query_words):
                relevant.append(f"{subject}: {predicate}")
        
        return relevant[:max_memories]
```

### Collaborative Agent

```python
class CollaborativeAgent(BaseAgent):
    """
    Agent that can work with other agents.
    """
    
    def __init__(self, agent_id: str, collaboration_mode: str = "sequential", **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.collaboration_mode = collaboration_mode
        self.peer_agents = {}
        self.shared_context = {}
        
    def register_peer(self, agent_id: str, agent: BaseAgent):
        """Register a peer agent for collaboration."""
        self.peer_agents[agent_id] = agent
        
    async def collaborate(self, task: str, peer_ids: List[str] = None) -> Dict:
        """Collaborate with peer agents on a task."""
        if self.collaboration_mode == "sequential":
            return await self._sequential_collaboration(task, peer_ids)
        elif self.collaboration_mode == "parallel":
            return await self._parallel_collaboration(task, peer_ids)
        else:
            raise ValueError(f"Unknown collaboration mode: {self.collaboration_mode}")
    
    async def _sequential_collaboration(self, task: str, peer_ids: List[str]) -> Dict:
        """Sequential collaboration where agents work one after another."""
        results = []
        current_context = {"task": task, "results": []}
        
        # Include self in the sequence
        all_agents = [self] + [self.peer_agents[pid] for pid in (peer_ids or [])]
        
        for i, agent in enumerate(all_agents):
            agent_input = self._prepare_agent_input(current_context, i)
            
            if agent == self:
                result = await self.process_task(agent_input)
            else:
                result = await agent.process_task(agent_input)
            
            results.append({
                "agent_id": agent.agent_id if hasattr(agent, 'agent_id') else f"agent_{i}",
                "result": result
            })
            
            current_context["results"] = results
        
        return {
            "collaboration_type": "sequential",
            "task": task,
            "results": results,
            "final_result": self._synthesize_results(results)
        }
    
    async def _parallel_collaboration(self, task: str, peer_ids: List[str]) -> Dict:
        """Parallel collaboration where agents work simultaneously."""
        import asyncio
        
        # Prepare tasks for all agents
        all_agents = [self] + [self.peer_agents[pid] for pid in (peer_ids or [])]
        agent_tasks = [agent.process_task(task) for agent in all_agents]
        
        # Execute in parallel
        results = await asyncio.gather(*agent_tasks)
        
        # Combine results
        combined_results = []
        for i, (agent, result) in enumerate(zip(all_agents, results)):
            combined_results.append({
                "agent_id": agent.agent_id if hasattr(agent, 'agent_id') else f"agent_{i}",
                "result": result
            })
        
        return {
            "collaboration_type": "parallel",
            "task": task,
            "results": combined_results,
            "final_result": self._synthesize_results(combined_results)
        }
    
    def _prepare_agent_input(self, context: Dict, agent_index: int) -> str:
        """Prepare input for an agent in sequential collaboration."""
        base_task = context["task"]
        
        if agent_index == 0:
            return base_task
        
        # Include previous results
        previous_results = context["results"]
        context_str = f"Task: {base_task}\n\nPrevious agent results:\n"
        
        for result in previous_results:
            context_str += f"- {result['agent_id']}: {result['result']}\n"
        
        context_str += f"\nNow continue or build upon these results."
        
        return context_str
    
    def _synthesize_results(self, results: List[Dict]) -> str:
        """Synthesize results from multiple agents."""
        if not results:
            return "No results to synthesize"
        
        if len(results) == 1:
            return results[0]["result"]
        
        # Simple synthesis (can be enhanced with LLM)
        synthesis = "Combined agent results:\n"
        for result in results:
            synthesis += f"- {result['agent_id']}: {result['result']}\n"
        
        return synthesis
```

## Agent Configuration

### Configuration Classes

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class AgentConfig:
    """Configuration class for agents."""
    model_name_or_path: str
    template: str
    tools: List[str]
    system_prompt: Optional[str] = None
    max_length: int = 8192
    max_steps: int = 10
    backend: str = "async_vllm"
    reward_name: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.9
    
    # Agent-specific configs
    reasoning_steps: int = 3
    memory_size: int = 100
    collaboration_mode: str = "sequential"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "template": self.template,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "max_length": self.max_length,
            "max_steps": self.max_steps,
            "backend": self.backend,
            "reward_name": self.reward_name,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

# Usage
config = AgentConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    template="qwen2.5-no-tool",
    tools=["code_interpreter", "google_search"],
    system_prompt="You are a helpful coding assistant."
)

agent = CustomReasoningAgent(**config.to_dict())
```

### Factory Pattern

```python
class AgentFactory:
    """Factory for creating different types of agents."""
    
    @staticmethod
    def create_agent(agent_type: str, config: AgentConfig) -> BaseAgent:
        """Create agent based on type."""
        base_kwargs = config.to_dict()
        
        if agent_type == "react":
            from agents.agents.react.react_agent import ReactAgent
            return ReactAgent(**base_kwargs)
        
        elif agent_type == "code":
            from agents.agents.specialized.code_agent import CodeAgent
            return CodeAgent(**base_kwargs)
        
        elif agent_type == "custom_reasoning":
            return CustomReasoningAgent(
                reasoning_steps=config.reasoning_steps,
                **base_kwargs
            )
        
        elif agent_type == "multimodal":
            return MultiModalAgent(**base_kwargs)
        
        elif agent_type == "planning":
            return PlanningAgent(
                max_plan_steps=config.reasoning_steps,
                **base_kwargs
            )
        
        elif agent_type == "memory":
            return MemoryAgent(
                memory_size=config.memory_size,
                **base_kwargs
            )
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

# Usage
config = AgentConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    template="qwen2.5-no-tool",
    tools=["code_interpreter"]
)

agent = AgentFactory.create_agent("custom_reasoning", config)
```

## Best Practices

### 1. Implement Error Handling

```python
class RobustAgent(BaseAgent):
    """Agent with comprehensive error handling."""
    
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """Generate with error handling."""
        try:
            return await self.llm_engine.generate_async(messages_list, **args)
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            # Return fallback responses
            return ["I apologize, but I encountered an error. Please try again."] * len(messages_list)
    
    def parse(self, responses: List[str], tools) -> List[Dict]:
        """Parse with error handling."""
        parsed_actions = []
        
        for response in responses:
            try:
                action = self._safe_parse(response, tools)
                parsed_actions.append(action)
            except Exception as e:
                self.logger.error(f"Parsing failed: {str(e)}")
                # Fallback action
                parsed_actions.append({
                    "response": response,
                    "tool_calls": [],
                    "should_continue": False,
                    "error": str(e)
                })
        
        return parsed_actions
```

### 2. Add Logging and Monitoring

```python
import logging
from typing import Any

class MonitoredAgent(BaseAgent):
    """Agent with comprehensive logging."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = {
            "total_generations": 0,
            "successful_generations": 0,
            "total_tool_calls": 0,
            "parsing_errors": 0
        }
    
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """Generate with monitoring."""
        self.metrics["total_generations"] += len(messages_list)
        
        start_time = time.time()
        try:
            responses = await self.llm_engine.generate_async(messages_list, **args)
            self.metrics["successful_generations"] += len(responses)
            
            generation_time = time.time() - start_time
            self.logger.info(f"Generated {len(responses)} responses in {generation_time:.2f}s")
            
            return responses
        except Exception as e:
            self.logger.error(f"Generation failed after {time.time() - start_time:.2f}s: {str(e)}")
            raise
    
    def parse(self, responses: List[str], tools) -> List[Dict]:
        """Parse with monitoring."""
        try:
            actions = super().parse(responses, tools)
            
            # Count tool calls
            tool_call_count = sum(len(action.get("tool_calls", [])) for action in actions)
            self.metrics["total_tool_calls"] += tool_call_count
            
            self.logger.info(f"Parsed {len(actions)} actions with {tool_call_count} tool calls")
            
            return actions
        except Exception as e:
            self.metrics["parsing_errors"] += 1
            self.logger.error(f"Parsing failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        success_rate = (
            self.metrics["successful_generations"] / self.metrics["total_generations"]
            if self.metrics["total_generations"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "avg_tool_calls_per_generation": (
                self.metrics["total_tool_calls"] / self.metrics["successful_generations"]
                if self.metrics["successful_generations"] > 0 else 0
            )
        }
```

### 3. Test Your Agents

```python
import unittest
from unittest.mock import Mock, AsyncMock

class TestCustomAgent(unittest.TestCase):
    """Test cases for custom agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = CustomReasoningAgent(
            model_name_or_path="test-model",
            template="test-template",
            tools=[],
            reasoning_steps=3
        )
        
        # Mock the LLM engine
        self.agent.llm_engine = Mock()
        self.agent.llm_engine.generate_async = AsyncMock()
    
    async def test_reasoning_prompt_addition(self):
        """Test that reasoning prompts are added correctly."""
        messages = [[{"role": "user", "content": "Test question"}]]
        
        # Mock LLM response
        self.agent.llm_engine.generate_async.return_value = ["Test response"]
        
        # Test generation
        result = await self.agent.generate_async(messages)
        
        # Verify reasoning prompt was added
        called_messages = self.agent.llm_engine.generate_async.call_args[0][0]
        self.assertTrue(any("Step 1" in str(msg) for msg in called_messages))
    
    def test_reasoning_extraction(self):
        """Test reasoning step extraction."""
        response = """
REASONING:
Step 1: Understand the problem
Step 2: Plan approach
Step 3: Execute

ACTION:
call_tool()
"""
        
        steps = self.agent._extract_reasoning_steps(response)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "Understand the problem")
    
    def test_parse_error_handling(self):
        """Test error handling in parsing."""
        invalid_response = "Invalid response format"
        
        actions = self.agent.parse([invalid_response], [])
        
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["tool_calls"], [])
```

### 4. Register Custom Agents

```python
# Register your custom agent with AutoAgent
from agents.agents.auto import AutoAgent

AutoAgent.register_agent("custom_reasoning", CustomReasoningAgent)
AutoAgent.register_agent("multimodal", MultiModalAgent)
AutoAgent.register_agent("planning", PlanningAgent)

# Now you can create them via AutoAgent
agent = AutoAgent.create(
    agent_type="custom_reasoning",
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    reasoning_steps=5
)
```

## Next Steps

Now that you understand agent customization, proceed to:
- [Tutorial 4: Data Preparation](04_data_preparation.md) to learn about data formatting
- [Tutorial 5: Template Configuration](05_template_configuration.md) to understand conversation templates

For a complete example, see [Tutorial 7: Complete Pipeline](07_complete_pipeline.md).