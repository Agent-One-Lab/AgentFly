# Tutorial 1: Defining Custom Tools

This tutorial covers how to define custom tools for your AgentFly agents. Tools are the primary way agents interact with external systems, APIs, and environments.

## Table of Contents

1. [Understanding Tools](#understanding-tools)
2. [Non-Stateful Tools](#non-stateful-tools)
3. [Stateful Tools](#stateful-tools)
4. [Tool Configuration](#tool-configuration)
5. [Advanced Tool Patterns](#advanced-tool-patterns)
6. [Best Practices](#best-practices)

## Understanding Tools

In AgentFly, tools are functions that agents can call to perform actions or retrieve information. There are two types of tools:

- **Non-Stateful Tools**: Simple functions that don't maintain state between calls
- **Stateful Tools**: Tools that interact with environments and maintain state

## Non-Stateful Tools

Non-stateful tools are the simplest type. They're just functions decorated with the `@tool` decorator.

### Basic Example

```python
from agents.tools.tool_base import tool

@tool(name="calculator", description="Performs basic arithmetic operations")
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    
    Args:
        expression (str): A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        str: The result of the calculation
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {
            k: v for k, v in __builtins__.items() 
            if k in ['abs', 'min', 'max', 'round', 'pow']
        }
        allowed_names.update({
            'sin': __import__('math').sin,
            'cos': __import__('math').cos,
            'tan': __import__('math').tan,
            'log': __import__('math').log,
            'sqrt': __import__('math').sqrt,
            'pi': __import__('math').pi,
            'e': __import__('math').e,
        })
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
```

### Weather API Tool

```python
import requests

@tool(name="weather_tool", description="Get current weather information")
def get_weather(city: str, api_key: str = None) -> dict:
    """
    Get current weather information for a specified city.
    
    Args:
        city (str): The name of the city
        api_key (str): OpenWeatherMap API key (optional if set as environment variable)
    
    Returns:
        dict: Weather information containing temperature, description, etc.
    """
    if not api_key:
        import os
        api_key = os.getenv('OPENWEATHER_API_KEY')
        
    if not api_key:
        return {"error": "API key not provided"}
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return {
            "observation": f"Weather in {city}: {data['main']['temp']}°C, {data['weather'][0]['description']}"
        }
    except requests.RequestException as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}
```

## Stateful Tools

Stateful tools interact with environments that maintain state between interactions. These are particularly useful for tasks like code execution, file manipulation, or interactive simulations.

### Code Execution Tool

```python
from agents.envs.python_env import PythonSandboxEnv
from agents.tools.tool_base import tool

@tool(
    name="code_interpreter",
    description="Execute Python code in a sandboxed environment",
    env_cls=PythonSandboxEnv,
    stateful=True,
    pool_size=8  # Number of concurrent environments
)
async def code_interpreter(code: str, env: PythonSandboxEnv) -> dict:
    """
    Execute Python code in a sandboxed environment.
    
    Args:
        code (str): Python code to execute
        env (PythonSandboxEnv): The environment instance
    
    Returns:
        dict: Execution result with output and any errors
    """
    try:
        result = await env.step(code)
        return {
            "observation": result.get("output", ""),
            "success": True
        }
    except Exception as e:
        return {
            "observation": f"Execution error: {str(e)}",
            "success": False
        }
```

### File System Tool

```python
from agents.envs.env_base import BaseEnv
from agents.tools.tool_base import tool
import os
import tempfile

class FileSystemEnv(BaseEnv):
    """Simple file system environment for file operations."""
    
    def __init__(self):
        super().__init__()
        self.temp_dir = None
        
    async def start(self):
        """Initialize the environment with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        
    async def reset(self):
        """Clean up files in the temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = tempfile.mkdtemp()
            
    async def step(self, action: str) -> dict:
        """Execute file system operations."""
        try:
            parts = action.split(':', 1)
            operation = parts[0].strip()
            args = parts[1].strip() if len(parts) > 1 else ""
            
            if operation == "write":
                filename, content = args.split('|', 1)
                filepath = os.path.join(self.temp_dir, filename.strip())
                with open(filepath, 'w') as f:
                    f.write(content.strip())
                return {"output": f"File {filename} written successfully"}
                
            elif operation == "read":
                filepath = os.path.join(self.temp_dir, args.strip())
                with open(filepath, 'r') as f:
                    content = f.read()
                return {"output": content}
                
            elif operation == "list":
                files = os.listdir(self.temp_dir)
                return {"output": f"Files: {', '.join(files)}"}
                
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": str(e)}
            
    async def aclose(self):
        """Clean up the environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
    def close(self):
        """Synchronous cleanup."""
        import asyncio
        asyncio.run(self.aclose())
        
    @staticmethod
    async def acquire():
        """Create a new environment instance."""
        env = FileSystemEnv()
        await env.start()
        return env

@tool(
    name="file_system",
    description="Perform file system operations",
    env_cls=FileSystemEnv,
    stateful=True,
    pool_size=4
)
async def file_system_tool(action: str, env: FileSystemEnv) -> dict:
    """
    Perform file system operations in a sandboxed environment.
    
    Available operations:
    - write: filename|content - Write content to a file
    - read: filename - Read content from a file
    - list: - List all files
    
    Args:
        action (str): The operation to perform
        env (FileSystemEnv): The environment instance
    
    Returns:
        dict: Operation result
    """
    result = await env.step(action)
    return {
        "observation": result.get("output", result.get("error", "Unknown error"))
    }
```

## Tool Configuration

### Tool Decorator Parameters

The `@tool` decorator accepts several parameters to configure tool behavior:

```python
@tool(
    name="my_tool",                    # Tool name (defaults to function name)
    description="Tool description",    # Description for the agent
    status="success",                  # Control flow: "success", "terminal", "continue"
    max_length=2048,                  # Maximum output length
    auto_register=True,               # Whether to register automatically
    stateful=False,                   # Whether the tool maintains state
    env_cls=None,                     # Environment class for stateful tools
    env_kwargs=None,                  # Environment initialization arguments
    pool_size=-1                      # Number of environment instances (stateful only)
)
def my_tool(arg1: str, arg2: int = 5) -> str:
    """Tool implementation"""
    pass
```

### Control Flow with Status

The `status` parameter controls how the agent continues after tool execution:

```python
# Terminal tool - stops the agent's reasoning chain
@tool(status="terminal")
def final_answer(answer: str) -> str:
    """Provide the final answer to the user's question."""
    return f"Final Answer: {answer}"

# Continue tool - agent continues reasoning after this tool
@tool(status="continue")
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return search_results
```

## Advanced Tool Patterns

### Tool with Multiple Return Formats

```python
@tool(name="data_analyzer")
def analyze_data(data: str, format: str = "summary") -> dict:
    """
    Analyze data and return results in different formats.
    
    Args:
        data (str): JSON string containing data to analyze
        format (str): Output format - "summary", "detailed", or "raw"
    
    Returns:
        dict: Analysis results
    """
    import json
    
    try:
        parsed_data = json.loads(data)
        
        # Perform analysis
        total_items = len(parsed_data)
        numeric_fields = [k for k, v in parsed_data[0].items() 
                         if isinstance(v, (int, float))]
        
        if format == "summary":
            return {
                "observation": f"Data contains {total_items} items with {len(numeric_fields)} numeric fields"
            }
        elif format == "detailed":
            stats = {}
            for field in numeric_fields:
                values = [item[field] for item in parsed_data]
                stats[field] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
            return {
                "observation": f"Detailed statistics: {json.dumps(stats, indent=2)}"
            }
        else:
            return {"observation": json.dumps(parsed_data, indent=2)}
            
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
```

### Tool with Validation

```python
@tool(name="email_validator")
def validate_email(email: str) -> dict:
    """
    Validate an email address format.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        dict: Validation result with details
    """
    import re
    
    # Email validation regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(pattern, email):
        domain = email.split('@')[1]
        return {
            "observation": f"Email {email} is valid (domain: {domain})",
            "valid": True,
            "domain": domain
        }
    else:
        return {
            "observation": f"Email {email} is invalid",
            "valid": False,
            "errors": ["Invalid email format"]
        }
```

## Best Practices

### 1. Clear Documentation

Always provide clear docstrings with:
- Purpose of the tool
- Parameter descriptions with types
- Return value description
- Example usage if complex

### 2. Error Handling

```python
@tool(name="robust_tool")
def robust_tool(input_data: str) -> dict:
    """A tool with proper error handling."""
    try:
        # Tool logic here
        result = process_data(input_data)
        return {"observation": result}
    except ValueError as e:
        return {"error": f"Invalid input: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
```

### 3. Input Validation

```python
@tool(name="age_validator")
def validate_age(age: str) -> dict:
    """Validate age input."""
    try:
        age_int = int(age)
        if age_int < 0:
            return {"error": "Age cannot be negative"}
        if age_int > 150:
            return {"error": "Age seems unrealistic"}
        return {"observation": f"Age {age_int} is valid"}
    except ValueError:
        return {"error": "Age must be a number"}
```

### 4. Consistent Return Format

Always return dictionaries with either:
- `"observation"` key for successful results
- `"error"` key for error cases

### 5. Resource Management for Stateful Tools

```python
@tool(
    name="database_tool",
    env_cls=DatabaseEnv,
    stateful=True,
    pool_size=4  # Limit concurrent database connections
)
async def database_query(query: str, env: DatabaseEnv) -> dict:
    """Execute database query with proper resource management."""
    try:
        result = await env.step(query)
        return {"observation": result}
    finally:
        # Ensure resources are properly released
        await env.cleanup_if_needed()
```

## Usage in Agents

Once defined, tools can be used in agents:

```python
from agents.agents.react.react_agent import ReactAgent

# Import your custom tools
from my_tools import calculator, weather_tool, code_interpreter

# Create agent with tools
agent = ReactAgent(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    tools=[calculator, weather_tool, code_interpreter],
    template="qwen2.5-no-tool"
)

# The agent can now use these tools during execution
```

## Next Steps

Now that you understand how to define tools, proceed to:
- [Tutorial 2: Defining Rewards](02_defining_rewards.md) to learn about reward functions
- [Tutorial 3: Customizing Agents](03_customizing_agents.md) to understand agent architecture

For a complete example, see [Tutorial 7: Complete Pipeline](07_complete_pipeline.md).