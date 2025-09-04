# Creating Custom Templates

## Overview

The Chat Template System is designed to be highly extensible, allowing you to create custom templates that fit your specific needs. This guide covers how to create, configure, and register custom templates.

## Template Components

### Core Template Fields



```python
from agentfly.agents.templates import Template

template = Template(
    name="my-custom-template",           # Unique identifier
    system_template="System: {system_message}",  # System message format
    system_message="You are a helpful assistant.", # Default system message
    user_template="User: {content}",           # User message format
    assistant_template="Assistant: {content}</s>",      # Assistant message format
    tool_template="Tool: {observation}",       # Tool response format
    stop_words=["</s>"]                 # Stop generation tokens
)
```

### Advanced Template Fields

```python
template = Template(
    # ... core fields ...
    
    # Tool support
    system_template_with_tools="System: {system_message}\n\nTools: {tools}",
    user_template_with_tools="User: {content}\n\nAvailable tools: {tools}",
    
    # Vision support
    vision_start="<vision>",
    vision_end="</vision>",
    image_token="<image>",
    video_token="<video>",
    
    # Custom chat template (Jinja)
    chat_template="{% for message in messages %}{{ message.content }}{% endfor %}"
)
```

## Template Creation Examples

### 1. Simple Chat Template

```python
simple_template = Template(
    name="simple-chat",
    system_template="You are a helpful assistant.\n",
    system_message="You are a helpful assistant.",
    user_template="User: {content}\n",
    assistant_template="Assistant: {content}\n",
    stop_words=["\n"]
)
```

### 2. XML-Style Template

```python
xml_template = Template(
    name="xml-style",
    system_template="<system>{system_message}</system>\n",
    system_message="You are an AI assistant.",
    user_template="<user>{content}</user>\n",
    assistant_template="<assistant>{content}</assistant>\n",
    stop_words=["</assistant>"]
)
```

### 3. Markdown-Style Template

```python
markdown_template = Template(
    name="markdown-style",
    system_template="# System\n{system_message}\n\n",
    system_message="You are a helpful AI assistant.",
    user_template="## User\n{content}\n\n",
    assistant_template="## Assistant\n{content}\n\n",
    stop_words=["\n\n"]
)
```

### 4. Tool-Enabled Template

```python
tool_template = Template(
    name="tool-enabled",
    system_template="System: {system_message}\n",
    system_template_with_tools="System: {system_message}\n\nAvailable Tools:\n{tools}\n",
    system_message="You are an AI assistant with access to tools.",
    user_template="User: {content}\n",
    user_template_with_tools="User: {content}\n\nTools: {tools}\n",
    assistant_template="Assistant: {content}\n",
    tool_template="Tool Response: {observation}\n",
    stop_words=["\n"]
)
```

### 5. Vision-Enabled Template

```python
vision_template = Template(
    name="vision-enabled",
    system_template="You are a vision-capable AI assistant.\n",
    system_message="You are a vision-capable AI assistant.",
    user_template="User: {content}\n",
    assistant_template="Assistant: {content}\n",
    vision_start="<vision>",
    vision_end="</vision>",
    image_token="<image>",
    video_token="<video>",
    stop_words=["\n"]
)
```

## Policy Configuration

### System Policy

```python
from agentfly.agents.templates.system_policy import SystemPolicy

# Basic system policy
system_policy = SystemPolicy(
    use_system=True,                           # Always include system message
    use_system_without_system_message=True,    # Include system even without explicit message
    content_processor=None                     # No content processing
)

# System policy with content processor
def add_timestamp(system_message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] {system_message}"

system_policy = SystemPolicy(
    use_system=True,
    use_system_without_system_message=True,
    content_processor=add_timestamp
)
```

### Tool Policy

```python
from agentfly.agents.templates.tool_policy import ToolPolicy, JsonFormatter, JsonIndentedFormatter
from agentfly.agents.templates.constants import ToolPlacement

# Basic tool policy
tool_policy = ToolPolicy(
    placement=ToolPlacement.SYSTEM,           # Place tools in system message
    formatter=JsonFormatter(indent=2)         # Pretty-printed JSON
)

# Advanced tool policy
tool_policy = ToolPolicy(
    placement=ToolPlacement.FIRST_USER,       # Place tools with first user message
    formatter=JsonIndentedFormatter(indent=4), # Indented JSON
    content_processor=None                    # No content processing
)
```

### Global Policy

```python
from agentfly.agents.templates import GlobalPolicy

global_policy = GlobalPolicy(
    prefix="<|begin_of_text|>"               # Add prefix to all prompts
)
```

## Complete Template Example

```python
from agentfly.agents.templates import Template, GlobalPolicy
from agentfly.agents.templates.system_policy import SystemPolicy
from agentfly.agents.templates.tool_policy import ToolPolicy, JsonIndentedFormatter
from agentfly.agents.templates.constants import ToolPlacement

# Create a comprehensive template
comprehensive_template = Template(
    name="comprehensive-example",
    
    # Basic templates
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a comprehensive AI assistant with multiple capabilities.",
    
    # Tool support
    system_template_with_tools="<|im_start|>system\n{system_message}\n\nAvailable Tools:\n{tools}<|im_end|>\n",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    user_template_with_tools="<|im_start|>user\n{content}\n\nTools: {tools}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    
    # Vision support
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    
    # Stop words
    stop_words=["<|im_end|>"],
    
    # Policies
    global_policy=GlobalPolicy(prefix="<|begin_of_text|>"),
    system_policy=SystemPolicy(
        use_system=True,
        use_system_without_system_message=True
    ),
    tool_policy=ToolPolicy(
        placement=ToolPlacement.SYSTEM,
        formatter=JsonIndentedFormatter(indent=2)
    )
)
```

## Template Registration

### Registering a Template

```python
from agentfly.agents.templates import register_template

# Register the template
register_template(comprehensive_template)

# Now you can use it
from agentfly.agents.templates import get_template
template = get_template("comprehensive-example")
```

### Overriding Existing Templates

```python
# Override an existing template
register_template(comprehensive_template, override=True)
```

### Template Registry Management

```python
from agentfly.agents.templates import TEMPLATES

# List all registered templates
print("Available templates:", list(TEMPLATES.keys()))

# Check if template exists
if "my-template" in TEMPLATES:
    print("Template exists")

# Remove a template
if "my-template" in TEMPLATES:
    del TEMPLATES["my-template"]
```

## Advanced Template Features

### Custom Jinja Templates

```python
# Use a custom Jinja template
custom_jinja_template = Template(
    name="custom-jinja",
    chat_template="""
    {% for message in messages %}
        {% if message.role == 'system' %}
            System: {{ message.content }}
        {% elif message.role == 'user' %}
            User: {{ message.content }}
        {% elif message.role == 'assistant' %}
            Assistant: {{ message.content }}
        {% endif %}
    {% endfor %}
    """,
    system_message="You are a helpful assistant."
)
```

### Template Inheritance

```python
# Create a base template
base_template = Template(
    name="base",
    system_template="Base: {system_message}\n",
    user_template="User: {content}\n",
    assistant_template="Assistant: {content}\n"
)

# Create a specialized template
specialized_template = Template(
    name="specialized",
    system_template="Specialized: {system_message}\n",
    user_template=base_template.user_template,      # Inherit user template
    assistant_template=base_template.assistant_template,  # Inherit assistant template
    system_message="You are a specialized assistant."
)
```

### Template Copying

```python
# Copy an existing template
original = get_template("qwen2.5")
modified = original.copy()
modified.name = "qwen2.5-modified"
modified.system_message = "Modified system message"

# Register the modified version
register_template(modified)
```

## Best Practices

### 1. **Template Naming**
- Use descriptive, unique names
- Include version information when appropriate
- Use consistent naming conventions across your project

### 2. **Template Structure**
- Keep templates focused and single-purpose
- Use consistent formatting patterns
- Document special tokens and their meanings

### 3. **Policy Configuration**
- Start with default policies and customize as needed
- Use appropriate tool placement strategies
- Consider the impact of content processors

### 4. **Vision Support**
- Only add vision tokens when needed
- Use appropriate vision start/end markers
- Consider token expansion implications

### 5. **Testing**
- Test templates with various message types
- Verify tool integration works correctly
- Test vision processing if applicable
- Validate Jinja template generation

## Example: Complete Custom Template

Here's a complete example of creating a custom template for a specific use case:

```python
from agentfly.agents.templates import Template, register_template
from agentfly.agents.templates.tool_policy import ToolPolicy, JsonCompactFormatter
from agentfly.agents.templates.constants import ToolPlacement

# Create a coding assistant template
coding_template = Template(
    name="coding-assistant",
    
    # System message
    system_template="""<|im_start|>system
You are an expert coding assistant. You help users write, debug, and understand code.
Always provide clear explanations and follow best practices.
{system_message}<|im_end|>
""",
    system_message="You are an expert coding assistant.",
    
    # User and assistant templates
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    
    # Tool support for code execution
    system_template_with_tools="""<|im_start|>system
You are an expert coding assistant with access to code execution tools.
Always think through the problem before writing code.
{system_message}

Available Tools:
{tools}<|im_end|>
""",
    user_template_with_tools="<|im_start|>user\n{content}\n\nTools: {tools}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    
    # Stop words
    stop_words=["<|im_end|>"],
    
    # Tool policy - place tools with first user message
    tool_policy=ToolPolicy(
        placement=ToolPlacement.FIRST_USER,
        formatter=JsonCompactFormatter()
    )
)

# Register the template
register_template(coding_template)

# Test the template
from agentfly.agents.templates import Chat

chat = Chat(template="coding-assistant", messages=[
    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
])

prompt = chat.prompt()
print(prompt)
```

This comprehensive guide should help you create custom templates that meet your specific requirements. Remember to test thoroughly and follow the best practices outlined above.
