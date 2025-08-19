# Basic Usage Guide

## Getting Started

The Chat Template System provides a simple yet powerful interface for creating and using conversation templates. This guide covers the fundamental operations you'll need to get started.

## Importing the System

```python
from agents.agents.agents.templates import Chat, get_template, Template
from agents.agents.agents.templates.tool_policy import ToolPolicy, JsonFormatter
from agents.agents.agents.templates.system_policy import SystemPolicy
```

## Using Pre-built Templates

### Available Templates

The system comes with several pre-built templates:

- **qwen2.5**: Standard Qwen2.5 format
- **qwen2.5-vl**: Qwen2.5 with vision support
- **qwen2.5-think**: Qwen2.5 with thinking process
- **llama-3.2**: Llama 3.2 format
- **glm-4**: GLM-4 format
- **phi-4**: Phi-4 format
- **nemotron**: Nemotron format

### Basic Template Usage

```python
# Get a pre-built template
template = get_template("qwen2.5")

# Create a chat instance
chat = Chat(template="qwen2.5", messages=[
    {"role": "user", "content": "Hello, how are you?"}
])

# Generate a prompt
prompt = chat.prompt()
print(prompt)
```

## Creating a Chat Instance

### Simple Chat

```python
# Basic chat with text messages
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Tell me more about Paris."}
]

chat = Chat(template="qwen2.5", messages=messages)
prompt = chat.prompt()
```

### Chat with Tools

```python
# Chat with tool definitions
tools = [
    {
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

chat = Chat(template="qwen2.5", messages=messages, tools=tools)
prompt = chat.prompt(tools=tools)
```

### Chat with Vision

```python
# Chat with image content
messages_with_image = [
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image": "/path/to/image.jpg"}
        ]
    }
]

chat = Chat(template="qwen2.5-vl", messages=messages_with_image)
```

## Template Operations

### Generating Prompts

```python
# Basic prompt generation
prompt = chat.prompt()

# With generation prompt (for inference)
prompt_with_gen = chat.prompt(add_generation_prompt=True)

# With tools
prompt_with_tools = chat.prompt(tools=tools)
```

### Tokenization

```python
# Tokenize the conversation
inputs = chat.tokenize(
    tokenizer=tokenizer,
    add_generation_prompt=True,
    tools=tools
)

# The result includes:
# - input_ids: Token IDs
# - attention_mask: Attention mask
# - labels: Labels for training (-100 for non-assistant tokens)
# - action_mask: Action mask for training (1 for assistant tokens)
```

### Adding Messages

```python
# Add a single message
chat.append({"role": "user", "content": "Another question"})

# Add multiple messages
chat.append([
    {"role": "user", "content": "Question 1"},
    {"role": "assistant", "content": "Answer 1"}
])
```

## Template Configuration

### Basic Template Structure

```python
template = Template(
    name="custom",
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a helpful assistant.",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_words=["<|im_end|>"]
)
```

### Template with Tools

```python
template_with_tools = Template(
    name="custom-with-tools",
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_template_with_tools="<|im_start|>system\n{system_message}\n\n# Tools\n{tools}<|im_end|>\n",
    system_message="You are a helpful assistant with access to tools.",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    stop_words=["<|im_end|>"]
)
```

### Template with Vision

```python
vision_template = Template(
    name="custom-vision",
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a helpful vision assistant.",
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    stop_words=["<|im_end|>"]
)
```

## Message Formats

### Standard Message Format

```python
# Simple text message
{"role": "user", "content": "Hello"}

# Assistant response
{"role": "assistant", "content": "Hi there!"}

# System message
{"role": "system", "content": "You are a helpful assistant."}
```

### Multi-Modal Message Format

```python
# Message with image
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "image": "/path/to/image.jpg"}
    ]
}

# Message with video
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze this video"},
        {"type": "video", "video": "/path/to/video.mp4"}
    ]
}

# Message with URL image
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
}
```

## Working with Tools

### Tool Definition Format

```python
tools = [
    {
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum results"}
                },
                "required": ["query"]
            }
        }
    }
]
```

### Tool Response Format

```python
# Tool response message
{
    "role": "tool",
    "content": "Search results: [results here]"
}
```

## Error Handling

### Common Issues

```python
try:
    # Get a template that doesn't exist
    template = get_template("nonexistent")
except KeyError as e:
    print(f"Template not found: {e}")

try:
    # Create chat with invalid template
    chat = Chat(template="invalid", messages=messages)
except KeyError as e:
    print(f"Invalid template: {e}")
```

### Validation

```python
# Check if template supports vision
if template.supports_vision():
    print("Template supports vision processing")

# Check if template supports tool calls
if template._supports_tool_call():
    print("Template supports tool calls")
```

## Best Practices

### 1. **Template Naming**
- Use descriptive names that indicate the model and capabilities
- Include version information when appropriate
- Use consistent naming conventions

### 2. **Message Structure**
- Always use the standard role/content format
- For multi-modal content, use the list format with type specifications
- Ensure content types match the template's capabilities

### 3. **Tool Integration**
- Define tools with clear, descriptive names and parameters
- Use appropriate tool placement strategies for your use case
- Test tool integration thoroughly before deployment

### 4. **Vision Processing**
- Use vision-enabled templates for image/video content
- Ensure proper image formats and sizes
- Handle vision token expansion appropriately

This basic usage guide should get you started with the Chat Template System. For more advanced features, see the [Advanced Features](./advanced_features.md) and [Vision Templates](./vision_templates.md) sections.
