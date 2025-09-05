# Basic Usage Guide

## Getting Started

The Chat Template System provides a simple yet powerful interface for creating and using conversation templates. This guide covers the fundamental operations you'll need to get started.

## Importing the System

```python
from agentfly.templates import Chat, get_template, Template, ToolPolicy, JsonFormatter, SystemPolicy
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

`Template` is the basic template class, consists of different components and responsible for forming the prompt. While `Chat` is the class we recommand for users to obtain prompts.

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
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
        ]
    }
]

chat = Chat(template="qwen2.5-vl", messages=messages_with_image)
prompt = chat.prompt()
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

Use `Chat.tokenize` method to tokenize the messages with the specified chat template.

```python
from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# Tokenize the conversation
inputs = chat.tokenize(
    tokenizer=tokenizer,
    add_generation_prompt=True,
    processor=processor,
    tools=tools
)
print(inputs.keys())

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
