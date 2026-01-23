# Examples and Use Cases

## Overview

This section provides comprehensive examples of how to use the Chat Template System in various scenarios. Each example demonstrates different features and capabilities of the system.

## Basic Examples

### Example 1: Simple Chat Template

```python
from agentfly.templates import Chat, get_template

# Get a pre-built template
template = get_template("qwen2.5")

# Create a simple conversation
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking! How can I help you today?"},
    {"role": "user", "content": "Can you explain what machine learning is?"}
]

# Create chat instance
chat = Chat(template="qwen2.5", messages=messages)

# Generate prompt
prompt = chat.prompt()
print("Generated Prompt:")
print(prompt)

# Generate prompt with generation prompt (for inference)
prompt_with_gen = chat.prompt(add_generation_prompt=True)
print("\nPrompt with Generation Prompt:")
print(prompt_with_gen)
```

**Output:**
```
Generated Prompt:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you for asking! How can I help you today?<|im_end|>
<|im_start|>user
Can you explain what machine learning is?<|im_end|>

Prompt with Generation Prompt:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you for asking! How can I help you today?<|im_end|>
<|im_start|>user
Can you explain what machine learning is?<|im_end|>
<|im_start|>assistant
```

### Example 2: Chat with Tools

```python
# Define tools
tools = [
    {
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    }
]

# Create chat with tools
chat = Chat(template="qwen2.5", messages=messages, tools=tools)

# Generate prompt with tools
prompt_with_tools = chat.prompt(tools=tools)
print("Prompt with Tools:")
print(prompt_with_tools)
```

**Output:**
```
Prompt with Tools:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"name": "search_web", "description": "Search the web for current information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum number of results"}, "required": ["query"]}}}
{"function": {"name": "calculate", "description": "Perform mathematical calculations", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Mathematical expression to evaluate"}, "required": ["expression"]}}}
</tools>

<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you for asking! How can I help you today?<|im_end|>
<|im_start|>user
Can you explain what machine learning is?<|im_end|>
```

### Example 3: Tokenization

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Tokenize the conversation
inputs = chat.tokenize(
    tokenizer=tokenizer,
    add_generation_prompt=True,
    tools=tools
)

print("Tokenization Results:")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")
print(f"Labels shape: {inputs['labels'].shape}")
print(f"Action mask shape: {inputs['action_mask'].shape}")

# Show token alignment
print(f"\nFirst 20 tokens: {inputs['input_ids'][0][:20]}")
print(f"First 20 labels: {inputs['labels'][0][:20]}")
print(f"First 20 action mask: {inputs['action_mask'][0][:20]}")
```

## Advanced Examples

### Example 4: Custom Template Creation

```python
from agentfly.templates import Template, register_template
from agentfly.templates.tool_policy import ToolPolicy, JsonIndentedFormatter
from agentfly.templates.constants import ToolPlacement

# Create a custom coding assistant template
coding_template = Template(
    name="coding-assistant",

    # System message
    system_template="""<|im_start|>system
You are an expert coding assistant. You help users write, debug, and understand code.
Always provide clear explanations and follow best practices.
{system_message}<|im_end|>
""",
    system_message="You are an expert coding assistant.",

    # Tool support for code execution
    system_template_with_tools="""<|im_start|>system
You are an expert coding assistant with access to code execution tools.
Always think through the problem before writing code.
{system_message}

Available Tools:
{tools}<|im_end|>
""",

    # User and assistant templates
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    user_template_with_tools="<|im_start|>user\n{content}\n\nTools: {tools}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",

    # Stop words
    stop_words=["<|im_end|>"],

    # Tool policy - place tools with first user message
    tool_policy=ToolPolicy(
        placement=ToolPlacement.FIRST_USER,
        formatter=JsonIndentedFormatter(indent=2)
    )
)

# Register the template
register_template(coding_template)

# Test the template
coding_chat = Chat(template="coding-assistant", messages=[
    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
])

prompt = coding_chat.prompt()
print("Coding Assistant Template:")
print(prompt)
```

### Example 5: Vision Template Usage

```python
# Create a vision-enabled chat
vision_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image? Please describe it in detail."},
            {"type": "image", "image": "/path/to/sample_image.jpg"}
        ]
    }
]

# Use a vision template
vision_chat = Chat(template="qwen2.5-vl", messages=vision_messages)

# Generate prompt
vision_prompt = vision_chat.prompt()
print("Vision Template Prompt:")
print(vision_prompt)

# Get vision inputs
vision_inputs = vision_chat.vision_inputs()
print(f"\nVision inputs: {list(vision_inputs.keys())}")
```
