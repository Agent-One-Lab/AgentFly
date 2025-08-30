# Chat Template System

To enable Agent-RL for various models, we have supported a powerful and flexible template system inspired by building block toys, designed to support various ways of forming conversation templates for large language models.

## Overview

The Chat Template System is a comprehensive framework that provides a modular, extensible approach to creating conversation templates. It's designed with the philosophy that complex templates can be built from simple, reusable components - much like building blocks that can be combined in countless ways.

## Key Features

- **Modular Design**: Templates are built from configurable components
- **Multi-Modal Support**: Built-in vision and video processing capabilities
- **Tool Integration**: Flexible tool placement and formatting strategies
- **Policy-Based Configuration**: System and tool policies for fine-grained control
- **Jinja Template Generation**: Automatic HuggingFace-compatible template generation
- **Extensible Architecture**: Easy to add new template types and processors

## Quick Start

```python
from agentfly.agents.templates import Chat, get_template

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

## Documentation Structure

- [**Architecture & Design**](./architecture.md) - System design philosophy and architecture
- [**Basic Usage**](./basic_usage.md) - Getting started with templates
- [**Custom Templates**](./custom_templates.md) - Creating your own templates
- [**Advanced Features**](./advanced_features.md) - Tool policies, system policies, and more
- [**Vision Templates**](./vision_templates.md) - Multi-modal template support
- [**Examples**](./examples.md) - Practical examples and use cases

## Core Concepts

### Template Components
- **System Template**: Defines the system message format
- **User Template**: How user messages are formatted
- **Assistant Template**: How assistant responses are formatted
- **Tool Template**: How tool responses are formatted

### Policies
- **System Policy**: Controls system message behavior
- **Tool Policy**: Manages tool placement and formatting
- **Global Policy**: Global template behavior settings

### Vision Support
- **Image Processing**: Automatic image token expansion
- **Video Processing**: Video frame extraction and processing
- **Multi-Modal Alignment**: Proper tensor alignment for training

## Getting Help

For questions and issues:
- Check the examples in the [Examples](./examples.md) section
- Review the [Advanced Features](./advanced_features.md) for complex use cases
- Examine the source code in `agents/agents/agents/templates/`

## Contributing

The template system is designed to be extensible. See [Custom Templates](./custom_templates.md) for guidance on adding new template types and processors.
