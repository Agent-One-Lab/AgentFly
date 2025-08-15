# Chat Template System Documentation

Welcome to the comprehensive documentation for the Chat Template System - a powerful and flexible framework for creating conversation templates inspired by building block toys.

## 📚 Documentation Structure

### 🏗️ [Architecture & Design](./architecture.md)
- **System Philosophy**: Building block approach to template design
- **Architecture Overview**: High-level system design and flow
- **Design Patterns**: Factory, Strategy, and Observer patterns
- **Extensibility Points**: How to extend the system
- **Key Design Decisions**: Rationale behind architectural choices

### 🚀 [Basic Usage](./basic_usage.md)
- **Getting Started**: Quick start guide and imports
- **Pre-built Templates**: Available templates and their usage
- **Chat Operations**: Creating chats, generating prompts, tokenization
- **Template Configuration**: Basic template structure and fields
- **Message Formats**: Standard and multi-modal message structures
- **Error Handling**: Common issues and validation

### 🛠️ [Custom Templates](./custom_templates.md)
- **Template Components**: Core and advanced template fields
- **Template Creation**: Step-by-step template building
- **Policy Configuration**: System, tool, and global policies
- **Template Registration**: How to register and manage templates
- **Advanced Features**: Jinja templates, inheritance, copying
- **Best Practices**: Template design and testing guidelines

### 🔧 [Advanced Features](./advanced_features.md)
- **Tool Policy System**: Placement strategies and formatters
- **System Policy System**: Message control and content processors
- **Global Policy Configuration**: Template-wide settings
- **Policy Composition**: Combining and inheriting policies
- **Advanced Tool Integration**: Custom placement and validation
- **Performance Optimization**: Caching and lazy evaluation

### 👁️ [Vision Templates](./vision_templates.md)
- **Vision Architecture**: Pipeline overview and key components
- **Creating Vision Templates**: Basic and advanced vision templates
- **Vision Processor Configuration**: Automatic registration and model inference
- **Processor Types**: Patch-based, Qwen-VL, LLaVA processors
- **Input Formats**: Image, video, and message formats
- **Token Calculation**: Image and video token computation
- **Advanced Features**: Custom processors and configuration options

### 💡 [Examples & Use Cases](./examples.md)
- **Basic Examples**: Simple chat, tools, tokenization
- **Advanced Examples**: Custom templates, vision usage, dynamic generation
- **Real-World Use Cases**: Customer support, education, content analysis
- **Testing & Validation**: Template comparison and validation
- **Complete Examples**: End-to-end implementation examples

## 🎯 Quick Start

```python
from agents.agents.agents.templates import Chat, get_template

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

## 🔑 Key Concepts

### Template Components
- **System Template**: Defines system message format
- **User Template**: How user messages are formatted
- **Assistant Template**: How assistant responses are formatted
- **Tool Template**: How tool responses are formatted

### Policies
- **System Policy**: Controls system message behavior
- **Tool Policy**: Manages tool integration strategy
- **Global Policy**: Template-wide behavior settings

### Vision Support
- **Image Processing**: Automatic image token expansion
- **Video Processing**: Video frame extraction and processing
- **Multi-Modal Alignment**: Proper tensor alignment for training

## 🚀 Getting Started

1. **Read the [Architecture](./architecture.md)** to understand the system design
2. **Follow [Basic Usage](./basic_usage.md)** for quick setup
3. **Explore [Examples](./examples.md)** to see practical implementations
4. **Create [Custom Templates](./custom_templates.md)** for your specific needs
5. **Leverage [Advanced Features](./advanced_features.md)** for complex use cases
6. **Add [Vision Support](./vision_templates.md)** for multi-modal capabilities

## 🎨 Design Philosophy

The Chat Template System is inspired by **building block toys** - where complex structures are created by combining simple, standardized components. This philosophy manifests in:

- **Modularity**: Interchangeable, composable, extensible components
- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Strategy Pattern**: Different behaviors can be selected at runtime
- **Policy-Based Configuration**: Flexible behavior control without hardcoding

## 🔧 System Architecture

```
Messages + Tools → Template Processing → Vision Processing → LLM-Ready Inputs
```

The system follows a **three-step rendering process**:
1. **Tool Insertion**: Decide where and how to inject tool definitions
2. **Turn Encoding**: Convert each conversation turn to its textual representation
3. **Generation Prompt**: Optionally append generation prefixes

## 🌟 Key Features

- **Modular Design**: Templates built from configurable components
- **Multi-Modal Support**: Built-in vision and video processing
- **Tool Integration**: Flexible tool placement and formatting strategies
- **Policy-Based Configuration**: Fine-grained control over behavior
- **Jinja Template Generation**: Automatic HuggingFace-compatible templates
- **Extensible Architecture**: Easy to add new template types and processors

## 📖 Additional Resources

- **Source Code**: `agents/agents/agents/templates/`
- **API Reference**: Check the source code for detailed method documentation
- **Issues & Discussions**: Use the project's issue tracker for questions

## 🤝 Contributing

The template system is designed to be extensible. See [Custom Templates](./custom_templates.md) for guidance on adding new template types and processors.

---

*This documentation covers the complete Chat Template System. Start with the architecture to understand the design, then follow the usage guides to implement your own templates.*
