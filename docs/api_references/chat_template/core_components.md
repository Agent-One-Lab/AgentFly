
<style>

.system {
  background-color: #cdb4db;
  border-radius: 6px;
  padding: 2px 6px;
}
.user {
  background-color: #ffc8dd;
  border-radius: 6px;
  padding: 2px 6px;
}
.assistant {
  background-color: #ffafcc;
  border-radius: 6px;
  padding: 2px 6px;
}
.tool {
  background-color: #bde0fe;
  border-radius: 6px;
  padding: 2px 6px;
}
</style>




# Core Components

### Core Chat Template Components

The Chat Template System is inspired by the art of building block toys - where complex structures are created by combining simple, standardized components. We identify some basic components from LLM's chat templates, and use them to form prompts from conversation messages. Below are some basic core compoenents:

`system_template`: Specify how system prompt is formatted in chat template.

`system_template_with_tools`: Specify how tools along with system prompt is formatted in chat template

`user_template`: Specify how user message is formatted in chat template

`assistant_template`: Specify how assistant is formatted in chat template

`tool_template`: Specify how tool response is formatted in chat template

Assume we have the following chat template, and messages
```
system_template = f"System: {system_message}\n"
system_template_with_tools = f"System: {system_message}\n#Tools: {tools}\n"
user_template = "User: {content}\n"
assistant_template = "User: {content}\n"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi, Can you help me search the information."},
    {"role": "assistant", "content": "tool call: search tool arguments: related query"}
    {"role": "tool", "content": "Searched inforamtion..."}
]

tools = [
    {
        "name": "search",
        "description": "Search the web."
    }
]
```

**Formatted Prompt**

<span class="system"> </span>: formatted system prompt; <span class="user"> </span>: formatted user message; <span class="assistant"> </span>: formatted assistant message; <span class="tool"> </span>: formatted tool message;
 
1. When combined, these create the complete prompt:

<span class="system">System: You are a helpful assistant.</span>

<span class="user">User: Hi, Can you help me search the information.</span>

<span class="assistant">Assistant: tool call: search\ntool arguments: related query</span>

<span class="tool">Tool: Searched inforamtion...</span>

2. When tools are included, the `system_template_with_tools` is used:

<span class="system">System: You are a helpful assistant.
<br>
#Tools: [{"name": "search", "description": "Search the web"}]</span>

<span class="user">User: Hi, Can you help me search the information.</span>

<span class="assistant">Assistant: tool call: search\ntool arguments: related query</span>

<span class="tool">Tool: Searched inforamtion...</span>


### High-Level Workflow

```
Messages + Tools → Template Processing → Vision Processing → LLM-Ready Inputs
```

The system follows a three-step rendering process:

1. **Tool Insertion**: Decide where and how to inject tool definitions
2. **Turn Encoding**: Convert each conversation turn to its textual representation
3. **Generation Prompt**: Optionally append generation prefixes

If we tokenize the input messages, the vision processor will do the following steps:

- **Template** → Human-readable prompt with vision tokens
- **Vision Processor** → Token expansion and multi-modal inputs
- **Result** → LLM-ready inputs with proper tensor alignment

### Core Class Components

#### Template
The central class that manages:
- Message formatting templates
- Policy configurations
- Jinja template generation

#### Chat
Recommended class for user usage:
- Store and format messages
- Get formatted prompts
- Tokenize formatted prompt

### Advanced Features

**1. Register & Obtain Template**

Templates are created and retrieved through a global registry:
```python
# Registration
register_template(Template(name="custom", ...))

# Retrieval
template = get_template("custom")
```

**2. Fine-grained Behavior Control**

Three levels of policy control:

1. **Global Policy**: Template-wide settings (e.g., prefix tokens)
2. **System Policy**: System message behavior and content processing
3. **Tool Policy**: Tool placement, formatting, and content processing

```python
# Tool formatting strategies
JsonFormatter(indent=4)
JsonCompactFormatter()
YamlFormatter()

# Tool placement strategies
ToolPlacement.SYSTEM
ToolPlacement.FIRST_USER
ToolPlacement.LAST_USER
```

**3. Vision Process**

Vision processors are automatically registered when vision tokens are detected:
```python
def _register_vision_processor(self):
    """Automatically register a vision processor for this template"""
    if self.image_token or self.video_token:
        # Auto-registration based on template configuration
```


**4. Jinja Template Generation**

Templates can generate HuggingFace-compatible Jinja templates:
- Enables use with external systems (vLLM, transformers tokenizers, etc.)
- Maintains consistency between Python and Jinja rendering
- Supports complex logic through Jinja macros
