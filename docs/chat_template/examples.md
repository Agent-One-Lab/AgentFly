# Examples and Use Cases

## Overview

This section provides comprehensive examples of how to use the Chat Template System in various scenarios. Each example demonstrates different features and capabilities of the system.

## Basic Examples

### Example 1: Simple Chat Template

```python
from agentfly.agents.templates import Chat, get_template

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
from agentfly.agents.templates import Template, register_template
from agentfly.agents.templates.tool_policy import ToolPolicy, JsonIndentedFormatter
from agentfly.agents.templates.constants import ToolPlacement

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

### Example 6: Dynamic Template Generation

```python
def create_specialized_template(base_name, capabilities):
    """Create template based on capabilities"""
    
    # Base templates
    system_base = "You are a helpful AI assistant."
    user_base = "User: {content}"
    assistant_base = "Assistant: {content}"
    
    # Add tool support if needed
    if "tools" in capabilities:
        system_base += "\n\nYou have access to tools: {tools}"
        user_base += "\n\nAvailable tools: {tools}"
    
    # Add vision support if needed
    if "vision" in capabilities:
        system_base += "\n\nYou can process images and videos."
    
    return Template(
        name=f"{base_name}-{'-'.join(capabilities)}",
        system_template=system_base,
        user_template=user_base,
        assistant_template=assistant_base,
        vision_start="<vision>" if "vision" in capabilities else None,
        vision_end="</vision>" if "vision" in capabilities else None,
        image_token="<image>" if "vision" in capabilities else None,
        video_token="<video>" if "vision" in capabilities else None
    )

# Create specialized templates
coding_template = create_specialized_template("coding", ["tools"])
vision_template = create_specialized_template("vision", ["vision"])
full_template = create_specialized_template("full", ["tools", "vision"])

# Register them
register_template(coding_template)
register_template(vision_template)
register_template(full_template)

print("Created specialized templates:")
print(f"- {coding_template.name}")
print(f"- {vision_template.name}")
print(f"- {full_template.name}")
```

## Real-World Use Cases

### Use Case 1: Customer Support Bot

```python
# Create a customer support template
support_template = Template(
    name="customer-support",
    system_template="""<|im_start|>system
You are a helpful customer support representative. You help customers with their questions and issues.
Always be polite, patient, and professional. If you need to escalate an issue, let the customer know.
{system_message}<|im_end|>
""",
    system_message="You are a helpful customer support representative.",
    
    # Tool support for looking up customer information
    system_template_with_tools="""<|im_start|>system
You are a helpful customer support representative with access to customer databases.
You can look up customer information and order history to better assist customers.
{system_message}

Available Tools:
{tools}<|im_end|>
""",
    
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    user_template_with_tools="<|im_start|>user\n{content}\n\nTools: {tools}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    
    stop_words=["<|im_end|>"],
    
    tool_policy=ToolPolicy(
        placement=ToolPlacement.SYSTEM,
        formatter=JsonIndentedFormatter(indent=2)
    )
)

register_template(support_template)

# Customer support conversation
support_messages = [
    {"role": "user", "content": "Hi, I have a question about my recent order #12345"},
    {"role": "assistant", "content": "Hello! I'd be happy to help you with your order. Let me look up the details for you."}
]

support_tools = [
    {
        "function": {
            "name": "lookup_order",
            "description": "Look up order details by order number",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "Order number to look up"}
                },
                "required": ["order_number"]
            }
        }
    }
]

support_chat = Chat(template="customer-support", messages=support_messages, tools=support_tools)
support_prompt = support_chat.prompt(tools=support_tools)
print("Customer Support Template:")
print(support_prompt)
```

### Use Case 2: Educational Assistant

```python
# Create an educational assistant template
education_template = Template(
    name="education-assistant",
    system_template="""<|im_start|>system
You are an educational assistant that helps students learn and understand various subjects.
Always provide clear explanations, examples, and encourage critical thinking.
Adapt your explanations to the student's level of understanding.
{system_message}<|im_end|>
""",
    system_message="You are an educational assistant.",
    
    # Tool support for educational resources
    system_template_with_tools="""<|im_start|>system
You are an educational assistant with access to educational resources and tools.
You can search for relevant materials, create practice problems, and provide detailed explanations.
{system_message}

Available Tools:
{tools}<|im_end|>
""",
    
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    user_template_with_tools="<|im_start|>user\n{content}\n\nTools: {tools}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    
    stop_words=["<|im_end|>"],
    
    tool_policy=ToolPolicy(
        placement=ToolPlacement.FIRST_USER,
        formatter=JsonIndentedFormatter(indent=2)
    )
)

register_template(education_template)

# Educational conversation
education_messages = [
    {"role": "user", "content": "Can you explain how photosynthesis works?"}
]

education_tools = [
    {
        "function": {
            "name": "search_educational_resources",
            "description": "Search for educational resources on a specific topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to search for"},
                    "grade_level": {"type": "string", "description": "Grade level of the student"}
                },
                "required": ["topic"]
            }
        }
    }
]

education_chat = Chat(template="education-assistant", messages=education_messages, tools=education_tools)
education_prompt = education_chat.prompt(tools=education_tools)
print("Educational Assistant Template:")
print(education_prompt)
```

### Use Case 3: Multi-Modal Content Analysis

```python
# Create a multi-modal analysis template
analysis_template = Template(
    name="content-analyzer",
    system_template="""<|im_start|>system
You are a content analysis expert. You can analyze text, images, and videos to provide insights.
Always provide detailed analysis with supporting evidence from the content.
{system_message}<|im_end|>
""",
    system_message="You are a content analysis expert.",
    
    # Vision support
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    
    # Tool support for analysis
    system_template_with_tools="""<|im_start|>system
You are a content analysis expert with access to analysis tools.
You can analyze text, images, and videos to provide comprehensive insights.
{system_message}

Available Tools:
{tools}<|im_end|>
""",
    
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    user_template_with_tools="<|im_start|>user\n{content}\n\nTools: {tools}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    
    stop_words=["<|im_end|>"],
    
    tool_policy=ToolPolicy(
        placement=ToolPlacement.SYSTEM,
        formatter=JsonIndentedFormatter(indent=2)
    )
)

register_template(analysis_template)

# Multi-modal analysis conversation
analysis_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image and provide insights about the content, style, and potential use cases."},
            {"type": "image", "image": "/path/to/analysis_image.jpg"}
        ]
    }
]

analysis_tools = [
    {
        "function": {
            "name": "image_analysis",
            "description": "Perform detailed image analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string", "enum": ["objects", "text", "emotions", "style"]}
                },
                "required": ["analysis_type"]
            }
        }
    }
]

analysis_chat = Chat(template="content-analyzer", messages=analysis_messages, tools=analysis_tools)
analysis_prompt = analysis_chat.prompt(tools=analysis_tools)
print("Content Analyzer Template:")
print(analysis_prompt)
```

## Testing and Validation

### Example: Template Comparison

```python
def compare_templates(template_names, messages, tools=None):
    """Compare multiple templates side by side"""
    
    print("Template Comparison:")
    print("=" * 80)
    
    for template_name in template_names:
        try:
            chat = Chat(template=template_name, messages=messages, tools=tools)
            prompt = chat.prompt(tools=tools)
            
            print(f"\n{template_name.upper()}:")
            print("-" * 40)
            print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
            
        except Exception as e:
            print(f"\n{template_name.upper()}: ERROR - {e}")
    
    print("\n" + "=" * 80)

# Compare different templates
templates_to_compare = ["qwen2.5", "llama-3.2", "glm-4"]
test_messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

compare_templates(templates_to_compare, test_messages)
```

### Example: Template Validation

```python
def validate_template(template_name, test_cases):
    """Validate a template with various test cases"""
    
    print(f"Validating template: {template_name}")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print("-" * 40)
        
        try:
            chat = Chat(template=template_name, messages=test_case['messages'])
            
            # Test basic prompt generation
            prompt = chat.prompt()
            print(f"✓ Basic prompt generated ({len(prompt)} characters)")
            
            # Test with tools if provided
            if 'tools' in test_case:
                prompt_with_tools = chat.prompt(tools=test_case['tools'])
                print(f"✓ Tool prompt generated ({len(prompt_with_tools)} characters)")
            
            # Test with generation prompt
            prompt_with_gen = chat.prompt(add_generation_prompt=True)
            print(f"✓ Generation prompt generated ({len(prompt_with_gen)} characters)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)

# Test cases for validation
test_cases = [
    {
        "description": "Basic conversation",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    },
    {
        "description": "Conversation with tools",
        "messages": [
            {"role": "user", "content": "Search for information"}
        ],
        "tools": [
            {
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            }
        ]
    }
]

# Validate a template
validate_template("qwen2.5", test_cases)
```

These examples demonstrate the versatility and power of the Chat Template System. Use them as starting points for your own implementations and adapt them to your specific needs.
