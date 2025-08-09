# Tutorial 5: Template Configuration

This tutorial covers how to configure and customize conversation templates in AgentFly. Templates control how messages are formatted for different language models and conversation styles.

## Table of Contents

1. [Understanding Templates](#understanding-templates)
2. [Predefined Templates](#predefined-templates)
3. [Creating Custom Templates](#creating-custom-templates)
4. [Template Components](#template-components)
5. [Advanced Template Features](#advanced-template-features)
6. [Best Practices](#best-practices)

## Understanding Templates

Templates in AgentFly handle the conversion from conversation messages to the specific format expected by different language models. They manage:

- System prompts and instructions
- User message formatting
- Assistant response formatting
- Tool call and response formatting
- Stop tokens and special sequences
- Multi-modal content (images, videos)

## Predefined Templates

AgentFly includes several predefined templates for popular models:

### Qwen Templates

```python
# Basic Qwen template without tools
template = "qwen2.5-no-tool"

# Qwen template with tool support
template = "qwen2.5"

# Qwen template with thinking capabilities
template = "qwen2.5-think"
```

### Using Predefined Templates

```python
from agents.agents.react.react_agent import ReactAgent

agent = ReactAgent(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    template="qwen2.5-no-tool",  # Use predefined template
    tools=[],
    system_prompt="You are a helpful assistant."
)
```

### Available Templates

```python
from agents.agents.templates.utils import get_template

# List available templates
available_templates = [
    "qwen2.5",
    "qwen2.5-no-tool", 
    "qwen2.5-think",
    "qwen2.5-no-system-tool",
    "deepseek-prover",
    # Add more as needed
]

# Get a specific template
template = get_template("qwen2.5")
print(template.name)
print(template.system_template)
```

## Creating Custom Templates

### Basic Custom Template

```python
from agents.agents.templates.templates import Template, register_template

# Define a custom template
custom_template = Template(
    name="my_custom_template",
    
    # System message formatting
    system_template="<|system|>\n{system_message}<|end|>\n",
    system_message="You are a helpful AI assistant.",
    
    # User message formatting
    user_template="<|user|>\n{content}<|end|>\n",
    
    # Assistant message formatting
    assistant_template="<|assistant|>\n{content}<|end|>\n",
    
    # Tool response formatting
    tool_template="<|tool|>\n{observation}<|end|>\n",
    
    # Stop tokens
    stop_words=["<|end|>"],
    
    # Tool support system prompt
    system_template_with_tools="""<|system|>
{system_message}

You have access to the following tools:
{tools}

To use a tool, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call><|end|>
"""
)

# Register the template
register_template(custom_template)
```

### Template for Code Generation

```python
code_template = Template(
    name="code_assistant_template",
    
    system_template="### SYSTEM ###\n{system_message}\n\n",
    system_message="""You are an expert programmer. When solving problems:
1. Think through the problem step by step
2. Write clean, well-commented code
3. Test your solution
4. Explain your approach""",
    
    user_template="### USER ###\n{content}\n\n",
    assistant_template="### ASSISTANT ###\n{content}\n\n",
    tool_template="### TOOL OUTPUT ###\n{observation}\n\n",
    
    stop_words=["### USER ###", "### SYSTEM ###"],
    
    system_template_with_tools="""### SYSTEM ###
{system_message}

Available tools:
{tools}

Use tools by writing:
```tool
{{"name": "tool_name", "arguments": {{"code": "your_code_here"}}}}
```

### ASSISTANT ###
I'll help you with programming. Let me think through this step by step.

"""
)

register_template(code_template)
```

### Multi-Modal Template

```python
multimodal_template = Template(
    name="multimodal_template",
    
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a helpful assistant that can understand both text and images.",
    
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\n{observation}<|im_end|>\n",
    
    stop_words=["<|im_end|>"],
    
    # Vision-specific tokens
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    image_token="<|image|>",
    video_token="<|video|>",
    
    system_template_with_tools="""<|im_start|>system
{system_message}

You can use these tools:
{tools}

For images, describe what you see before answering questions.
For tool usage, format as: {{"name": "tool_name", "arguments": {{"param": "value"}}}}
<|im_end|>
"""
)

register_template(multimodal_template)
```

## Template Components

### System Template Configuration

```python
# Simple system template
system_template = "{system_message}"

# System template with role indicators
system_template = "<|system|>{system_message}<|end|>"

# System template with tools
system_template_with_tools = """<|system|>
{system_message}

# Available Tools
{tools}

# Usage Instructions
To use a tool, format your response as:
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
<|end|>"""
```

### Message Templates

```python
# User message template
user_template = "<|user|>{content}<|end|>"

# Assistant message template  
assistant_template = "<|assistant|>{content}<|end|>"

# Tool response template
tool_template = "<|tool|>{observation}<|end|>"
```

### Stop Words and Tokens

```python
# String stop words
stop_words = ["<|end|>", "\n\n###"]

# List of stop words
stop_words = ["<|endoftext|>", "<|im_end|>", "<|end|>"]

# Can also be a single string
stop_words = "<|end|>"
```

## Advanced Template Features

### Dynamic System Prompts

```python
class DynamicTemplate(Template):
    """Template with dynamic system prompts based on task type."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_prompts = {
            "math": "You are a mathematics expert. Show your work step by step.",
            "code": "You are a programming expert. Write clean, efficient code.",
            "qa": "You are a knowledgeable assistant. Provide accurate answers.",
            "creative": "You are a creative writing assistant. Be imaginative and engaging."
        }
    
    def render(self, messages, tools=None, add_generation_prompt=False, task_type="general"):
        """Render with dynamic system prompt based on task type."""
        
        # Update system message based on task type
        if task_type in self.task_prompts:
            original_message = self.system_message
            self.system_message = self.task_prompts[task_type]
        
        # Call parent render method
        result = super().render(messages, tools, add_generation_prompt)
        
        # Restore original system message
        if task_type in self.task_prompts:
            self.system_message = original_message
        
        return result

# Register dynamic template
dynamic_template = DynamicTemplate(
    name="dynamic_template",
    system_template="<|system|>{system_message}<|end|>",
    user_template="<|user|>{content}<|end|>", 
    assistant_template="<|assistant|>{content}<|end|>",
    stop_words=["<|end|>"]
)

register_template(dynamic_template)
```

### Conditional Formatting

```python
class ConditionalTemplate(Template):
    """Template with conditional formatting based on content."""
    
    def render(self, messages, tools=None, add_generation_prompt=False):
        """Render with conditional formatting."""
        
        # Check if messages contain code
        has_code = any("```" in str(msg.get("content", "")) for msg in messages)
        
        if has_code:
            # Use code-specific formatting
            self.assistant_template = "<|assistant|>\n```\n{content}\n```\n<|end|>"
        else:
            # Use regular formatting
            self.assistant_template = "<|assistant|>{content}<|end|>"
        
        return super().render(messages, tools, add_generation_prompt)

conditional_template = ConditionalTemplate(
    name="conditional_template",
    system_template="<|system|>{system_message}<|end|>",
    user_template="<|user|>{content}<|end|>",
    assistant_template="<|assistant|>{content}<|end|>",
    stop_words=["<|end|>"]
)

register_template(conditional_template)
```

### Template with Custom Tool Formatting

```python
class CustomToolTemplate(Template):
    """Template with custom tool call formatting."""
    
    def render(self, messages, tools=None, add_generation_prompt=False):
        """Render with custom tool formatting."""
        
        if tools:
            # Create custom tool descriptions
            tool_descriptions = []
            for tool in tools:
                tool_desc = f"""
Tool: {tool.name}
Description: {tool.description}
Usage: {tool.name}(arguments)
"""
                tool_descriptions.append(tool_desc)
            
            tools_text = "\n".join(tool_descriptions)
            
            # Update system template with custom tool formatting
            self.system_template_with_tools = f"""<|system|>
{{system_message}}

=== AVAILABLE TOOLS ===
{tools_text}

=== USAGE INSTRUCTIONS ===
To use a tool, write: TOOL_CALL[tool_name](arguments)
Example: TOOL_CALL[calculator](2 + 2)
<|end|>"""
        
        return super().render(messages, tools, add_generation_prompt)

custom_tool_template = CustomToolTemplate(
    name="custom_tool_template",
    system_template="<|system|>{system_message}<|end|>",
    user_template="<|user|>{content}<|end|>",
    assistant_template="<|assistant|>{content}<|end|>",
    tool_template="<|tool|>RESULT: {observation}<|end|>",
    stop_words=["<|end|>"]
)

register_template(custom_tool_template)
```

### Template Factory

```python
class TemplateFactory:
    """Factory for creating templates with common patterns."""
    
    @staticmethod
    def create_chat_template(
        name: str,
        start_token: str = "<|start|>",
        end_token: str = "<|end|>",
        role_prefix: bool = True
    ) -> Template:
        """Create a chat-style template."""
        
        if role_prefix:
            system_template = f"{start_token}system\n{{system_message}}{end_token}\n"
            user_template = f"{start_token}user\n{{content}}{end_token}\n"
            assistant_template = f"{start_token}assistant\n{{content}}{end_token}\n"
            tool_template = f"{start_token}tool\n{{observation}}{end_token}\n"
        else:
            system_template = f"{start_token}{{system_message}}{end_token}\n"
            user_template = f"{start_token}{{content}}{end_token}\n"
            assistant_template = f"{start_token}{{content}}{end_token}\n"
            tool_template = f"{start_token}{{observation}}{end_token}\n"
        
        return Template(
            name=name,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            tool_template=tool_template,
            stop_words=[end_token]
        )
    
    @staticmethod
    def create_instruct_template(
        name: str,
        instruction_prefix: str = "### Instruction:",
        response_prefix: str = "### Response:",
        separator: str = "\n\n"
    ) -> Template:
        """Create an instruction-following template."""
        
        return Template(
            name=name,
            system_template=f"{instruction_prefix}\n{{system_message}}{separator}",
            user_template=f"{instruction_prefix}\n{{content}}{separator}",
            assistant_template=f"{response_prefix}\n{{content}}{separator}",
            tool_template=f"### Tool Output:\n{{observation}}{separator}",
            stop_words=[instruction_prefix, response_prefix]
        )
    
    @staticmethod
    def create_alpaca_template(name: str) -> Template:
        """Create an Alpaca-style template."""
        
        return Template(
            name=name,
            system_template="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{system_message}\n\n",
            user_template="### Instruction:\n{content}\n\n",
            assistant_template="### Response:\n{content}\n\n",
            tool_template="### Tool Output:\n{observation}\n\n",
            stop_words=["### Instruction:", "### Response:"]
        )

# Usage
factory = TemplateFactory()

# Create different template styles
chat_template = factory.create_chat_template("my_chat", "<|start|>", "<|end|>")
instruct_template = factory.create_instruct_template("my_instruct")
alpaca_template = factory.create_alpaca_template("my_alpaca")

# Register templates
register_template(chat_template)
register_template(instruct_template)
register_template(alpaca_template)
```

## Template Testing and Validation

### Template Tester

```python
class TemplateTester:
    """Test template formatting and behavior."""
    
    def __init__(self, template_name: str):
        self.template = get_template(template_name)
    
    def test_basic_conversation(self):
        """Test basic conversation formatting."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Can you help me with math?"}
        ]
        
        rendered = self.template.render(messages, add_generation_prompt=True)
        print("Basic Conversation:")
        print(rendered)
        print("-" * 50)
        
        return rendered
    
    def test_with_tools(self):
        """Test conversation with tools."""
        from agents.tools.src.code.tools import code_interpreter
        
        messages = [
            {"role": "user", "content": "Calculate 2 + 2"},
            {"role": "assistant", "content": "I'll calculate that for you."},
            {"role": "tool", "content": "4"}
        ]
        
        tools = [code_interpreter]
        rendered = self.template.render(messages, tools=tools, add_generation_prompt=True)
        print("With Tools:")
        print(rendered)
        print("-" * 50)
        
        return rendered
    
    def test_multimodal(self):
        """Test multimodal content."""
        if not (self.template.image_token or self.template.vision_start):
            print("Template doesn't support multimodal content")
            return
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": "base64_image_data"}
                ]
            }
        ]
        
        rendered = self.template.render(messages, add_generation_prompt=True)
        print("Multimodal:")
        print(rendered)
        print("-" * 50)
        
        return rendered
    
    def test_edge_cases(self):
        """Test edge cases."""
        edge_cases = [
            # Empty messages
            [],
            # Only system message
            [{"role": "system", "content": "You are helpful"}],
            # Very long content
            [{"role": "user", "content": "x" * 1000}],
            # Special characters
            [{"role": "user", "content": "Special chars: <|>{}[]()"}]
        ]
        
        for i, messages in enumerate(edge_cases):
            try:
                rendered = self.template.render(messages)
                print(f"Edge case {i+1}: ✅ OK")
            except Exception as e:
                print(f"Edge case {i+1}: ❌ Error - {str(e)}")
    
    def run_all_tests(self):
        """Run all template tests."""
        print(f"Testing template: {self.template.name}")
        print("=" * 60)
        
        self.test_basic_conversation()
        self.test_with_tools()
        self.test_multimodal()
        self.test_edge_cases()

# Usage
tester = TemplateTester("qwen2.5")
tester.run_all_tests()
```

### Template Validator

```python
def validate_template(template: Template) -> Dict[str, Any]:
    """Validate template configuration."""
    
    errors = []
    warnings = []
    
    # Check required components
    if not template.name:
        errors.append("Template must have a name")
    
    if not template.user_template:
        errors.append("Template must have user_template")
    
    if not template.assistant_template:
        errors.append("Template must have assistant_template")
    
    # Check for placeholder consistency
    required_placeholders = {
        "system_template": ["{system_message}"],
        "user_template": ["{content}"],
        "assistant_template": ["{content}"],
        "tool_template": ["{observation}"]
    }
    
    for template_attr, placeholders in required_placeholders.items():
        template_value = getattr(template, template_attr, None)
        if template_value:
            for placeholder in placeholders:
                if placeholder not in template_value:
                    warnings.append(f"{template_attr} missing placeholder: {placeholder}")
    
    # Check stop words
    if not template.stop_words:
        warnings.append("Template has no stop words defined")
    
    # Check tool support
    if template.system_template_with_tools:
        if "{tools}" not in template.system_template_with_tools:
            warnings.append("system_template_with_tools missing {tools} placeholder")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

# Validate a template
result = validate_template(custom_template)
if result["valid"]:
    print("✅ Template is valid")
else:
    print("❌ Template has errors:")
    for error in result["errors"]:
        print(f"  - {error}")

if result["warnings"]:
    print("⚠️ Warnings:")
    for warning in result["warnings"]:
        print(f"  - {warning}")
```

## Best Practices

### 1. Template Design Principles

```python
def template_design_principles():
    """Best practices for template design."""
    
    principles = {
        "clarity": [
            "Use clear role indicators",
            "Make start/end tokens obvious",
            "Avoid ambiguous formatting"
        ],
        "consistency": [
            "Use consistent token patterns",
            "Maintain uniform spacing",
            "Keep role formatting consistent"
        ],
        "compatibility": [
            "Test with target model",
            "Validate tokenization",
            "Check for token conflicts"
        ],
        "flexibility": [
            "Support both with/without tools",
            "Handle multimodal content",
            "Graceful error handling"
        ]
    }
    
    return principles
```

### 2. Testing Templates

```python
def comprehensive_template_test(template_name: str):
    """Comprehensive template testing."""
    
    test_cases = [
        # Basic conversation
        {
            "name": "basic_chat",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        
        # With system prompt
        {
            "name": "with_system",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Help me"}
            ]
        },
        
        # Multi-turn
        {
            "name": "multi_turn",
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2"}
            ]
        },
        
        # With tools
        {
            "name": "with_tools",
            "messages": [
                {"role": "user", "content": "Calculate something"},
                {"role": "assistant", "content": "Let me calculate"},
                {"role": "tool", "content": "Result: 42"}
            ],
            "tools": ["calculator"]
        }
    ]
    
    template = get_template(template_name)
    results = {}
    
    for test_case in test_cases:
        try:
            rendered = template.render(
                test_case["messages"],
                tools=test_case.get("tools"),
                add_generation_prompt=True
            )
            results[test_case["name"]] = {
                "status": "success",
                "length": len(rendered),
                "rendered": rendered[:200] + "..." if len(rendered) > 200 else rendered
            }
        except Exception as e:
            results[test_case["name"]] = {
                "status": "error",
                "error": str(e)
            }
    
    return results
```

### 3. Template Migration

```python
def migrate_template(old_template_name: str, new_template_name: str, 
                    conversion_mapping: Dict[str, str] = None):
    """Migrate from one template to another."""
    
    old_template = get_template(old_template_name)
    new_template = get_template(new_template_name)
    
    # Default conversion mapping
    if conversion_mapping is None:
        conversion_mapping = {
            "system_role": "system",
            "user_role": "user",
            "assistant_role": "assistant",
            "tool_role": "tool"
        }
    
    def convert_messages(messages: List[Dict]) -> List[Dict]:
        """Convert messages from old to new format."""
        converted = []
        
        for message in messages:
            converted_message = message.copy()
            
            # Apply role conversions
            if "role" in converted_message:
                old_role = converted_message["role"]
                if old_role in conversion_mapping:
                    converted_message["role"] = conversion_mapping[old_role]
            
            converted.append(converted_message)
        
        return converted
    
    return convert_messages

# Usage for migration
conversion_map = {
    "human": "user",
    "ai": "assistant",
    "system": "system"
}

migrate_fn = migrate_template("old_template", "new_template", conversion_map)
```

### 4. Performance Considerations

```python
import time
from typing import List

def benchmark_template(template_name: str, test_messages: List[List[Dict]], 
                      iterations: int = 100):
    """Benchmark template rendering performance."""
    
    template = get_template(template_name)
    
    # Warmup
    for messages in test_messages[:5]:
        template.render(messages)
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(iterations):
        for messages in test_messages:
            template.render(messages, add_generation_prompt=True)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_render = total_time / (iterations * len(test_messages))
    
    return {
        "template": template_name,
        "total_time": total_time,
        "iterations": iterations,
        "messages_per_iteration": len(test_messages),
        "avg_time_per_render": avg_time_per_render,
        "renders_per_second": 1 / avg_time_per_render
    }

# Benchmark templates
test_messages = [
    [{"role": "user", "content": "Short message"}],
    [{"role": "user", "content": "Much longer message " * 50}],
    [
        {"role": "user", "content": "Multi-turn"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Follow-up"}
    ]
]

results = benchmark_template("qwen2.5", test_messages)
print(f"Template {results['template']}:")
print(f"  Average time per render: {results['avg_time_per_render']:.4f}s")
print(f"  Renders per second: {results['renders_per_second']:.1f}")
```

## Usage in Training

Templates are specified in agent configuration:

```python
# Direct template specification
agent = ReactAgent(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    template="my_custom_template",  # Use your custom template
    tools=[...],
    system_prompt="Custom system prompt"
)

# In training configuration
python -m verl.trainer.main_ppo \
    agent.template=my_custom_template \
    agent.model_name_or_path=Qwen/Qwen2.5-7B-Instruct \
    ...
```

## Next Steps

Now that you understand template configuration, proceed to:
- [Tutorial 6: Training Setup](06_training_setup.md) to learn about starting training
- [Tutorial 7: Complete Pipeline](07_complete_pipeline.md) for the end-to-end workflow

You now have all the components needed to build a complete agent training pipeline!