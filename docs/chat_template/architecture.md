# Architecture & Design Philosophy

## Inspiration: Building Block Toys

The Chat Template System is inspired by the art of building block toys - where complex structures are created by combining simple, standardized components. This philosophy manifests in several key design principles:

### 1. **Modularity**
Just as building blocks can be snapped together in different configurations, template components are designed to be:
- **Interchangeable**: Different formatting strategies can be swapped without changing the core logic
- **Composable**: Complex templates are built from simple, reusable parts
- **Extensible**: New block types can be added without modifying existing ones

### 2. **Separation of Concerns**
Each component has a single, well-defined responsibility:
- **Templates** handle message formatting and structure
- **Policies** control behavior and configuration
- **Processors** handle specialized input contents (images, videos, etc.)
- **Formatters** manage tool representation

### 3. **Strategy Pattern**
The system uses the strategy pattern extensively, allowing different behaviors to be selected at runtime:
- **Tool Formatters**: JSON, YAML, custom formats
- **Tool Placement**: System, first user, last user...
- **Vision Processors**: Patch-based, model-specific implementations

## System Architecture

### High-Level Flow

```
Messages + Tools → Template Processing → Vision Processing → LLM-Ready Inputs
```

The system follows a three-step rendering process:

1. **Tool Insertion**: Decide where and how to inject tool definitions
2. **Turn Encoding**: Convert each conversation turn to its textual representation
3. **Generation Prompt**: Optionally append generation prefixes

### Core Components

#### Template Class
The central orchestrator that manages:
- Message formatting templates
- Policy configurations
- Vision processor registration
- Jinja template generation

#### Policy System
Three levels of policy control:

1. **Global Policy**: Template-wide settings (e.g., prefix tokens)
2. **System Policy**: System message behavior and content processing
3. **Tool Policy**: Tool placement, formatting, and content processing

#### Vision Processing
Separate from template processing to maintain clean separation:
- **Template** → Human-readable prompt with vision tokens
- **Vision Processor** → Token expansion and multi-modal inputs
- **Result** → LLM-ready inputs with proper tensor alignment

### Design Patterns

#### Factory Pattern
Templates are created and retrieved through a global registry:
```python
# Registration
register_template(Template(name="custom", ...))

# Retrieval
template = get_template("custom")
```

#### Strategy Pattern
Different behaviors are encapsulated in strategy classes:
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

#### Observer Pattern
Vision processors are automatically registered when vision tokens are detected:
```python
def _register_vision_processor(self):
    """Automatically register a vision processor for this template"""
    if self.image_token or self.video_token:
        # Auto-registration based on template configuration
```

## Key Design Decisions

### 1. **Template vs. Processor Separation**
- **Templates** handle text formatting and structure
- **Processors** handle specialized input types (images, videos, tools)
- This separation allows templates to focus on their core responsibility

### 2. **Policy-Based Configuration**
Instead of hardcoded behavior, templates use configurable policies:
- **System Policy**: Controls when and how system messages appear
- **Tool Policy**: Manages tool integration strategy
- **Global Policy**: Template-wide behavior settings

### 3. **Automatic Vision Registration**
Vision processors are automatically registered based on template configuration:
```python
def __post_init__(self):
    if self.image_token or self.video_token:
        self._register_vision_processor()
```

### 4. **Jinja Template Generation**
Templates can generate HuggingFace-compatible Jinja templates:
- Enables use with external systems (vLLM, etc.)
- Maintains consistency between Python and Jinja rendering
- Supports complex logic through Jinja macros

## Extensibility Points

### Adding New Template Types
1. Create a new `Template` instance
2. Configure templates, policies, and vision settings
3. Register with `register_template()`

### Adding New Tool Formatters
1. Inherit from `ToolFormatter` base class
2. Implement `format()` and `jinja()` methods
3. Use in `ToolPolicy` configuration

### Adding New Vision Processors
1. Inherit from `VisionProcessor` base class
2. Implement required abstract methods
3. Register with `register_processor()`

### Adding New System Processors
1. Inherit from `SystemContentProcessor` base class
2. Implement `__call__()` and `jinja()` methods
3. Use in `SystemPolicy` configuration

## Benefits of This Architecture

### 1. **Maintainability**
- Clear separation of concerns
- Single responsibility principle
- Easy to modify individual components

### 2. **Flexibility**
- Multiple strategies for each behavior
- Runtime configuration through policies
- Easy to add new capabilities

### 3. **Reusability**
- Components can be shared across templates
- Policies can be reused and combined
- Vision processors work with any template

### 4. **Consistency**
- Unified interface for all template types
- Consistent policy configuration
- Standardized vision processing pipeline

This architecture makes the template system both powerful and easy to use, while maintaining the flexibility to handle diverse use cases and requirements.
