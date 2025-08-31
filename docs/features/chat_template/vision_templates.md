# Vision Templates

## Overview

The Chat Template System provides comprehensive support for multi-modal templates that can process images and videos alongside text. This guide covers how to create, configure, and use vision-enabled templates.

## Vision Architecture

### Pipeline Overview

The vision processing follows a clear separation of concerns:

```
Messages → Template Processing → Vision Processor → LLM-Ready Inputs
```

1. **Template Processing**: Creates human-readable prompts with vision tokens
2. **Vision Processing**: Handles image/video processing and token expansion
3. **Final Output**: LLM-ready inputs with proper tensor alignment

### Key Components

- **Vision Tokens**: Placeholders in prompts that get expanded to actual tokens
- **Vision Processors**: Specialized classes that handle multi-modal input processing
- **Token Expansion**: Converting vision tokens to their actual token representations
- **Tensor Alignment**: Ensuring all tensors (input_ids, attention_mask, labels, action_mask) are properly aligned

## Creating Vision Templates

### Basic Vision Template

```python
from agentfly.agents.templates import Template

vision_template = Template(
    name="vision-enabled",
    system_template="You are a vision-capable AI assistant.\n",
    system_message="You are a vision-capable AI assistant.",
    user_template="User: {content}\n",
    assistant_template="Assistant: {content}\n",
    
    # Vision configuration
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    
    stop_words=["\n"]
)
```

### Vision Template with Tools

```python
vision_tool_template = Template(
    name="vision-tool-enabled",
    system_template="You are a vision-capable AI assistant.\n",
    system_template_with_tools="You are a vision-capable AI assistant with tools.\n\nTools: {tools}\n",
    system_message="You are a vision-capable AI assistant with tools.",
    user_template="User: {content}\n",
    user_template_with_tools="User: {content}\n\nTools: {tools}\n",
    assistant_template="Assistant: {content}\n",
    tool_template="Tool: {observation}\n",
    
    # Vision configuration
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    
    stop_words=["\n"]
)
```

## Vision Processor Configuration

### Automatic Registration

Vision processors are automatically registered when vision tokens are detected:

```python
# This happens automatically in __post_init__
def _register_vision_processor(self):
    """Automatically register a vision processor for this template"""
    if self.image_token or self.video_token:
        from .vision_processor import VisionProcessorConfig, register_processor
        
        # Determine model type based on template name
        model_type = self._infer_model_type()
        
        # Create vision config
        config = VisionProcessorConfig(
            model_type=model_type,
            image_token=self.image_token or "",
            video_token=self.video_token or "",
            vision_start=self.vision_start or "",
            vision_end=self.vision_end or "",
            processor_class="AutoProcessor",
            expansion_strategy="patch_based"
        )
        
        # Register the processor
        register_processor(self.name, config)
```

### Model Type Inference

The system automatically infers the appropriate vision processor based on template name:

```python
def _infer_model_type(self) -> str:
    """Infer model type from template name"""
    name_lower = self.name.lower()
    
    if "qwen" in name_lower:
        return "qwen_vl"
    elif "llava" in name_lower:
        return "llava"
    elif "gemma" in name_lower:
        return "gemma3"
    elif "paligemma" in name_lower:
        return "paligemma"
    elif "internvl" in name_lower:
        return "internvl"
    elif "minicpm" in name_lower:
        return "minicpm"
    elif "mllama" in name_lower:
        return "mllama"
    elif "pixtral" in name_lower:
        return "pixtral"
    elif "video" in name_lower:
        return "video_llava"
    else:
        # Default to patch-based for unknown models
        return "patch_based"
```

## Vision Processor Types

### Patch-Based Processor

The default processor used by most vision models:

```python
from agentfly.agents.templates.vision_processor import PatchBasedProcessor

# Automatically used for most models
# Supports multiple image input formats
# Handles token calculation and expansion
```

### Qwen-VL Processor

Specialized processor for Qwen-VL models:

```python
from agentfly.agents.templates.vision_processor import QwenVLProcessor

# Qwen-VL specific image preprocessing
# Custom token calculation using grid-based approach
# Optimized for Qwen-VL architecture
```

### LLaVA Processor

Specialized processor for LLaVA models:

```python
from agentfly.agents.templates.vision_processor import LlavaProcessor

# LLaVA specific token calculation
# Optimized for LLaVA architecture
```

## Input Formats

### Image Input Formats

The system supports multiple image input formats:

```python
# File path
image_path = "/path/to/image.jpg"

# URL
image_url = "https://example.com/image.jpg"

# Base64 string (data URL)
image_base64 = "data:image/jpeg;base64,/9j/4AAQ..."

# Raw base64 string
raw_base64 = "iVBORw0KGgoAAAANSUhEUgAA..."

# PIL Image object
from PIL import Image
pil_image = Image.open("image.jpg")

# Bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

# File-like object
with open("image.jpg", "rb") as f:
    image_file = f

# Dict format
image_dict = {"path": "/path/to/image.jpg"}
# or
image_dict = {"bytes": b"image_data"}
```

### Video Input Formats

```python
# Video file path
video_path = "/path/to/video.mp4"

# File-like object
with open("video.mp4", "rb") as f:
    video_file = f

# List of image frames
video_frames = [
    "/path/to/frame1.jpg",
    "/path/to/frame2.jpg",
    "/path/to/frame3.jpg"
]

# List of PIL Image objects
from PIL import Image
video_frames = [
    Image.open("frame1.jpg"),
    Image.open("frame2.jpg"),
    Image.open("frame3.jpg")
]
```

### Message Format with Vision

```python
# Message with image
message_with_image = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "image": "/path/to/image.jpg"}
    ]
}

# Message with video
message_with_video = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze this video"},
        {"type": "video", "video": "/path/to/video.mp4"}
    ]
}

# Message with URL image
message_with_url = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
}
```

## Using Vision Templates

### Basic Vision Chat

```python
from agentfly.agents.templates import Chat

# Create chat with vision template
chat = Chat(template="qwen2.5-vl", messages=[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in this image"},
            {"type": "image", "image": "/path/to/image.jpg"}
        ]
    }
])

# Generate prompt
prompt = chat.prompt()
print(prompt)
```

### Vision Chat with Tools

```python
# Vision chat with tool definitions
tools = [
    {
        "function": {
            "name": "analyze_image",
            "description": "Analyze image content",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string", "enum": ["objects", "text", "emotions"]}
                }
            }
        }
    }
]

chat = Chat(template="qwen2.5-vl", messages=messages_with_image, tools=tools)
prompt = chat.prompt(tools=tools)
```

### Tokenization with Vision

```python
# Tokenize vision-enabled conversation
inputs = chat.tokenize(
    tokenizer=tokenizer,
    processor=processor,  # Required for vision processing
    add_generation_prompt=True,
    tools=tools
)

# Result includes:
# - input_ids: Token IDs with vision tokens expanded
# - attention_mask: Attention mask
# - labels: Labels for training (-100 for non-assistant tokens)
# - action_mask: Action mask for training (1 for assistant tokens)
# - pixel_values: Image/video tensors
# - image_grid_thw: Grid information (for some models)
```

## Vision Processing Pipeline

### Step 1: Template Processing

```python
# Template creates prompt with vision tokens
prompt, elements, roles = template.render(messages, tools=tools)
# Result: "User: Describe what you see in this image <|image_pad|>"
```

### Step 2: Vision Token Expansion

```python
# Vision processor expands tokens based on actual image content
expanded_prompt = vision_processor.expand_vision_tokens(
    prompt=prompt,
    images=images,
    videos=videos,
    processor=processor
)
# Result: "User: Describe what you see in this image <|image_pad|><|image_pad|><|image_pad|>..."
```

### Step 3: Multi-Modal Input Generation

```python
# Generate vision inputs
mm_inputs = vision_processor.get_mm_inputs(images, videos, processor)
# Result: {"pixel_values": tensor, "image_grid_thw": tensor}
```

### Step 4: Final Tokenization

```python
# Tokenize expanded prompt with proper alignment
final_inputs = vision_processor.process_for_llm(
    prompt=prompt,
    elements=elements,
    mask_flags=mask_flags,
    images=images,
    videos=videos,
    processor=processor,
    tokenizer=tokenizer
)
```

## Token Calculation

### Image Token Calculation

```python
def calculate_image_tokens(self, image_data, processor):
    """Calculate tokens needed for an image"""
    
    if "pixel_values" in image_data:
        # Try grid-based calculation first (HuggingFace method)
        if "image_grid_thw" in image_data:
            grid_info = image_data["image_grid_thw"]
            grid_prod = grid_info.prod().item()
            
            # Get merge_size from processor
            merge_size = getattr(processor, "merge_size", 1)
            merge_length = merge_size ** 2
            
            num_image_tokens = grid_prod // merge_length
            return max(1, num_image_tokens)
        
        # Fallback to patch-based calculation
        height, width = get_image_size(image_data["pixel_values"][0])
        image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
        
        # Add additional tokens if specified
        if hasattr(processor, 'num_additional_image_tokens'):
            image_seqlen += processor.num_additional_image_tokens
        
        # Adjust for feature selection strategy
        if (hasattr(processor, 'vision_feature_select_strategy') and 
            processor.vision_feature_select_strategy == "default"):
            image_seqlen -= 1
        
        return image_seqlen
    
    return 1
```

### Video Token Calculation

```python
def calculate_video_tokens(self, video_data, processor):
    """Calculate tokens needed for a video"""
    
    if "pixel_values" in video_data:
        video_tensor = video_data["pixel_values"][0]
        
        if len(video_tensor.shape) > 3:  # Has frame dimension
            num_frames = video_tensor.shape[0]
            height, width = get_image_size(video_tensor[0])
            frame_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
            
            # Add additional tokens if specified
            if hasattr(processor, 'num_additional_image_tokens'):
                frame_seqlen += processor.num_additional_image_tokens
            
            # Adjust for feature selection strategy
            if (hasattr(processor, 'vision_feature_select_strategy') and 
                processor.vision_feature_select_strategy == "default"):
                frame_seqlen -= 1
            
            return frame_seqlen * num_frames
        else:
            # Single frame video
            return self.calculate_image_tokens(video_data, processor)
    
    return 1
```

## Advanced Vision Features

### Custom Vision Processors

```python
from agentfly.agents.templates.vision_processor import VisionProcessor, VisionProcessorConfig

class CustomVisionProcessor(VisionProcessor):
    """Custom vision processor for specific needs"""
    
    def preprocess_images(self, images, processor):
        """Custom image preprocessing"""
        # Custom preprocessing logic
        processed_images = []
        for image in images:
            # Apply custom transformations
            processed_image = self._custom_transform(image)
            processed_images.append(processed_image)
        
        # Use processor's image processor
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise ValueError("Image processor not found")
        
        return image_processor(processed_images, return_tensors="pt")
    
    def calculate_image_tokens(self, image_data, processor):
        """Custom token calculation"""
        # Custom token calculation logic
        base_tokens = super().calculate_image_tokens(image_data, processor)
        return base_tokens * 2  # Example: double the tokens
    
    def expand_vision_tokens(self, prompt, images, videos, processor):
        """Custom token expansion"""
        # Custom expansion logic
        expanded = super().expand_vision_tokens(prompt, images, videos, processor)
        return f"<vision_start>{expanded}<vision_end>"

# Register custom processor
config = VisionProcessorConfig(
    model_type="custom",
    image_token="<custom_image>",
    video_token="<custom_video>",
    vision_start="<custom_vision_start>",
    vision_end="<custom_vision_end>"
)

from agentfly.agents.templates.vision_processor import register_processor
register_processor("custom-template", config, CustomVisionProcessor)
```

### Vision Configuration Options

```python
from agentfly.agents.templates.vision_processor import VisionProcessorConfig

config = VisionProcessorConfig(
    model_type="qwen_vl",
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    vision_start="<|vision_start|>",
    vision_end="<|vision_end|>",
    processor_class="AutoProcessor",
    expansion_strategy="patch_based",
    image_max_pixels=16384 * 28 * 28,  # Maximum image size
    image_min_pixels=4 * 28 * 28,      # Minimum image size
    video_max_pixels=16384 * 28 * 28,  # Maximum video size
    video_min_pixels=4 * 28 * 28,      # Minimum video size
    video_fps=2.0,                     # Video frame rate
    video_maxlen=128                    # Maximum video length
)
```

## Best Practices

### 1. **Template Design**
- Use descriptive vision token names
- Ensure vision tokens are unique and recognizable
- Consider token expansion implications

### 2. **Image Processing**
- Use appropriate image formats (JPEG, PNG)
- Consider image size and resolution
- Handle various input formats gracefully

### 3. **Video Processing**
- Use appropriate video formats (MP4, AVI)
- Consider frame rate and length
- Handle both file and frame-based inputs

### 4. **Token Management**
- Understand token calculation for your model
- Consider memory implications of large images/videos
- Use appropriate token limits

### 5. **Error Handling**
- Validate image/video inputs
- Handle processing failures gracefully
- Provide meaningful error messages

### 6. **Performance**
- Cache processed images when possible
- Use appropriate image sizes for your use case
- Consider batch processing for multiple images

## Example: Complete Vision Template

Here's a complete example of creating and using a vision template:

```python
from agentfly.agents.templates import Template, register_template, Chat
from agentfly.agents.templates.tool_policy import ToolPolicy, JsonFormatter
from agentfly.agents.templates.constants import ToolPlacement

# Create a comprehensive vision template
vision_template = Template(
    name="comprehensive-vision",
    
    # Basic templates
    system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
    system_message="You are a comprehensive vision-capable AI assistant.",
    
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
    
    # Tool policy
    tool_policy=ToolPolicy(
        placement=ToolPlacement.SYSTEM,
        formatter=JsonFormatter(indent=2)
    )
)

# Register the template
register_template(vision_template)

# Create chat with vision content
chat = Chat(template="comprehensive-vision", messages=[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image and describe what you see"},
            {"type": "image", "image": "/path/to/image.jpg"}
        ]
    }
])

# Generate prompt
prompt = chat.prompt()
print(prompt)

# Tokenize with vision processing
inputs = chat.tokenize(
    tokenizer=tokenizer,
    processor=processor,
    add_generation_prompt=True
)

print("Input shape:", inputs["input_ids"].shape)
print("Vision inputs:", list(inputs.keys()))
```

This comprehensive guide covers all aspects of vision templates in the Chat Template System. Use these features to create powerful multi-modal templates that can handle images, videos, and text seamlessly.
