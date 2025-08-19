## Installation

```bash
pip install -e .
pip install -e '.[verl]' --no-build-isolation
pip install git+https://github.com/huggingface/diffusers.git
```

## Basic Usage


### Creating an Agent

```python
from agents.agents.specialized.image_agent.image_agent import ImageEditingAgent

# Create an agent with asynchronous vllm backends
agent = ImageEditingAgent(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    template="qwen2.5-vl-system-tool",
    backend="async_vllm",
    streaming="console"
)
```

### Available Tools

The ImageEditingAgent contain three tools (currently):

1. **`detect_objects_tool`**: Detects objects in images using GroundingDINO
2. **`inpaint_image_tool`**: Fills in masked areas using AI generation
3. **`auto_inpaint_image_tool`**: Combines detection and inpainting in one operation

### Running the Agent

```python
# Prepare your messages with images
messages_list = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                    {
                        "type": "text",
                        "text": "Find what animal is in the image, then inpaint it with a cat."
                    }
                ]
            }
        ]
    }
]

# Run the agent
await agent.run_async(
    start_messages=messages_list,
    max_steps=4,
    num_chains=1,
    enable_streaming=False
)
```

## Streaming Support

The ImageEditingAgent supports real-time streaming of responses and tool executions. This is useful for monitoring the agent's progress and debugging.

### Enabling Streaming

```python
# Create agent with streaming enabled
agent = ImageEditingAgent(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    template="qwen2.5-vl-system-tool",
    backend="async_vllm",
    streaming="console"  # Enables console streaming
)

# Run with streaming enabled
await agent.run_async(
    start_messages=messages_list,
    max_steps=4,
    num_chains=1,
    enable_streaming=True  # Must be True for streaming
)
```

**Note**: When using streaming, `num_chains` must be 1 due to console streaming limitations.



## Getting Trajectories and Messages

After running the agent, you can access the complete conversation history and trajectories.

### Accessing Messages

```python
# Get all messages from all chains
messages = agent.get_messages()

# Print messages for a specific chain
agent.print_messages(index=0)

# Access specific message content
for message in messages[0]["messages"]:
    role = message["role"]
    content = message["content"]
    print(f"{role}: {content}")
```

### Understanding Message Structure

Messages follow this structure:
- **User messages**: Contain image and text content
- **Assistant messages**: Contain generated responses and tool calls
- **Tool messages**: Contain tool execution results and observations

### Extracting Final Responses

```python
# Get the final response from a trajectory
final_response = agent.extract_final_response(messages[0]["messages"])
print(f"Final response: {final_response}")
```

## Tokenized Inputs

For training and analysis purposes, you can tokenize the conversation trajectories.

### Basic Tokenization

```python
# Tokenize trajectories with default tokenizer
inputs, other_info = agent.tokenize_trajectories()

# Access tokenized data
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
position_ids = inputs["position_ids"]
```

### Advanced Tokenization

```python
# Use custom tokenizer
from transformers import AutoTokenizer
custom_tokenizer = AutoTokenizer.from_pretrained("your-model")

# Tokenize with reward mask
inputs, other_info = agent.tokenize_trajectories(
    tokenizer=custom_tokenizer,
    return_reward_mask=True
)

# Access reward mask if available
if "reward_mask" in inputs:
    reward_mask = inputs["reward_mask"]
```

### Tokenization Output

The `tokenize_trajectories` method returns:
- **`inputs`**: Dictionary containing tokenized data
  - `input_ids`: Token IDs
  - `attention_mask`: Attention mask
  - `position_ids`: Position IDs
  - `reward_mask`: Reward mask (if enabled)
- **`other_info`**: List of metadata for each trajectory

## Using the testing example

Go to agents directory first:
```bash
cd agents
```
Test the tools
```bash
python -m pytest tests/unit/agents/test_image_agent/test_image_tools.py -s
```
Test agents
```bash
python -m pytest tests/unit/agents/test_image_agent/test_image_agent.py -s
```

## Advanced Features

### Custom Tool Parameters

```python
# Customize detection parameters
detection_result = await agent.detect_objects_tool(
    image_id="your_image_id",
    text_prompt="a dog",
    box_threshold=0.5,      # Higher confidence threshold
    text_threshold=0.3,     # Higher text matching threshold
    auto_mask_dilate=2,     # Dilate mask by 2 pixels
    auto_mask_feather=3     # Feather mask by 3 pixels
)

# Customize inpainting parameters
inpaint_result = await agent.inpaint_image_tool(
    image_id="your_image_id",
    mask_id="your_mask_id",
    prompt="a beautiful cat",
    guidance_scale=7.5,        # Higher guidance for better quality
    num_inference_steps=50,    # More steps for better quality
    strength=0.8,              # Lower strength for subtle changes
    seed=42                    # Fixed seed for reproducibility
)
```

### Image Management

```python
# Store an image and get its ID
image_id = agent._store_image(your_pil_image)

# Retrieve an image by ID
image = agent._get_image(image_id)

# Save an image to disk
agent.save_image(image_id, "output_image.jpg")
```
