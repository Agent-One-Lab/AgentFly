## Installation

```bash
pip install -e .
pip install -e '.[verl]' --no-build-isolation
pip install git+https://github.com/huggingface/diffusers.git
```

## Basic Usage

### Creating an Agent

The ImageEditingAgent supports multiple backend configurations for different use cases:

#### Using Client-Based Image Editing (Remote Service)

Instead of loading image editing models locally, you can use a remote image editing service:

```python
from agentfly.agents import ImageEditingAgent

# Create an agent with client-based image editing
agent = ImageEditingAgent(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    client_nodes=[("localhost", 8000), ("localhost", 8001)],  # Remote image editing servers
    max_requests_per_minute=50,  # Rate limiting
    timeout=600  # Request timeout
)
```

This approach:
- **Reduces memory usage** - No need to load image editing models locally
- **Enables scaling** - Use multiple remote servers for load balancing
- **Improves performance** - Dedicated GPU resources for image editing
- **Supports multiple models** - Different servers can run different image editing models

**Setting up the remote image editing service:**

```bash
# Start the image editing API server
python -m agentfly.agents.specialized.image_agent.utils.diffuser_apis \
    --model-type qwen \
    --model-path Qwen/Qwen-Image-Edit \
    --gpu-ids 0 1 \
    --use-fast \
    --host 0.0.0.0 \
    --port 8000
```

The service provides:
- Multi-GPU support with automatic load balancing
- OpenAI-compatible API endpoints
- Rate limiting and timeout handling
- Health monitoring and statistics

#### Using Local Image Editing (Traditional)

The traditional approach loads models locally:

#### Using AsyncVLLM Backend (Local)

```python
from agentfly.agents import ImageEditingAgent
from agentfly.agents.llm_backends import AsyncVLLMConfig

# Create an agent with asynchronous vllm backend
agent = ImageEditingAgent(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    template="qwen2.5-vl-system-tool",
    backend="async_vllm",
    backend_config=AsyncVLLMConfig(
        pipeline_parallel_size=4,
        gpu_memory_utilization=0.5
    ),
    streaming="console"
)
```

#### Using Client Backend (Local Server)

#### Deploying the model
```bash
# Take qwen2.5 as an example
# Note you must set the template since qwen2.5 does not inherently support tools in official chat template
python -m agentfly.utils.deploy --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct --template qwen2.5-vl-system-tool --tp 2 --dp 2
```

```python
from agentfly.agents import ImageEditingAgent
from agentfly.agents.llm_backends import ClientConfig

# Create an agent with client backend for local server
agent = ImageEditingAgent(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    backend="client",
    backend_config=ClientConfig(
        base_url="http://0.0.0.0:8000/v1",
    ),
    streaming="console"
)
```

#### Using OpenAI Backend (Cloud)

```python
from agentfly.agents import ImageEditingAgent
from agentfly.agents.llm_backends import ClientConfig

# Create an agent with OpenAI backend
agent = ImageEditingAgent(
    model_name_or_path="gpt-5-mini",
    backend="client",
    backend_config=ClientConfig(
        base_url="https://api.openai.com/v1",
        api_key="your-openai-api-key",
    ),
    streaming="console"
)
```

### Available Tools

The ImageEditingAgent contains four tools:

1. **`detect_objects_tool`**: Detects objects in images using GroundingDINO
2. **`inpaint_image_tool`**: Fills in masked areas using AI generation
3. **`auto_inpaint_image_tool`**: Combines detection and inpainting in one operation
4. **`qwen_edit_image_tool`**: Edits images using Qwen-Image-Edit model with natural language instructions

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
await agent.run(
    messages=messages_list,
    max_turns=4,
    num_chains=1,
    enable_streaming=True
)

# Print the conversation
agent.print_messages(index=0)
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
    backend_config=AsyncVLLMConfig(
        pipeline_parallel_size=4,
        gpu_memory_utilization=0.5
    ),
    streaming="console"  # Enables console streaming
)

# Run with streaming enabled
await agent.run(
    messages=messages_list,
    max_turns=4,
    num_chains=1,
    enable_streaming=True  # Must be True for streaming
)
```

**Note**: When using streaming, `num_chains` must be 1 due to console streaming limitations.

## Backend Configuration Options

### AsyncVLLM Backend

The AsyncVLLM backend is optimized for local inference with GPU acceleration:

```python
from agentfly.agents.llm_backends import AsyncVLLMConfig

backend_config = AsyncVLLMConfig(
    pipeline_parallel_size=4,        # Number of pipeline parallel stages
    gpu_memory_utilization=0.5,      # GPU memory utilization (0.0-1.0)
    max_model_len=8192,              # Maximum sequence length
    trust_remote_code=True,          # Trust remote code for custom models
    dtype="auto"                     # Data type for model weights
)
```

### Client Backend

The Client backend connects to external API servers:

```python
from agentfly.agents.llm_backends import ClientConfig

# For local server
backend_config = ClientConfig(
    base_url="http://0.0.0.0:8000/v1",
    timeout=60.0
)

# For OpenAI
backend_config = ClientConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    timeout=60.0
)
```

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


## Using the testing example

Test the tools
```bash
pytest agentfly/tests/unit/agents/test_image_agent/test_image_tools.py -s
```
Test agents
```bash
pytest agentfly/tests/unit/agents/test_image_agent/test_image_agent.py -s
```

Test with different backends
```
# Test client backend
pytest agentfly/tests/unit/agents/test_image_agent/test_image_agent_backends.py::test_image_agent_client -s

# Test async vllm backend
pytest agentfly/tests/unit/agents/test_image_agent/test_image_agent_backends.py::test_image_agent_async_vllm -s
```

Run the test_run file
```
python -m agentfly.tests.unit.agents.test_image_agent.test_run
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

# Customize Qwen image editing parameters
edit_result = await agent.qwen_edit_image_tool(
    image_id="your_image_id",
    prompt="Change the background to a sunset",
    negative_prompt="blurry, low quality",
    true_cfg_scale=6.0,        # Higher CFG scale for stronger adherence
    num_inference_steps=75,    # More steps for better quality
    seed=123                   # Fixed seed for reproducibility
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

## Complete Example

Here's a complete example showing how to use the ImageEditingAgent with different backends:

```python
import asyncio
from agentfly.agents import ImageEditingAgent
from agentfly.agents.llm_backends import AsyncVLLMConfig, ClientConfig

async def main():
    # Choose your backend configuration
    # Option 1: Local AsyncVLLM
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend="async_vllm",
        backend_config=AsyncVLLMConfig(
            pipeline_parallel_size=4,
            gpu_memory_utilization=0.5
        ),
        streaming="console"
    )

    # Option 2: Local Server
    # agent = ImageEditingAgent(
    #     model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    #     backend="client",
    #     backend_config=ClientConfig(
    #         base_url="http://0.0.0.0:8000/v1",
    #     ),
    #     streaming="console"
    # )

    # Option 3: OpenAI
    # agent = ImageEditingAgent(
    #     model_name_or_path="gpt-4o-mini",
    #     backend="client",
    #     backend_config=ClientConfig(
    #         base_url="https://api.openai.com/v1",
    #         api_key="your-openai-api-key",
    #     ),
    #     streaming="console"
    # )

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

    await agent.run(
        messages=messages_list,
        max_turns=4,
        num_chains=1,
        enable_streaming=True
    )

    agent.print_messages(index=0)

if __name__ == "__main__":
    asyncio.run(main())
```
