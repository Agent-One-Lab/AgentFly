# Streaming Functionality for LLM Agent Reinforcement Learning

This document describes the streaming functionality added to the LLM agent reinforcement learning framework, which allows real-time monitoring of agent responses and tool observations.

## Overview

The streaming functionality provides:

1. **Real-time LLM response streaming** - See tokens as they are generated
2. **Tool observation streaming** - Monitor tool calls and their results in real-time
3. **Event-based architecture** - Flexible observer pattern for different use cases
4. **Multiple output formats** - Console, JSON, WebSocket, and custom callbacks
5. **Async support** - Non-blocking streaming with proper async/await patterns
6. **Multi-chain support** - Handle multiple chains without mixing outputs

## Architecture

### Core Components

1. **StreamEvent** - Represents a streaming event with metadata
2. **StreamObserver** - Abstract base class for event observers
3. **StreamingManager** - Manages observers and event distribution
4. **ChainGeneration** - Enhanced with streaming support

### Event Types

- `LLM_GENERATION_START` - LLM generation begins
- `LLM_GENERATION_CHUNK` - Individual token/chunk generated
- `LLM_GENERATION_END` - LLM generation completes
- `TOOL_CALL_START` - Tool call begins
- `TOOL_CALL_END` - Tool call completes
- `TOOL_OBSERVATION` - Tool observation received
- `CHAIN_START` - Agent chain begins
- `CHAIN_END` - Agent chain completes
- `ERROR` - Error occurred

### Multi-Chain Problem and Solutions

**Problem**: When running multiple chains with streaming, outputs from different chains get mixed together, making it impossible to follow which events belong to which chain.

**Solutions**:

1. **Automatic Color Coding** - Each chain gets a different color in console output
2. **Chain Filtering** - Filter events to only show specific chains
3. **Chain-Specific Observers** - Create separate observers for each chain
4. **Separate Async Generators** - Process each chain's events independently
5. **Multi-Chain Observer** - Organize observers by chain ID

## Usage Examples

### Basic Console Streaming

```python
from agents.agents.agents.agent_base import BaseAgent
from agents.agents.agents.chain.streaming_observer import ConsoleStreamObserver

# Initialize agent
agent = YourAgent(model_name="your-model", tools=[...])

# Add console observer
console_observer = ConsoleStreamObserver(show_timestamps=True)
agent.streaming_manager.add_observer(console_observer)

# Run with streaming
await agent.run_async(
    max_steps=5,
    start_messages=your_messages,
    num_chains=1,
    enable_streaming=True
)
```

### Multi-Chain Streaming Solutions

When running multiple chains, the streaming output can become mixed. Here are several solutions:

#### 1. Colored Output (Automatic)

```python
# Each chain gets a different color automatically
console_observer = ConsoleStreamObserver(show_timestamps=True)
agent.streaming_manager.add_observer(console_observer)

await agent.run_async(
    max_steps=5,
    start_messages=your_messages,
    num_chains=3,  # Multiple chains
    enable_streaming=True
)
```

#### 2. Chain-Specific Observers

```python
from agents.agents.agents.chain.streaming_observer import MultiChainStreamObserver, ChainSpecificStreamObserver

# Create multi-chain observer
multi_observer = MultiChainStreamObserver()

# Add observers for specific chains
for chain_id in ["chain_0", "chain_1", "chain_2"]:
    console_observer = ConsoleStreamObserver(show_timestamps=True)
    chain_observer = ChainSpecificStreamObserver(chain_id, console_observer)
    multi_observer.add_chain_observer(chain_id, chain_observer)

agent.streaming_manager.add_observer(multi_observer)
```

#### 3. Filter by Chain

```python
# Only observe events from a specific chain
filtered_observer = ConsoleStreamObserver(
    show_timestamps=True, 
    chain_filter="chain_0"  # Only show chain_0 events
)
agent.streaming_manager.add_observer(filtered_observer)
```

#### 4. Separate Async Generators per Chain

```python
from agents.agents.agents.chain.streaming_observer import AsyncGeneratorStreamObserver

# Create separate generators for each chain
chain_generators = {}
for i in range(3):
    chain_id = f"chain_{i}"
    async_observer = AsyncGeneratorStreamObserver(chain_filter=chain_id)
    agent.streaming_manager.add_observer(async_observer)
    chain_generators[chain_id] = async_observer.events()

# Process each chain separately
async def process_chain(chain_id, generator):
    async for event in generator:
        print(f"{chain_id}: {event.event_type.value}")

# Run all chains concurrently
tasks = [process_chain(chain_id, generator) for chain_id, generator in chain_generators.items()]
await asyncio.gather(*tasks)
```

### JSON Logging

```python
from agents.agents.agents.chain.streaming_observer import JSONStreamObserver

# Add JSON observer
json_observer = JSONStreamObserver(file_path="events.jsonl")
agent.streaming_manager.add_observer(json_observer)
```

### Custom Streaming Callback

```python
async def custom_callback(chunk: str):
    print(f"🔄 {chunk}", end="", flush=True)

await agent.run_async(
    max_steps=5,
    start_messages=your_messages,
    num_chains=1,
    enable_streaming=True,
    streaming_callback=custom_callback
)
```

### WebSocket Streaming

```python
from agents.agents.agents.chain.websocket_streaming import WebSocketStreamingServer

# Start WebSocket server
server = WebSocketStreamingServer(host="localhost", port=8765)
await server.start()

# Add WebSocket observer
agent.streaming_manager.add_observer(server.get_observer())

# Run agent
await agent.run_async(..., enable_streaming=True)
```

### Async Generator Events

```python
from agents.agents.agents.chain.streaming_observer import AsyncGeneratorStreamObserver

# Create async generator observer
async_observer = AsyncGeneratorStreamObserver()
agent.streaming_manager.add_observer(async_observer)

# Start agent run
run_task = asyncio.create_task(agent.run_async(..., enable_streaming=True))

# Process events as they arrive
async for event in async_observer.events():
    print(f"Event: {event.event_type.value}")
    if event.event_type.value == "llm_generation_chunk":
        print(f"Content: {event.data.get('content', '')}")

await run_task
```

## Backend Support

### Transformers Backend

The Transformers backend supports streaming through token-by-token generation:

```python
agent = YourAgent(
    model_name="your-model",
    backend="transformers",
    # ... other args
)
```

### Async vLLM Backend

The Async vLLM backend provides efficient streaming:

```python
agent = YourAgent(
    model_name="your-model", 
    backend="async_vllm",
    # ... other args
)
```

## Event Structure

Each streaming event contains:

```python
@dataclass
class StreamEvent:
    event_type: StreamEventType
    chain_id: str
    timestamp: float
    data: Dict[str, Any]
    step: Optional[int] = None
    depth: Optional[int] = None
```

### Example Events

**LLM Generation Chunk:**
```json
{
    "event_type": "llm_generation_chunk",
    "chain_id": "uuid-123",
    "timestamp": 1234567890.123,
    "data": {"content": "def factorial"},
    "step": 1,
    "depth": 1
}
```

**Tool Observation:**
```json
{
    "event_type": "tool_observation",
    "chain_id": "uuid-123", 
    "timestamp": 1234567890.456,
    "data": {
        "tool_name": "code_interpreter",
        "observation": "120",
        "status": "success"
    },
    "step": 1,
    "depth": 1
}
```

## Performance Considerations

1. **Memory Usage** - Streaming events are lightweight but can accumulate
2. **Network Overhead** - WebSocket streaming adds minimal overhead
3. **Backend Compatibility** - Not all backends support streaming equally
4. **Observer Performance** - Heavy observers can slow down the main loop

## Best Practices

1. **Use appropriate observers** - Console for debugging, JSON for logging, WebSocket for web apps
2. **Handle errors gracefully** - Implement error handling in custom observers
3. **Clean up resources** - Properly close WebSocket connections and file handles
4. **Monitor performance** - Watch for memory leaks in long-running streams
5. **Test thoroughly** - Streaming adds complexity, test edge cases

## Integration with Existing Code

The streaming functionality is designed to be non-intrusive:

- Existing code continues to work without changes
- Streaming is opt-in via the `run_async_streaming` method
- Observers can be added/removed at runtime
- No performance impact when streaming is disabled

## Troubleshooting

### Common Issues

1. **No streaming output** - Check if backend supports streaming
2. **WebSocket connection issues** - Verify port availability and firewall settings
3. **Memory leaks** - Ensure observers are properly cleaned up
4. **Performance issues** - Consider using fewer observers or lighter implementations

### Debug Mode

Enable debug logging to troubleshoot streaming issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Filtering** - Event filtering based on type, chain_id, etc.
2. **Batching** - Batch multiple events for efficiency
3. **Compression** - Compress WebSocket messages for large-scale deployments
4. **Authentication** - Add authentication to WebSocket connections
5. **Metrics** - Built-in streaming metrics and monitoring 