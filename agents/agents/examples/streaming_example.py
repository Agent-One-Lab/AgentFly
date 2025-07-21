#!/usr/bin/env python3
"""
Example script demonstrating streaming functionality for LLM agent reinforcement learning.

This example shows how to:
1. Set up streaming observers
2. Run agent chains with streaming
3. Handle real-time events
"""

import asyncio
import json
from typing import List, Dict, Any
from agents.agents.agents.agent_base import BaseAgent
from agents.agents.agents.chain.streaming_observer import (
    StreamingManager, 
    ConsoleStreamObserver, 
    JSONStreamObserver,
    AsyncGeneratorStreamObserver,
    StreamEvent
)
from agents.agents.tools.src.code.tools import code_interpreter


class StreamingAgent(BaseAgent):
    """Example agent with streaming support"""
    
    def parse(self, responses: List[str], tools: List[Any], **args) -> List[Dict[str, Any]]:
        """Parse tool calls from responses"""
        # Simple parsing - in practice you'd use a more sophisticated parser
        parsed_responses = []
        for response in responses:
            # Check if response contains tool call
            if "```python" in response:
                # Extract code between ```python and ```
                start = response.find("```python") + 9
                end = response.find("```", start)
                if end != -1:
                    code = response[start:end].strip()
                    parsed_responses.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "code_interpreter",
                                "arguments": json.dumps({"code": code})
                            }
                        }]
                    })
                else:
                    parsed_responses.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}]
                    })
            else:
                parsed_responses.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                })
        
        return parsed_responses


async def main():
    """Main example function"""
    
    # Initialize the agent
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",  # Replace with your model
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",  # or "async_vllm" for streaming support
        debug=True
    )
    
    # Set up streaming observers
    console_observer = ConsoleStreamObserver(show_timestamps=True)
    json_observer = JSONStreamObserver(file_path="streaming_events.jsonl")
    
    # Add observers to the streaming manager
    agent.streaming_manager.add_observer(console_observer)
    agent.streaming_manager.add_observer(json_observer)
    
    # Example messages
    start_messages = [
        {
            "messages": [
                {"role": "user", "content": "Write a Python function to calculate the factorial of a number and test it with input 5."}
            ],
            "info": {"task": "factorial_calculation"},
            "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)\nprint(factorial(5))"
        }
    ]
    
    print("Starting streaming agent run...")
    print("=" * 50)
    
    # Run with streaming
    await agent.run_async(
        max_steps=5,
        start_messages=start_messages,
        num_chains=1,
        generation_config={"temperature": 0.7, "max_new_tokens": 512},
        enable_streaming=True
    )
    
    print("=" * 50)
    print("Streaming run completed!")
    
    # Print final results
    print("\nFinal trajectories:")
    for i, trajectory in enumerate(agent.trajectories):
        print(f"\nChain {i}:")
        for msg in trajectory["messages"]:
            if msg["role"] == "assistant":
                print(f"Assistant: {msg['content'][0]['text'][:100]}...")
            elif msg["role"] == "tool":
                print(f"Tool: {msg['content'][0]['text'][:100]}...")


async def streaming_with_custom_callback():
    """Example with custom streaming callback"""
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Custom streaming callback
    async def custom_streaming_callback(chunk: str):
        """Custom callback to handle streaming chunks"""
        print(f"🔄 Streaming chunk: {chunk}", end="", flush=True)
    
    # Add console observer
    console_observer = ConsoleStreamObserver(show_timestamps=True)
    agent.streaming_manager.add_observer(console_observer)
    
    start_messages = [
        {
            "messages": [
                {"role": "user", "content": "Write a simple Python function to add two numbers."}
            ],
            "info": {"task": "simple_addition"},
            "answer": "def add(a, b): return a + b"
        }
    ]
    
    print("Starting streaming with custom callback...")
    print("=" * 50)
    
    await agent.run_async(
        max_steps=3,
        start_messages=start_messages,
        num_chains=1,
        generation_config={"temperature": 0.7, "max_new_tokens": 256},
        enable_streaming=True,
        streaming_callback=custom_streaming_callback
    )


async def async_generator_example():
    """Example using AsyncGeneratorStreamObserver"""
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Create async generator observer
    async_observer = AsyncGeneratorStreamObserver()
    agent.streaming_manager.add_observer(async_observer)
    
    start_messages = [
        {
            "messages": [
                {"role": "user", "content": "Write a Python function to check if a number is prime."}
            ],
            "info": {"task": "prime_check"},
            "answer": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"
        }
    ]
    
    print("Starting async generator example...")
    print("=" * 50)
    
    # Start the agent run in background
    run_task = asyncio.create_task(
        agent.run_async(
            max_steps=3,
            start_messages=start_messages,
            num_chains=1,
            generation_config={"temperature": 0.7, "max_new_tokens": 256},
            enable_streaming=True
        )
    )
    
    # Process events as they arrive
    async for event in async_observer.events():
        print(f"📡 Event: {event.event_type.value} - Chain: {event.chain_id}")
        if event.event_type.value == "llm_generation_chunk":
            print(f"   Content: {event.data.get('content', '')}")
        elif event.event_type.value == "tool_observation":
            print(f"   Tool: {event.data.get('tool_name', '')} - {event.data.get('observation', '')[:50]}...")
    
    # Wait for the run to complete
    await run_task


if __name__ == "__main__":
    print("Streaming Agent Example")
    print("=" * 50)
    
    # Run different examples
    asyncio.run(main())
    print("\n" + "=" * 50)
    
    asyncio.run(streaming_with_custom_callback())
    print("\n" + "=" * 50)
    
    asyncio.run(async_generator_example()) 