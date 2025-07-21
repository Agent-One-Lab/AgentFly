#!/usr/bin/env python3
"""
Example demonstrating how to handle multiple chains with streaming
without mixing outputs from different chains.
"""

import asyncio
import json
from typing import List, Dict, Any
from agents.agents.agents.agent_base import BaseAgent
from agents.agents.agents.chain.streaming_observer import (
    ConsoleStreamObserver, 
    JSONStreamObserver,
    ChainSpecificStreamObserver,
    MultiChainStreamObserver,
    AsyncGeneratorStreamObserver
)
from agents.agents.tools.src.code.tools import code_interpreter


class StreamingAgent(BaseAgent):
    """Example agent with streaming support"""
    
    def parse(self, responses: List[str], tools: List[Any], **args) -> List[Dict[str, Any]]:
        """Parse tool calls from responses"""
        parsed_responses = []
        for response in responses:
            if "```python" in response:
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


async def example_1_colored_output():
    """Example 1: Use colored output to distinguish chains"""
    print("=== Example 1: Colored Output for Multiple Chains ===")
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Add colored console observer
    console_observer = ConsoleStreamObserver(show_timestamps=True)
    agent.streaming_manager.add_observer(console_observer)
    
    start_messages = [
        {
            "messages": [{"role": "user", "content": "Write a function to calculate factorial of 5."}],
            "info": {"task": "factorial"},
            "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)\nprint(factorial(5))"
        },
        {
            "messages": [{"role": "user", "content": "Write a function to calculate fibonacci of 10."}],
            "info": {"task": "fibonacci"},
            "answer": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))"
        }
    ]
    
    await agent.run_async(
        max_steps=3,
        start_messages=start_messages,
        num_chains=2,  # Run 2 chains per message = 4 total chains
        enable_streaming=True
    )


async def example_2_chain_specific_observers():
    """Example 2: Use chain-specific observers"""
    print("\n=== Example 2: Chain-Specific Observers ===")
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Create a multi-chain observer
    multi_observer = MultiChainStreamObserver()
    
    # Add chain-specific observers
    for i in range(4):  # We'll have 4 chains
        chain_id = f"chain_{i}"
        console_observer = ConsoleStreamObserver(show_timestamps=True)
        json_observer = JSONStreamObserver(file_path=f"chain_{i}_events.jsonl")
        
        # Create chain-specific observers
        chain_console = ChainSpecificStreamObserver(chain_id, console_observer)
        chain_json = ChainSpecificStreamObserver(chain_id, json_observer)
        
        multi_observer.add_chain_observer(chain_id, chain_console)
        multi_observer.add_chain_observer(chain_id, chain_json)
    
    agent.streaming_manager.add_observer(multi_observer)
    
    start_messages = [
        {
            "messages": [{"role": "user", "content": "Write a function to add two numbers."}],
            "info": {"task": "addition"},
            "answer": "def add(a, b): return a + b"
        },
        {
            "messages": [{"role": "user", "content": "Write a function to multiply two numbers."}],
            "info": {"task": "multiplication"},
            "answer": "def multiply(a, b): return a * b"
        }
    ]
    
    await agent.run_async(
        max_steps=3,
        start_messages=start_messages,
        num_chains=2,
        enable_streaming=True
    )


async def example_3_filter_by_chain():
    """Example 3: Filter events by specific chain"""
    print("\n=== Example 3: Filter by Specific Chain ===")
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Only observe events from a specific chain
    target_chain_id = "chain_0"  # We'll focus on the first chain
    filtered_observer = ConsoleStreamObserver(
        show_timestamps=True, 
        chain_filter=target_chain_id
    )
    
    agent.streaming_manager.add_observer(filtered_observer)
    
    start_messages = [
        {
            "messages": [{"role": "user", "content": "Write a function to check if a number is prime."}],
            "info": {"task": "prime_check"},
            "answer": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"
        }
    ]
    
    await agent.run_async(
        max_steps=3,
        start_messages=start_messages,
        num_chains=3,  # Run 3 chains but only observe one
        enable_streaming=True
    )


async def example_4_async_generator_per_chain():
    """Example 4: Use async generators for each chain"""
    print("\n=== Example 4: Async Generator per Chain ===")
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Create async generators for each chain
    chain_generators = {}
    for i in range(2):
        chain_id = f"chain_{i}"
        async_observer = AsyncGeneratorStreamObserver(chain_filter=chain_id)
        agent.streaming_manager.add_observer(async_observer)
        chain_generators[chain_id] = async_observer.events()
    
    start_messages = [
        {
            "messages": [{"role": "user", "content": "Write a function to reverse a string."}],
            "info": {"task": "string_reverse"},
            "answer": "def reverse_string(s): return s[::-1]"
        }
    ]
    
    # Start the agent run
    run_task = asyncio.create_task(
        agent.run_async(
            max_steps=3,
            start_messages=start_messages,
            num_chains=2,
            enable_streaming=True
        )
    )
    
    # Process events for each chain separately
    async def process_chain_events(chain_id: str, generator):
        print(f"\n--- Processing events for {chain_id} ---")
        async for event in generator:
            print(f"{chain_id}: {event.event_type.value} - {event.data.get('content', '')[:50]}...")
    
    # Process all chains concurrently
    tasks = [
        process_chain_events(chain_id, generator) 
        for chain_id, generator in chain_generators.items()
    ]
    
    await asyncio.gather(*tasks)
    await run_task


async def example_5_web_interface_simulation():
    """Example 5: Simulate web interface with separate streams"""
    print("\n=== Example 5: Web Interface Simulation ===")
    
    agent = StreamingAgent(
        model_name_or_path="microsoft/DialoGPT-medium",
        template="chatml",
        tools=[code_interpreter],
        backend="transformers",
        debug=True
    )
    
    # Simulate web interface with separate streams per chain
    web_streams = {}
    
    async def create_web_stream(chain_id: str):
        """Simulate a web stream for a specific chain"""
        print(f"🌐 Web stream created for {chain_id}")
        
        async def web_stream_handler(event):
            # Simulate sending to web client
            web_message = {
                "chain_id": event.chain_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "data": event.data
            }
            print(f"📡 Web client {chain_id}: {web_message['event_type']}")
        
        return web_stream_handler
    
    # Create web streams for each chain
    for i in range(2):
        chain_id = f"web_chain_{i}"
        handler = await create_web_stream(chain_id)
        web_streams[chain_id] = handler
    
    # Add observers that send to web streams
    for chain_id, handler in web_streams.items():
        chain_observer = ChainSpecificStreamObserver(chain_id, handler)
        agent.streaming_manager.add_observer(chain_observer)
    
    start_messages = [
        {
            "messages": [{"role": "user", "content": "Write a function to sort a list."}],
            "info": {"task": "sorting"},
            "answer": "def sort_list(lst): return sorted(lst)"
        }
    ]
    
    await agent.run_async(
        max_steps=3,
        start_messages=start_messages,
        num_chains=2,
        enable_streaming=True
    )


if __name__ == "__main__":
    print("Multi-Chain Streaming Examples")
    print("=" * 50)
    
    # Run all examples
    asyncio.run(example_1_colored_output())
    asyncio.run(example_2_chain_specific_observers())
    asyncio.run(example_3_filter_by_chain())
    asyncio.run(example_4_async_generator_per_chain())
    asyncio.run(example_5_web_interface_simulation())
    
    print("\n" + "=" * 50)
    print("All examples completed!") 