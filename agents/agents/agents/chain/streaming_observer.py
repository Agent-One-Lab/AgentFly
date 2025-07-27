import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator, Set
from enum import Enum
import json
import time
from termcolor import colored


class StreamEventType(Enum):
    """Types of streaming events"""
    LLM_GENERATION_START = "llm_generation_start"
    LLM_GENERATION_CHUNK = "llm_generation_chunk"
    LLM_GENERATION_END = "llm_generation_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_OBSERVATION = "tool_observation"
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A streaming event with metadata"""
    event_type: StreamEventType
    chain_id: str
    timestamp: float
    data: Dict[str, Any]
    step: Optional[int] = None
    depth: Optional[int] = None

    def __post_init__(self):
        # Add a unique identifier for this event
        if not hasattr(self, 'event_id'):
            self.event_id = f"{self.chain_id}_{self.timestamp}_{self.event_type.value}"


class StreamObserver(ABC):
    """Abstract base class for stream observers"""
    
    @abstractmethod
    async def on_event(self, event: StreamEvent) -> None:
        """Handle a streaming event"""
        pass
    
    async def on_error(self, error: Exception, chain_id: str) -> None:
        """Handle an error event"""
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            chain_id=chain_id,
            timestamp=time.time(),
            data={"error": str(error), "error_type": type(error).__name__}
        )
        await self.on_event(event)


class StreamingManager:
    """Manages streaming observers and event distribution"""
    
    def __init__(self):
        self.observers: List[StreamObserver] = []
        self.enabled = False
        self.active_chains: Set[str] = set()
        self.chain_events: Dict[str, List[StreamEvent]] = {}
    
    def add_observer(self, observer: StreamObserver) -> None:
        """Add a streaming observer"""
        self.observers.append(observer)
        self.enabled = True
    
    def remove_observer(self, observer: StreamObserver) -> None:
        """Remove a streaming observer"""
        if observer in self.observers:
            self.observers.remove(observer)
        if not self.observers:
            self.enabled = False
    
    async def emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all observers"""
        if not self.enabled:
            return
        
        # Track active chains
        if event.event_type == StreamEventType.CHAIN_START:
            self.active_chains.add(event.chain_id)
            self.chain_events[event.chain_id] = []
        elif event.event_type == StreamEventType.CHAIN_END:
            self.active_chains.discard(event.chain_id)
        
        # Store event for this chain
        if event.chain_id in self.chain_events:
            self.chain_events[event.chain_id].append(event)
        
        tasks = [observer.on_event(event) for observer in self.observers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def emit_error(self, error: Exception, chain_id: str) -> None:
        """Emit an error event to all observers"""
        if not self.enabled:
            return
        
        tasks = [observer.on_error(error, chain_id) for observer in self.observers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_chain_events(self, chain_id: str) -> List[StreamEvent]:
        """Get all events for a specific chain"""
        return self.chain_events.get(chain_id, [])
    
    def get_active_chains(self) -> Set[str]:
        """Get all currently active chain IDs"""
        return self.active_chains.copy()


class ConsoleStreamObserver(StreamObserver):
    """Simple console-based stream observer for debugging"""
    
    def __init__(self, show_timestamps: bool = True, chain_filter: Optional[str] = None):
        self.show_timestamps = show_timestamps
        self.chain_filter = chain_filter  # Only show events for this chain_id
        self.chain_colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]
        self.chain_id_data = {}

    
    async def on_event(self, event: StreamEvent) -> None:
        # Filter by chain if specified
        if self.chain_filter and event.chain_id != self.chain_filter:
            return

        turn_info = f" (turn {event.step})" if event.step is not None else ""
        
        # Use different colors for different chains
        chain_index = hash(event.chain_id) % len(self.chain_colors)
        chain_color = self.chain_colors[chain_index]
        if event.chain_id not in self.chain_id_data:
            self.chain_id_data[event.chain_id] = {
                "color": chain_color,
                "timestamp": event.timestamp,
                "step": event.step,
                "depth": event.depth,
                "event_type": event.event_type.value,
                "content_buffer": ""
            }

        if event.event_type == StreamEventType.LLM_GENERATION_START:
            print(f"{event.timestamp - self.chain_id_data[event.chain_id]['timestamp']:.2f}s {turn_info}".center(80, "="), flush=True)
        elif event.event_type == StreamEventType.LLM_GENERATION_CHUNK:
            content = event.data.get("content", "")
            if content:
                # clear the terminal
                if self.chain_id_data[event.chain_id]["event_type"] == StreamEventType.LLM_GENERATION_CHUNK:
                    print(colored(f"""{content[len(self.chain_id_data[event.chain_id]["content_buffer"]):]}""", color=chain_color), end="", flush=True)
                    self.chain_id_data[event.chain_id]["content_buffer"] = content
                else:
                    self.chain_id_data[event.chain_id]["content_buffer"] = content
                    print(colored(f"{content}", color=chain_color), end="", flush=True)
                    self.chain_id_data[event.chain_id]["event_type"] = StreamEventType.LLM_GENERATION_CHUNK
        elif event.event_type == StreamEventType.LLM_GENERATION_END:
            print(colored(f"\n{event.timestamp - self.chain_id_data[event.chain_id]['timestamp']:.2f}s", color=chain_color), flush=True)
            self.chain_id_data[event.chain_id]["event_type"] = StreamEventType.LLM_GENERATION_END
        elif event.event_type == StreamEventType.TOOL_OBSERVATION:
            observation = event.data.get("observation", "")
            tool_name = event.data.get("tool_name", "")
            print(colored(f"Tool: [{tool_name}] {observation[:1024]}{'...' if len(observation) > 200 else ''}", color=chain_color))
            print(f"".center(80, "="), flush=True)
            self.chain_id_data[event.chain_id]["event_type"] = StreamEventType.TOOL_OBSERVATION
        elif event.event_type == StreamEventType.ERROR:
            error_msg = event.data.get("error", "")
            print(colored(f"  ❌ Error: {error_msg}", color=chain_color))
            self.chain_id_data[event.chain_id]["event_type"] = StreamEventType.ERROR


class AsyncGeneratorStreamObserver(StreamObserver):
    """Stream observer that yields events as an async generator"""
    
    def __init__(self, chain_filter: Optional[str] = None):
        self.queue = asyncio.Queue()
        self.chain_filter = chain_filter
    
    async def on_event(self, event: StreamEvent) -> None:
        # Filter by chain if specified
        if self.chain_filter and event.chain_id != self.chain_filter:
            return
            
        await self.queue.put(event)
    
    async def events(self) -> AsyncGenerator[StreamEvent, None]:
        """Yield events as they arrive"""
        while True:
            try:
                event = await self.queue.get()
                if event.event_type == StreamEventType.CHAIN_END:
                    # Send the final event and stop
                    yield event
                    break
                yield event
            except asyncio.CancelledError:
                break


class ChainSpecificStreamObserver(StreamObserver):
    """Stream observer that only handles events for a specific chain"""
    
    def __init__(self, target_chain_id: str, base_observer: StreamObserver):
        self.target_chain_id = target_chain_id
        self.base_observer = base_observer
    
    async def on_event(self, event: StreamEvent) -> None:
        if event.chain_id == self.target_chain_id:
            await self.base_observer.on_event(event)
    
    async def on_error(self, error: Exception, chain_id: str) -> None:
        if chain_id == self.target_chain_id:
            await self.base_observer.on_error(error, chain_id)


class MultiChainStreamObserver(StreamObserver):
    """Stream observer that organizes events by chain"""
    
    def __init__(self):
        self.chain_observers: Dict[str, List[StreamObserver]] = {}
        self.global_observers: List[StreamObserver] = []
    
    def add_chain_observer(self, chain_id: str, observer: StreamObserver) -> None:
        """Add an observer for a specific chain"""
        if chain_id not in self.chain_observers:
            self.chain_observers[chain_id] = []
        self.chain_observers[chain_id].append(observer)
    
    def add_global_observer(self, observer: StreamObserver) -> None:
        """Add an observer for all chains"""
        self.global_observers.append(observer)
    
    async def on_event(self, event: StreamEvent) -> None:
        # Send to chain-specific observers
        if event.chain_id in self.chain_observers:
            tasks = [obs.on_event(event) for obs in self.chain_observers[event.chain_id]]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Send to global observers
        tasks = [obs.on_event(event) for obs in self.global_observers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def on_error(self, error: Exception, chain_id: str) -> None:
        # Send to chain-specific observers
        if chain_id in self.chain_observers:
            tasks = [obs.on_error(error, chain_id) for obs in self.chain_observers[chain_id]]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Send to global observers
        tasks = [obs.on_error(error, chain_id) for obs in self.global_observers]
        await asyncio.gather(*tasks, return_exceptions=True) 