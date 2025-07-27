"""
WebSocket-based streaming interface for real-time agent interactions.
This module provides a WebSocket server that can stream agent events to web clients.
"""

import asyncio
import json
import websockets
from typing import Dict, Set, Optional, Callable
from .streaming_observer import StreamObserver, StreamEvent, StreamEventType
import logging

logger = logging.getLogger(__name__)


class WebSocketStreamObserver(StreamObserver):
    """Stream observer that broadcasts events to WebSocket clients"""
    
    def __init__(self):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.lock = asyncio.Lock()
    
    async def on_event(self, event: StreamEvent) -> None:
        """Broadcast event to all connected WebSocket clients"""
        if not self.clients:
            return
        
        # Convert event to JSON
        event_data = {
            "event_type": event.event_type.value,
            "chain_id": event.chain_id,
            "timestamp": event.timestamp,
            "step": event.step,
            "depth": event.depth,
            "data": event.data
        }
        
        message = json.dumps(event_data)
        
        # Broadcast to all clients
        disconnected_clients = set()
        async with self.lock:
            for client in self.clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected_clients
    
    async def add_client(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Add a new WebSocket client"""
        async with self.lock:
            self.clients.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.clients)}")
    
    async def remove_client(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """Remove a WebSocket client"""
        async with self.lock:
            self.clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.clients)}")


class WebSocketStreamingServer:
    """WebSocket server for streaming agent events"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.observer = WebSocketStreamObserver()
        self.server = None
    
    async def handle_client(self, websocket, path):
        """Handle individual WebSocket client connection"""
        await self.observer.add_client(websocket)
        try:
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle client messages if needed
                    logger.info(f"Received message from client: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {message}")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.observer.remove_client(websocket)
    
    async def start(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        return self.server
    
    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
    
    def get_observer(self) -> WebSocketStreamObserver:
        """Get the WebSocket stream observer"""
        return self.observer


class WebSocketStreamingClient:
    """WebSocket client for receiving streaming events"""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Connect to the WebSocket server"""
        self.websocket = await websockets.connect(self.uri)
        logger.info(f"Connected to WebSocket server at {self.uri}")
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from WebSocket server")
    
    async def receive_events(self, event_handler: Optional[Callable] = None):
        """Receive and handle streaming events"""
        if not self.websocket:
            await self.connect()
        
        try:
            async for message in self.websocket:
                try:
                    event_data = json.loads(message)
                    if event_handler:
                        await event_handler(event_data)
                    else:
                        # Default event handling
                        event_type = event_data.get("event_type")
                        chain_id = event_data.get("chain_id")
                        data = event_data.get("data", {})
                        
                        if event_type == "llm_generation_chunk":
                            content = data.get("content", "")
                            print(f"🤖 Chain {chain_id}: {content}", end="", flush=True)
                        elif event_type == "tool_observation":
                            tool_name = data.get("tool_name", "")
                            observation = data.get("observation", "")
                            print(f"\n🔧 {tool_name}: {observation[:100]}...")
                        elif event_type == "chain_end":
                            print(f"\n✅ Chain {chain_id} completed!")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling event: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")


# Example usage functions
async def start_websocket_server():
    """Start the WebSocket streaming server"""
    server = WebSocketStreamingServer()
    await server.start()
    return server


async def run_agent_with_websocket_streaming(agent, start_messages, max_steps=5, num_chains=1):
    """Run an agent with WebSocket streaming"""
    
    # Start WebSocket server
    server = await start_websocket_server()
    
    # Add WebSocket observer to agent
    agent.streaming_manager.add_observer(server.get_observer())
    
    try:
        # Run the agent
        await agent.run_async(
            max_steps=max_steps,
            start_messages=start_messages,
            num_chains=num_chains,
            enable_streaming=True
        )
    finally:
        # Stop the server
        await server.stop()


async def connect_and_monitor():
    """Connect to WebSocket server and monitor events"""
    client = WebSocketStreamingClient()
    
    async def event_handler(event_data):
        """Custom event handler"""
        event_type = event_data.get("event_type")
        chain_id = event_data.get("chain_id")
        
        if event_type == "llm_generation_start":
            print(f"🚀 Chain {chain_id}: Starting LLM generation...")
        elif event_type == "tool_call_start":
            tool_name = event_data.get("data", {}).get("tool_name", "")
            print(f"🔧 Chain {chain_id}: Calling tool {tool_name}...")
        elif event_type == "chain_end":
            final_depth = event_data.get("data", {}).get("final_depth", 0)
            reward = event_data.get("data", {}).get("reward")
            print(f"✅ Chain {chain_id}: Completed in {final_depth} steps (reward: {reward})")
    
    await client.receive_events(event_handler)


if __name__ == "__main__":
    # Example: Start WebSocket server
    asyncio.run(start_websocket_server()) 