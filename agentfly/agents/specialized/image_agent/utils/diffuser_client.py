#!/usr/bin/env python3
"""
Multi-node client for the Multi-GPU Image Edit API.
Provides load balancing across multiple nodes with concurrency control.
Follows the same pattern as OpenAI API client for consistency.
"""

import asyncio
import base64
import time
import aiohttp
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
import statistics
from dataclasses import dataclass
import threading
from queue import Queue
import random
from functools import partial
import logging
from .utils import ImageEditRequest, ImageEditResponse, ErrorResponse

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a node."""
    host: str
    port: int
    url: str
    active_requests: int = 0
    last_used: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

class DiffuserClient:
    """Multi-node client with load balancing and concurrency control.
    
    This client provides a thin wrapper around the Multi-GPU Image Edit API,
    supporting both synchronous and asynchronous operations. It includes built-in
    rate limiting and retry mechanisms for reliable API communication.
    """
    
    def __init__(
        self,
        nodes: List[Tuple[str, int]],
        max_requests_per_minute: int = 100,
        timeout: int = 600,
        **kwargs
    ):
        """Initialize MultiNodeImageEditClient.
        
        Args:
            nodes: List of (host, port) tuples for available nodes
            max_requests_per_minute: Rate limiting for API requests. Defaults to 100.
            timeout: Request timeout in seconds. Defaults to 600.
            **kwargs: Additional configuration parameters.
        """
        # Initialize nodes
        self.nodes = []
        for host, port in nodes:
            node_info = NodeInfo(
                host=host,
                port=port,
                url=f"http://{host}:{port}"
            )
            self.nodes.append(node_info)
        
        # Connection settings
        self.max_requests_per_minute = max_requests_per_minute
        self.timeout = timeout
        
        # Rate limiting (token bucket, allow burst up to max_requests_per_minute)
        # Start with full bucket to allow initial burst
        self._tokens = asyncio.Semaphore(max_requests_per_minute)
        self._max_tokens = max_requests_per_minute
        self._refill_task = None  # started lazily
        
        # Statistics
        self.stats_lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        
        logger.debug(f"Initialized multi-node client with {len(self.nodes)} nodes")
        logger.debug(f"Max requests per minute: {max_requests_per_minute}")
    
    def load_image_as_base64(self, image_path: str) -> str:
        """Load a local image file and convert to base64."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to load image {image_path}: {str(e)}")
    
    def get_best_node(self) -> Optional[NodeInfo]:
        """Get the best available node based on load balancing."""
        if not self.nodes:
            return None
        
        # Simply return the node with the lowest active requests
        return min(self.nodes, key=lambda node: node.active_requests)
    
    def get_node_load_info(self) -> Dict[str, Any]:
        """Get detailed load information for all nodes."""
        node_info = {}
        for i, node in enumerate(self.nodes):
            success_rate = (node.successful_requests / node.total_requests * 100) if node.total_requests > 0 else 0
            time_since_last_use = time.time() - node.last_used if node.last_used > 0 else float('inf')
            
            node_info[f"node_{i}"] = {
                "host": node.host,
                "port": node.port,
                "active_requests": node.active_requests,
                "total_requests": node.total_requests,
                "successful_requests": node.successful_requests,
                "failed_requests": node.failed_requests,
                "success_rate": success_rate,
                "time_since_last_use": time_since_last_use,
                "last_used": node.last_used
            }
        return node_info
    
    # --------------------------------------------------------------------- #
    # Low‑level single request (runs in threadpool so it doesn't block loop)
    # --------------------------------------------------------------------- #
    def _blocking_call(self, request: ImageEditRequest) -> Dict[str, Any]:
        """Blocking call to edit image (runs in threadpool)."""
        start_time = time.time()
        
        try:
            # Get best available node
            node = self.get_best_node()
            if node is None:
                return {
                    "status": "error",
                    "error": "No available nodes",
                    "response_time": 0.0,
                    "node": "none"
                }
            
            logger.debug(f"Using node: {node.url}")
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error selecting node: {str(e)}",
                "response_time": 0.0,
                "node": "none"
            }
        
        # Update node stats
        with self.stats_lock:
            node.active_requests += 1
            node.last_used = time.time()
            node.total_requests += 1
        
        try:
            # Prepare request payload
            payload = {
                "image": request.image,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "true_cfg_scale": request.true_cfg_scale,
                "num_inference_steps": request.num_inference_steps,
                "seed": request.seed,
                "timeout": request.timeout
            }
            
            # Send request synchronously using requests
            import requests
            timeout = request.timeout or self.timeout
            try:
                logger.debug(f"Sending request to {node.url}")
                response = requests.post(
                    f"{node.url}/v1/images/edits",
                    json=payload,
                    timeout=request.timeout
                )
            except requests.exceptions.Timeout:
                return {
                    "status": "timeout",
                    "response_time": time.time() - start_time,
                    "status_code": 0,
                    "node": f"{node.host}:{node.port}",
                    "error": f"Request timeout after {timeout} seconds"
                }
            except requests.exceptions.ConnectionError as e:
                return {
                    "status": "connection_error",
                    "response_time": time.time() - start_time,
                    "status_code": 0,
                    "node": f"{node.host}:{node.port}",
                    "error": f"Connection error: {str(e)}"
                }
            except requests.exceptions.RequestException as e:
                return {
                    "status": "request_error",
                    "response_time": time.time() - start_time,
                    "status_code": 0,
                    "node": f"{node.host}:{node.port}",
                    "error": f"Request error: {str(e)}"
                }
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "node": f"{node.host}:{node.port}",
                    "has_image": "image" in result,
                    "image_size": len(result.get("image", "")) if "image" in result else 0,
                    "result": result
                }
            else:
                error_text = response.text
                return {
                    "status": "error",
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "node": f"{node.host}:{node.port}",
                    "error": error_text
                }
                        
        except Exception as e:
            end_time = time.time()
            return {
                "status": "exception",
                "response_time": end_time - start_time,
                "status_code": 0,
                "node": f"{node.host}:{node.port}",
                "error": str(e)
            }
        finally:
            # Release node capacity
            with self.stats_lock:
                node.active_requests -= 1
    
    async def _call(self, request: ImageEditRequest) -> Dict[str, Any]:
        """Async call with rate limiting."""
        # Update statistics
        with self.stats_lock:
            self.total_requests += 1
        
        try:
            # Acquire a rate‑limit token
            async with self._tokens:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, partial(self._blocking_call, request))
                
                # Update statistics
                with self.stats_lock:
                    if result["status"] == "success":
                        self.successful_requests += 1
                    else:
                        self.failed_requests += 1
                    
                    self.response_times.append(result["response_time"])
                
                return result
                
        except Exception as e:
            # Update statistics for exceptions
            with self.stats_lock:
                self.failed_requests += 1
            
            # Return error result instead of raising
            return {
                "status": "exception",
                "error": str(e),
                "response_time": 0.0,
                "node": "unknown",
                "status_code": 0
            }
    
    # Public API ‑‑ sync or async depending on caller's context
    def edit_image(
        self,
        request: ImageEditRequest,
    ) -> Union[Dict[str, Any], asyncio.Task]:
        """
        Edit a single image.
        
        Returns:
          • In an *async* context → **awaitable Task** (so caller writes `await client.edit_image(...)`).
          • In a *sync* context  → real result dict (blocks until done).
        """
        async def _runner():
            # Ensure refiller is running in this event loop
            self._ensure_refiller_running()
            return await self._call(request)
        
        try:
            loop = asyncio.get_running_loop()  # ➊ already inside a loop?
        except RuntimeError:
            # --- synchronous caller: spin a loop just for this call
            return asyncio.run(_runner())
        
        # --- asynchronous caller: schedule task & hand it back
        # (don't block the caller's event loop)
        return loop.create_task(_runner())
    
    def batch_edit_images(
        self,
        requests: List[ImageEditRequest],
    ) -> Union[List[Dict[str, Any]], asyncio.Task]:
        """
        Edit multiple images concurrently.
        
        Returns:
          • In an *async* context → **awaitable Task** (so caller writes `await client.batch_edit_images(...)`).
          • In a *sync* context  → real list of results (blocks until done).
        """
        async def _runner():
            # Ensure refiller is running in this event loop
            self._ensure_refiller_running()
            tasks = [asyncio.create_task(self._call(request)) for request in requests]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        try:
            loop = asyncio.get_running_loop()  # ➊ already inside a loop?
        except RuntimeError:
            # --- synchronous caller: spin a loop just for this call
            return asyncio.run(_runner())
        
        # --- asynchronous caller: schedule task & hand it back
        # (don't block the caller's event loop)
        return loop.create_task(_runner())
    
    async def edit_image_async(self, request: ImageEditRequest) -> Dict[str, Any]:
        """Async version of edit_image."""
        return await self.edit_image(request)
    
    async def batch_edit_images_async(self, requests: List[ImageEditRequest]) -> List[Dict[str, Any]]:
        """Async version of batch_edit_images."""
        return await self.batch_edit_images(requests)
    
    # Background token‑bucket refill (one token each 60/max_rpm seconds)
    async def _refill_tokens(self):
        """Background task to refill rate limiting tokens."""
        interval = 60 / self._max_tokens
        while True:
            await asyncio.sleep(interval)
            if self._tokens._value < self._max_tokens:
                self._tokens.release()
    
    def _ensure_refiller_running(self):
        """Ensure the token refiller is running."""
        if self._refill_task is None or self._refill_task.done():
            try:
                # Try to get running loop first
                loop = asyncio.get_running_loop()
                self._refill_task = loop.create_task(self._refill_tokens())
            except RuntimeError:
                # No event loop running, this will be handled by the caller
                # The refiller will be started when we're in an event loop
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self.stats_lock:
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            
            stats = {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "max_requests_per_minute": self.max_requests_per_minute,
                "available_tokens": self._tokens._value
            }
            
            if self.response_times:
                stats["response_time_stats"] = {
                    "min": min(self.response_times),
                    "max": max(self.response_times),
                    "mean": statistics.mean(self.response_times),
                    "median": statistics.median(self.response_times),
                    "p95": self.percentile(self.response_times, 95),
                    "p99": self.percentile(self.response_times, 99)
                }
            
            # Node statistics
            stats["nodes"] = []
            for node in self.nodes:
                node_stats = {
                    "host": node.host,
                    "port": node.port,
                    "active_requests": node.active_requests,
                    "total_requests": node.total_requests,
                    "successful_requests": node.successful_requests,
                    "failed_requests": node.failed_requests,
                    "success_rate": (node.successful_requests / node.total_requests * 100) if node.total_requests > 0 else 0,
                    "last_used": node.last_used,
                    "time_since_last_use": time.time() - node.last_used if node.last_used > 0 else float('inf')
                }
                stats["nodes"].append(node_stats)
            
            # Add detailed load information
            stats["node_load_info"] = self.get_node_load_info()
            
            return stats
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calculate the given percentile of the data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("MULTI-NODE CLIENT STATISTICS")
        print("=" * 60)
        
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']} ({stats['success_rate']:.1f}%)")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Max requests per minute: {stats['max_requests_per_minute']}")
        print(f"Available tokens: {stats['available_tokens']}")
        
        if 'response_time_stats' in stats:
            rt_stats = stats['response_time_stats']
            print(f"\nResponse times:")
            print(f"  Min: {rt_stats['min']:.2f}s")
            print(f"  Max: {rt_stats['max']:.2f}s")
            print(f"  Mean: {rt_stats['mean']:.2f}s")
            print(f"  Median: {rt_stats['median']:.2f}s")
            print(f"  95th percentile: {rt_stats['p95']:.2f}s")
            print(f"  99th percentile: {rt_stats['p99']:.2f}s")
        
        print(f"\nNode status:")
        for node in stats['nodes']:
            print(f"  {node['host']}:{node['port']} - "
                  f"Active: {node['active_requests']}, "
                  f"Total: {node['total_requests']}, "
                  f"Success: {node['success_rate']:.1f}%")
        
        print("=" * 60)

async def main():
    """python -m agentfly.agents.specialized.image_agent.utils.diffuser_client --nodes 10.24.3.201:8000 10.24.0.92:8000 --image SC02.png --requests 40 --rate 5 --timeout 600"""
    parser = argparse.ArgumentParser(description="Multi-node Image Edit Client")
    parser.add_argument("--nodes", nargs="+", required=True, 
                       help="List of nodes in format 'host:port' (e.g., '192.168.1.100:8000 192.168.1.101:8000')")
    parser.add_argument("--image", required=True, help="Path to the local image file")
    parser.add_argument("--requests", type=int, default=1, help="Number of requests to send")
    parser.add_argument("--rate", type=int, default=200, help="Maximum requests per minute")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Parse nodes
    nodes = []
    for node_str in args.nodes:
        if ':' not in node_str:
            raise ValueError(f"Invalid node format: {node_str}. Use 'host:port'")
        host, port = node_str.split(':', 1)
        nodes.append((host, int(port)))
    
    # Create client
    client = DiffuserClient(
        nodes, 
        max_requests_per_minute=args.rate
    )
    
    # Create all requests concurrently
    tasks = []
    for i in range(args.requests):
        request = ImageEditRequest(
            image=client.load_image_as_base64(args.image),
            prompt="Make this image more artistic and dramatic",
            negative_prompt="",
            true_cfg_scale=7.5,
            num_inference_steps=20,
            seed=42,
            timeout=args.timeout,
        )
        task = client.edit_image(request)
        tasks.append(task)
    
    # Wait for all requests to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Log results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Request {i} failed: {result}")
        else:
            logger.info(f"Request {i} completed with status: {result.get('status', 'unknown')}")
    
    # Print statistics
    client.print_statistics()
    

if __name__ == "__main__":
    asyncio.run(main())