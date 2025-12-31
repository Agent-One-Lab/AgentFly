import time
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from ..reward_base import reward
import os
import json
import glob
import logging
import asyncio
import sys
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from ... import AGENT_HOME

logger = logging.getLogger(__name__)

def get_server_ips(model: str) -> List[str]:
    """Get list of server IPs from the most recent complete server instances file for a specific model"""
    # Get the directory where this utils.py file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_status_dir = os.path.join(Path(AGENT_HOME).parent.parent, "data-process", "vllm_server", "server_status")
    
    # Clean model name for filename matching (replace / and - with _)
    model_clean = model.replace('/', '_').replace('-', '_')
    search_pattern = os.path.join(server_status_dir, f"server_instances_complete_{model_clean}_*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        # Fallback: try to find any server instances file and filter by model in the JSON content
        fallback_pattern = os.path.join(server_status_dir, "server_instances_complete_*.json")
        all_files = glob.glob(fallback_pattern)
        
        for file in all_files:
            try:
                with open(file, 'r') as f:
                    server_info = json.load(f)
                
                # Check if any server in this file matches our model
                matching_servers = [info for info in server_info if info.get('model') == model]
                if matching_servers:
                    json_files = [file]
                    logger.info(f"Found servers for model '{model}' in fallback file: {file}")
                    break
            except Exception as e:
                logger.warning(f"Error reading file {file}: {e}")
                continue
        
        if not json_files:
            raise RuntimeError(f"No server instances file found for model '{model}' in {search_pattern}")
    
    # Get the most recent file
    latest_file = max(json_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        server_info = json.load(f)
    
    # Filter servers by model and extract IPs
    ips = []
    for info in server_info:
        # Check if this server entry matches our model
        if info.get('model') == model and 'ip' in info:
            ips.append(info['ip'])
    
    if not ips:
        raise RuntimeError(f"No IPs found for model '{model}' in server instances file {latest_file}")
    
    logger.info(f"Found {len(ips)} server instances for model '{model}': {ips}")
    return ips

class RateLimiter:
    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.semaphore = asyncio.Semaphore(max_window_size)
        logger.info(f"Created rate limiter with max_window_size={max_window_size}")
    
    async def acquire(self):
        await self.semaphore.acquire()
        logger.info(f"Successfully acquired slot. Available slots: {self.semaphore._value}/{self.max_window_size}")
    
    async def release(self):
        self.semaphore.release()
        logger.info(f"Successfully released slot. Available slots: {self.semaphore._value}/{self.max_window_size}")

class RoundRobinClient:
    def __init__(self, ips: List[str], port: int, api_key: str, timeout: int, rate_limiters: List[RateLimiter]):
        self.ips = ips
        self.current_index = 0
        self.port = port
        self.api_key = api_key
        self.clients = [
            AsyncOpenAI(
                base_url=f"http://{ip}:{port}/v1",
                api_key=api_key,
                timeout=timeout,
            ) for ip in ips
        ]
        self.rate_limiters = rate_limiters
        logger.info(f"Initialized RoundRobinClient with {len(ips)} instances on port {port}")
        for i, ip in enumerate(ips):
            logger.info(f"Instance {i}: {ip}:{port} with rate limit {rate_limiters[i].max_window_size}")
    
    async def get_next_available_client(self) -> tuple[AsyncOpenAI, RateLimiter]:
        # Find the instance with the most available slots
        max_available = -1
        best_client = None
        best_limiter = None
        best_index = -1
        
        # Log current state of all instances
        logger.info("Current instance states:")
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            used = self.rate_limiters[i].max_window_size - available
            logger.info(f"Instance {i}: {self.ips[i]}:{self.port} - Used: {used}, Available: {available}/{self.rate_limiters[i].max_window_size}")
        
        # Check all instances to find the one with most available slots
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            if available > max_available and available > 0:
                max_available = available
                best_client = self.clients[i]
                best_limiter = self.rate_limiters[i]
                best_index = i
                logger.info(f"Selected instance {best_index} ({self.ips[best_index]}:{self.port}) with {max_available} available slots")
        
        if best_client is not None:
            # Found an instance with available capacity
            await best_limiter.acquire()
            return best_client, best_limiter
        else:
            # All instances are at capacity, distribute waiting requests across all instances
            logger.info("All instances at capacity, distributing wait across all instances")
            
            # Create a list of all instances that we can wait on
            wait_tasks = []
            for i in range(len(self.clients)):
                task = asyncio.create_task(self.rate_limiters[i].acquire())
                wait_tasks.append((i, task))
            
            # Wait for any instance to become available
            try:
                # Wait for the first instance to become available
                done, pending = await asyncio.wait(
                    [task for _, task in wait_tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel all other waiting tasks
                for _, task in wait_tasks:
                    if task not in done:
                        task.cancel()
                
                # Find which instance became available
                for i, task in wait_tasks:
                    if task in done:
                        logger.info(f"Instance {i} ({self.ips[i]}:{self.port}) became available")
                        return self.clients[i], self.rate_limiters[i]
                
                # This should never happen, but just in case
                raise RuntimeError("No instance became available despite wait completion")
            except Exception as e:
                logger.error(f"Error while waiting for instances: {e}")
                raise

    def check_availability(self) -> Dict[str, Any]:
        """Check current availability status across all instances"""
        total_capacity = 0
        total_available = 0
        total_used = 0
        instance_details = []
        
        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            capacity = self.rate_limiters[i].max_window_size
            used = capacity - available
            
            total_capacity += capacity
            total_available += available
            total_used += used
            
            instance_details.append({
                'instance_id': i,
                'ip': self.ips[i],
                'port': self.port,
                'available_slots': available,
                'used_slots': used,
                'total_slots': capacity,
                'utilization_percent': (used / capacity) * 100 if capacity > 0 else 0
            })
        
        return {
            'total_instances': len(self.clients),
            'total_capacity': total_capacity,
            'total_available': total_available,
            'total_used': total_used,
            'overall_utilization_percent': (total_used / total_capacity) * 100 if total_capacity > 0 else 0,
            'has_available_slots': total_available > 0,
            'instances': instance_details
        }
    
    async def wait_for_availability(self, min_slots: int = 1, timeout_seconds: Optional[int] = None) -> bool:
        """Wait until at least min_slots are available across all instances
        
        Args:
            min_slots: Minimum number of slots to wait for
            timeout_seconds: Maximum time to wait (None for no timeout)
            
        Returns:
            True if slots became available, False if timeout
        """
        start_time = time.time()
        
        while True:
            availability = self.check_availability()
            if availability['total_available'] >= min_slots:
                logger.info(f"Availability achieved: {availability['total_available']} slots available (needed {min_slots})")
                return True
            
            # Check timeout
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning(f"Timeout waiting for availability after {elapsed:.2f}s")
                    return False
            
            # Wait a short time before checking again
            await asyncio.sleep(0.1)


class LLMClient:
    def __init__(self, 
                 model: str,
                 timeout_seconds: int = 30,
                 max_window_size_per_instance: int = 10,
                 port: int = 8000,
                 api_key: str = "token-abc123"):
        """Initialize LLMClient with configurable parameters
        
        Args:
            model: Model name to find the correct server instances
            timeout_seconds: Request timeout in seconds
            max_window_size_per_instance: Maximum concurrent requests per server instance
            port: Port number for server instances
            api_key: API key for authentication
        """
        self.timeout_seconds = timeout_seconds
        server_ips = get_server_ips(model)
        # Create a rate limiter for each instance
        rate_limiters = [RateLimiter(max_window_size_per_instance) for _ in server_ips]
        self.client_manager = RoundRobinClient(server_ips, port, api_key, timeout_seconds, rate_limiters)
        logger.info(f"Initialized LLMClient for model '{model}' with {len(server_ips)} instances on port {port}, {max_window_size_per_instance} slots per instance")

    async def single_call(self, inputs, model, **kwargs):
        try:
            start_time = time.time()
            logger.info("Starting to get next available client...")
            # Get next available client and its rate limiter (already acquired)
            client, rate_limiter = await self.client_manager.get_next_available_client()
            logger.info(f"Got client for {client.base_url}")
            try:
                logger.info(f"Making request to {client.base_url}")
                response = await client.chat.completions.create(
                    model=model, 
                    messages=inputs,
                    timeout=self.timeout_seconds,
                    **kwargs
                )
                duration = time.time() - start_time
                
                if duration > self.timeout_seconds * 0.9:  # If request took more than 90% of timeout
                    logger.warning(f"Request took {duration:.2f}s (close to timeout {self.timeout_seconds}s)")
                
                logger.info(f"Request completed in {duration:.2f}s")
                return response.choices[0].message.content
            finally:
                await rate_limiter.release()
                logger.info(f"Released rate limiter slot. Available slots: {rate_limiter.semaphore._value}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {self.timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return None

    async def process_all_inputs(self, inputs_list, num_generations=1, model=None, **kwargs):
        try:
            # Create N tasks for each input
            all_tasks = []
            for inputs in inputs_list:
                tasks = []
                for _ in range(num_generations):
                    tasks.append(self.single_call(inputs, model=model, **kwargs))
                all_tasks.extend(tasks)
            
            # Run all tasks concurrently with progress bar
            responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing requests", file=sys.stdout)
            
            # Group responses by input
            grouped_responses = []
            for i in range(0, len(responses), num_generations):
                grouped_responses.append(responses[i:i + num_generations])
            
            return grouped_responses
        finally:
            pass  # No need to stop the rate limiters as they're managed by the acquire/release calls

    def check_availability(self) -> Dict[str, Any]:
        """Check current availability status across all server instances
        
        Returns:
            Dictionary containing detailed availability information
        """
        return self.client_manager.check_availability()
    
    async def wait_for_availability(self, min_slots: int = 1, timeout_seconds: Optional[int] = None) -> bool:
        """Wait until at least min_slots are available across all instances
        
        Args:
            min_slots: Minimum number of slots to wait for
            timeout_seconds: Maximum time to wait (None for no timeout)
            
        Returns:
            True if slots became available, False if timeout
        """
        return await self.client_manager.wait_for_availability(min_slots, timeout_seconds)
    
    def is_available(self) -> bool:
        """Quick check if any slots are currently available
        
        Returns:
            True if at least one slot is available across all instances
        """
        availability = self.check_availability()
        return availability['has_available_slots']
    
    def get_capacity_info(self) -> Dict[str, int]:
        """Get basic capacity information
        
        Returns:
            Dictionary with total_capacity, total_available, total_used
        """
        availability = self.check_availability()
        return {
            'total_capacity': availability['total_capacity'],
            'total_available': availability['total_available'],
            'total_used': availability['total_used'],
            'utilization_percent': availability['overall_utilization_percent']
        }


SystemPromptMath = """You are a math judge. Your task is to determine whether the *reference answer* is mathematically equivalent to the *golden answer*. The answers may be expressed in different formats, including LaTeX, decimal numbers, fractions, or expressions. Do not rely on string matching—evaluate whether the two answers represent the same mathematical value.

If they are mathematically equivalent, output 1.0. If not, output 0.0.

reference_answer = {reference_answer}  
golden_answer = {golden_answer}

Output the result in the following JSON format:  
{{"score": 1.0}} or {{"score": 0.0}}
"""

import re
import json

def extract_score(output_str: str) -> dict:
    """
    Extract the score JSON from a string and return it as a dictionary.
    The string is expected to contain something like: {"score": 1.0}

    Args:
        output_str (str): The raw output string from the LLM.

    Returns:
        dict: A dictionary like {"score": 1.0} or {"score": 0.0}

    Raises:
        ValueError: If no valid score JSON is found.
    """
    match = re.search(r'\{\s*"score"\s*:\s*(0\.0|1\.0)\s*\}', output_str)
    if not match:
        raise ValueError(f"Score JSON not found in output: {output_str}")
    return json.loads(match.group(0))

@reward(name="llm_as_judge_math_reward")
async def llm_as_judge_client_math_reward(prediction: str, answer: str) -> float:
    client = LLMClient(
        model="Qwen/Qwen2.5-72B-Instruct",
    )
    for i in range(10):
        if not client.is_available():
            time.sleep(1)
    
    messages = [
        {
            "role": "user",
            "content": SystemPromptMath.format(
                golden_answer=answer,
                reference_answer=prediction,
            )
        }
    ]
    responses = await client.process_all_inputs([messages], model="Qwen/Qwen2.5-72B-Instruct")
    if responses and responses[0] and responses[0][0]:
        output_text = responses[0][0]
        try:
            json_output = extract_score(output_text)
            score = json_output["score"]
        except Exception as e:
            logger.error(f"Error extracting score from output: {e}")
            score = 0.0
        return {
            "reward": score
        }
    else:
        return {
            "reward": 0.0,
        }

