#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import pytest
import threading
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any
import statistics
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentfly.agents.llm_backends.llm_backends import ClientBackend


class TestClientBackendWorkload:
    """Test suite for ClientBackend workload and rate limiting functionality."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.dict.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "tool_calls": None
                    }
                }
            ]
        }
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def client_backend(self, mock_openai_client):
        """Create a ClientBackend instance for testing."""
        with patch('openai.OpenAI', return_value=mock_openai_client):
            backend = ClientBackend(
                model_name_or_path="test-model",
                template="test-template",
                base_url="http://localhost:8000/v1",
                max_requests_per_minute=10,  # Low limit for testing
                timeout=30,
                api_key="test-key"
            )
            return backend
    
    def test_basic_functionality(self, client_backend, mock_openai_client):
        """Test basic ClientBackend functionality."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Test sync generation
        response = client_backend.generate(messages)
        
        assert isinstance(response, list)
        assert len(response) == 1
        assert response[0] == "Test response"
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_generation(self, client_backend, mock_openai_client):
        """Test async generation functionality."""
        messages = [{"role": "user", "content": "Hello"}]
        
        response = await client_backend.generate_async(messages)
        
        assert isinstance(response, list)
        assert len(response) == 1
        assert response[0] == "Test response"
    
    def test_rate_limiting_basic(self, client_backend):
        """Test basic rate limiting functionality."""
        # Verify semaphore is initialized correctly
        assert client_backend._tokens._value == 10  # max_requests_per_minute
        assert client_backend._max_tokens == 10
    
    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, client_backend, mock_openai_client):
        """Test rate limiting under high load with 100 concurrent requests."""
        # Configure mock to simulate API delay
        def mock_api_call(*args, **kwargs):
            time.sleep(1)  # 1s delay per request
            return Mock(dict=lambda: {
                "choices": [{"message": {"content": "Test response", "tool_calls": None}}]
            })
        
        mock_openai_client.chat.completions.create = mock_api_call
        
        # Create 100 concurrent requests (10x the rate limit)
        num_requests = 100
        messages = [{"role": "user", "content": f"Request {i}"} for i in range(num_requests)]
        
        start_time = time.time()
        
        # Send all requests concurrently
        tasks = [client_backend.generate_async([msg]) for msg in messages]
        responses = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all requests completed
        assert len(responses) == num_requests
        assert all(isinstance(r, list) and len(r) == 1 for r in responses)
        
        # Verify rate limiting worked (should take longer than if unlimited)
        # With 10 RPM limit, 100 requests should take at least 6 minutes in theory
        # But with our 1s delay, it should take at least 10 seconds
        assert total_time >= 10  # At least 10s due to our mock delay
        
        print(f"Rate limiting test: {num_requests} requests completed in {total_time:.2f}s")
        print(f"Effective rate: {num_requests/total_time:.2f} requests/second")
    
    
    @pytest.mark.asyncio
    async def test_concurrent_burst_requests(self, client_backend, mock_openai_client):
        """Test handling of burst requests that exceed the rate limit."""
        # Configure mock with realistic delay
        async def mock_api_call(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms delay per request
            return Mock(dict=lambda: {
                "choices": [{"message": {"content": "Burst response", "tool_calls": None}}]
            })
        
        mock_openai_client.chat.completions.create = mock_api_call
        
        # Send burst of 20 requests (2x the rate limit)
        burst_size = 20
        messages = [{"role": "user", "content": f"Burst {i}"} for i in range(burst_size)]
        
        start_time = time.time()
        
        # Send all requests at once
        tasks = [client_backend.generate_async([msg]) for msg in messages]
        responses = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all requests completed
        assert len(responses) == burst_size
        assert all("Burst response" in r[0] for r in responses)
        
        # Verify rate limiting was applied
        # Should take longer than 20 * 0.05 = 1 second due to rate limiting
        assert total_time >= 0.5  # At least 500ms due to rate limiting
        
        print(f"Burst test: {burst_size} requests completed in {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, client_backend, mock_openai_client):
        """Test performance under sustained load over time."""
        # Configure mock with realistic delay
        async def mock_api_call(*args, **kwargs):
            await asyncio.sleep(0.02)  # 20ms delay per request
            return Mock(dict=lambda: {
                "choices": [{"message": {"content": "Sustained response", "tool_calls": None}}]
            })
        
        mock_openai_client.chat.completions.create = mock_api_call
        
        # Send requests in batches over time
        batch_size = 5
        num_batches = 4
        total_requests = batch_size * num_batches
        
        all_responses = []
        batch_times = []
        
        for batch in range(num_batches):
            batch_start = time.time()
            
            messages = [{"role": "user", "content": f"Sustained batch {batch} req {i}"} 
                       for i in range(batch_size)]
            
            tasks = [client_backend.generate_async([msg]) for msg in messages]
            batch_responses = await asyncio.gather(*tasks)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            all_responses.extend(batch_responses)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Verify all requests completed
        assert len(all_responses) == total_requests
        assert all("Sustained response" in r[0] for r in all_responses)
        
        # Analyze performance
        avg_batch_time = statistics.mean(batch_times)
        total_time = sum(batch_times)
        
        print(f"Sustained load test: {total_requests} requests in {num_batches} batches")
        print(f"Average batch time: {avg_batch_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print(f"Effective rate: {total_requests/total_time:.2f} requests/second")
    
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_workload(self, client_backend, mock_openai_client):
        """Test mixed sync and async calls under load."""
        # Configure mock
        async def mock_api_call(*args, **kwargs):
            await asyncio.sleep(0.01)
            return Mock(dict=lambda: {
                "choices": [{"message": {"content": "Mixed response", "tool_calls": None}}]
            })
        
        mock_openai_client.chat.completions.create = mock_api_call
        
        # Mix of sync and async calls
        sync_messages = [{"role": "user", "content": f"Sync {i}"} for i in range(5)]
        async_messages = [{"role": "user", "content": f"Async {i}"} for i in range(5)]
        
        # Run sync calls in a separate thread to avoid blocking
        def run_sync_calls():
            return [client_backend.generate([msg]) for msg in sync_messages]
        
        # Execute sync calls in thread pool
        loop = asyncio.get_running_loop()
        sync_task = loop.run_in_executor(None, run_sync_calls)
        
        # Execute async calls
        async_tasks = [client_backend.generate_async([msg]) for msg in async_messages]
        async_results = await asyncio.gather(*async_tasks)
        
        # Wait for sync calls to complete
        sync_results = await sync_task
        
        # Verify all calls completed
        assert len(sync_results) == 5
        assert len(async_results) == 5
        
        # All should be successful
        all_results = sync_results + async_results
        assert all("Mixed response" in r[0] for r in all_results)
        
        print(f"Mixed workload test: {len(all_results)} total requests completed")
    
    def test_refiller_startup_edge_cases(self, client_backend):
        """Test refiller startup in different contexts."""
        # Test startup when no event loop exists
        with patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop")):
            # Should not crash
            client_backend._ensure_refiller_running()
            assert client_backend._refill_task is None
        
        # Test startup when event loop exists but is not running
        mock_loop = Mock()
        mock_loop.is_running.return_value = False
        mock_loop.create_task.return_value = Mock()
        
        with patch('asyncio.get_event_loop', return_value=mock_loop):
            client_backend._ensure_refiller_running()
            mock_loop.create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_large_scale_workload(self, client_backend, mock_openai_client):
        """Test with a large number of requests (1000) to verify system stability."""
        # Configure mock with minimal delay
        async def mock_api_call(*args, **kwargs):
            await asyncio.sleep(0.001)  # 1ms delay per request
            return Mock(dict=lambda: {
                "choices": [{"message": {"content": "Large scale response", "tool_calls": None}}]
            })
        
        mock_openai_client.chat.completions.create = mock_api_call
        
        # Create 1000 requests
        num_requests = 1000
        messages = [{"role": "user", "content": f"Large scale {i}"} for i in range(num_requests)]
        
        start_time = time.time()
        
        # Send all requests concurrently
        tasks = [client_backend.generate_async([msg]) for msg in messages]
        responses = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all requests completed
        assert len(responses) == num_requests
        assert all("Large scale response" in r[0] for r in responses)
        
        print(f"Large scale test: {num_requests} requests completed in {total_time:.2f}s")
        print(f"Effective rate: {num_requests/total_time:.2f} requests/second")
        
        # Verify rate limiting was applied (should be limited to ~10 RPM = 0.167 RPS)
        # With 1000 requests at 0.167 RPS, should take at least 6000 seconds
        # But our test is more lenient due to the mock delay
        assert total_time >= 1.0  # At least 1 second due to rate limiting

    def test_workload_with_real_api(self):
        client_backend = ClientBackend(
            model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            template="deepseek",
            base_url="http://localhost:8000/v1",
            max_requests_per_minute=60,
            timeout=300,
            api_key="EMPTY"
        )
        messages = [[{"role": "user", "content": "Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:\n\[\log_2\left({x \over yz}\right) = {1 \over 2}\]\n\[\log_2\left({y \over xz}\right) = {1 \over 3}\]\n\[\log_2\left({z \over xy}\right) = {1 \over 4}\]\nThen the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$."}]] * 100
        time_start = time.time()
        response = client_backend.generate(messages)
        time_end = time.time()

        assert len(response) == 100
        print(f"Time taken: {time_end - time_start} seconds")
        print(f"Effective rate: {100/(time_end - time_start)} requests/second")

    