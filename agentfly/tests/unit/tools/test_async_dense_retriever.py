import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import tempfile
import json
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys

# Add the parent directory to the Python path to fix import issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Fix imports to handle both versions properly
try:
    from ....tools.src.search.async_dense_retriever import DenseRetriever as AsyncDenseRetriever
    from ....tools.src.search.async_dense_retriever import dense_retrieve as async_dense_retrieve
except ImportError as e:
    print(f"Error importing async_dense_retriever: {e}")
    AsyncDenseRetriever = None
    async_dense_retrieve = None

try:
    from ....tools.src.search.dense_retriever import DenseRetriever as SyncDenseRetriever
    from ....tools.src.search.dense_retriever import dense_retrieve as sync_dense_retrieve
except ImportError as e:
    print(f"Error importing dense_retriever: {e}")
    SyncDenseRetriever = None
    sync_dense_retrieve = None

# Mock data for testing
MOCK_CORPUS_DATA = [
    {"id": "1", "contents": "The quick brown fox jumps over the lazy dog"},
    {"id": "2", "contents": "Python is a high-level programming language"},
    {"id": "3", "contents": "Machine learning is a subset of artificial intelligence"},
    {"id": "4", "contents": "Deep learning models require large amounts of data"},
    {"id": "5", "contents": "Natural language processing enables computers to understand human language"}
]

# @pytest.fixture
# def mock_corpus_file():
#     """Create a temporary corpus file for testing"""
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
#         for item in MOCK_CORPUS_DATA:
#             f.write(json.dumps(item) + '\n')
#         temp_path = f.name
#     yield temp_path
#     os.unlink(temp_path)


# @pytest.fixture
# def mock_index_file():
#     """Create a temporary index file path for testing"""
#     with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as f:
#         temp_path = f.name
#     yield temp_path
#     if os.path.exists(temp_path):
#         os.unlink(temp_path)


# @pytest.mark.skipif(sync_dense_retrieve is None or async_dense_retrieve is None, 
#                     reason="Both retrievers need to be available")
# def test_schema():
#     """Test that both retrievers have the same schema"""
#     sync_schema = sync_dense_retrieve.schema
#     async_schema = async_dense_retrieve.schema
    
#     # Correct schema access
#     assert sync_schema['function']['name'] == async_schema['function']['name']
#     assert sync_schema['function']['description'] == async_schema['function']['description']
#     print(f"Schema: {async_schema}")

# @pytest.mark.asyncio
# @pytest.mark.skipif(AsyncDenseRetriever is None, reason="AsyncDenseRetriever not available")
# async def test_basic_functionality(mock_corpus_file, mock_index_file):
#     """Test basic retrieval functionality"""
#     # Mock the model and tokenizer to avoid downloading
#     with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
#          patch('transformers.AutoModel.from_pretrained') as mock_model, \
#          patch('agents.tools.src.search.async_dense_retriever.load_corpus') as mock_load_corpus, \
#          patch('torch.cuda.is_available', return_value=False), \
#          patch('faiss.read_index') as mock_faiss_read:
        
#         # Setup tokenizer mock
#         mock_tokenizer_instance = MagicMock()
#         mock_tokenizer_instance.return_value = {
#             'input_ids': torch.tensor([[1, 2, 3]]),
#             'attention_mask': torch.tensor([[1, 1, 1]])
#         }
#         mock_tokenizer.return_value = mock_tokenizer_instance
        
#         # Setup model mock
#         mock_model_instance = MagicMock()
#         mock_model_instance.eval = MagicMock()
#         mock_model_instance.to = MagicMock(return_value=mock_model_instance)
#         mock_output = MagicMock()
#         mock_output.last_hidden_state = torch.randn(1, 3, 768)
#         mock_model_instance.return_value = mock_output
#         mock_model.return_value = mock_model_instance
        
#         # Mock corpus
#         mock_corpus = MagicMock()
#         mock_corpus.__getitem__ = MagicMock(side_effect=lambda idx: MOCK_CORPUS_DATA[idx] if isinstance(idx, int) else [item['id'] for item in MOCK_CORPUS_DATA])
#         mock_load_corpus.return_value = mock_corpus
        
#         # Mock FAISS index
#         mock_index = MagicMock()
#         mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
#         mock_faiss_read.return_value = mock_index
        
#         # Test retriever
#         retriever = AsyncDenseRetriever(mock_corpus_file, mock_index_file)
#         results = await retriever.search(["query: python programming"], top_k=3)
        
#         assert len(results) == 1
#         assert len(results[0]) == 3
#         print(f"Basic search results: {results}")

# @pytest.mark.asyncio
# @pytest.mark.skipif(AsyncDenseRetriever is None, reason="AsyncDenseRetriever not available")
# async def test_concurrent_searches(mock_corpus_file, mock_index_file):
#     """Test multiple concurrent searches"""
#     with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
#          patch('transformers.AutoModel.from_pretrained') as mock_model, \
#          patch('agents.tools.src.search.async_dense_retriever.load_corpus') as mock_load_corpus, \
#          patch('torch.cuda.is_available', return_value=False), \
#          patch('faiss.read_index') as mock_faiss_read:
        
#         # Setup mocks similar to test_basic_functionality
#         mock_tokenizer_instance = MagicMock()
#         mock_tokenizer_instance.return_value = {
#             'input_ids': torch.tensor([[1, 2, 3]]),
#             'attention_mask': torch.tensor([[1, 1, 1]])
#         }
#         mock_tokenizer.return_value = mock_tokenizer_instance
        
#         mock_model_instance = MagicMock()
#         mock_model_instance.eval = MagicMock()
#         mock_model_instance.to = MagicMock(return_value=mock_model_instance)
#         mock_output = MagicMock()
#         mock_output.last_hidden_state = torch.randn(1, 3, 768)
#         mock_model_instance.return_value = mock_output
#         mock_model.return_value = mock_model_instance
        
#         # Mock corpus with proper method signature
#         mock_corpus = MagicMock()
#         def corpus_getitem(key):
#             if isinstance(key, int):
#                 return MOCK_CORPUS_DATA[key % len(MOCK_CORPUS_DATA)]
#             elif key == "id":
#                 return [item['id'] for item in MOCK_CORPUS_DATA]
#             else:
#                 return None
#         mock_corpus.__getitem__.side_effect = corpus_getitem
#         mock_load_corpus.return_value = mock_corpus
        
#         # Fix: Mock FAISS index with fixed indices (no undefined 'i')
#         mock_index = MagicMock()
#         mock_index.search.return_value = (
#             np.array([[0.9, 0.8, 0.7]]), 
#             np.array([[0, 1, 2]])  # ← Fixed: Use static indices instead of undefined 'i'
#         )
#         mock_faiss_read.return_value = mock_index
        
#         retriever = AsyncDenseRetriever(mock_corpus_file, mock_index_file)
        
#         # Perform multiple concurrent searches
#         queries = [
#             "query: machine learning",
#             "query: deep learning", 
#             "query: natural language processing",
#             "query: python programming",
#             "query: artificial intelligence"
#         ]
        
#         start_time = time.time()
#         results = await asyncio.gather(*[
#             retriever.search([query], top_k=3) for query in queries
#         ])
#         async_time = time.time() - start_time
        
#         assert len(results) == len(queries)
#         for result in results:
#             assert len(result[0]) == 3
        
#         print(f"Concurrent search time: {async_time:.4f}s for {len(queries)} queries")

# @pytest.mark.asyncio
# @pytest.mark.skipif(sync_dense_retrieve is None or async_dense_retrieve is None,
#                     reason="Both retrievers need to be available")
# async def test_performance_comparison():
#     """Compare performance between sync and async versions"""
#     # Create mock data
#     import agents.tools.src.search.async_dense_retriever as async_module
#     import agents.tools.src.search.dense_retriever as sync_module
    
#     with patch.object(async_module, 'GLOBAL_RETRIEVER', None), \
#          patch.object(sync_module, 'GLOBAL_RETRIEVER', None), \
#          patch.object(async_module, 'AGENT_DATA_DIR', '.'), \
#          patch.object(sync_module, 'AGENT_DATA_DIR', '.'), \
#          patch.object(async_module, 'DenseRetriever') as mock_async_retriever, \
#          patch.object(sync_module, 'DenseRetriever') as mock_sync_retriever:
        
#         # Setup mock async retriever
#         mock_async_instance = MagicMock()
#         async def mock_search(queries, top_k):
#             await asyncio.sleep(0.1)  # Simulate some processing time
#             return [[{"contents": f"Result {i} for {q}"} for i in range(top_k)] for q in queries]
#         mock_async_instance.search = mock_search
#         mock_async_retriever.return_value = mock_async_instance
        
#         # Setup mock sync retriever
#         mock_sync_instance = MagicMock()
#         async def mock_sync_search(queries, top_k):
#             await asyncio.sleep(0.1)  # Simulate same processing time
#             return [[{"contents": f"Result {i} for {q}"} for i in range(top_k)] for q in queries]
#         mock_sync_instance.search = mock_sync_search
#         mock_sync_retriever.return_value = mock_sync_instance
        
#         queries = ["query1", "query2", "query3", "query4", "query5"]
        
#         start_time = time.time()
#         async_results = await asyncio.gather(*[
#             async_dense_retrieve(query=query) for query in queries
#         ])
#         async_time = time.time() - start_time
        
#         start_time = time.time()
#         sync_results = []
#         for query in queries:
#             result = await sync_dense_retrieve(query=query)
#             sync_results.append(result)
#         sync_time = time.time() - start_time
        
#         print(f"\nPerformance Comparison:")
#         print(f"Async (concurrent): {async_time:.4f}s")
#         print(f"Sync (sequential): {sync_time:.4f}s")
#         print(f"Speedup: {sync_time/async_time:.2f}x")
        
#         assert len(async_results) == len(sync_results)

# @pytest.mark.asyncio
# @pytest.mark.skipif(async_dense_retrieve is None, reason="async_dense_retrieve not available")
# async def test_global_retriever_singleton():
#     """Test that the global retriever is created only once"""
#     import agents.tools.src.search.async_dense_retriever as async_module
    
#     with patch.object(async_module, 'GLOBAL_RETRIEVER', None), \
#          patch.object(async_module, 'AGENT_DATA_DIR', '.'), \
#          patch.object(async_module, 'DenseRetriever') as mock_retriever:
        
#         mock_instance = MagicMock()
#         mock_instance.search = AsyncMock(return_value=[[{"contents": "test"}]])
#         mock_retriever.return_value = mock_instance
        
#         await async_dense_retrieve(query="test query 1")
#         assert mock_retriever.call_count == 1
        
#         await async_dense_retrieve(query="test query 2")
#         assert mock_retriever.call_count == 1

# @pytest.mark.asyncio
# @pytest.mark.skipif(async_dense_retrieve is None, reason="async_dense_retrieve not available")
# async def test_query_prefix_handling():
#     """Test that 'query:' prefix is added when missing"""
#     import agents.tools.src.search.async_dense_retriever as async_module
    
#     mock_retriever = MagicMock()
#     called_queries = []
    
#     async def capture_query(queries, top_k):
#         called_queries.extend(queries)
#         return [[{"contents": "test"}]]
    
#     mock_retriever.search = capture_query
    
#     with patch.object(async_module, 'GLOBAL_RETRIEVER', mock_retriever):
#         await async_dense_retrieve(query="test without prefix")
#         assert called_queries[-1] == "query: test without prefix"
        
#         await async_dense_retrieve(query="query: test with prefix")
#         assert called_queries[-1] == "query: test with prefix"
        
#         print(f"Query prefix handling test passed: {called_queries}")

# @pytest.mark.asyncio
# @pytest.mark.skipif(AsyncDenseRetriever is None, reason="AsyncDenseRetriever not available")
# async def test_thread_pool_efficiency():
#     """Test that the thread pool is being used efficiently"""
#     with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
#          patch('transformers.AutoModel.from_pretrained') as mock_model, \
#          patch('agents.tools.src.search.async_dense_retriever.load_corpus') as mock_load_corpus, \
#          patch('torch.cuda.is_available', return_value=False), \
#          patch('faiss.read_index') as mock_faiss_read:
        
#         # Track thread pool usage
#         executor_calls = []
#         original_run_in_executor = None
        
#         async def mock_run_in_executor(executor, func, *args):
#             executor_calls.append((func.__name__, args))
#             # Call the original function to test the actual logic
#             if func.__name__ == '_embed_sync':
#                 return np.random.rand(1, 768)  # Mock embedding
#             elif func.__name__ == '_faiss_sync':
#                 return [[(0.9, 0), (0.8, 1), (0.7, 2)]]  # Mock FAISS results
#             return func(*args)
        
#         # Setup mocks
#         mock_tokenizer_instance = MagicMock()
#         mock_tokenizer_instance.return_value = {
#             'input_ids': torch.tensor([[1, 2, 3]]),
#             'attention_mask': torch.tensor([[1, 1, 1]])
#         }
#         mock_tokenizer.return_value = mock_tokenizer_instance
        
#         mock_model_instance = MagicMock()
#         mock_model_instance.eval = MagicMock()
#         mock_model_instance.to = MagicMock(return_value=mock_model_instance)
#         mock_output = MagicMock()
#         mock_output.last_hidden_state = torch.randn(1, 3, 768)
#         mock_model_instance.return_value = mock_output
#         mock_model.return_value = mock_model_instance
        
#         # Mock corpus
#         mock_corpus = MagicMock()
#         mock_corpus.__getitem__ = MagicMock(side_effect=lambda idx: MOCK_CORPUS_DATA[idx] if isinstance(idx, int) else [item['id'] for item in MOCK_CORPUS_DATA])
#         mock_load_corpus.return_value = mock_corpus
        
#         # Mock FAISS index
#         mock_index = MagicMock()
#         mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
#         mock_faiss_read.return_value = mock_index
        
#         with patch('asyncio.get_running_loop') as mock_loop:
#             mock_loop.return_value.run_in_executor = mock_run_in_executor
            
#             retriever = AsyncDenseRetriever("mock_corpus.jsonl", "mock_index.index")
#             await retriever.search(["test query"], top_k=3)
            
#         # Verify that both embedding and FAISS search use the thread pool
#         assert len(executor_calls) >= 2
#         func_names = [call[0] for call in executor_calls]
#         assert '_embed_sync' in func_names
#         assert '_faiss_sync' in func_names
        
#         print(f"Thread pool usage verified: {func_names}")

# @pytest.mark.asyncio
# @pytest.mark.skipif(async_dense_retrieve is None, reason="async_dense_retrieve not available")
# async def test_error_handling():
#     """Test error handling in async retriever"""
#     import agents.tools.src.search.async_dense_retriever as async_module
    
#     # Reset the global retriever to ensure it gets recreated
#     with patch.object(async_module, 'GLOBAL_RETRIEVER', None):
#         # Create a mock that raises an exception when DenseRetriever is instantiated
#         with patch(
#             'agents.tools.src.search.async_dense_retriever.DenseRetriever',
#             side_effect=FileNotFoundError("Cannot find corpus or index files")
#         ), patch.object(async_module, 'AGENT_DATA_DIR', '/non/existent/path'):
#             # This should now raise an error during GLOBAL_RETRIEVER creation
#             with pytest.raises(FileNotFoundError):
#                 await async_dense_retrieve(query="test query")
            
#             assert exc_info.value is not None
#             print(f"Error handling test passed: {type(exc_info.value).__name__}: {exc_info.value}")
# @pytest.mark.asyncio
# @pytest.mark.skipif(async_dense_retrieve is None, reason="async_dense_retrieve not available")
# async def test_large_batch_performance():
#     """Test performance with large batch of queries"""
#     import agents.tools.src.search.async_dense_retriever as async_module
    
#     call_count = 0
    
#     async def mock_search(queries, top_k):
#         nonlocal call_count
#         call_count += 1
#         await asyncio.sleep(0.01)  # Simulate processing
#         return [[{"contents": f"Result for {q}"} for _ in range(top_k)] for q in queries]
    
#     mock_retriever = MagicMock()
#     mock_retriever.search = mock_search
    
#     with patch.object(async_module, 'GLOBAL_RETRIEVER', mock_retriever):
#         # Create a large batch of queries
#         num_queries = 50
#         queries = [f"query {i}" for i in range(num_queries)]
        
#         start_time = time.time()
#         results = await asyncio.gather(*[
#             async_dense_retrieve(query=query) for query in queries
#         ])
#         total_time = time.time() - start_time
        
#         assert len(results) == num_queries
#         assert call_count == num_queries  # Each query should trigger one search
        
#         print(f"Large batch test: {num_queries} queries in {total_time:.4f}s")
#         print(f"Average time per query: {total_time/num_queries:.4f}s")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])