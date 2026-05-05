import asyncio
import time
import numpy as np
# Import the module, not just the variables
import statistics
import agentfly.tools.src.search.async_dense_retriever as async_dense_retriever
from agentfly.tools.src.search.async_dense_retriever import GLOBAL_RETRIEVER, async_dense_retrieve

async def profile_search(retriever, query_batch):
    """
    Measures the time spent in each phase of a single search call.
   
    """
    metrics = {}
    loop = asyncio.get_running_loop()

    # Phase 1: Embedding
    start = time.perf_counter()
    async with retriever._semaphore:
        mid_1 = time.perf_counter()
        embeddings = await loop.run_in_executor(retriever._POOL, retriever._get_embeddings, query_batch)
        mid_2 = time.perf_counter()
        
        # Phase 2: FAISS Search
        score_ids_list = await loop.run_in_executor(
            retriever._POOL, retriever.indexer.search, embeddings, 3
        )
        mid_3 = time.perf_counter()

    # Phase 3: Doc Retrieval (I/O)
    all_results = []
    for score_ids in score_ids_list:
        ids = [sid[1] for sid in score_ids]
        docs = await loop.run_in_executor(retriever._POOL, retriever._get_docs_by_ids, ids)
        all_results.append(docs)
    end = time.perf_counter()

    metrics['queue_wait'] = mid_1 - start
    metrics['embedding'] = mid_2 - mid_1
    metrics['faiss_search'] = mid_3 - mid_2
    metrics['doc_retrieval'] = end - mid_3
    metrics['total'] = end - start
    
    return metrics

async def run_benchmark(num_concurrent_users=50):
    # 1. This call will initialize the retriever inside the module
    print("Initializing retriever...")
    await async_dense_retrieve(query="warmup")
    
    # 2. Access the instance via the module namespace to get the updated value
    retriever = async_dense_retriever.GLOBAL_RETRIEVER
    
    if retriever is None:
        print("Error: Retriever failed to initialize.")
        return

    # 3. Now this will work correctly
    print("Waiting for corpus to load...")
    await retriever._corpus_loading_event.wait()
    
    print(f"--- Starting Benchmark: {num_concurrent_users} concurrent users ---")
    test_queries = [f"What is the history of {i}?" for i in range(num_concurrent_users)]
    
    tasks = [profile_search(retriever, [q]) for q in test_queries]
    
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - start_time

    # Aggregate Metrics
    avg_metrics = {k: statistics.mean([r[k] for r in results]) for k in results[0].keys()}
    
    print(f"\nResults (Averages per request):")
    print(f"  - Semaphore Queue Wait: {avg_metrics['queue_wait']*1000:.2f}ms")
    print(f"  - Embedding Time:       {avg_metrics['embedding']*1000:.2f}ms")
    print(f"  - FAISS Search Time:    {avg_metrics['faiss_search']*1000:.2f}ms")
    print(f"  - Doc Retrieval (I/O):  {avg_metrics['doc_retrieval']*1000:.2f}ms")
    print(f"  - Total Latency:        {avg_metrics['total']*1000:.2f}ms")
    print(f"\nSystem Throughput: {len(results)/total_duration:.2f} requests/sec")

if __name__ == "__main__":
    asyncio.run(run_benchmark(num_concurrent_users=50))