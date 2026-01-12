# Retrieval Tools

## Overview

The Retrieval tools provide semantic search and document retrieval capabilities using dense vector embeddings and FAISS indexing. These tools enable agents to query knowledge bases and retrieve relevant information for question answering and knowledge-grounded reasoning.

## Components

- [Tools](tools.md) - Retrieval tool implementations

## Quick Start

```python
from agentfly.tools.src.search.async_dense_retriever import asyncdense_retrieve

# Retrieve relevant documents
result = await asyncdense_retrieve("What is machine learning?")
print(result)
```

## Key Features

* **Dense Vector Search**: E5-base-v2 embeddings for semantic similarity
* **FAISS Indexing**: Efficient similarity search across large document corpora  
* **Async Performance**: Optimized for concurrent queries
* **Auto Loading**: Automatic model and corpus initialization

