.. _retrieval_tools:

Retrieval Tools
===============

The Retrieval tools provide semantic search capabilities for document retrieval from large corpora using dense vector embeddings.

Tools Reference
---------------

.. currentmodule:: agents.tools.src.search

asyncdense_retrieve
~~~~~~~~~~~~~~~~~~~

.. autofunction:: agents.tools.src.search.async_dense_retriever.asyncdense_retrieve

**Function Signature:**

.. code-block:: python

    async def asyncdense_retrieve(query: str) -> str

**Description:** Retrieve relevant documents using dense vector search with E5 embeddings and FAISS indexing

**Parameters:**
    - **query** (str): Search query string. Natural language questions work best. Automatically prepends "query: " for E5 model optimization.

**Returns:**
    str: Formatted string with numbered documents::

        ### 1: [Document 1 content]
        ### 2: [Document 2 content] 
        ### 3: [Document 3 content]

**Example:**

.. code-block:: python

    # Basic retrieval
    result = await asyncdense_retrieve("What is quantum computing?")
    print(result)
    
    # Scientific queries
    result = await asyncdense_retrieve("How does photosynthesis work?")
    print(result)

**Features:**
    - High-performance async implementation
    - Thread-safe global retriever instance
    - Automatic model and corpus loading
    - GPU acceleration when available
    - Returns top-3 most relevant documents

dense_retrieve
~~~~~~~~~~~~~~

.. autofunction:: agents.tools.src.search.dense_retriever.dense_retrieve

**Function Signature:**

.. code-block:: python

    async def dense_retrieve(query: str) -> str

**Description:** Retrieve relevant documents using dense vector search (synchronous version)

**Parameters:**
    - **query** (str): Search query string. Automatically prepends "query: " for E5 model optimization.

**Returns:**
    str: Formatted string with numbered documents

**Example:**

.. code-block:: python

    # Simple retrieval
    result = await dense_retrieve("artificial intelligence applications")
    print(result)

**Use Cases:**
    - Simple retrieval tasks
    - Development and debugging
    - Single-query scenarios

Technical Details
-----------------

**Model:** intfloat/e5-base-v2 (768-dim embeddings)
**Corpus:** Wikipedia-18 (18M+ passages)  
**Index:** FAISS Flat index
**Memory:** ~2-8GB RAM required 