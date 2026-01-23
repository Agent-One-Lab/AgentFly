# Retrieval Tools

The Retrieval tools provide semantic search capabilities for document retrieval from large corpora using dense vector embeddings.

## Tools Reference

### asyncdense_retrieve

::: agentfly.tools.src.search.async_dense_retriever.asyncdense_retrieve
    options:
      show_source: true

**Tool Signature:**

```python
@tool(
    name="asyncdense_retrieve",
    description="Use a dense retriever to retrieve documents from a corpus.",
    max_length=4096,
)
async def asyncdense_retrieve(query: str)
```

**Description:**

Retrieves relevant documents from a Wikipedia corpus using dense vector embeddings and semantic similarity search. Automatically initializes the retriever model and FAISS index on first use.

**Call Signature:**
- **For this non-stateful tool:** ``asyncdense_retrieve(query=...)``

**Parameters:**
- **query** (str): Search query string. Automatically prepends "query: " prefix for E5 model optimization if not already present.

**Returns:**
Tool result dict with:
- **observation** (str): Formatted string with numbered documents (max 4096 chars):

    ```
    ### 1: [Document 1 content]
    ### 2: [Document 2 content]
    ### 3: [Document 3 content]
    ```

- **status** (str): "success"
- **name** (str): "asyncdense_retrieve"
- **arguments** (dict): Input parameters used
- **info** (dict): Additional metadata

**Example:**

```python
# Using the tool wrapper
result = await asyncdense_retrieve(query="What is quantum computing?")
print(result["observation"])

# Direct function call (returns formatted string)
docs = await asyncdense_retrieve("How does photosynthesis work?")
```

**Implementation Features:**
- Global retriever instance (thread-safe)
- Lazy initialization on first call
- E5-base-v2 embeddings (768-dim)
- FAISS Flat index for similarity search
- Returns top-3 most relevant documents
- Automatic corpus and model loading from AGENT_DATA_DIR

### dense_retrieve

::: agentfly.tools.src.search.dense_retriever.dense_retrieve
    options:
      show_source: true

**Tool Signature:**

```python
@tool(
    name="dense_retrieve",
    description="Use a dense retriever to retrieve documents from a corpus.",
    max_length=4096
)
async def dense_retrieve(query: str)
```

**Description:**

Retrieves relevant documents from a Wikipedia corpus using dense vector embeddings and semantic similarity search. Similar to asyncdense_retrieve with slightly different internal implementation.

**Call Signature:**
- **For this non-stateful tool:** ``dense_retrieve(query=...)``

**Parameters:**
- **query** (str): Search query string. Automatically prepends "query: " prefix for E5 model optimization if not already present.

**Returns:**
Tool result dict with:
- **observation** (str): Formatted string with numbered documents (max 4096 chars):

    ```
    ### 1: [Document 1 content]
    ### 2: [Document 2 content]
    ### 3: [Document 3 content]
    ```

- **status** (str): "success"
- **name** (str): "dense_retrieve"
- **arguments** (dict): Input parameters used
- **info** (dict): Additional metadata

**Example:**

```python
# Using the tool wrapper
result = await dense_retrieve(query="artificial intelligence applications")
print(result["observation"])
```

**Implementation Features:**
- Global retriever instance (thread-safe)
- Lazy initialization on first call
- E5-base-v2 embeddings (768-dim)
- FAISS Flat index for similarity search
- Returns top-3 most relevant documents
- Automatic corpus and model loading from AGENT_DATA_DIR

## Technical Details

**Model:** intfloat/e5-base-v2 (768-dim embeddings)
**Corpus:** Wikipedia-18 (18M+ passages)
**Index:** FAISS Flat index
**Memory:** ~2-8GB RAM required
