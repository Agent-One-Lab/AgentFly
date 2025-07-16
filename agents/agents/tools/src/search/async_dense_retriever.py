import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import multiprocessing
import hashlib
import pickle
from collections import deque
try:
    from torch import Tensor, inference_mode
    import torch
    from torch.quantization import quantize_dynamic
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    Tensor = None
    inference_mode = None
    torch = None

import datasets
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    AutoTokenizer = None
    AutoModel = None

from .faiss_indexer import Indexer
from ...tool_base import tool
from ....__init__ import AGENT_DATA_DIR
import builtins
import numpy as np
import importlib
import threading

"""
This module provides a high-performance, asynchronous dense retriever for document retrieval tasks.

It leverages FAISS for efficient similarity search, sentence-transformers for generating
dense vector embeddings, and various optimization techniques including caching, quantization,
ONNX runtime, and asynchronous I/O to ensure fast and scalable performance.

Key Components:
- ``DenseRetriever``: The main class that encapsulates model loading, embedding, indexing, and searching logic.
- ``asyncdense_retrieve``: A tool-decorated asynchronous function that serves as a high-level API for performing retrievals.
- ``load_corpus``: A utility function for efficiently loading and caching document corpora from JSONL files.

Dependencies:
- PyTorch (torch)
- Transformers (transformers)
- Datasets (datasets)
- FAISS (faiss-cpu or faiss-gpu)
- ONNX Runtime (onnxruntime) - Optional, for CPU acceleration.
"""

def load_corpus(corpus_path: str):
    """Loads a document corpus from a JSONL file with caching.

    This function first checks for a pre-processed pickle cache ('.cache.pkl').
    If a valid cache exists and is newer than the source file, it is loaded
    directly. Otherwise, it parses the JSONL source file, builds a dictionary
    of documents, and saves it to the cache for future use.

    The function is robust to JSON parsing errors in individual lines and can
    fall back to the `datasets` library if the primary line-by-line reading fails.

    :param corpus_path: The file path to the corpus in JSONL format.
                        Each line should be a JSON object with at least an 'id'
                        and a 'contents' field.
    :type corpus_path: str
    :return: A dictionary mapping document IDs to document objects.
    :rtype: dict[int, dict]
    """
    cache_path = corpus_path + ".cache.pkl"
    
    if os.path.exists(cache_path):
        # Check if cache is newer than source
        if os.path.getmtime(cache_path) > os.path.getmtime(corpus_path):
            print(f"Loading corpus from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                print("Cache corrupted, rebuilding...")
    
    print(f"Building corpus dict from {corpus_path}")
    
    corpus_dict = {}
    try:
        import json
        with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print(f"Loaded {i} documents...")
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc_id = int(doc.get('id', i))
                    # Ensure 'contents' field exists
                    if 'contents' not in doc:
                        print(f"Warning: Document {doc_id} has no 'contents' field")
                        doc['contents'] = doc.get('text', doc.get('content', ''))
                    corpus_dict[doc_id] = doc
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i}: {e}")
                    continue
    except Exception as e:
        print(f"Line reading failed: {e}")
        print("Falling back to datasets loader...")
        corpus = datasets.load_dataset(
            "json",
            data_files=corpus_path,
            split="train",
            num_proc=1,
        )
        for doc in corpus:
            doc_id = int(doc.get('id', 0))
            if 'contents' not in doc:
                doc['contents'] = doc.get('text', doc.get('content', ''))
            corpus_dict[doc_id] = doc
    
    print(f"Loaded {len(corpus_dict)} documents")
    
    print(f"Saving corpus cache to {cache_path}")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(corpus_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return corpus_dict

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    """Performs average pooling on the last hidden states of a transformer model.

    This function computes a single vector representation for each item in a batch
    by averaging its token embeddings, taking the attention mask into account to
    ignore padding tokens.

    :param last_hidden_states: The output token embeddings from the model.
                               Shape: (batch_size, sequence_length, hidden_size)
    :type last_hidden_states: torch.Tensor
    :param attention_mask: The attention mask for the input tokens.
                           Shape: (batch_size, sequence_length)
    :type attention_mask: torch.Tensor
    :return: The pooled embeddings. Shape: (batch_size, hidden_size)
    :rtype: torch.Tensor
    """
    hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return hidden.sum(1) / attention_mask.sum(1)[..., None]

class DenseRetriever:
    """An optimized, asynchronous dense retriever.

    This class manages the entire retrieval pipeline: loading a transformer model
    and tokenizer, embedding queries into vectors, searching a FAISS index for
    nearest neighbors, and fetching the corresponding documents. It is designed
    for high throughput and low latency, employing several layers of caching,
    asynchronous execution, and hardware-specific optimizations (e.g., ONNX,
    quantization, FP16).

    The document corpus is lazy-loaded to speed up initialization.

    :ivar corpus_file: Path to the document corpus file.
    :vartype corpus_file: str
    :ivar indexer: The FAISS indexer instance for vector search.
    :vartype indexer: Indexer
    :ivar model: The pre-trained transformer model for embedding.
    :vartype model: transformers.AutoModel
    :ivar tokenizer: The tokenizer corresponding to the model.
    :vartype tokenizer: transformers.AutoTokenizer
    :ivar device: The PyTorch device (e.g., 'cuda' or 'cpu') the model is on.
    :vartype device: torch.device
    """
    _POOL = ThreadPoolExecutor(max_workers=20)  # Increase parallelism!

    def __init__(self, corpus_file: str, index_file: str):
        """Initializes the DenseRetriever.

        Sets up paths, loads the FAISS index, and initializes the transformer model
        and tokenizer. Applies device-specific optimizations like quantization for
        CPU or FP16 for GPU. It also attempts to load or create an ONNX model
        for accelerated CPU inference.

        :param corpus_file: The path to the JSONL corpus file.
        :type corpus_file: str
        :param index_file: The path to the FAISS index file.
        :type index_file: str
        """
        print("Initializing AsyncDenseRetriever...")
        
        # Store paths for lazy loading
        self.corpus_file = corpus_file
        self.corpus_dict = None  # Lazy load
        self._corpus_loading = False
        self._corpus_lock = threading.Lock()
        
        #small hot cache of recent documents in memory
        self._hot_doc_cache = {}  # doc_id -> doc
        self._hot_cache_size = 10000  # Keep 10k most recent docs
        
        print(f"Loading FAISS index from {index_file}...")
        
        try:
            import faiss
            self.indexer = Indexer(index_file=index_file, ids=None)
            
            # Load IDs separately
            ids_file = index_file + ".ids" 
            if os.path.exists(ids_file):
                print("Loading document IDs...")
                with open(ids_file, 'rb') as f:
                    ids = pickle.load(f)
                self.indexer.ids = ids
            else:
                # We'll need to load corpus to get IDs
                print("No cached IDs found, will load corpus...")
                self._ensure_corpus_loaded()
                ids = list(self.corpus_dict.keys())
                # Save for next time
                with open(ids_file, 'wb') as f:
                    pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.indexer.ids = ids
                
            print(f"FAISS index loaded: {self.indexer.index.ntotal} vectors")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            self._ensure_corpus_loaded()
            ids = list(self.corpus_dict.keys())
            self.indexer = Indexer(index_file=index_file, ids=ids)

        model_name = "intfloat/e5-base-v2"
        print(f"Using {model_name} (768D embeddings)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"Loading tokenizer '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")

        print(f"Loading model '{model_name}'...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Apply optimizations based on device
        if self.device.type == "cpu":
            # Use all CPU cores efficiently
            torch.set_num_threads(multiprocessing.cpu_count())
            # Apply dynamic quantization
            self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            print(f"CPU optimizations applied: {multiprocessing.cpu_count()} threads, INT8 quantization")
        else:
            # GPU optimizations
            self.model.half()  # Use FP16 for faster inference
            torch.backends.cudnn.benchmark = True
            print("GPU optimizations applied: FP16 mode")
            
        # Compile if available (PyTorch 2.0+)
        if hasattr(torch, "compile") and os.getenv("TORCH_COMPILE", "1") == "1":
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            
        self.model.eval()
        print("Model loaded.")
        
        self._exact_cache = {}  # Exact string match cache
        self._embedding_cache = {}  # Cache computed embeddings
        self._cache_hits = 0
        self._total_queries = 0
        
        self._prefetch_queue = deque(maxlen=10)
        self._prefetch_embeddings = {}
        
        #ONNX if available
        self._setup_onnx()
        
        print("AsyncDenseRetriever initialization complete.")

    def _setup_onnx(self):
        """Initializes an ONNX session for accelerated CPU inference.

        If an ONNX model file does not exist, it may be exported from the
        PyTorch model if the ``EXPORT_ONNX`` environment variable is set.
        If ONNX Runtime is not available or setup fails, it falls back to PyTorch.
        """
        self.onnx_session = None
        try:
            import onnxruntime as ort
            onnx_path = os.path.join(os.path.dirname(__file__), f"retriever_model.onnx")
            
            if not os.path.exists(onnx_path) and os.getenv("EXPORT_ONNX", "0") == "1":
                print("Exporting model to ONNX...")
                dummy_input = self.tokenizer("dummy", return_tensors="pt", padding=True, truncation=True, max_length=512)
                dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                
                torch.onnx.export(
                    self.model,
                    (dummy_input['input_ids'], dummy_input['attention_mask']),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['last_hidden_state'],
                    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'},
                                  'attention_mask': {0: 'batch', 1: 'sequence'},
                                  'last_hidden_state': {0: 'batch', 1: 'sequence'}},
                    opset_version=14
                )
                print("ONNX export complete.")
            
            if os.path.exists(onnx_path):
                print("Loading ONNX model...")
                providers = ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                print("ONNX model loaded.")
        except Exception as e:
            print(f"ONNX setup failed (will use PyTorch): {e}")

    def _normalize_query(self, query: str) -> str:
        """Normalizes a query string for consistent caching.

        Performs lowercasing, trims whitespace, and collapses multiple spaces.

        :param query: The raw query string.
        :type query: str
        :return: The normalized query string.
        :rtype: str
        """
        # Remove extra whitespace, lowercase, strip punctuation at ends
        normalized = ' '.join(query.lower().strip().split())
        return normalized

    def _get_query_hash(self, query: str) -> str:
        """Computes a fast MD5 hash of a query for caching.

        :param query: The query string to hash.
        :type query: str
        :return: The hex digest of the MD5 hash.
        :rtype: str
        """
        return hashlib.md5(query.encode()).hexdigest()

    def _embed_sync(self, queries: list[str]):
        """Synchronously computes embeddings for a list of queries.

        Checks a local cache for pre-computed embeddings before running inference.
        Uses the ONNX session if available for faster CPU performance, otherwise
        falls back to the PyTorch model. Caches new embeddings.

        :param queries: A list of query strings to embed.
        :type queries: list[str]
        :return: A numpy array of computed embeddings.
        :rtype: numpy.ndarray
        """
        uncached_queries = []
        uncached_indices = []
        cached_embeddings = []
        
        for i, q in enumerate(queries):
            q_hash = self._get_query_hash(q)
            if q_hash in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[q_hash]))
            else:
                uncached_queries.append(q)
                uncached_indices.append(i)
        
        if not uncached_queries:
            # All queries were cached
            embeddings = np.zeros((len(queries), cached_embeddings[0][1].shape[0]))
            for idx, emb in cached_embeddings:
                embeddings[idx] = emb
            return embeddings
        
        # Compute only uncached embeddings
        batch = self.tokenizer(
            uncached_queries, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        
        if self.onnx_session is not None:
            # Use ONNX for inference (2-4x faster on CPU)
            ort_inputs = {
                'input_ids': batch['input_ids'].numpy(),
                'attention_mask': batch['attention_mask'].numpy()
            }
            outputs = self.onnx_session.run(None, ort_inputs)
            last_hidden_state = torch.from_numpy(outputs[0])
            embs = average_pool(last_hidden_state, batch["attention_mask"]).numpy()
        else:
            # Fallback to PyTorch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.inference_mode():
                out = self.model(**batch)
                embs = average_pool(out.last_hidden_state, batch["attention_mask"]).cpu().numpy()
        
        # Cache the new embeddings
        for i, (q, emb) in enumerate(zip(uncached_queries, embs)):
            q_hash = self._get_query_hash(q)
            self._embedding_cache[q_hash] = emb
            # Limit cache size
            if len(self._embedding_cache) > 10000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
        
        # Combine cached and new embeddings
        all_embeddings = np.zeros((len(queries), embs.shape[1]))
        for idx, emb in cached_embeddings:
            all_embeddings[idx] = emb
        for i, idx in enumerate(uncached_indices):
            all_embeddings[idx] = embs[i]
        
        return all_embeddings

    def _faiss_sync(self, embs, k: int):
        """Synchronously performs a search on the FAISS index.

        :param embs: A numpy array of query embeddings.
        :type embs: numpy.ndarray
        :param k: The number of nearest neighbors to retrieve.
        :type k: int
        :return: A list of lists, where each inner list contains (score, id)
                 tuples for a query.
        :rtype: list[list[tuple[float, int]]]
        """
        return self.indexer.search(embs, k)

    @lru_cache(maxsize=8192)
    def _search_sync(self, query: str, top_k: int):
        """Synchronously performs a full search for a single query.

        This method is decorated with ``@lru_cache`` to provide a fast in-memory
        cache for identical (query, top_k) pairs. It handles embedding,
        FAISS search, and document lookup.

        :param query: The query string.
        :type query: str
        :param top_k: The number of documents to retrieve.
        :type top_k: int
        :return: A list of retrieved document dictionaries.
        :rtype: list[dict]
        """
        cache_key = f"{query}:{top_k}"
        if cache_key in self._exact_cache:
            self._cache_hits += 1
            return self._exact_cache[cache_key]
        
        self._total_queries += 1
        
        q_hash = self._get_query_hash(query)
        if q_hash in self._prefetch_embeddings:
            embs = self._prefetch_embeddings[q_hash].reshape(1, -1)
            del self._prefetch_embeddings[q_hash]  # Use once
        else:
            embs = self._embed_sync([query])
        
        score_ids_list = self._faiss_sync(embs, top_k)
        ids = [score_id[1] for score_id in score_ids_list[0]]
        
        if self.corpus_dict is None:
            docs = self._load_documents_by_ids(ids)
        else:
            docs = [self.corpus_dict[doc_id] for doc_id in ids]
        
        self._exact_cache[cache_key] = docs
        if len(self._exact_cache) > 5000:
            oldest = next(iter(self._exact_cache))
            del self._exact_cache[oldest]
        
        return docs

    def _load_documents_by_ids(self, doc_ids):
        """Loads document content for a given list of document IDs.

        Features a "hot cache" to keep recently accessed documents in memory,
        reducing redundant lookups in the main corpus dictionary, which might
        not be fully loaded. Triggers the main corpus load if necessary.

        :param doc_ids: A list of document IDs to retrieve.
        :type doc_ids: list[int]
        :return: A list of document dictionaries corresponding to the IDs.
        :rtype: list[dict]
        """
        docs = []
        missing_ids = []
        
        for doc_id in doc_ids:
            if doc_id in self._hot_doc_cache:
                docs.append(self._hot_doc_cache[doc_id])
            else:
                missing_ids.append(doc_id)
                docs.append(None) 
        
        if missing_ids:
            self._ensure_corpus_loaded()
            
            for i, doc_id in enumerate(doc_ids):
                if docs[i] is None: 
                    doc = self.corpus_dict[doc_id]
                    docs[i] = doc
                    
                    self._hot_doc_cache[doc_id] = doc
                    
                    if len(self._hot_doc_cache) > self._hot_cache_size:
                        oldest_key = next(iter(self._hot_doc_cache))
                        del self._hot_doc_cache[oldest_key]
        
        return docs

    async def search(self, queries: list[str], top_k: int):
        """Asynchronously searches for documents matching a list of queries.

        This is the main public entry point for retrieval. It normalizes queries
        and offloads the synchronous, CPU/GPU-bound work (embedding and searching)
        to a thread pool to avoid blocking the asyncio event loop. It handles
        both single and batch queries efficiently.

        :param queries: A list of query strings.
        :type queries: list[str]
        :param top_k: The number of documents to return for each query.
        :type top_k: int
        :return: A list of lists of document dictionaries. Each inner list
                 corresponds to a query.
        :rtype: list[list[dict]]
        """
        loop = asyncio.get_running_loop()

        queries = [self._normalize_query(q) for q in queries]

        if len(queries) == 1:
            docs = await loop.run_in_executor(self._POOL, self._search_sync, queries[0], top_k)
            return [docs]

        # Original batch path, now with optimizations
        embs = await loop.run_in_executor(self._POOL, self._embed_sync, queries)
        score_ids_list = await loop.run_in_executor(self._POOL, self._faiss_sync, embs, top_k)

        docs_list = [
            [self.corpus_dict[idx] for _, idx in score_ids] for score_ids in score_ids_list
        ]
        return docs_list

    def _load_corpus_thread(self, corpus_file):
        """Target function to load the corpus in a background thread.

        Calls the `load_corpus` utility and stores the result in the instance's
        `corpus_dict` attribute.

        :param corpus_file: Path to the corpus file.
        :type corpus_file: str
        """
        print(f"Loading corpus in background from {corpus_file}...")
        self.corpus_dict = load_corpus(corpus_file)
        print(f"Background corpus load complete: {len(self.corpus_dict)} documents")

    def _ensure_corpus_loaded(self):
        """Thread-safely triggers lazy loading of the corpus.

        If the corpus is not already loaded or being loaded, this method
        spawns a background thread to load it. A lock prevents multiple threads
        from attempting to load the corpus simultaneously.
        """
        with self._corpus_lock:
            if not self._corpus_loading:
                self._corpus_loading = True
                self._load_corpus_thread(self.corpus_file)

GLOBAL_RETRIEVER = None

@tool(
    name="asyncdense_retrieve",
    description="Use a dense retriever to retrieve documents from a corpus.",
    max_length=4096,
)
async def asyncdense_retrieve(query: str):
    global GLOBAL_RETRIEVER, AGENT_DATA_DIR

    if not query.startswith("query:"):
        query = "query: " + query

    # build once per process/rank
    if GLOBAL_RETRIEVER is None:
        GLOBAL_RETRIEVER = DenseRetriever(
            corpus_file=os.path.join(
                AGENT_DATA_DIR, "search", "wiki-18.jsonl"
            ),
            index_file=os.path.join(
                AGENT_DATA_DIR, "search", "e5_Flat.index"
            ),
        )

    docs = (await GLOBAL_RETRIEVER.search([query], top_k=3))[0]

    # small optmization building the string x)
    return "\n".join(f"### {i + 1}: {doc['contents']}" for i, doc in enumerate(docs)) + "\n"


    