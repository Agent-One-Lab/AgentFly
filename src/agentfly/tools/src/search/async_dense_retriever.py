import asyncio
import logging
import multiprocessing
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import datasets
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .... import AGENT_CACHE_DIR
from ...decorator import tool
from .faiss_indexer import Indexer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Computes mean pooling for embeddings."""
    hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(1) / attention_mask.sum(1)[..., None]


class DenseRetriever:
    """Optimized, thread-safe asynchronous dense retriever."""

    _POOL = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)

    def __init__(self, corpus_file: str, index_file: str, max_concurrent: int = 4):
        """
        DenseRetriever is a class that provides a thread-safe asynchronous dense retriever.
        """
        self.corpus_file = corpus_file
        self.corpus_dict = None
        self._corpus_loading_event = asyncio.Event()

        # Caching layers
        self._hot_doc_cache = {}
        self._embedding_cache = {}
        self._max_cache = 10000

        # 1. Load Model & Tokenizer
        model_name = "intfloat/e5-base-v2"
        # Use CPU for now
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )

        # 2. Apply Hardware Optimizations
        if self.device.type == "cpu":
            # self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            pass
        else:
            self.model.half()
        self.model.eval()

        # 3. Load Index and IDs
        self.indexer = Indexer(index_file=index_file, use_gpu=True, num_gpus=8)
        ids_file = index_file + ".ids"
        if os.path.exists(ids_file):
            with open(ids_file, "rb") as f:
                self.indexer.ids = pickle.load(f)

        loop = asyncio.get_event_loop()

        # Pass the loop as an argument to the background thread
        threading.Thread(
            target=self._load_corpus_task, args=(loop,), daemon=True
        ).start()

        # Limit the number of concurrent heavy-compute tasks
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Use more workers in the pool than the semaphore limit to handle doc I/O
        self._POOL = ThreadPoolExecutor(max_workers=20)

    def _load_corpus_task(self, loop):
        """Internal task to load corpus without blocking main thread."""
        # This prevents the .cache.pkl file from being used with the new loading mechanism
        # In a real scenario, you might want more robust cache invalidation

        corpus = datasets.load_dataset(
            "json",
            data_files=self.corpus_file,
            split="train",
            num_proc=4,
        )

        temp_dict = {}
        for doc in corpus:
            doc_id = int(doc.get("id"))
            # Ensure contents field is populated
            if doc.get("contents") is None:
                doc["contents"] = doc.get("text", "")
            temp_dict[doc_id] = doc
        self.corpus_dict = temp_dict

        loop.call_soon_threadsafe(self._corpus_loading_event.set)

    def _get_embeddings(self, queries: list[str]):
        """Computes embeddings with E5 prefixing and caching."""
        # Ensure queries have E5 prefix
        prefixed = [f"query: {q}" if not q.startswith("query:") else q for q in queries]

        batch = self.tokenizer(
            prefixed, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.inference_mode():
            out = self.model(**batch)
            return (
                average_pool(out.last_hidden_state, batch["attention_mask"])
                .cpu()
                .numpy()
            )

    def _get_docs_by_ids(self, doc_ids: list[int]):
        """Fetch docs from dictionary (Wait is now handled in search())."""
        results = []
        for d_id in doc_ids:
            # Check hot cache
            if d_id in self._hot_doc_cache:
                results.append(self._hot_doc_cache[d_id])
            else:
                # Safe access to the dictionary loaded by the background thread
                doc = self.corpus_dict.get(d_id, {"contents": "[Error: Doc not found]"})
                self._hot_doc_cache[d_id] = doc
                results.append(doc)
        return results

    async def search(self, queries: list[str], top_k: int = 3):
        """Safe asynchronous search with concurrency throttling."""

        # NEW: Wait for the corpus to be ready BEFORE doing any retrieval work
        await self._corpus_loading_event.wait()

        loop = asyncio.get_running_loop()

        async with self._semaphore:
            # 1. Embed
            embeddings = await loop.run_in_executor(
                self._POOL, self._get_embeddings, queries
            )

            # 2. FAISS Search
            score_ids_list = await loop.run_in_executor(
                self._POOL, self.indexer.search, embeddings, top_k
            )
            logger.debug(f"Retrieved score_ids_list: {score_ids_list}")

        # 3. Doc retrieval
        all_results = []
        for score_ids in score_ids_list:
            ids = [sid[1] for sid in score_ids]
            docs = await loop.run_in_executor(self._POOL, self._get_docs_by_ids, ids)
            all_results.append(docs)

        return all_results


GLOBAL_RETRIEVER = None


@tool(
    name="async_dense_retrieve", description="Retrieve wiki documents.", max_length=8192
)
async def async_dense_retrieve(query: str):
    global GLOBAL_RETRIEVER
    if GLOBAL_RETRIEVER is None:
        GLOBAL_RETRIEVER = DenseRetriever(
            corpus_file=os.path.join(
                AGENT_CACHE_DIR, "data", "search", "wiki-18.jsonl"
            ),
            index_file=os.path.join(AGENT_CACHE_DIR, "data", "search", "e5_Flat.index"),
        )

    # Simplified call
    results = await GLOBAL_RETRIEVER.search([query], top_k=3)
    docs = results[0]

    return "\n".join(f"Doc {i + 1}: {d['contents']}" for i, d in enumerate(docs)) + "\n"
