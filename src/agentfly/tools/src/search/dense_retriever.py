"""Dense Wikipedia retrieval, simple variant: a synchronous-model E5 + FAISS retriever and the ``dense_retrieve`` tool.

Same data as :mod:`.async_dense_retriever` (wiki-18 corpus, E5 embeddings, FAISS index) without
its background loading and caching.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ...decorator import tool
from .faiss_indexer import Indexer

# torch / transformers / datasets are imported where a retriever is built (see async_dense_retriever).
if TYPE_CHECKING:
    from torch import Tensor


def load_corpus(corpus_path: str):
    import datasets

    corpus = datasets.load_dataset(
        "json", data_files=corpus_path, split="train", num_proc=4
    )
    return corpus


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class DenseRetriever:
    def __init__(self, corpus_file: str, index_file: str):
        self.corpus = load_corpus(corpus_file)
        ids = [int(id) for id in self.corpus["id"]]
        self.indexer = Indexer(index_file=index_file, ids=ids)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
        self.model = AutoModel.from_pretrained("intfloat/e5-base-v2")

    async def search(self, queries: list[str], top_k: int):
        batch_dict = self.tokenizer(
            queries, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        outputs = self.model(**batch_dict)
        embeddings = (
            average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            .detach()
            .numpy()
        )

        score_ids_list = self.indexer.search(embeddings, top_k)
        ids_list = []
        for score_ids in score_ids_list:
            ids_list.append([score_id[1] for score_id in score_ids])

        docs_list = []
        for ids in ids_list:
            docs_list.append([self.corpus[id] for id in ids])

        return docs_list


GLOBAL_RETRIEVER = None


@tool(
    name="dense_retrieve",
    description="Use a dense retriever to retrieve documents from a corpus.",
    max_length=4096,
)
async def dense_retrieve(query: str):
    global AF_CACHE_DIR
    if not query.startswith("query:"):
        query = "query: " + query
    global GLOBAL_RETRIEVER
    if GLOBAL_RETRIEVER is None:
        from ....utils.download import ensure_tool_data

        ensure_tool_data("async_dense_retrieve")  # same corpus + index, if missing (approval-gated)
        GLOBAL_RETRIEVER = DenseRetriever(
            corpus_file=os.path.join(
                AF_CACHE_DIR, "data", "search", "wiki-18.jsonl"
            ),
            index_file=os.path.join(AF_CACHE_DIR, "data", "search", "e5_Flat.index"),
        )
    doc_list = await GLOBAL_RETRIEVER.search(query, 3)
    doc_list = doc_list[0]
    content = ""
    for i, doc in enumerate(doc_list):
        content += f"### {i + 1}: {doc['contents']}\n"
    return content
