import os
from .faiss_indexer import Indexer
import datasets
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from ...tool_base import tool
from ....__init__ import AGENT_DATA_DIR

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class DenseRetriever:
    def __init__(self, corpus_file: str, index_file: str):
        self.corpus = load_corpus(corpus_file)
        ids = [int(id) for id in self.corpus["id"]]
        self.indexer = Indexer(index_file=index_file, ids=ids)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')

    async def search(self, queries: list[str], top_k: int):
        batch_dict = self.tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"]).detach().numpy()

        score_ids_list = self.indexer.search(embeddings, top_k)
        ids_list = []
        for score_ids in score_ids_list:
            ids_list.append([score_id[1] for score_id in score_ids])

        docs_list = []
        for ids in ids_list:
            docs_list.append([self.corpus[id] for id in ids])

        return docs_list

GLOBAL_RETRIEVER = None

@tool(name="dense_retrieve", description="Use a dense retriever to retrieve documents from a corpus.", max_length=4096)
async def dense_retrieve(query: str):
    global AGENT_DATA_DIR
    if not query.startswith("query:"):
        query = "query: " + query
    global GLOBAL_RETRIEVER
    if GLOBAL_RETRIEVER is None:
        GLOBAL_RETRIEVER = DenseRetriever(corpus_file=os.path.join(AGENT_DATA_DIR, "search", "wiki-18.jsonl"), index_file=os.path.join(AGENT_DATA_DIR, "search", "e5_Flat.index"))
    doc_list = await GLOBAL_RETRIEVER.search(query, 3)
    doc_list = doc_list[0]
    content = ""
    for i, doc in enumerate(doc_list):
        content += f"### {i+1}: {doc['contents']}\n"
    return content

