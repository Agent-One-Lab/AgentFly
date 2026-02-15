"""
Standalone HTTP API server for the dense retriever. Deploy this separately so that
the retriever runs in a process with GPU access; clients use async_dense_retrieve_api
to call this API via HTTP.

Run:
  python -m agentfly.tools.src.search.retriever_server
  # or with env:
  RETRIEVER_CORPUS_FILE=... RETRIEVER_INDEX_FILE=... RETRIEVER_HOST=0.0.0.0 RETRIEVER_PORT=8765 python -m agentfly.tools.src.search.retriever_server
"""
import os
import logging
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .... import AGENT_CACHE_DIR
from .async_dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)

# Config from env (same defaults as async_dense_retrieve tool)
DEFAULT_CORPUS = os.path.join(AGENT_CACHE_DIR, "data", "search", "wiki-18.jsonl")
DEFAULT_INDEX = os.path.join(AGENT_CACHE_DIR, "data", "search", "e5_Flat.index")

app = FastAPI(title="Dense Retriever API", version="1.0")
retriever: DenseRetriever | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(3, ge=1, le=100, description="Number of documents to return")


class DocResult(BaseModel):
    contents: str


class SearchResponse(BaseModel):
    results: list[DocResult]


@app.on_event("startup")
async def startup():
    global retriever
    corpus_file = os.environ.get("RETRIEVER_CORPUS_FILE", DEFAULT_CORPUS)
    index_file = os.environ.get("RETRIEVER_INDEX_FILE", DEFAULT_INDEX)
    logger.info("Loading retriever: corpus=%s index=%s", corpus_file, index_file)
    retriever = DenseRetriever(corpus_file=corpus_file, index_file=index_file, max_concurrent=8)
    logger.info("Retriever ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "retriever_loaded": retriever is not None}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    try:
        results = await retriever.search([req.query], top_k=req.top_k)
        docs = results[0]
        return SearchResponse(results=[DocResult(contents=d.get("contents", "")) for d in docs])
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    host = os.environ.get("RETRIEVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RETRIEVER_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
