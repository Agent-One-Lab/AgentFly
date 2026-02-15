"""
Client tool that calls a separately deployed dense retriever API via HTTP.
Use this when the retriever is run as a service (e.g. retriever_server.py) so that
the retriever process can use GPU while the trainer/agent process does not need to.
"""
import os
import logging

import httpx
from ...decorator import tool

logger = logging.getLogger(__name__)

# Base URL of the retriever API (set when deploying the server)
DEFAULT_RETRIEVER_API_URL = os.environ.get("RETRIEVER_API_URL", "http://localhost:8765")
SEARCH_PATH = "/search"
TIMEOUT_SEC = 300.0


def _format_retriever_error(exc: Exception, response: httpx.Response | None = None) -> str:
    """Build a non-empty error message from exception and optional response body."""
    parts = []
    if response is not None:
        parts.append(f"status={response.status_code}")
        try:
            body = response.json()
            if isinstance(body, dict) and "detail" in body:
                parts.append(str(body["detail"]))
            else:
                text = response.text
                if text and text.strip():
                    parts.append(text.strip()[:500])
        except Exception:
            if response.text and response.text.strip():
                parts.append(response.text.strip()[:500])
    msg = str(exc).strip() if str(exc).strip() else f"{type(exc).__name__}"
    parts.append(msg)
    return "; ".join(parts)


@tool(
    name="async_dense_retrieve_api",
    description="Retrieve wiki documents via the deployed retriever API. Same as async_dense_retrieve but calls a remote service.",
    max_length=8192,
)
async def async_dense_retrieve_api(query: str):
    """Call the dense retriever API and return concatenated doc contents. Set RETRIEVER_API_URL to point to the server (default http://localhost:8765)."""
    url = DEFAULT_RETRIEVER_API_URL.rstrip("/") + SEARCH_PATH
    payload = {"query": query, "top_k": 3}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning("Retriever API request failed: %s", e)
        err_msg = _format_retriever_error(e, getattr(e, "response", None))
        return f"[Retriever API error: {err_msg}]"
    except httpx.HTTPError as e:
        logger.warning("Retriever API request failed: %s", e)
        err_msg = _format_retriever_error(e)
        return f"[Retriever API error: {err_msg}]"
    results = data.get("results") or []
    return "\n".join(f"Doc {i+1}: {r.get('contents', '')}" for i, r in enumerate(results)) + "\n"
