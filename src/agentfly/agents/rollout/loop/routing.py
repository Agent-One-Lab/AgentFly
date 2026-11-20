"""What the rollout engine reports about its own sampling, carried to training: the log-probability of every
sampled token, and (mixture-of-experts models) the experts chosen at every token position.

Why. The trainer scores a sampled token again with its own forward pass. For a routed model the two disagree
unless the trainer picks the same experts: the choice is a hard top-k, and rounding differences flip it. So the
engine's choices are recorded per position and replayed in training ("rollout routing replay"); the engine's
log-probabilities are kept to measure the remaining gap every step.

Layout, per trajectory. Rows are indexed by token position in the sequence the engine ran: row ``p`` holds the
choices made at position ``p`` (the computation that predicts token ``p + 1``), so ``N`` tokens have ``N - 1``
rows, ``[layers, k]`` each, one byte per expert id. Context positions (system, user, tool results) are covered
too: their states are what later tokens attend to.

Several turns. Each generation request returns only the rows the caller does not hold yet
(``routed_experts_prompt_start``): the context added since the last turn, then the turn's own sampled tokens. The
chunk is kept on the assistant message of that turn (:class:`RoutedRows`), next to its sampled ``token_ids``.
Rows of an earlier turn are never replaced by a later request, which may compute a position again with other
choices: a sampled token is scored with the choices it was sampled under.

A chunk is only valid for the token ids it was recorded for. Before a request, and again when the training row is
built, the ids are compared (a digest of the prompt, and the sampled ids themselves); on a difference the request
starts over from position 0, and a training row whose ids do not match its record is left out of the loss.
"""

import hashlib
from array import array
from typing import Any, Dict, List, Optional

import numpy as np

ROUTES = "routed_experts"  # assistant message key: RoutedRows
LOGPROBS = "rollout_log_probs"  # assistant message key: the engine's log-probability of each sampled token
PAD = 255  # no recorded choice


def digest(token_ids) -> str:
    """A short fingerprint of a token-id sequence."""
    return hashlib.blake2b(array("I", [int(t) for t in token_ids]).tobytes(), digest_size=12).hexdigest()


class RoutedRows:
    """The rows one generation request returned: positions ``start`` .. ``start + len(rows) - 1`` of the sequence,
    recorded for a prompt of ``prompt_len`` tokens with fingerprint ``prompt_digest`` followed by the sampled
    tokens. Never modified in place, so copies of the conversation share it."""

    __slots__ = ("rows", "start", "prompt_len", "prompt_digest")

    def __init__(self, rows: np.ndarray, start: int, prompt_len: int, prompt_digest: str):
        self.rows = rows
        self.start = int(start)
        self.prompt_len = int(prompt_len)
        self.prompt_digest = prompt_digest

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f"RoutedRows(rows={tuple(self.rows.shape)}, start={self.start}, prompt_len={self.prompt_len})"

    def cut_to(self, sampled: int) -> "RoutedRows":
        """The rows for a turn of which only the first ``sampled`` tokens are kept."""
        keep = (self.prompt_len - self.start) + max(sampled - 1, 0)
        return self if keep >= len(self.rows) else RoutedRows(self.rows[:keep], self.start, self.prompt_len, self.prompt_digest)


def held_rows(messages: List[Dict[str, Any]], prompt_ids: List[int]) -> int:
    """How many rows the conversation already holds for ``prompt_ids``, i.e. where the next request starts.

    The last recorded turn must be a prefix of the prompt, token for token; otherwise 0 (start over)."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        routes, sampled = msg.get(ROUTES), msg.get("token_ids")
        if not isinstance(routes, RoutedRows) or sampled is None:
            return 0
        end = routes.prompt_len + len(sampled)
        if (
            routes.start + len(routes.rows) != end - 1
            or len(prompt_ids) < end
            or [int(t) for t in prompt_ids[routes.prompt_len : end]] != [int(t) for t in sampled]
            or digest(prompt_ids[: routes.prompt_len]) != routes.prompt_digest
        ):
            return 0
        return end - 1
    return 0


def attach_rollout_signals(messages, responses) -> None:
    """Put the engine's log-probabilities and routed-expert rows (when the backend returned them) on each new
    assistant message."""
    if not isinstance(responses, dict):
        return
    for key, values in ((LOGPROBS, responses.get("response_logprobs")), (ROUTES, responses.get("routed_experts"))):
        for msg, value in zip(messages or [], values or []):
            if isinstance(msg, dict) and value is not None:
                msg[key] = value


def sync_rollout_signals(messages) -> None:
    """Keep the signals in step with the sampled ``token_ids`` after the ids were cut or dropped: both describe
    the sampled tokens one for one, so without ids there is nothing to align them to."""
    for msg in messages or []:
        if not isinstance(msg, dict) or (LOGPROBS not in msg and ROUTES not in msg):
            continue
        ids = msg.get("token_ids")
        if ids is None:
            msg.pop(LOGPROBS, None)
            msg.pop(ROUTES, None)
            continue
        if LOGPROBS in msg:
            msg[LOGPROBS] = list(msg[LOGPROBS][: len(ids)])
        if isinstance(msg.get(ROUTES), RoutedRows):
            msg[ROUTES] = msg[ROUTES].cut_to(len(ids))


def drop_rollout_signals(messages) -> None:
    for msg in messages or []:
        if isinstance(msg, dict):
            msg.pop(LOGPROBS, None)
            msg.pop(ROUTES, None)


def _sampled_spans(action_mask: np.ndarray) -> List[tuple]:
    """``[start, end)`` of each run of trained tokens in a row (one run per assistant turn)."""
    padded = np.concatenate([[0], action_mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def assemble_row(messages: List[Dict[str, Any]], input_ids: np.ndarray, action_mask: np.ndarray) -> Dict[str, Any]:
    """One training row's record, aligned with its ``input_ids``.

    Returns ``log_probs`` (float32 ``[L]``, NaN where the engine reported none; the value for the token at position
    ``p`` is at ``p``), ``routes`` (uint8 ``[L, layers, k]`` filled with ``PAD``, or None when no turn carries
    rows) and ``status``: ``"ok"``, ``"none"`` (no turn carries rows) or the reason the record does not fit the row.
    """
    length = len(input_ids)
    log_probs = np.full(length, np.nan, dtype=np.float32)
    turns = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("token_ids") is not None]
    spans = _sampled_spans(action_mask)
    chunks = [m.get(ROUTES) for m in turns if isinstance(m.get(ROUTES), RoutedRows)]
    routes = None
    if chunks:
        routes = np.full((length, *chunks[0].rows.shape[1:]), PAD, dtype=np.uint8)
    status = "ok" if chunks else "none"
    expected_start = 0
    for msg in turns:
        sampled = [int(t) for t in msg["token_ids"]]
        chunk = msg.get(ROUTES)
        position = chunk.prompt_len if isinstance(chunk, RoutedRows) else None
        if position is None:
            # No rows for this turn: find its span by its ids, for the log-probabilities only.
            position = next((b for b, _ in spans if input_ids[b : b + len(sampled)].tolist() == sampled), None)
            if chunks:
                status = "turn_without_rows"
        if position is None or input_ids[position : position + len(sampled)].tolist() != sampled:
            status = "sampled_ids_moved" if chunks else status
            continue
        if msg.get(LOGPROBS) is not None:
            values = np.asarray(msg[LOGPROBS], dtype=np.float32)[: len(sampled)]
            log_probs[position : position + len(values)] = values
        if not isinstance(chunk, RoutedRows):
            continue
        if chunk.start == 0:
            expected_start = 0  # this request started over: earlier rows describe another sequence
            routes[:] = PAD
        if chunk.start != expected_start:
            status = "rows_not_contiguous"
        elif digest(input_ids[: chunk.prompt_len]) != chunk.prompt_digest:
            status = "prompt_ids_differ"
        elif chunk.start + len(chunk.rows) > length:
            status = "rows_beyond_row"
        else:
            routes[chunk.start : chunk.start + len(chunk.rows)] = chunk.rows
        expected_start = chunk.start + len(chunk.rows)
    return {"log_probs": log_probs, "routes": routes if status == "ok" else None, "status": status}


def assemble_batch(conversations: List[List[Dict[str, Any]]], inputs: Dict[str, Any], rows_total: int) -> Optional[Dict[str, Any]]:
    """The batch's record: tensors aligned with ``inputs["input_ids"]`` for the first ``len(conversations)`` rows
    (the others, divisor padding, carry none), plus counts. None when no row carries anything.

    ``tensors``: ``engine_log_probs`` (float32 ``[B, L]``, NaN where absent) and, when any row has rows,
    ``routed_experts`` (uint8 ``[B, L, layers, k]``, ``PAD`` where absent). ``mismatched``: the indices of rows
    that carry routed-expert rows which do not fit the row's token ids; the caller leaves them out of the loss.
    """
    import torch  # lazy: heavy import

    input_ids = inputs["input_ids"].cpu().numpy()
    action_mask = inputs["action_mask"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()
    results = []
    for i, messages in enumerate(conversations):
        n = int(attention_mask[i].sum())
        if n and not attention_mask[i, :n].all():
            raise ValueError("rollout record: training rows must be right-padded")
        results.append(assemble_row(messages, input_ids[i, :n], action_mask[i, :n]))
    has_routes = any(r["status"] != "none" for r in results)
    has_log_probs = any(np.isfinite(r["log_probs"]).any() for r in results)
    if not has_routes and not has_log_probs:
        return None
    width = input_ids.shape[1]
    tensors = {}
    log_probs = np.full((rows_total, width), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        log_probs[i, : len(r["log_probs"])] = r["log_probs"]
    tensors["engine_log_probs"] = torch.from_numpy(log_probs)
    statuses: Dict[str, int] = {}
    mismatched = []
    if has_routes:
        shape = next(r["routes"].shape[1:] for r in results if r["routes"] is not None) if any(
            r["routes"] is not None for r in results
        ) else None
        if shape is not None:
            routes = np.full((rows_total, width, *shape), PAD, dtype=np.uint8)
            for i, r in enumerate(results):
                if r["routes"] is not None:
                    routes[i, : len(r["routes"])] = r["routes"]
            tensors["routed_experts"] = torch.from_numpy(routes)
        for i, r in enumerate(results):
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            if r["status"] not in ("ok", "none"):
                mismatched.append(i)
    return {"tensors": tensors, "mismatched": mismatched, "statuses": statuses}
