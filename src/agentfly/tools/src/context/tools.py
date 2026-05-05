import re
from typing import Any, Dict, List
from copy import deepcopy
from ...decorator import tool

# Pattern for a previously added summary (from an earlier fold).
_PREVIOUS_SUMMARY_PATTERN = re.compile(r"\s*\[Previous Summary\]:.*", re.DOTALL)


@tool(name="summarize", description="Summarize the history.")
def summarize(summary: str):
    """
    Summarize the history.

    Args:
        summary (str): The summary of the history.
    """
    return f"[Previous Summary]: {summary}"


def fold_messages_with_summarize(
    turns: List[Dict[str, Any]],
    summary: str,
    keep_previous_summary: bool = False,
) -> List[Dict[str, Any]]:
    """
    Folding logic for the summarize context tool.

    We assume:
    - System + user messages before the first assistant turn are kept verbatim.
    - All turns from the first assistant turn onwards are replaced by a single
      assistant summary message.

    Args:
        turns: Full list of message turns (dicts with 'role' and 'content').
        summary: Summary string returned by the summarize tool.
        keep_previous_summary: If True, do not remove any previous summary content from
            the preserved prefix; the new summary will be appended alongside existing
            summary text. If False (default), remove any previous summary so only the
            latest summary remains.

    Returns:
        A new list of turns representing the folded conversation.
    """
    first_assistant_idx = None
    for i, t in enumerate(turns):
        if t.get("role") == "assistant":
            first_assistant_idx = i
            break

    if first_assistant_idx is None:
        prefix = deepcopy(turns)
    else:
        prefix = deepcopy(turns[:first_assistant_idx])

    context = prefix[-1]["content"][0]["text"]
    if not keep_previous_summary:
        # Remove previous summary so we can add the new one without stacking.
        context = _PREVIOUS_SUMMARY_PATTERN.sub("", context).strip()

    if prefix[-1]["role"] == "user":
        prefix[-1]["content"] = [{"type": "text", "text": f"{context} {summary}"}]

    return prefix
