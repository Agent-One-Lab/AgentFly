import re
from typing import Any, Dict, List

from ...decorator import tool

# Pattern for a previously added summary (from an earlier fold).
_PREVIOUS_SUMMARY_PATTERN = re.compile(r"\s*\[Known information\]:.*", re.DOTALL)


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

    Returns:
        A new list of turns representing the folded conversation.
    """
    first_assistant_idx = None
    for i, t in enumerate(turns):
        if t.get("role") == "assistant":
            first_assistant_idx = i
            break

    if first_assistant_idx is None:
        prefix = turns
    else:
        prefix = turns[:first_assistant_idx]

    context = prefix[-1]["content"][0]["text"]
    # Remove previous summary so we can add the new one without stacking.
    context = _PREVIOUS_SUMMARY_PATTERN.sub("", context).strip()

    if prefix[-1]["role"] == "user":
        prefix[-1]["content"] = [{"type": "text", "text": f"{context} {summary}"}]

    return prefix
