"""Console inspection of recorded trajectories, independent of agent state."""

from ..types import Trajectory


def print_trajectory(trajectory: Trajectory) -> None:
    """Print a trajectory's messages, keeping its context segments separate.

    Multiple segments have zero-based labels. Repeated context is printed as
    recorded, and empty or context-only segments are not filtered out. Text and
    tool calls are shown directly; images and other non-text content use
    placeholders instead of dumping their payloads. The trajectory is not changed,
    and runtime-only ``steps`` are not needed (JSON-restored trajectories work).
    """
    if not trajectory.segments:
        print("(no segments)")
        return

    for index, segment in enumerate(trajectory.segments):
        if len(trajectory.segments) > 1:
            print(f"Segment {index}:")
        if not segment.messages:
            print("(empty segment)")
            continue

        for message in segment.messages:
            content = message.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        text += part.get("text") or ""
                    elif part["type"] in ("image", "image_url"):
                        text += "[Image]"
                    else:
                        text += f"[{part['type']}]"
            print(f"{message['role']}: {text}")

            for tool_call in message.get("tool_calls") or []:
                function = tool_call["function"]
                print(
                    f"  Tool call: {function['name']} "
                    f"Arguments: {function.get('arguments', '')}"
                )
