"""Dataset rows become agent inputs the same way in training and in ``agent.run``."""

import io
import json

import pytest
from PIL import Image

from agentfly.agents.utils.messages import MessagesList, MessagesValidationError, row_to_messages


def png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


ROWS = [
    {"question": "What is 2+2?", "answer": "4"},
    {"prompt": "Solve for x", "answer": "1"},
    {"instruction": "Do it", "task_id": 7},
    {"problem": "Prove it", "level": 3},
    {"question": "Q", "messages": [{"role": "user", "content": "own messages"}], "task_id": "t"},
]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: "+".join(sorted(r)))
def test_row_fields(row):
    item = row_to_messages(row)
    key = next(k for k in ("prompt", "question", "instruction", "problem") if k in row)
    assert item["question"] == row[key]
    assert {k: v for k, v in item.items() if k not in ("messages", "question")} == {
        k: v for k, v in row.items() if k not in ("messages", "question")
    }
    if "messages" in row:
        assert item["messages"] == row["messages"]  # a row's own messages win
    else:
        assert item["messages"] == [{"role": "user", "content": [{"type": "text", "text": row[key]}]}]


def test_image_is_attached_and_large_images_are_resized():
    item = row_to_messages({"question": "Describe", "image": png_bytes(3000, 2000)})
    parts = item["messages"][0]["content"]
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    import base64

    img = Image.open(io.BytesIO(base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])))
    assert img.width * img.height <= 1536 * 1536


def test_run_accepts_dataset_rows_and_mixes():
    ms = MessagesList.from_data(ROWS)
    assert len(ms) == len(ROWS)
    assert MessagesList.from_data({"question": "single row"})  # a single row works too


def test_row_without_prompt_gets_a_clear_error():
    with pytest.raises(MessagesValidationError, match="Item 1: Row has no prompt field"):
        MessagesList.from_data([{"question": "ok"}, {"answer": "no prompt"}])


def test_training_dataset_builds_the_same_inputs(tmp_path):
    pytest.importorskip("torch")
    from agentfly.verl.utils.dataset.rl_dataset import RLHFAgentDataset

    rows = [r for r in ROWS if any(k in r for k in ("prompt", "question", "instruction", "problem"))]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(rows))
    dataset = RLHFAgentDataset(str(path), tokenizer=None)
    for i, row in enumerate(rows):
        assert dataset[i]["messages"] == row_to_messages(row)
