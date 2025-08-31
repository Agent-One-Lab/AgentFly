from .....agents.utils.messages import Messages, MessagesList, MessagesValidationError
import pytest

def test_messages_init():
    # 1) List of dicts with "messages"
    data1 = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
            "run_id": 1,
        },
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
            "run_id": 2,
            "other_key": "x",
            "other_key2": "y",
        }
    ]
    print(MessagesList.from_data(data1).to_list())

    # 2) List of lists (each inner list is a turn list)
    data2 = [
        [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
        ],
        [
            {"role": "user", "content": [{"type": "image", "image": "BASE64..."}, {"type": "text", "text": "Describe"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Looks like..."}]},
        ],
    ]
    print(MessagesList.from_data(data2, default_meta={"source": "batch-42"}).to_list())

    # 3) Dict with "messages"
    data3 = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "ping"}]},
        ],
        "other_key": "x",
    }
    print(MessagesList.from_data(data3).to_list())

    # 4) List of turn dicts (single item)
    data4 = [
        {"role": "user", "content": [{"type": "image", "image_url": "https://example.com/cat.png"}]},
        {"role": "user", "content": [{"type": "text", "text": "Describe the image"}]},
    ]
    print(MessagesList.from_data(data4, default_meta={"dataset": "demo"}).to_list())

    # Programmatic building (with helpers)
    m = Messages.from_turns([], tag="manual")
    m.add("user", [{"type": "text", "text": "Tell me a joke."}])
    ms = MessagesList(strict=True)
    ms.append(m)
    print(ms.to_list())


def test_messages_error_detection():
    # Empty list
    data = [
        {}
    ]
    with pytest.raises(MessagesValidationError):
        MessagesList.from_data(data)

    # List of dicts with "messages" key but no "content" key
    data = [
        {"messages": [{"role": "user"}]},
    ]
    with pytest.raises(MessagesValidationError):
        MessagesList.from_data(data)
