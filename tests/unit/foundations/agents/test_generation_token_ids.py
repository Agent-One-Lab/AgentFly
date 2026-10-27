"""generate_response carries backend ``response_ids`` onto assistant messages as ``token_ids``."""
from agentfly.agents.rollout.loop.generation import attach_token_ids


def _msgs(n):
    return [{"role": "assistant", "content": f"r{i}", "tool_calls": []} for i in range(n)]


def test_attach_token_ids_sets_one_list_per_message():
    msgs = _msgs(2)
    attach_token_ids(msgs, {"response_texts": ["a", "b"], "response_ids": [[1, 2, 3], [4, 5]]})
    assert msgs[0]["token_ids"] == [1, 2, 3]
    assert msgs[1]["token_ids"] == [4, 5]


def test_attach_token_ids_copies_not_aliases():
    src = [7, 8, 9]
    msgs = _msgs(1)
    attach_token_ids(msgs, {"response_ids": [src]})
    msgs[0]["token_ids"].append(10)
    assert src == [7, 8, 9]


def test_attach_token_ids_noop_when_backend_has_no_ids():
    # Backends without ids (client / async_vllm) leave the message untouched,
    # so tokenization falls back to re-encoding text exactly as before.
    for responses in ({"response_texts": ["a"]}, {"response_ids": None}, {"response_ids": []}, None):
        msgs = _msgs(1)
        attach_token_ids(msgs, responses)
        assert "token_ids" not in msgs[0]


def test_attach_token_ids_skips_none_entries():
    msgs = _msgs(2)
    attach_token_ids(msgs, {"response_ids": [None, [1]]})
    assert "token_ids" not in msgs[0]
    assert msgs[1]["token_ids"] == [1]
