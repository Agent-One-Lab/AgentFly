"""AsyncVerlBackend: generated token ids are returned with padding stripped."""
import pytest

torch = pytest.importorskip("torch")

from agentfly.utils.llm_backends.llm_backends import extract_response_ids

IM_END = 151645


def _batch(with_prompts=True):
    P, R = 3, 6
    prompts = torch.zeros(2, P, dtype=torch.long)
    # right-padded responses: row0 has 3 valid ids, row1 has 5
    responses = torch.tensor(
        [[11, 12, IM_END, 0, 0, 0], [21, 22, 23, 24, IM_END, 0]], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long
    )
    b = {"responses": responses, "attention_mask": attention_mask}
    if with_prompts:
        b["prompts"] = prompts
    return b


def test_extract_response_ids_strips_padding():
    ids = extract_response_ids(_batch())
    assert ids == [[11, 12, IM_END], [21, 22, 23, 24, IM_END]]
    # eos is kept (it is a sampled token), padding is not
    assert all(r[-1] == IM_END for r in ids)


def test_extract_response_ids_infers_prompt_width_without_prompts_key():
    ids = extract_response_ids(_batch(with_prompts=False))
    assert ids == [[11, 12, IM_END], [21, 22, 23, 24, IM_END]]


def test_extract_response_ids_truncated_row_has_no_eos():
    b = _batch()
    # row0 hit max length: 6 valid tokens, none of them eos
    b["responses"][0] = torch.tensor([11, 12, 13, 14, 15, 16])
    b["attention_mask"][0] = torch.ones(9, dtype=torch.long)
    ids = extract_response_ids(b)
    assert ids[0] == [11, 12, 13, 14, 15, 16]
    assert ids[0][-1] != IM_END
