"""``AsyncVerlBackend`` samples from the training render.

The backend renders each prompt with the same chat-bricks call ``tokenize_trajectories``
makes (earlier assistant turns spliced from their sampled ``token_ids``) and hands verl
the ids, so the prompt vLLM samples from is the training row's prefix by construction.
``prompt_check`` then compares those ids against the ``prompts`` verl echoes back and
reports on stdout, where the Ray log shows it.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("chat_bricks")

from agentfly.utils.llm_backends.llm_backends import AsyncVerlBackend  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IM_END = 151645
SAMPLED_THINK = [27, 26865, 29]  # how Qwen2.5 *samples* "<think>"
TEXT_THINK = [13708, 766, 29]  # how "<think>" re-tokenizes from text


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer for {MODEL} unavailable: {e}")


def _contains(seq, sub):
    return any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def multi_turn_messages(tokenizer):
    """One completed assistant turn whose sampled ids carry the drifting ``<think>`` form."""
    turn0 = "<think> look around </think>\n<action> go to fridge 1 </action>"
    assert tokenizer.encode(turn0, add_special_tokens=False)[:3] == TEXT_THINK
    sampled = SAMPLED_THINK + tokenizer.encode(turn0[len("<think>"):], add_special_tokens=False) + [IM_END]
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": [{"type": "text", "text": "Task: put a mug in the fridge."}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": turn0}],
            "tool_calls": [{"id": "call_0", "type": "function",
                            "function": {"name": "alfworld_step", "arguments": '{"action": "go to fridge 1"}'}}],
            "token_ids": sampled,
            "loss": True,
            "status": "continue",
        },
        {"role": "tool", "tool_call_id": "call_0", "tool_name": "alfworld_step",
         "content": [{"type": "text", "text": "You arrive at fridge 1. It is closed."}]},
    ]


class EchoEngine:
    """Fake verl engine: emits a fixed response and echoes the prompt ids it was given,
    left-padded, the way verl's agent loop returns ``prompts``. ``tamper`` rewrites the
    echoed prompt to simulate verl re-rendering the prompt from text."""

    def __init__(self, tokenizer, response_ids, tamper=None):
        self.tokenizer, self.response_ids, self.tamper = tokenizer, response_ids, tamper
        self.calls = []

    async def generate_sequences_async(self, batch):
        from agentfly.verl.protocol import DataProto

        nt = batch.non_tensor_batch
        self.calls.append({k: nt[k] for k in nt})
        if "prompt_ids" in nt:
            rows = [list(r) for r in nt["prompt_ids"]]
        else:  # verl would render raw_prompt itself; any ids will do for the fallback test
            rows = [[1, 2, 3] for _ in nt["raw_prompt"]]
        if self.tamper:
            rows = [self.tamper(r) for r in rows]
        pad = self.tokenizer.pad_token_id
        width = max(len(r) for r in rows)
        prompts = torch.full((len(rows), width), pad, dtype=torch.long)
        pmask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            prompts[i, width - len(r):] = torch.tensor(r)
            pmask[i, width - len(r):] = 1
        responses = torch.tensor([self.response_ids] * len(rows), dtype=torch.long)
        rmask = torch.ones_like(responses)
        return DataProto.from_single_dict(
            {"prompts": prompts, "responses": responses, "attention_mask": torch.cat([pmask, rmask], dim=1)}
        )


def _backend(tokenizer, engine, **kw):
    return AsyncVerlBackend(llm_engine=engine, model_name_or_path=MODEL, template=None, **kw)


def _training_prefix(tokenizer, messages):
    """The training render of the same conversation (``tokenize_trajectories``' call), as ids."""
    from chat_bricks import tokenize_conversations

    out = tokenize_conversations(
        [messages], tokenizer=tokenizer, template=MODEL, processor=None, max_length=None,
        add_generation_prompt=True, tools=None, ignore_tool_calls=True,
    )
    return out["input_ids"][0].tolist()


def test_sampling_prompt_is_the_training_render(tokenizer):
    msgs = multi_turn_messages(tokenizer)
    response = SAMPLED_THINK + tokenizer.encode(" open fridge 1", add_special_tokens=False) + [IM_END]
    engine = EchoEngine(tokenizer, response)
    out = asyncio.run(_backend(tokenizer, engine).generate_async([msgs], return_dict=True, tools=[]))

    sent = list(engine.calls[0]["prompt_ids"][0])
    assert sent == _training_prefix(tokenizer, msgs), "prompt sent to verl must be the training row's prefix"
    assert _contains(sent, SAMPLED_THINK) and not _contains(sent, TEXT_THINK), "earlier turn is spliced from sampled ids"

    # The old path (verl rendering raw_prompt from text) would have produced the drifted form.
    text_render = tokenizer.apply_chat_template(
        [{"role": m["role"], "content": (m["content"] if isinstance(m["content"], str) else m["content"][0]["text"])}
         for m in msgs if m["role"] != "tool"] + [{"role": "user", "content": "You arrive at fridge 1. It is closed."}],
        tokenize=True, add_generation_prompt=True)
    assert _contains(list(text_render), TEXT_THINK) and list(text_render) != sent

    # The sampled response is still passed through verbatim.
    assert out["response_ids"] == [response]
    # raw_prompt is still shipped (verl keeps it in extra_fields / reads images from it).
    assert "raw_prompt" in engine.calls[0]


def test_prompt_check_reports_identical_with_turn_depth(tokenizer, capsys):
    msgs = multi_turn_messages(tokenizer)
    engine = EchoEngine(tokenizer, [IM_END])
    asyncio.run(_backend(tokenizer, engine).generate_async([msgs], return_dict=True, tools=[]))
    out = capsys.readouterr().out
    assert "prompt_check: 1/1 rows identical" in out
    assert "assistant turns=[1]" in out


def test_prompt_check_prints_mismatch_when_verl_rerenders(tokenizer, capsys):
    """Simulate the bug: verl ignores the ids and re-tokenizes the earlier turn from text."""
    msgs = multi_turn_messages(tokenizer)

    def retokenize(row):
        k = next(i for i in range(len(row)) if row[i : i + 3] == SAMPLED_THINK)
        return row[:k] + TEXT_THINK + row[k + 3:]

    engine = EchoEngine(tokenizer, [IM_END], tamper=retokenize)
    backend = _backend(tokenizer, engine)
    asyncio.run(backend.generate_async([msgs], return_dict=True, tools=[]))
    out = capsys.readouterr().out
    assert "PROMPT MISMATCH" in out and "1/1 rows differ" in out
    assert backend.prompt_check_stats["checked"] == 1 and backend.prompt_check_stats["mismatched"] == 1


def test_prompt_check_strict_raises(tokenizer):
    msgs = multi_turn_messages(tokenizer)
    engine = EchoEngine(tokenizer, [IM_END], tamper=lambda row: row[:-1] + [0])
    backend = _backend(tokenizer, engine, check_prompt_consistency="strict")
    with pytest.raises(RuntimeError, match="PROMPT MISMATCH"):
        asyncio.run(backend.generate_async([msgs], return_dict=True, tools=[]))


def test_prompt_check_can_be_disabled(tokenizer, capsys):
    msgs = multi_turn_messages(tokenizer)
    engine = EchoEngine(tokenizer, [IM_END], tamper=lambda row: row[:-1] + [0])
    backend = _backend(tokenizer, engine, check_prompt_consistency=False)
    asyncio.run(backend.generate_async([msgs], return_dict=True, tools=[]))
    assert "prompt_check" not in capsys.readouterr().out and backend.prompt_check_stats["checked"] == 0


def test_vision_rows_fall_back_to_verl_render(tokenizer, capsys):
    """Ids are text-only for now: a row with vision inputs is left for verl to render."""
    msgs = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                     {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}]},
    ]
    engine = EchoEngine(tokenizer, [IM_END])
    backend = _backend(tokenizer, engine, check_prompt_consistency=False)  # the guard is not under test here
    asyncio.run(backend.generate_async([msgs], return_dict=True, tools=[]))
    assert "prompt_ids" not in engine.calls[0]
    assert "vision inputs" in capsys.readouterr().out
