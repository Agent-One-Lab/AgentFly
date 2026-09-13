"""Tests for StepRollout (single-turn / bounded-history rollout) — Phases 1–2.

Covers the parts that don't need a GPU/container: registry resolution, the bounded
ChatWindowPromptBuilder, and the per-step `to_dataproto` layout (tokenizer + DataProto
mocked so the test targets the layout logic, not tensor plumbing).
"""
import sys
import types

import numpy as np
import pytest

from agentfly.agents.agent_base import BaseAgent
from agentfly.agents.rollout import conversion
from agentfly.agents.rollout.base import Rollout
from agentfly.agents.rollout.strategies.step_rollout import StepRollout
from agentfly.agents.rollout.structures import Step
from agentfly.agents.rollout.registry import ROLLOUT_REGISTRY, resolve_rollout
from agentfly.agents.rollout.strategies.prompt_builders import (
    ChatWindowPromptBuilder,
    StepPromptBuilder,
    resolve_prompt_builder,
)
from agentfly.agents.types import RunResult, Segment, Trajectory
from agentfly.agents.utils.messages import Messages
from agentfly.tools.types import ToolResult


# ---- (a) registry ---------------------------------------------------------

def test_step_registered_and_is_rollout():
    assert ROLLOUT_REGISTRY["step"] is StepRollout
    assert issubclass(StepRollout, Rollout)


def test_resolve_step_by_name_and_instance():
    r = resolve_rollout("step")
    assert isinstance(r, StepRollout)
    inst = StepRollout(history_length=3)
    assert resolve_rollout(inst) is inst
    assert inst.history_length == 3


def test_prompt_builder_resolution():
    pb = resolve_prompt_builder("chat_window", max_prompt_length=1234)
    assert isinstance(pb, ChatWindowPromptBuilder)
    assert pb.max_prompt_length == 1234
    assert resolve_prompt_builder(pb) is pb


# ---- (b) ChatWindowPromptBuilder -----------------------------------------

class _FakeAgent(types.SimpleNamespace):
    # Rollouts ask the agent which tool schemas are rendered into the prompt
    # (RolloutAgent.prompt_tools); these fakes render none.
    def prompt_tools(self):
        return []


def _h(obs, action):
    """A structured history entry (chat window reads obs/response)."""
    return {"obs": obs, "response": action, "action": action, "raw_obs": obs, "admissible": []}


def _cur(obs, raw_obs=None, admissible=None, task=""):
    return {"obs": obs, "raw_obs": raw_obs if raw_obs is not None else obs,
            "admissible": admissible or [], "task_description": task}


def test_chat_window_bounds_history():
    agent = _FakeAgent(system_prompt="SYS", tokenizer=None)
    pb = ChatWindowPromptBuilder(max_prompt_length=10_000)  # large -> no truncation
    history = [_h("obs0", "act0"), _h("obs1", "act1"), _h("obs2", "act2")]
    msgs = pb.build(agent, history=history, current=_cur("CUR"), history_length=2)
    # system + last-2 (obs,action) pairs + current obs
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    # only the LAST 2 history turns are kept (obs1/act1, obs2/act2), then CUR
    assert [m["content"] for m in msgs] == ["SYS", "obs1", "act1", "obs2", "act2", "CUR"]


def test_chat_window_truncates_when_over_budget():
    agent = _FakeAgent(system_prompt="SYS", tokenizer=None)
    # tiny budget (char/4 fallback) forces dropping all history -> system + current only
    pb = ChatWindowPromptBuilder(max_prompt_length=1)
    history = [_h("a" * 40, "b" * 40), _h("c" * 40, "d" * 40)]
    msgs = pb.build(agent, history=history, current=_cur("CUR"), history_length=2)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert [m["content"] for m in msgs] == ["SYS", "CUR"]


def test_chat_window_no_system_prompt():
    agent = _FakeAgent(system_prompt=None, tokenizer=None)
    pb = ChatWindowPromptBuilder(max_prompt_length=10_000)
    msgs = pb.build(agent, history=[], current=_cur("CUR"), history_length=2)
    assert [m["role"] for m in msgs] == ["user"]


# ---- (b2) FlatAlfworldPromptBuilder — verl-agent byte-match ---------------

from agentfly.agents.rollout.strategies.prompt_builders import (  # noqa: E402
    FlatAlfworldPromptBuilder, ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS,
)


def _golden(task, step_index, window, current_raw, admissible):
    """Independent re-implementation of verl-agent's ALFWorld prompt rendering."""
    adm = "\n ".join(f"'{s}'" for s in admissible if s != "help")
    if not window:
        return ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=current_raw, admissible_actions=adm)
    start = step_index - len(window)
    lines = "\n".join(
        f"[Observation {start+j+1}: '{h['raw_obs']}', Action {start+j+1}: '{h['action']}']"
        for j, h in enumerate(window)
    )
    return ALFWORLD_TEMPLATE.format(
        task_description=task, step_count=step_index, history_length=len(window),
        action_history=lines, current_step=step_index + 1,
        current_observation=current_raw, admissible_actions=adm,
    )


def test_flat_no_history_byte_match():
    pb = FlatAlfworldPromptBuilder()
    cur = _cur("obs C", raw_obs="obs C", admissible=["go to desk 1", "help", "open drawer 2"], task="put a mug")
    msgs = pb.build(agent=None, history=[], current=cur, history_length=2, step_index=0)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"                 # single flat user message
    assert msgs[0]["content"] == _golden("put a mug", 0, [], "obs C", cur["admissible"])


def test_flat_with_history_byte_match():
    pb = FlatAlfworldPromptBuilder()
    window = [{"raw_obs": "obs A", "action": "go to desk 1"},
              {"raw_obs": "obs B", "action": "open drawer 1"}]
    cur = _cur("obs C", raw_obs="obs C", admissible=["go to desk 1", "help", "open drawer 2"], task="put a mug")
    msgs = pb.build(agent=None, history=list(window), current=cur, history_length=2, step_index=2)
    assert msgs[0]["content"] == _golden("put a mug", 2, window, "obs C", cur["admissible"])


def test_flat_bounded_window_and_step_numbers():
    pb = FlatAlfworldPromptBuilder()
    history = [{"raw_obs": f"o{i}", "action": f"a{i}"} for i in range(3)]
    cur = _cur("oc", raw_obs="oc", admissible=["x"], task="t")
    content = pb.build(agent=None, history=history, current=cur, history_length=2, step_index=3)[0]["content"]
    assert "[Observation 2: 'o1', Action 2: 'a1']" in content
    assert "[Observation 3: 'o2', Action 3: 'a2']" in content
    assert "Observation 1" not in content                 # oldest bounded out
    assert "you have already taken 3 step(s)" in content  # step_count = step_index
    assert "now at step 4" in content                     # current_step = step_index + 1


def test_flat_obs_tuple_normalized():
    pb = FlatAlfworldPromptBuilder()
    cur = _cur("x", raw_obs=("You arrive at desk 1.",), admissible=["x"], task="t")
    content = pb.build(agent=None, history=[], current=cur, history_length=2, step_index=0)[0]["content"]
    assert "You arrive at desk 1." in content
    assert "('You arrive" not in content                  # tuple-repr unwrapped


def test_resolve_alfworld_flat():
    pb = resolve_prompt_builder("alfworld_flat", max_prompt_length=1024)
    assert isinstance(pb, FlatAlfworldPromptBuilder)
    assert pb.max_prompt_length == 1024


# ---- (c) per-step to_dataproto layout ------------------------------------

def _install_fake_verl(monkeypatch):
    """Fake agentfly.verl(.protocol) so to_dataproto's DataProto import is cheap."""
    captured = {}

    class FakeDataProto:
        @classmethod
        def from_single_dict(cls, inputs, meta_info=None):
            captured["inputs"] = inputs
            captured["meta_info"] = meta_info
            obj = cls()
            obj.inputs = inputs
            obj.meta_info = meta_info
            return obj

    verl_mod = types.ModuleType("agentfly.verl")
    protocol_mod = types.ModuleType("agentfly.verl.protocol")
    protocol_mod.DataProto = FakeDataProto
    verl_mod.protocol = protocol_mod
    monkeypatch.setitem(sys.modules, "agentfly.verl", verl_mod)
    monkeypatch.setitem(sys.modules, "agentfly.verl.protocol", protocol_mod)
    return captured


def _fake_tokenize(n_rows, L=4):
    import torch

    # reward_mask: 1 only on the last response token of each row
    reward_mask = torch.zeros(n_rows, L)
    reward_mask[:, -1] = 1.0
    return {
        "input_ids": torch.ones(n_rows, L, dtype=torch.long),
        "attention_mask": torch.ones(n_rows, L, dtype=torch.long),
        "action_mask": torch.ones(n_rows, L, dtype=torch.long),
        "reward_mask": reward_mask,
        "position_ids": torch.zeros(n_rows, L, dtype=torch.long),
    }


def _step(i, reward, invalid, anchor):
    """One Step carrying its env ToolResult (post-action anchor + signals).

    ``is_action_valid`` is verl-agent's FORMAT check on the model output (needs a
    ``<think>`` and an ``<action>`` block), not the env-semantic ``invalid_action``
    flag — so a format-invalid step is one whose response lacks the tags.
    """
    content = f"act{i}" if invalid else f"<think>t{i}</think><action>act{i}</action>"
    return Step(
        type="Action Input",
        messages=Messages.from_turns(
            [
                {"role": "user", "content": f"obs{i}"},
                {"role": "assistant", "content": content},
            ]
        ),
        tool_result=ToolResult(
            name="t",
            arguments={},
            observation=f"obs{i}",
            anchor=anchor,
            step_reward=reward,
            metrics={"invalid_action": invalid},
        ),
    )


def _traj(traj_uid, chain_idx, outcome, steps):
    return Trajectory(
        segments=[Segment(messages=s.messages.messages) for s in steps],
        reward=outcome,
        metrics={"trajectory/accuracy": float(outcome > 0)},   # a reward-fn extra metric
        group_id="G",
        chain_id=traj_uid,
        group_idx=0,
        chain_idx=chain_idx,
        steps=steps,
    )


def test_to_dataproto_per_step_layout(monkeypatch):
    captured = _install_fake_verl(monkeypatch)

    # 2 trajectories sharing one task uid; outcomes 1.0 and 0.0; 3 and 2 steps.
    # Steps carry POST-action anchors; the layout right-shifts them into the
    # pre-action grouping anchor (step-0 -> the shared "__init__" marker).
    tA = _traj("tA", 0, 1.0, [
        _step(0, 0.0, 0.0, "a0"),
        _step(1, 0.0, 1.0, "a1"),
        _step(2, 1.0, 0.0, "a2"),   # winning step
    ])
    tB = _traj("tB", 1, 0.0, [
        _step(0, 0.0, 0.0, "b0"),
        _step(1, 0.0, 0.0, "b1"),
    ])
    run_result = RunResult(trajectories=[tA, tB], rollout="step")

    monkeypatch.setattr(
        conversion, "tokenize_trajectories",
        lambda agent, messages_list, **kw: _fake_tokenize(len(messages_list)),
    )
    agent = _FakeAgent(tokenizer=None, processor=None, template=None, max_model_len=4096)

    BaseAgent.to_verl_dataproto(agent, run_result)
    inputs = captured["inputs"]
    meta = captured["meta_info"]

    # one row per step
    assert len(inputs["uid"]) == 5
    assert list(inputs["traj_uid"]) == ["tA", "tA", "tA", "tB", "tB"]
    assert list(inputs["uid"]) == ["G"] * 5                     # shared task group
    # pre-action anchors: right-shifted post-anchors, step-0 -> "__init__"
    assert list(inputs["anchor_obs"]) == ["__init__", "a0", "a1", "__init__", "b0"]
    assert list(inputs["step_env_reward"]) == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert list(inputs["is_action_valid"]) == [1.0, 0.0, 1.0, 1.0, 1.0]
    assert meta["layout"] == "per_step"

    # outcome broadcast to every step-row of the trajectory (on the reward-mask token)
    rm_last = inputs["rm_scores"][:, -1].tolist()
    assert rm_last == [1.0, 1.0, 1.0, 0.0, 0.0]  # tA rows -> 1.0, tB rows -> 0.0

    # reward-fn extra metric broadcast per step-row as rm_<key> (mirrors ChainRollout)
    assert list(inputs["rm_trajectory/accuracy"]) == [1.0, 1.0, 1.0, 0.0, 0.0]

    # repeat_times = per-trajectory step counts (aligns the input batch to the step-rows,
    # the SAME mechanism folding uses); no divisor padding here so no absorption
    assert meta["repeat_times"] == [3, 2]


def test_to_dataproto_raises_without_records(monkeypatch):
    _install_fake_verl(monkeypatch)
    with pytest.raises(ValueError, match="No trainable segments"):
        BaseAgent.to_verl_dataproto(
            _FakeAgent(tokenizer=None, processor=None),
            run_result=RunResult(trajectories=[], rollout="step"),
        )


def test_to_dataproto_reads_canonical_segments_and_pads_signals(monkeypatch):
    captured = _install_fake_verl(monkeypatch)
    traj = _traj("t", 0, 1.0, [_step(0, 0.5, 1.0, "a0")])
    # A consumer can replace the training conversation without touching live Steps.
    messages = [
        {"role": "user", "content": "updated context"},
        {"role": "assistant", "content": "<think>t</think><action>act</action>",
         "token_ids": [10, 11, 2]},
    ]
    traj.segments[0] = Segment(messages=messages)
    seen = []

    def tokenize(agent, messages_list, **kwargs):
        seen.extend(messages_list)
        return _fake_tokenize(len(messages_list))

    monkeypatch.setattr(conversion, "tokenize_trajectories", tokenize)
    BaseAgent.to_verl_dataproto(
        _FakeAgent(tokenizer=None, processor=None),
        run_result=RunResult(trajectories=[traj], rollout="step"),
        pad_to_multiple_of=2,
        log_token_drift=False,
    )
    assert seen == [messages]
    inputs = captured["inputs"]
    assert inputs["rm_scores"][:, -1].tolist() == [1.0, 1.0]
    assert inputs["step_env_reward"].tolist() == [0.5, 0.5]
    assert inputs["is_action_valid"].tolist() == [1.0, 1.0]
    assert inputs["active_masks"].tolist() == [1.0, 0.0]
    assert inputs["action_mask"][1].tolist() == [0, 0, 0, 0]
    assert captured["meta_info"]["repeat_times"] == [2]


def test_to_dataproto_rejects_misaligned_segments_and_runtime_signals(monkeypatch):
    _install_fake_verl(monkeypatch)
    traj = _traj("t", 0, 1.0, [_step(0, 0.0, 0.0, "a0")])
    traj.segments.append(Segment(messages=[]))
    with pytest.raises(ValueError, match="one runtime step per segment"):
        BaseAgent.to_verl_dataproto(
            _FakeAgent(tokenizer=None, processor=None),
            run_result=RunResult(trajectories=[traj], rollout="step"),
        )


# ---- (d) token_ids splice + drift diagnostic through to_dataproto -----------
#
# Uses the REAL Qwen2.5 tokenizer + HF chat template (the alfworld ``template=null``
# path) so this exercises the actual chat-bricks splice, not a mocked tokenizer.

HF_TMPL = "Qwen/Qwen2.5-1.5B-Instruct"
DRIFT_THINK = [27, 26865, 29]  # what the model samples for "<think>"
IM_END = 151645


def _real_tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    return transformers.AutoTokenizer.from_pretrained(HF_TMPL)


def _ids_step(i, content, token_ids=None):
    asst = {"role": "assistant", "content": content}
    if token_ids is not None:
        asst["token_ids"] = token_ids
    return Step(
        type="Action Input",
        messages=Messages.from_turns([{"role": "user", "content": f"obs{i}"}, asst]),
        tool_result=ToolResult(
            name="t", arguments={}, observation=f"obs{i}", anchor=f"a{i}",
            step_reward=0.0, metrics={"invalid_action": 0.0},
        ),
    )


def _has_sub(seq, sub):
    return any(seq[k:k + len(sub)] == sub for k in range(len(seq) - len(sub) + 1))


@pytest.mark.parametrize("entrypoint", ["converter", "agent"])
def test_to_dataproto_splices_token_ids_and_reports_drift(monkeypatch, entrypoint):
    captured = _install_fake_verl(monkeypatch)
    tok = _real_tokenizer()
    if tok.encode("<think>", add_special_tokens=False) == DRIFT_THINK:
        pytest.skip("tokenizer does not drift on <think>")
    content = "<think>x</think><action>go</action>"
    # the sampled sequence: drifted <think> + rest + eos
    sampled = DRIFT_THINK + tok.encode("x</think><action>go</action>", add_special_tokens=False) + [IM_END]

    traj = _traj("t", 0, 1.0, [_ids_step(0, content, sampled), _ids_step(1, content, sampled)])
    agent = _FakeAgent(tokenizer=tok, processor=None, template=HF_TMPL, max_model_len=4096)
    result = RunResult(trajectories=[traj], rollout="step")
    if entrypoint == "agent":
        BaseAgent.to_verl_dataproto(agent, result)
    else:
        conversion.step_to_dataproto(agent, result)

    inputs, meta = captured["inputs"], captured["meta_info"]
    for r in range(2):
        ids = inputs["input_ids"][r].tolist()
        # the sampled (drifted) ids are in the training input verbatim ...
        assert _has_sub(ids, DRIFT_THINK)
        # ... and the trained span ends with the sampled eos (reward token position)
        span = inputs["input_ids"][r][inputs["action_mask"][r] == 1].tolist()
        assert IM_END in span

    drift = meta["token_drift"]
    assert drift["token_drift/frac_rows_with_ids"] == 1.0
    assert drift["token_drift/frac_ids_end_with_eos"] == 1.0
    assert drift["token_drift/frac_rows_differ"] == 1.0      # text path would re-encode <think>
    assert drift["token_drift/frac_tokens_differ"] > 0.0
    assert drift["token_drift/n_sampled"] == 2.0


def test_to_dataproto_without_token_ids_reports_none_and_is_text_path(monkeypatch):
    captured = _install_fake_verl(monkeypatch)
    tok = _real_tokenizer()
    content = "<think>x</think><action>go</action>"
    traj = _traj("t", 0, 0.0, [_ids_step(0, content)])
    agent = _FakeAgent(tokenizer=tok, processor=None, template=HF_TMPL, max_model_len=4096)
    BaseAgent.to_verl_dataproto(agent, RunResult(trajectories=[traj], rollout="step"))

    drift = captured["meta_info"]["token_drift"]
    assert drift == {"token_drift/frac_rows_with_ids": 0.0}
    # no ids -> canonical re-encoding: the drifted form must NOT appear
    assert not _has_sub(captured["inputs"]["input_ids"][0].tolist(), DRIFT_THINK)


def test_log_token_drift_can_be_disabled(monkeypatch):
    captured = _install_fake_verl(monkeypatch)
    tok = _real_tokenizer()
    traj = _traj("t", 0, 0.0, [_ids_step(0, "<think>x</think><action>go</action>", [1, 2, IM_END])])
    agent = _FakeAgent(tokenizer=tok, processor=None, template=HF_TMPL, max_model_len=4096)
    BaseAgent.to_verl_dataproto(
        agent, RunResult(trajectories=[traj], rollout="step"), log_token_drift=False,
    )
    assert "token_drift" not in captured["meta_info"]
