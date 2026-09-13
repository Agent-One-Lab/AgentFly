"""The flat WebShop step prompt is verl-agent's ``WEBSHOP_TEMPLATE`` byte-for-byte, and the
WebShop first-step hook feeds the step rollout what that template needs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentfly.agents.rollout.strategies.prompt_builders import (
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    FlatWebshopPromptBuilder,
    resolve_prompt_builder,
)

TASK = "Find me a sulfate free shampoo with price lower than 50.00 dollars"
PAGE0 = "'Search'"
PAGE1 = "'Back to Search' [SEP] 'Page 1 (Total results: 50)' [SEP] 'Next >' [SEP] 'B07B6DVG41' [SEP] 'Glossy Locks Repair Shampoo' [SEP] '$27.0'"
ACTS0 = ["search[<your query>]", "click[search]"]
ACTS1 = ["click[back to search]", "click[next >]", "click[b07b6dvg41]", "click[buy now]"]


def test_registered_under_webshop_flat():
    assert isinstance(resolve_prompt_builder("webshop_flat"), FlatWebshopPromptBuilder)


def test_first_step_uses_no_history_template_verbatim():
    msgs = FlatWebshopPromptBuilder().build(
        None, history=[], current={"raw_obs": PAGE0, "obs": "", "admissible": ACTS0, "task_description": TASK},
        history_length=2, step_index=0,
    )
    assert [m["role"] for m in msgs] == ["user"], "one flat user message, no system role"
    expected = WEBSHOP_TEMPLATE_NO_HIS.format(
        task_description=TASK, current_observation=PAGE0,
        available_actions="'search[<your query>]',\n'click[search]',",
    )
    assert msgs[0]["content"] == expected
    assert "Your task is to: " + TASK + ".\n" in expected and "[\n'search[<your query>]',\n'click[search]',\n]." in expected


def test_history_template_matches_verl_agent_memory_format():
    history = [
        {"raw_obs": PAGE0, "obs": "", "response": "<think>..</think><action>search[shampoo]</action>", "action": "search[shampoo]", "admissible": ACTS0},
    ]
    msgs = FlatWebshopPromptBuilder().build(
        None, history=history, current={"raw_obs": PAGE1, "obs": "", "admissible": ACTS1, "task_description": TASK},
        history_length=2, step_index=1,
    )
    expected = WEBSHOP_TEMPLATE.format(
        task_description=TASK, step_count=1, history_length=1,
        action_history=f"[Observation 1: '{PAGE0}', Action 1: 'search[shampoo]']",
        current_step=2, current_observation=PAGE1,
        available_actions="\n".join(f"'{a}'," for a in ACTS1),
    )
    assert msgs[0]["content"] == expected


def test_history_window_uses_absolute_step_numbers():
    history = [
        {"raw_obs": f"'p{i}'", "obs": "", "response": "", "action": f"a{i}", "admissible": []} for i in range(4)
    ]
    text = FlatWebshopPromptBuilder().build(
        None, history=history, current={"raw_obs": "'p4'", "obs": "", "admissible": [], "task_description": TASK},
        history_length=2, step_index=4,
    )[0]["content"]
    assert "[Observation 3: ''p2'', Action 3: 'a2']\n[Observation 4: ''p3'', Action 4: 'a3']" in text
    assert "taken 4 step(s)" in text and "most recent 2 observations" in text and "now at step 5" in text


def test_first_step_hook_seeds_prompt_and_flat_metadata():
    from agentfly.agents.specialized.action_agent import ActionAgent
    from agentfly.agents.rollout.structures import Step
    from agentfly.agents.utils.messages import Messages

    raw = f"WebShop [SEP] Instruction: [SEP] {TASK} [SEP] Search"

    class FakeEnv:
        observation = raw
        async def reset(self, env_args=None): self.reset_args = env_args
        def get_available_actions(self): return {"has_search_bar": True, "clickables": ["search"]}
        def get_instruction_text(self): return TASK

    env = FakeEnv()

    async def acquire(**kwargs): return env

    ctx = SimpleNamespace(metadata={"task_id": "7"}, acquire_resource=acquire)
    step = Step(type="Action Input", messages=Messages.from_turns([{"role": "user", "content": "placeholder"}]))
    asyncio.run(ActionAgent._prepare_webshop_first_step(object.__new__(ActionAgent), ctx, step))

    first_user = step.messages.messages[0]
    text = first_user["content"] if isinstance(first_user["content"], str) else "".join(p["text"] for p in first_user["content"])
    assert text.startswith(TASK) and "'Search'" in text and "'click[search]'," in text
    assert env.reset_args == {"task_id": "7", "_flat_initial_obs": PAGE0, "_flat_initial_admissible": ACTS0, "_flat_task_description": TASK} or ctx.metadata["_flat_task_description"] == TASK
    assert ctx.metadata["_flat_initial_obs"] == PAGE0
    assert ctx.metadata["_flat_initial_admissible"] == ACTS0
