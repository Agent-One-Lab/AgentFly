"""``webshop_browser_action`` returns the verl-agent-aligned step contract (no container needed)."""

from __future__ import annotations

import pytest

from agentfly.core import Context
from agentfly.tools.src.webshop import tools as ws
from agentfly.tools.types import ToolResult

TASK = "Find me a sulfate free shampoo with price lower than 50.00 dollars"
RAW_INDEX = f"WebShop [SEP] Instruction: [SEP] {TASK} [SEP] Search"
RAW_RESULTS = f"WebShop [SEP] Instruction: [SEP] {TASK} [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B07B6DVG41 [SEP] Glossy Locks Repair Shampoo [SEP] $27.0"


class FakeWebShopEnv:
    """Minimal stand-in for ``WebShopEnv``: a search page, one results page, a Buy page."""

    def __init__(self, score=1.0):
        self.observation = RAW_INDEX
        self.state = {"url": "/index/0"}
        self.score = score
        self.calls = []

    def get_available_actions(self):
        if "/index" in self.state["url"]:
            return {"has_search_bar": True, "clickables": ["search"]}
        if "/search_results" in self.state["url"]:
            return {"has_search_bar": False, "clickables": ["back to search", "next >", "b07b6dvg41", "buy now"]}
        return {"has_search_bar": False, "clickables": []}

    def get_instruction_text(self):
        return TASK

    async def step(self, action, task_id=None):
        self.calls.append((action, task_id))
        if action == "get_reward":
            return {"observation": "Thank you for shopping with us!", "reward": self.score}
        if action.startswith("search["):
            self.state["url"] = "/search_results/0/shampoo/1"
            self.observation = RAW_RESULTS
            return self.observation + "\nClickables: [...]"
        if action == "click[buy now]":
            self.state["url"] = "/done/0/B07B6DVG41/{}"
            self.observation = "Thank you for shopping with us!"
            return self.observation
        if action.startswith("click["):
            return "Invalid action, action argument should be one of the clickables: [...]"
        return "Invalid action, action name should be 'click' or 'search'."


@pytest.fixture
def ctx(monkeypatch):
    env = FakeWebShopEnv()
    c = Context(rollout_id="test_webshop_tool")
    c.metadata = {"task_id": "0"}

    async def acquire(**kwargs):
        return env

    monkeypatch.setattr(c, "acquire_resource", acquire)
    monkeypatch.setattr(c, "is_spec_acquired", lambda spec: True)
    return c, env


# ---- pure helpers (verl-agent's env_manager formatting) ------------------------

def test_extract_task_and_format_page():
    assert ws.extract_task(RAW_RESULTS) == TASK
    assert ws.extract_task("Thank you for shopping with us!") is None
    page = ws.format_page(RAW_RESULTS, TASK)
    assert page == "'Back to Search' [SEP] 'Page 1 (Total results: 50)' [SEP] 'Next >' [SEP] 'B07B6DVG41' [SEP] 'Glossy Locks Repair Shampoo' [SEP] '$27.0'"
    assert ws.format_page("no task here", TASK) == "no task here"


def test_format_avail_actions_matches_verl_agent():
    assert ws.format_avail_actions({"has_search_bar": True, "clickables": ["search"]}) == ["search[<your query>]", "click[search]"]
    assert ws.format_avail_actions({"has_search_bar": False, "clickables": ["buy now"]}) == ["click[buy now]"]


def test_format_observation_lists_one_quoted_action_per_line():
    text = ws.format_observation("'Search'", ["search[<your query>]", "click[search]"])
    assert text == "'Search'\n\nAdmissible actions:\n[\n'search[<your query>]',\n'click[search]',\n]"
    assert ws.format_observation("page", []) == "page"


# ---- the tool ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_step_returns_page_anchor_and_menu(ctx):
    c, env = ctx
    r = await ws.webshop_browser_action(action="search[shampoo]", context=c)
    assert isinstance(r, ToolResult)
    assert r.anchor == ws.format_page(RAW_RESULTS, TASK), "anchor is the undecorated verl-agent page"
    assert r.observation.startswith(r.anchor) and "'click[buy now]'," in r.observation
    assert r.control is None and r.step_reward == 0.0
    assert r.metrics["done"] == 0.0 and r.metrics["won"] == 0.0 and r.metrics["invalid_action"] == 0.0
    assert r.metrics["admissible_actions"] == ["click[back to search]", "click[next >]", "click[b07b6dvg41]", "click[buy now]"]
    assert ("get_reward", "0") not in env.calls, "score is only fetched on the Buy step"


@pytest.mark.asyncio
async def test_rejected_action_is_flagged_and_page_unchanged(ctx):
    c, env = ctx
    r = await ws.webshop_browser_action(action="click[nonexistent]", context=c)
    assert r.metrics["invalid_action"] == 1.0
    assert r.anchor == ws.format_page(RAW_INDEX, TASK) == "'Search'"
    assert r.control is None and r.step_reward == 0.0


@pytest.mark.asyncio
async def test_buy_step_ends_episode_with_dense_score(ctx):
    c, env = ctx
    await ws.webshop_browser_action(action="search[shampoo]", context=c)
    r = await ws.webshop_browser_action(action="click[buy now]", context=c)
    assert r.control == "end"
    assert r.step_reward == 10.0, "verl-agent's success reward"
    assert r.metrics["won"] == 1.0 and r.metrics["done"] == 1.0 and r.metrics["task_score"] == 1.0
    assert ("get_reward", "0") in env.calls, "the dense score is fetched for THIS task id"
    assert r.observation == "Thank you for shopping with us!" and r.metrics["admissible_actions"] == []


@pytest.mark.asyncio
async def test_partial_score_is_not_a_win(ctx, monkeypatch):
    c, env = ctx
    env.score = 0.75
    await ws.webshop_browser_action(action="search[shampoo]", context=c)
    r = await ws.webshop_browser_action(action="click[buy now]", context=c)
    assert r.step_reward == 0.0, "a partially-right purchase earns no reward (verl-agent binarizes)"
    assert r.metrics["task_score"] == 0.75 and r.metrics["won"] == 0.0 and r.control == "end"


@pytest.mark.asyncio
async def test_choose_alias_maps_to_click(ctx):
    c, env = ctx
    await ws.webshop_browser_action(action="search[shampoo]", context=c)
    r = await ws.webshop_browser_action(action="choose[buy now]", context=c)
    assert r.control == "end" and env.calls[-2][0] == "click[buy now]"


@pytest.mark.asyncio
async def test_action_is_lowercased_like_verl_agent(ctx):
    """verl-agent lowercases the extracted action: ``Search[...]`` / ``Click[Buy Now]`` are valid."""
    c, env = ctx
    r = await ws.webshop_browser_action(action="  Search[Shampoo] ", context=c)
    assert env.calls[-1][0] == "search[shampoo]" and r.metrics["invalid_action"] == 0.0
    r = await ws.webshop_browser_action(action="Click[Buy Now]", context=c)
    assert env.calls[-2][0] == "click[buy now]" and r.control == "end" and r.step_reward == 10.0
