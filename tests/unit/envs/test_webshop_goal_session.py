"""WebShopEnv serves caller-provided goals (verl-agent-replayed tasks) through POST /session.

No container: the simulator server is replaced by an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from urllib.parse import unquote

import httpx
import pytest

from agentfly.envs import webshop_text_env as wse

GOAL = {"asin": "B07B6DVG41", "instruction_text": "Find me a shampoo, and price lower than 40.00 dollars",
        "attributes": ["sulfate free"], "goal_options": {}, "price_upper": 40.0,
        "query": "shampoo", "product_category": "Beauty", "name": "Shampoo", "category": "beauty"}
PRICES = {"B07B6DVG41": 34.0, "B08532HWVM": 12.5}
INDEX_HTML = '<html><body><div id="instruction-text"><h4>Instruction: <br>x</h4></div><input id="search_input"></body></html>'


def make_env(handler):
    env = object.__new__(wse.WebShopEnv)
    env._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://webshop")
    env.session_id = "legacysess"
    env.goal_session_id = "gabcdefghij"
    env.session_goal = False
    env.observation_mode = "text"
    env.home_html = INDEX_HTML
    env.state = {"html": INDEX_HTML, "url": "/index/legacysess"}
    env.text_to_clickable = None
    return env


@pytest.mark.asyncio
async def test_goal_reset_posts_goal_and_prices_then_scores_without_task_id():
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, unquote(request.url.path), request.url.query.decode(), request.content))
        if request.method == "POST":
            return httpx.Response(200, json={"ok": True})
        if request.url.path.startswith("/done/"):
            return httpx.Response(200, json={"observation": "<html>done</html>", "reward": 1.0})
        return httpx.Response(200, text=INDEX_HTML)

    env = make_env(handler)
    await env.reset(env_args={"task_id": 7, "goal_json": json.dumps(GOAL), "prices_json": json.dumps(PRICES)})

    method, path, _, body = seen[0]
    assert (method, path) == ("POST", "/session/gabcdefghij")
    assert json.loads(body) == {"goal": GOAL, "prices": PRICES}
    assert seen[1][:2] == ("GET", "/index/gabcdefghij")
    assert env.session_goal and env.session_id == "gabcdefghij"

    # The reward request uses the session's goal: no task_id query (which would select the
    # server's own goal list instead).
    env.state = {"html": "<html></html>", "url": "/done/gabcdefghij/B07B6DVG41/{}"}
    out = await env.step("get_reward", 7)
    assert out["reward"] == 1.0
    done_path, done_query = seen[-1][1], seen[-1][2]
    assert done_path == "/done/gabcdefghij/B07B6DVG41/{}" and done_query == ""


@pytest.mark.asyncio
async def test_goal_reset_raises_when_server_lacks_session_route():
    env = make_env(lambda request: httpx.Response(404, text="not found"))
    with pytest.raises(RuntimeError, match="POST /session"):
        await env.reset(env_args={"task_id": 1, "goal_json": json.dumps(GOAL)})


@pytest.mark.asyncio
async def test_task_id_reset_keeps_server_goal_path():
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, request.url.path, request.url.query.decode()))
        if request.url.path.startswith("/done/"):
            return httpx.Response(200, json={"observation": "<html>done</html>", "reward": 0.5})
        return httpx.Response(200, text=INDEX_HTML)

    env = make_env(handler)
    env.session_goal = True  # from a previous goal episode
    await env.reset(env_args={"task_id": 3})
    assert seen[0][:2] == ("GET", "/index/3") and env.session_goal is False
    env.state = {"html": "<html></html>", "url": "/done/3/B07B6DVG41/{}"}
    await env.step("get_reward", 3)
    assert seen[-1][2] == "task_id=3"


def test_server_mount(monkeypatch):
    monkeypatch.delenv("AF_WEBSHOP_SERVER_MOUNT", raising=False)
    mount = wse.server_mount()
    (key, options), = mount.items()
    source, target, fstype = key.split(":")
    assert source.endswith("dockers/webshop_env/webshop_simulator_server.py")
    assert (target, fstype, options) == ("/app/webshop_simulator_server.py", "none", "bind,ro")
    monkeypatch.setenv("AF_WEBSHOP_SERVER_MOUNT", "0")
    assert wse.server_mount() == {}
