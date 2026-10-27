"""``checkout_task_commit``: SWE tools put the repo on the task's commit once per rollout."""

import pytest

from agentfly.core import Context
from agentfly.tools.utils.task_repo import _RC_MARKER, checkout_task_commit


class FakeContainer:
    def __init__(self, fail_first_checkouts=0):
        self.cmds, self.fail = [], fail_first_checkouts

    async def run_cmd(self, cmd, timeout=None, workdir=None):
        self.cmds.append((cmd, workdir))
        if "checkout" in cmd and self.fail > 0:
            self.fail -= 1
            return f"error: pathspec did not match\n{_RC_MARKER}1"
        return f"\n{_RC_MARKER}0"


@pytest.mark.asyncio
async def test_checks_out_once_per_rollout_without_fetch():
    c, ctx = FakeContainer(), Context(rollout_id="r", metadata={"git_commit_hash": "task-branch"})
    assert await checkout_task_commit(c, ctx) is None
    assert await checkout_task_commit(c, ctx) is None  # second tool call: nothing to do
    assert len(c.cmds) == 1 and "checkout -q task-branch" in c.cmds[0][0] and c.cmds[0][1] == "/testbed"


@pytest.mark.asyncio
async def test_fetches_only_when_checkout_fails():
    c = FakeContainer(fail_first_checkouts=1)
    assert await checkout_task_commit(c, Context(rollout_id="r", metadata={"git_commit_hash": "b"})) is None
    assert ["checkout" in x[0] for x in c.cmds] == [True, False, True] and "fetch" in c.cmds[1][0]


@pytest.mark.asyncio
async def test_failure_is_reported_not_ignored():
    c = FakeContainer(fail_first_checkouts=2)
    error = await checkout_task_commit(c, Context(rollout_id="r", metadata={"git_commit_hash": "missing"}))
    assert error and "could not check out" in error and "missing" in error


@pytest.mark.asyncio
async def test_no_commit_means_no_git():
    c = FakeContainer()
    assert await checkout_task_commit(c, Context(rollout_id="r", metadata={"image_id": "x"})) is None
    assert c.cmds == []
