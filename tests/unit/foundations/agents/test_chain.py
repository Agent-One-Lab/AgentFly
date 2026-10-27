from agentfly.agents.rollout.structures import Chain, Step
from agentfly.agents.utils.messages import Messages
from agentfly.tools.types import ToolResult


def test_step_creation():
    # ``observation`` is derived from the step's ToolResult (env output).
    step = Step(
        is_terminal=False,
        type="Thought",
        description="This is a test thought",
        tool_result=ToolResult(name="t", arguments={}, observation="Test observation"),
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    assert step.is_terminal == False
    assert step.type == "Thought"
    assert step.description == "This is a test thought"
    assert step.observation == "Test observation"
    assert step.depth == 0
    assert len(step.children) == 0


def test_step_to_json():
    step = Step(
        is_terminal=False,
        type="Action",
        description="google_search",
        tool_result=ToolResult(name="google_search", arguments={}, observation="Test result"),
        messages=[{"role": "user", "content": "test"}],
    )

    json_data = step.to_json(use_messages=True)

    assert json_data["is_terminal"] == False
    assert json_data["type"] == "Action"
    assert json_data["description"] == "google_search"
    assert json_data["observation"] == "Test result"
    assert len(json_data["messages"]) == 1
    assert json_data["messages"][0]["role"] == "user"


def test_chain_creation():
    chain = Chain(info={"question": "test question"})

    assert chain.info["question"] == "test question"
    assert chain.root is None


def test_chain_add_step():
    chain = Chain(info={"question": "test question"})

    # Add root step
    root = chain.add_step(
        type="Thought",
        description="Initial thought",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    assert chain.root == root
    assert root.type == "Thought"
    assert root.description == "Initial thought"

    # Add child step
    child = chain.add_step(
        type="Action",
        description="google_search",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    assert len(root.children) == 1
    assert root.children[0] == child
    assert child.parent == root
    assert child.depth == 1


def test_chain_to_json():
    chain = Chain(info={"question": "test question"})
    chain.add_step(
        type="Thought",
        description="Initial thought",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )
    chain.add_step(
        type="Action",
        description="google_search",
        messages=Messages.from_turns([{"role": "user", "content": "test"}]),
    )

    json_data = chain.to_json()

    assert len(json_data) == 2
    assert json_data[0]["type"] == "Thought"
    assert json_data[1]["type"] == "Action"
