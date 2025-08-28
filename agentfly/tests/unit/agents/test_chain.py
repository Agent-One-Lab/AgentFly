import pytest
from ....agents.chain.chain_base import Chain, Node, ChainRollout


def test_node_creation():
    node = Node(
        is_terminal=False,
        type="Thought",
        description="This is a test thought",
        observation="Test observation"
    )
    
    assert node.is_terminal == False
    assert node.type == "Thought"
    assert node.description == "This is a test thought"
    assert node.observation == "Test observation"
    assert node.depth == 0
    assert len(node.children) == 0


def test_node_to_json():
    node = Node(
        is_terminal=False,
        type="Action",
        description="google_search",
        observation="Test result",
        messages=[{"role": "user", "content": "test"}]
    )
    
    json_data = node.to_json(use_messages=True)
    
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


def test_chain_add_node():
    chain = Chain(info={"question": "test question"})
    
    # Add root node
    root = chain.add_node(
        type="Thought",
        description="Initial thought"
    )
    
    assert chain.root == root
    assert root.type == "Thought"
    assert root.description == "Initial thought"
    
    # Add child node
    child = chain.add_node(
        type="Action",
        description="google_search"
    )
    
    assert len(root.children) == 1
    assert root.children[0] == child
    assert child.parent == root
    assert child.depth == 1


def test_chain_to_json():
    chain = Chain(info={"question": "test question"})
    chain.add_node(type="Thought", description="Initial thought")
    chain.add_node(type="Action", description="google_search")
    
    json_data = chain.to_json()
    
    assert len(json_data) == 2
    assert json_data[0]["type"] == "Thought"
    assert json_data[1]["type"] == "Action"


