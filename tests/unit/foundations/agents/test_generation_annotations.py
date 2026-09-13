"""Generation annotations must resolve using the module's declared imports."""

import ast
import inspect
from typing import get_type_hints

import pytest

from agentfly.agents.rollout.agent import RolloutAgent
from agentfly.agents.rollout.loop import generation


@pytest.mark.parametrize(
    "function",
    [generation.prepare_generation_config, generation.generate_response],
)
def test_agent_annotation_resolves_from_type_checking_imports(function):
    # Evaluate the actual static imports in an isolated namespace. Supplying
    # RolloutAgent directly to get_type_hints would hide a missing import.
    namespace = vars(generation).copy()
    tree = ast.parse(inspect.getsource(generation))
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            imports = ast.Module(body=node.body, type_ignores=[])
            exec(compile(imports, generation.__file__, "exec"), namespace)

    assert get_type_hints(function, globalns=namespace)["agent"] is RolloutAgent
    # Resolving static annotations must not make this a runtime import.
    assert "RolloutAgent" not in vars(generation)
