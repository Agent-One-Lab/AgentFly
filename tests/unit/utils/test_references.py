"""Unit tests for the component-reference resolver."""
import textwrap

import pytest

from agentfly.utils.references import is_reference, load_object, resolve_reference


REGISTRY = {"miniswe": "AGENT_OBJ", "bash": "TOOL_OBJ"}


def test_is_reference():
    assert not is_reference("miniswe")
    assert not is_reference("qa_em_reward")
    assert is_reference("pkg.mod:attr")
    assert is_reference("/abs/path.py:attr")
    assert is_reference("./rel.py:attr")
    assert is_reference("dir/file.py")


def test_registered_name_wins():
    assert resolve_reference("miniswe", REGISTRY, "agent") == "AGENT_OBJ"
    assert resolve_reference("MiniSWE", REGISTRY, "agent") == "AGENT_OBJ"  # case-insensitive


def test_unknown_bare_name_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_reference("nope", REGISTRY, "reward")


def test_module_attribute_reference():
    # os.getcwd is a stable importable attribute
    import os
    assert resolve_reference("os:getcwd", {}, "tool") is os.getcwd
    assert load_object("os.path:join") is os.path.join


def test_file_attribute_reference(tmp_path):
    mod = tmp_path / "my_component.py"
    mod.write_text(textwrap.dedent("""
        SENTINEL = "loaded-from-file"
        def my_reward():
            return 1.0
    """))
    assert resolve_reference(f"{mod}:SENTINEL", {}, "reward") == "loaded-from-file"
    fn = resolve_reference(f"{mod}:my_reward", {}, "reward")
    assert fn() == 1.0


def test_file_scheme_prefix(tmp_path):
    mod = tmp_path / "c.py"
    mod.write_text("VALUE = 42\n")
    assert resolve_reference(f"file://{mod}:VALUE", {}, "tool") == 42


def test_bad_reference_shapes():
    with pytest.raises(ValueError):
        load_object("no_attribute_here")           # missing ':attr'
    with pytest.raises(ValueError):
        load_object(":attr")                        # missing location
    with pytest.raises(ModuleNotFoundError):
        load_object("no.such.module:attr")
    with pytest.raises(AttributeError):
        load_object("os:definitely_not_here")
    with pytest.raises(FileNotFoundError):
        load_object("/no/such/file.py:attr")


def test_registered_name_checked_before_import():
    # a name that is ALSO shaped like a path still prefers the registry
    reg = {"pkg.mod:attr": "REGISTERED"}
    assert resolve_reference("pkg.mod:attr", reg, "tool") == "REGISTERED"
