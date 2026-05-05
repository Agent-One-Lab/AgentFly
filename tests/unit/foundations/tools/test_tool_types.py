"""Unit tests for the typed tool result class and its from_raw factory."""
import pytest

from agentfly.tools.types import ToolResult, ToolReturn


class TestToolResultFromRawString:
    def test_basic(self):
        tr = ToolResult.from_raw("hello", name="echo", arguments={"x": 1})
        assert tr.observation == "hello"
        assert tr.name == "echo"
        assert tr.arguments == {"x": 1}
        assert tr.info == {}
        assert tr.image is None
        assert tr.status == "success"

    def test_max_length_truncation(self):
        tr = ToolResult.from_raw("a" * 100, name="t", arguments={}, max_length=10)
        assert tr.observation == "a" * 10

    def test_no_truncation_when_max_length_none(self):
        tr = ToolResult.from_raw("a" * 100, name="t", arguments={})
        assert len(tr.observation) == 100


class TestToolResultFromRawDict:
    def test_basic(self):
        tr = ToolResult.from_raw(
            {"observation": "see"}, name="t", arguments={"a": 1}
        )
        assert tr.observation == "see"
        assert tr.info == {}
        assert tr.image is None

    def test_image_extracted_to_field(self):
        tr = ToolResult.from_raw(
            {"observation": "see", "image": "http://x", "score": 0.5},
            name="t",
            arguments={"a": 1},
        )
        assert tr.observation == "see"
        assert tr.image == "http://x"
        assert tr.info == {"score": 0.5}

    def test_max_length_truncation_with_marker(self):
        tr = ToolResult.from_raw(
            {"observation": "a" * 100}, name="t", arguments={}, max_length=10
        )
        assert tr.observation == "a" * 10 + "...(truncated)"

    def test_does_not_mutate_caller_dict(self):
        d = {"observation": "see", "image": "http://x", "score": 0.5}
        before = dict(d)
        ToolResult.from_raw(d, name="t", arguments={})
        assert d == before, "from_raw must not mutate the caller's dict"


class TestToolResultFromRawErrors:
    def test_dict_missing_observation_raises(self):
        with pytest.raises(ValueError, match="observation key"):
            ToolResult.from_raw({"foo": "bar"}, name="t", arguments={})

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            ToolResult.from_raw(42, name="t", arguments={})


class TestToolResultPassthrough:
    def test_existing_tool_result_returned_as_is(self):
        original = ToolResult(name="t", arguments={}, observation="x")
        assert ToolResult.from_raw(original, name="ignored", arguments={}) is original


class TestToolResultToDict:
    def test_basic(self):
        tr = ToolResult(name="t", arguments={"x": 1}, observation="hi")
        d = tr.to_dict()
        assert d == {
            "name": "t",
            "arguments": {"x": 1},
            "observation": "hi",
            "status": "success",
            "info": {},
        }

    def test_with_image(self):
        tr = ToolResult(name="t", arguments={}, observation="hi", image="http://x")
        d = tr.to_dict()
        assert d["image"] == "http://x"

    def test_image_field_omitted_when_none(self):
        tr = ToolResult(name="t", arguments={}, observation="hi")
        assert "image" not in tr.to_dict()

    def test_legacy_shape_round_trip(self):
        # Build from a legacy-style dict, then to_dict must reproduce a dict that
        # has all the keys the existing chain-side consumers expect.
        legacy_in = {"observation": "see", "image": "http://x", "score": 0.5}
        tr = ToolResult.from_raw(legacy_in, name="t", arguments={"a": 1})
        d = tr.to_dict()
        assert set(d) == {"name", "arguments", "observation", "status", "info", "image"}
        assert d["info"] == {"score": 0.5}


class TestToolReturn:
    def test_typed_dict_is_annotation_only(self):
        x: ToolReturn = {"observation": "see", "image": "http://x"}
        assert isinstance(x, dict)
