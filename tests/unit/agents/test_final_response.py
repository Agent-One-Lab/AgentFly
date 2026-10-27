"""``final_response``: what rewards grade as the agent's answer."""

from agentfly.agents import HFAgent
from agentfly.tools import answer_math, calculator


def make_agent():
    return HFAgent(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        tools=[calculator, answer_math],
        backend_config={"backend": "client", "base_url": "http://localhost:1/v1", "api_key": "EMPTY"},
        monitors=[],
    )


def text(s):
    return [{"type": "text", "text": s}]


QUESTION = {"role": "user", "content": text("What is 48 + 24?")}
CALL = {"role": "assistant", "content": text("Let me compute."), "tool_calls": [{"function": {"name": "calculator"}}]}
CALC_RESULT = {"role": "tool", "tool_name": "calculator", "content": text("72")}


def test_answer_tool_output_is_the_answer():
    answer = {"role": "tool", "tool_name": "answer_math", "content": text("\\boxed{72}")}
    msgs = [QUESTION, CALL, CALC_RESULT, {"role": "assistant", "content": text("Submitting.")}, answer]
    assert make_agent().extract_final_response(msgs) == "\\boxed{72}"


def test_model_answer_after_tool_result_is_the_answer():
    msgs = [QUESTION, CALL, CALC_RESULT, {"role": "assistant", "content": text("The answer is \\boxed{72}.")}]
    assert make_agent().extract_final_response(msgs) == "The answer is \\boxed{72}."


def test_ordinary_tool_output_is_never_the_answer():
    # The episode stopped right after the calculator (e.g. max_turns): grade the model, not the tool.
    assert make_agent().extract_final_response([QUESTION, CALL, CALC_RESULT]) == "Let me compute."


def test_plain_string_content():
    assert make_agent().extract_final_response([QUESTION, {"role": "assistant", "content": "72"}]) == "72"
