from agentfly.agents import HFAgent, ClientConfig
from agentfly.tools import code_interpreter


def test_agent_tool_parser():
    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tool_parser_name="hermes",
        tools=[code_interpreter],
        backend="client",
        backend_config=ClientConfig(
            base_url="http://localhost:8000/v1", api_key="EMPTY"
        ),
    )

    responses = [
        "<think>I need to search for information.</think><tool_call>google_search</tool_call><tool_input>query: test query</tool_input>",
        """<tool_call>{"name": "google_search", "arguments": {"query": "test query"}}</tool_call>""",
    ]

    result = agent.parse(responses)
    print(result)
