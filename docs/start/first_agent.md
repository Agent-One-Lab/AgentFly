# Build an Agent

A simplest agent can be build by initializing the agent instance with tools. The following shows a small example to build an agent using Qwen2.5.

```python
--8<-- "tests/docs/start/quick_example.py:agent_init"
```

Then, we can use the agent to do the task (or, say *rollout* in reinforcement learning scenario). The main interface is `run` method, which is fully asynchronous. You may use `asyncio.run` or `await` for the method.

```python
--8<-- "tests/docs/start/quick_example.py:agent_run"
```
Here, `max_turns` specifies the maximal number of rounds that the agent can iteract with the environment. `num_chains` specifies how many chains/trajectories the agent will run for a single query. After the running, we can obtain the results by getting its trajectories.

```python
--8<-- "tests/docs/start/quick_example.py:agent_trajectories"
```

It is in ShareGPT/OpenAI's input messages, and will look like something to this:
```
{
    'messages': [
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'What is the result of 1 + 1?'}]
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1 + 1"}}\n</tool_call>'}
            ],
            'tool_calls': [
                    {
                        'id': None, 'type': 'function',
                        'function': {
                            'name': 'calculator',
                            'arguments': {'expression': '1 + 1'}
                        }
                    }
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': None,
            'tool_name': 'calculator',
            'content': [
                {'type': 'text', 'text': '2'}
            ]
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'The result of 1 + 1 is 2.'}
            ],
            'tool_calls': [],
        }
    ]
}
```

To inspect the tokenization of a recorded segment without building a training
batch, use the standalone utility with explicit messages and a tokenizer:

```python
from agentfly.agents.utils.tokenizer import tokenize_trajectories

inputs = tokenize_trajectories(
    agent,
    messages_list=[result.trajectories[0].segments[0].messages],
    tokenizer=agent.tokenizer,
)
# input_ids, attention_mask, labels, action_mask, position_ids
```

For a scored result, `agent.to_verl_dataproto(result)` performs the complete
training conversion, including tokenization, rewards, and row alignment.
There is no need to call the tokenizer separately before conversion.

Now we have this built and run the agent. However, to run agent reinforcement learning, we still need several steps: define and get the tool to use, define reward functions, and finally, run the training.
