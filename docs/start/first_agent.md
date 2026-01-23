# Build an Agent

A simplest agent can be build by initializing the agent instance with tools. The following shows a small example to build an agent using Qwen2.5.

```python
from agentfly.agents import HFAgent
from agentfly.tools import calculator
agent = HFAgent(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    tools=[calculator],
    template="qwen2.5",
    backend="async_vllm",
)
```

Then, we can use the agent to do the task (or, say *rollout* in reinforcement learning scenario). The main interface is `run` method, which is fully asynchronous. You may use `asyncio.run` or `await` for the method.

```python
messages = [{"role": "user", "content": "What is the result of 1 + 1?"}]
await agent.run(
    messages=messages,
    max_turns=3,
    num_chains=1
)
```
Here, `max_turns` specifies the maximal number of rounds that the agent can iteract with the environment. `num_chains` specifies how many chains/trajectories the agent will run for a single query. After the running, we can obtain the results by getting its trajectories.

```python
trajectories = agent.trajectories
print(trajectories)
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

To help the training, we can obtain the tokenized trajectories by calling `tokenize_trajectories` method
```python
inputs = agent.tokenize_trajectories()
>>> # 'input_ids': tokenized ids of trajectories
    # 'attention_mask': attention_mask of ids
    # 'labels': used for supervised finetuning
    # 'action_mask': mask where llm generated response are set '1', otherwise '0'
```

Now we have this built and run the agent. However, to run agent reinforcement learning, we still need several steps: define and get the tool to use, define reward functions, and finally, run the training.
