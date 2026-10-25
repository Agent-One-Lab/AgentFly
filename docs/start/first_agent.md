# Build an Agent

A simplest agent can be build by initializing the agent instance with tools. The following shows a small example to build an agent using Qwen2.5.

```python
--8<-- "tests/docs/start/quick_example.py:agent_init"
```

`template` names a chat template from [chat-bricks](https://github.com/Agent-One-Lab/chat-bricks),
the library that renders conversations (including tool calls and images) into model prompts.
List the available names (chat-bricks' templates plus those AgentFly adds) with:

```bash
python -c "import agentfly.agents; from chat_bricks.registry import TEMPLATES; print(sorted(TEMPLATES))"
```

Then, we can use the agent to do the task (or, say *rollout* in reinforcement learning scenario). The main interface is `run` method, which is fully asynchronous. You may use `asyncio.run` or `await` for the method.

```python
--8<-- "tests/docs/start/quick_example.py:agent_run"
```
Here, `max_turns` specifies the maximal number of rounds that the agent can iteract with the environment. `num_chains` specifies how many chains/trajectories the agent will run for a single query.

## A complete script

The snippets above run inside an async function. As a script, wrap them in `asyncio.run(main())`
under an `if __name__ == "__main__":` guard. **The guard is required**: vLLM starts worker
processes that re-import your script, and without it the run fails with *"An attempt has been
made to start a new process before the current process has finished its bootstrapping phase"*.

```python
--8<-- "tests/docs/start/first_agent_script.py:full_script"
```

`gpu_memory_utilization` is the fraction of GPU memory vLLM reserves (default 0.92). Lower it
when another process (a second agent, a reward model, a notebook) shares the GPU.

### Running a batch

`messages` accepts one conversation (a list of messages, as above) or a batch: a list of dicts,
each with a `messages` list. Other fields of a task, such as `answer`, are kept on its
trajectories as `metadata` and passed to the reward function by name.

```python
--8<-- "tests/docs/start/first_agent_script.py:batch_run"
```

### Reading the result

`run` returns a [`RunResult`](../api_references/agents/agent.md): a list of trajectories, one
per (task, chain), grouped by task. `result.rewards` lists their rewards (`None` without a
`reward_fn`). Each trajectory holds:

| Field | Meaning |
|---|---|
| `segments[-1].messages` | The conversation of this attempt (see below). Usually there is one segment; context folding and step rollouts record several. |
| `group_idx`, `chain_idx` | Which task (position in the batch) and which of its `num_chains` attempts. |
| `reward`, `metrics` | The reward and any extra values the reward function returned. |
| `finish_reason` | Why it stopped: `no_tool_calls` (the model answered without calling a tool), `max_turns`, `terminal` (a finishing tool), `max_model_len`, ... |
| `num_turns`, `tool_call_counts` | Model generations, and tool calls per tool. |
| `metadata` | The task's extra fields (e.g. `answer`). |

`agent.extract_final_response(messages)` gives the agent's final answer, the same text a
reward function receives as `final_response`.

```python
--8<-- "tests/docs/start/first_agent_script.py:read_result"
```

Output from a run with Qwen2.5-3B-Instruct:

```
4 trajectories
task 0 chain 0: answer='The result of 123 * 45 is 5535.' turns=2 finish=no_tool_calls tools={'calculator': 1}
task 0 chain 1: answer='The result of 123 * 45 is 5535.' turns=2 finish=no_tool_calls tools={'calculator': 1}
task 1 chain 0: answer='The result of \\(2^{10} - 24\\) is 1000.' turns=2 finish=no_tool_calls tools={'calculator': 1}
task 1 chain 1: answer='The result of the expression \\(2 ** 10 - 24\\) is 1000.' turns=2 finish=no_tool_calls tools={'calculator': 1}
```

The messages are in OpenAI chat format. Assistant messages also record the parsed `tool_calls`,
whether they are trained on (`loss`), and their `status`:

```json
[
  {"role": "user", "content": [{"type": "text", "text": "What is 123 * 45?"}]},
  {
    "role": "assistant",
    "content": [{"type": "text", "text": "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"123 * 45\"}}\n</tool_call>"}],
    "tool_calls": [{"id": "chatcmpl-tool-912e...", "type": "function",
                    "function": {"name": "calculator", "arguments": "{\"expression\": \"123 * 45\"}"}}],
    "loss": true,
    "status": "continue"
  },
  {"role": "tool", "tool_call_id": "chatcmpl-tool-912e...", "tool_name": "calculator",
   "content": [{"type": "text", "text": "5535"}]},
  {
    "role": "assistant",
    "content": [{"type": "text", "text": "The result of 123 * 45 is 5535."}],
    "tool_calls": [],
    "loss": true,
    "status": "terminal"
  }
]
```

### Saving the result

`RunResult` is a pydantic model, so it saves as JSON directly:

```python
--8<-- "tests/docs/start/first_agent_script.py:save_result"
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
There is no need to call the tokenizer separately before conversion. It builds
verl's training batch, so it needs the training extra (`uv sync --extra verl`, or
`uv pip install "agentfly-rl[verl]"`).

Now we have this built and run the agent. However, to run agent reinforcement learning, we still need several steps: define and get the tool to use, define reward functions, and finally, run the training.
