Agent Rollout
==============

!!! warning
    The findings discussed here are based on preliminary observations and have not been rigorously validated through controlled experiments.

## Rollout for LLM
In reinforcement learning, rollout refers to the stage where the LLM generates the responses to queries. The queries and responses will be concatenated later in the training stage, to calculate the advantages and update the LLM. To maximize the trainig efficiency, the rollout stage is a balance between exploration and exploitation.

Traditionaly, this is achieved by setting a proper temperature (e.g. 1.0). A higher temperature encourage the model to generate more diverse responses (explore more), while a lower temperature limits the model to generate more accurate responses.

## Rollout for Agent
However, things become more complicated for agents: The rollout for agents is naturally multi-turn, where agents first generate responses, call tools, then get the observation from the tool. The rollout is not just affected by previous generated content, but will also be diverged by tool observations. This makes agent rollout more complex and flexible. How do we set the rollout strategy, to maximize the training efficiency?

Unfortunately, there are not many studies explored the agent reinforcement learning, not to say the best rollout strategy. We provide some initial thoughts on how we can do agent rollout for reinforcement learning.

### Chain-Search
Intuitively, for each query, we generate one response, and call the tool, append the observation... Repeating this process will form a chain-like agent interaction trajectory. This is what AgentFly adopts in the rollout stage. Although intuitive and simple, as there are more and more turns, the rollout trajectories will diverge more and more, possibly making the training unstable.


### Tree-Search
In contrast to chain-search, tree-search generates multiple responses in each turn, trying to explore more action space for each query and obtain tree-based trajectories. Compared to single chain, tree-search will more likely to obtain both successful and failed trajectories, therefore better for RL learning.

### Filtering
Some studies show that some trajectories with specific patterns will lead to failture of agent reinforcement learning, even if their rewards are given accurately. SimpleTIR found a pattern, which they call void turns that "contain fragmented code or repetitive sentences and are often triggered by the premature generation of an eos token" [1].

### Single-Turn v.s. Multi-Turn
We found that training with shorter turns are more stable then training with longer turns, as shown in our report. This is also validated in other studies like GiGPO [1] and SimpleTIR. Another proposal is to convert multi-turn rollout into single-turn, which have the following benifits: (1) Memory efficient: we can only include the tool call and observations in history. Advantages, losses are only calculated in current turn response. (2) Stable: Training on single turn is much more stable than multi-turn.

### Our strategy
Currently, we adopt chain-based rollout. For each query, we initially generate *n* responses. Therefore we maintain *n* chains. For all generations, we use a fixed temperature. We will explore more on this and we also welcome your contributions!


[1] SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning

[2] GiGPO: Group-in-Group Policy Optimization for LLM Agent Training


## Returned Data

`agent.run(...)` returns a `RunResult` containing a list of `Trajectory` objects.
Each trajectory represents one task attempt and contains ordered `Segment` objects.
Each segment holds its message sequence in `segment.messages`:

```python
result = await agent.run(messages=messages, max_turns=10, num_chains=1)
for trajectory in result:
    for segment in trajectory.segments:
        for message in segment.messages:
            print(message)
```

The same `Segment` type is used by both strategies:

| Rollout | Segments per trajectory |
|---|---|
| Chain without context folding | One full conversation |
| Chain with context folding | Completed pre-fold views plus the final view |
| Step | One bounded prompt plus response per generation |

Segments may repeat earlier messages as context. They are not disjoint transcript
slices, and concatenating them does not reconstruct the original episode.
Message dictionaries retain their existing fields, including sampled `token_ids`,
tool calls, and multimodal content. The segment wrapper does not change loss masks
or reward assignment.

To construct results directly, import `Segment`, `Trajectory`, and `RunResult` from
`agentfly.agents`. Replace the former bare segment lists with
`Segment(messages=[...])`; serialized segments now have the shape
`{"messages": [...]}`. Consumers must use `.messages` rather than iterate the
segment object itself.

Both strategies preserve task fields in `Trajectory.metadata` (excluding reserved
fields such as the separately stored trajectory/group identifiers). Execution time
is recorded in `rollout_time_sec`, from episode setup through resource cleanup,
excluding time waiting for a concurrency slot. With no reward function configured,
`reward` is `None`; a computed zero reward remains `0.0`.

The former `agent.timing_data` accessor is removed; read
`result.trajectories[i].rollout_time_sec` for an episode's duration. This is not
the total batch wall time.

Both strategies report `agent/rollout/time/min`, `max`, and `avg` from these
trajectory durations. Missing durations (`None`) are skipped, while `0.0` is valid.
If no duration is available, timing summaries and slowest-trajectory logging are
skipped; an empty batch emits no end-of-rollout summary events.

The existing `agent/rollout/slowest_message` event now contains a serialized
`Trajectory`, like `agent/rollout/sample_trajectory`, for both chain and step.
Its payload uses `segments` instead of the former top-level `messages`; metadata,
reward, metrics, and identifiers follow the `Trajectory` schema, and internal
`steps` remain excluded. Duration ties select the first trajectory in result order.
Log consumers expecting the old chain-specific payload must update accordingly.

Execution counts are also stored on each trajectory: `num_turns` counts completed
model generations, and `tool_call_counts` maps each tool-result name to its number
of recorded attempts. Invalid/error results count; proposed calls that never
execute do not. Total calls are `sum(trajectory.tool_call_counts.values())` when
counts are available. No separate total is stored.

Both rollouts derive these fields from runtime records, not segment messages:
chain counts its generation nodes and tool results, while step counts its generation
records and attached tool results. Repeated history, context folding, and observation
roles do not affect the counts. They survive serialization even though `steps` do not.
No tool-execution behavior changes: step still executes only the first proposed
call in a generation, while chain can execute multiple calls.

`agent/rollout/avg_turns`, `avg_tool_calls`, and `tool_calls/<name>` now aggregate
these fields. `None` means unavailable (including older results) and is excluded
from the corresponding average; `0` turns and `{}` tool counts are measured zeros
and are included. Per-tool averages use all trajectories with known tool counts,
including those with zero calls to that tool. If no counts are available, the
corresponding events are omitted; reporting does not infer counts from messages.
`agent/rollout/avg_segments` remains the average number of context segments.

### The metrics step

Every `agent/rollout/*` event is logged against the custom axis `agent/rollout/step`.
A rollout strategy is constructed per `agent.run(...)` call, so it cannot count
training steps on its own; a caller that has a step counter passes it:

```python
result = await agent.run(messages=messages, max_turns=10, global_step=trainer_step)
```

The verl trainer passes its `global_steps` for both training and validation rollouts,
so the agent's curves share the trainer's axis, validation does not advance it, and a
resumed run continues from its checkpoint's step. Without `global_step` the strategy
falls back to counting its own runs — for a per-call strategy that is always `1`, and
a metric with a single x value renders as a bar rather than a line. The step is fixed
before any chain starts, so per-chain records (`agent/rollout/trajectory`,
`agent/rollout/info`) and the end-of-rollout summaries share one x.

### Per-task pass-rate distribution

When a batch contains several rollouts per task, each task has a *pass rate*: the
fraction of its samples whose reward reaches `RolloutMetrics.success_threshold`
(default `1.0`). The shape of that distribution over tasks says whether the pool is
worth training on — mass piled at 0 and 1 (a U-shape) means the sampled tasks are
mostly always-failed or always-solved, and a task whose samples all score alike gives
GRPO no advantage signal at all, however healthy the mean reward looks.

| Metric | Meaning |
| --- | --- |
| `agent/rollout/group/pass_rate_hist` | Histogram of per-task pass rates for this step |
| `agent/rollout/group/pass_rate_mean` | Mean per-task pass rate |
| `agent/rollout/group/frac_all_fail` | Share of tasks no sample solved |
| `agent/rollout/group/frac_all_pass` | Share of tasks every sample solved |
| `agent/rollout/group/frac_mixed` | Share of tasks with both outcomes — the ones that produce gradient |
| `agent/rollout/group/frac_zero_advantage` | Share of tasks whose samples all scored identically |
| `agent/rollout/group/reward_std_mean` | Mean within-task reward spread |
| `agent/rollout/group/num_groups`, `samples_per_task` | Size of the batch this was computed over |

Samples are grouped by `Trajectory.group_id`, which both strategies stamp on every
rollout of a task. Rollouts without a reward are skipped rather than counted as
failures, and a rollout with no `group_id` forms its own single-sample task. The block
is emitted only when some task actually has more than one sample: with one rollout per
task (validation, or `num_chains=1`) every pass rate is 0 or 1 by construction and the
distribution carries no information about difficulty. `frac_zero_advantage` and
`reward_std_mean` read raw rewards, so they stay meaningful for continuous rewards
that never reach the pass threshold.

For a scored trajectory, both training converters broadcast the full outcome reward
`R` to every retained training segment: `[R, ..., R]`, including earlier
context-folded chain views.
The reward is placed on each segment's reward-mask token; it is not divided by the
number of segments or discounted here. Raw environment step rewards and downstream
advantage computation are unchanged.

`Trajectory.metrics` contains reward-function extras and numeric tool averages in
both strategies. Tool metrics are grouped by tool name and metric key, using
`tool/<name>/<key>`; missing and non-numeric values are ignored, and booleans count
as 0/1. The first-class `ToolResult.step_reward` is included as a logging average
when supplied; the original per-call rewards remain unchanged in the runtime
records. Tool aggregates take precedence over reward extras with the same
`tool/...` key, matching the existing chain convention. They are collected even
when no episode reward function is configured.

This migration unifies the conversation container, not the entire training-signal
protocol. `Trajectory.steps` remains an internal, non-serialized compatibility
channel for existing advantage estimators. Its rollout-specific semantics and the
runtime step records are unchanged.

Both exporters use the internal `rollout.conversion.build_segment_batch` helper
for segment flattening, tokenization, equal outcome rewards, metric expansion,
and row padding. It returns a single trajectory/segment index mapping, including
padding copies, which each exporter uses to align its own step signals. The helper
does not inspect runtime `steps` or rollout state, and introduces no new result
types. Existing exporter-specific missing-value rules and trainer layouts remain
unchanged. Multimodal metadata is repeated alongside its padded token row.

### Explicit training conversion

`RunResult.rollout` is the producer identifier: `"chain"` or `"step"`, never a
rollout instance. Built-in rollouts set it, and agent postprocessing preserves it.

```python
result = await agent.run(messages=messages, max_turns=10, rollout="step")
batch = agent.to_verl_dataproto(result, pad_to_multiple_of=8)
```

Conversion dispatches from this field, not the agent's latest-run caches. An
earlier result can be converted after another run, including a different rollout.
Keep the agent's tokenization configuration unchanged (tokenizer, processor,
template, prompt tools, and maximum length).

`run()` records execution, independently of training. Its result retains empty
and context-only segments, including a freshly folded context that was never
followed by a generation. ActionAgent's experimental postprocessing policies are
retained but disabled by default.

Only conversion selects training rows. It skips segments without assistant
messages before tokenization, then skips rows whose action mask contains no
policy tokens (for example, after truncation). Segments ending in a tool
observation are still eligible if they contain assistant targets. Conversion
does not mutate trajectories, segments, messages, or runtime steps.

Row mappings retain the original trajectory/segment indices. `repeat_times`
counts retained rows per original trajectory, with zero for those contributing
none. Divisor padding repeats the last retained row and increases its owning
trajectory's count, even when later trajectories have no training rows.
Per-row multimodal data and step signals follow the same selection. If no
trainable rows remain, conversion raises a clear `ValueError`; it does not
silently return an empty training batch.

The no-argument conversion API is no longer supported. Manually constructed
results default to `rollout=None` for inference/inspection; training conversion
rejects missing and unsupported identifiers explicitly. Custom execution
strategies are still supported, but registering one does not automatically add a
stateless training converter; its result must use a supported protocol or callers
must provide their own conversion outside the rollout interface.

The only required rollout method is `run()`, returning a `RunResult`.
Inference-only custom rollouts do not need any training methods. The former
`Rollout.to_dataproto` and `Rollout.tokenize_trajectories` methods, including the
built-in strategy wrappers, have been removed. Training callers use
`agent.to_verl_dataproto(result)`; tokenization-only inspection and tests can
call `agentfly.agents.utils.tokenizer.tokenize_trajectories` directly with
explicit conversations and the agent's tokenizer. The training converter
already calls that utility internally; callers do not need to tokenize first.

The marker survives serialization, but internal `Trajectory.steps` do not.
Use the in-memory result for training; JSON is not a complete training snapshot.
This removes conversion's last-run dependency, not all shared state from agent
execution, and does not make concurrent calls to `agent.run` generally safe.

## Loop Control

Every rollout — chain or step — is a loop: *generate → (maybe) call a tool → observe → repeat*. **Loop control** decides, at each turn, whether the loop should **continue**, **end** the episode, or **raise** (fail fast). The decision lives in one place (the rollout) and uses a single vocabulary, so `chain` and `step` behave identically.

### The control vocabulary

| Control | The loop… | Trajectory outcome |
|---|---|---|
| `continue` | runs another turn | — |
| `end` | ends the episode **gracefully** | kept, reward computed, trained on |
| `raise` | throws, unwinds the rollout | discarded — not a training sample |

### Who decides, and in what order

At a turn boundary the rollout resolves exactly one control, with this precedence:

1. **The tool wins.** If a tool result carries an explicit `control`, the rollout honors it verbatim (e.g. an environment tool that reached a terminal state returns `control="end"`).
2. Otherwise the tool result's `status` — a pure *diagnosis* — routes to a run policy:
    - malformed call (`status="invalid"`: bad tool name / arguments) → `on_invalid_tool_call`
    - exception inside the tool body (`status="error"`) → `on_tool_error`
    - a clean result → `continue`
3. A turn with **no tool call at all** → `on_no_tool_call`.
4. The model may also end explicitly through its message `status` (`"terminal"` / `"finish"`), e.g. an "end task" sentinel.

Hard limits are independent of all of the above and are never overridable: the loop always ends at `max_turns`, and (for `chain`) when the prompt would exceed the model's context length. The `step` rollout uses a bounded prompt window, so `max_turns` is its only hard cap.

### Termination labels

Both strategies use the same `Trajectory.finish_reason` vocabulary. These labels
describe a decision already made by the loop; they do not change control policies.

| Label | Meaning |
|---|---|
| `terminal` | Explicit model stop through the terminal-message branch |
| `tool_control_end` | A tool explicitly returned `control="end"`, regardless of its status |
| `invalid_end` | An invalid tool call was routed to `end` by policy |
| `tool_error_end` | A tool error was routed to `end` by policy |
| `tool_end` | Other tool stop, for example from a custom control policy |
| `no_tool_calls` | A response without tool calls was routed to `end` |
| `max_turns` | The generation-turn budget was exhausted |
| `max_model_len` | The chain context-length guard stopped execution |

No-tool responses still follow `on_no_tool_call` before terminal-message handling.
A `raise` control still raises an exception rather than returning a trajectory
with an error label. A context-length stop in the chain retains `max_model_len`,
even if the tool also requested an end.

Previously, chain rollouts reported all tool stops as `terminal`. Consumers that
used that label for tool/environment completion should now handle the specific
tool-stop labels above. Existing step-rollout label strings are unchanged.

### The run policies

Three knobs, each `continue | end | raise`, set under `agent.run_config.rollout_config`:

| Policy | Fires when | Default |
|---|---|---|
| `on_no_tool_call` | the turn produced no tool call | `end` |
| `on_invalid_tool_call` | a malformed call (unknown name / bad args) | `end` |
| `on_tool_error` | an exception was raised inside a tool body | `continue` |

```yaml
agent:
  run_config:
    rollout: step
    rollout_config:
      prompt_builder: alfworld_flat
      on_no_tool_call: end          # e.g. "continue" to let the model take a tool-less "thinking" turn
      on_invalid_tool_call: end
      on_tool_error: continue       # feed the error text back as an observation and keep going
```

or programmatically:

```python
await agent.run(
    messages=...,
    max_turns=50,
    rollout="step",
    rollout_config={"on_no_tool_call": "continue", "on_tool_error": "raise"},
)
```

### Tools that steer the loop

By default a tool does **not** control the flow — it only reports what happened via `status` (`success` / `error` / `invalid`). A tool opts in to steering by returning `control` in its result dict:

```python
@tool(name="alfworld_step", stateful=True)
async def alfworld_step(action: str, context: "Context"):
    obs, reward, done, info = await env.step(action)
    won = bool(info.get("won"))
    return {
        "observation": ...,
        "control": "end" if won else None,   # end the episode once the task is solved
    }
```

`None` (the default) means "don't steer" — the rollout's run policy decides. This keeps a clean separation: **tools diagnose and may declare intent; the rollout decides and acts.**


## Training on What Was Sampled

The rollout carries the **token ids the model actually generated** on each assistant message (`token_ids`), so `agent.to_verl_dataproto(result)` trains on the sampled sequence instead of re-tokenizing decoded text (which drifts — e.g. `<think>` re-encodes differently on Qwen2.5). It also takes the tool schemas rendered into the prompt from one agent-owned source (`agent.prompt_tools()`) for both generation and training, so the prompt the model was conditioned on is the prompt it is trained under. Both come with per-batch diagnostics (`token_drift`, `prompt_check`). See [Training / Inference Consistency](training_consistency.md).


## Asynchronous Implementation

To make the full rollout pipeline asynchronous, there are three main components consuming time: *Generation*, *Tool Calling*, and *Reward Calculation*.

- For generation, we directly wrap verl's asynchronous rollout worker.

- For tool calling, we define each execution function to be asynchronous. For tools that require environments, we also ensure the envionment's methods to be asynchronous.

- For reward calculation, we adopt similar design as tool for them to be asynchronous.

For details, refer to the specific sections in the documentation.
