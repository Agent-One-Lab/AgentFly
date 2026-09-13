# Rollout

## Returned Data

Both chain and step rollouts return the same hierarchy:
`RunResult.trajectories` → `Trajectory.segments` → `Segment.messages`.

`RunResult.rollout` identifies the producer as `"chain"` or `"step"`.
Training conversion takes the result explicitly:

```python
result = await agent.run(messages=messages, max_turns=10, rollout="chain")
batch = agent.to_verl_dataproto(result)
```

The converter does not consult the agent's last-run caches. Missing or unsupported
rollout identifiers raise an error. Use the in-memory result: serialization
preserves the identifier, but excludes internal runtime steps.

Rollout results retain empty and context-only segments. Conversion selects only
rows with policy-token targets, leaving the result unchanged. `repeat_times`
counts selected rows (including padding) per original trajectory, not the raw
number of segments. A result with no trainable rows raises `ValueError`.

Rollout subclasses only need to implement `run()`. Training conversion is
separate; the rollout-bound conversion/tokenization methods have been removed.
The shared low-level tokenizer remains available as
`agentfly.agents.utils.tokenizer.tokenize_trajectories`.

::: agentfly.agents.types.Segment

::: agentfly.agents.types.Trajectory

::: agentfly.agents.types.RunResult

## Chain Generation

Base class for chain-based generation:

::: agentfly.agents.rollout.strategies.chain_rollout.ChainRollout
    options:
      show_inheritance: true
