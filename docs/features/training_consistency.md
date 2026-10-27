# Training / Inference Consistency

RL updates are only on-policy if the tokens the trainer scores are the tokens the model was
actually conditioned on and sampled. In a multi-turn agent framework there are two ways that
silently breaks, and AgentFly guards both:

| where | what goes wrong | AgentFly mechanism |
|---|---|---|
| **response** | the sampled response is decoded to text (to run tools / step the env) and later *re-tokenized* for training; re-encoding is not the identity | sampled ids ride on the message as `token_ids` and are spliced verbatim |
| **prompt** | generation and training render the prompt differently (e.g. one side includes a tools block the other doesn't) | the agent is the single source of prompt tools for both; a per-run guard compares the sampled prompt ids to the training render |

Both were found while reproducing verl-agent's ALFWorld results with AgentFly.

## Response side: train on the sampled token ids

### The problem — token drift

Tokenizers are not injective on text. With Qwen2.5 the model samples `<think>` as
`["<", "think", ">"]` = `[27, 26865, 29]`, but `tokenizer.encode("<think>")` gives
`["<th", "ink", ">"]` = `[13708, 766, 29]`. Measured on Qwen2.5-1.5B-Instruct, **100% of sampled
sequences and ~4% of tokens** re-tokenize differently. Native tool calls are worse: the
structured `tool_calls` field is re-serialized by the chat template with its own JSON spacing.
Training on the re-encoded text means `old_log_probs`, the importance ratio, and the KL to
the reference are all computed on a sequence the policy never produced — every update is
off-policy, and the KL runs away.

### What AgentFly does

1. **The backend keeps the ids.** `AsyncVerlBackend.generate_async(..., return_dict=True)`
   returns `response_ids` — verl's `responses` with padding stripped, exactly the sampled
   sequence, eos included when the model stopped on it (verl passes vLLM's ids through
   untouched).
2. **The ids ride on the message.** `generate_response` calls `attach_token_ids`, which sets
   `message["token_ids"]` on the assistant message it appends. `Messages`, `Step`, and
   trajectory storage preserve the key, so nothing downstream changes.
3. **Tokenization splices them.** `tokenize_trajectories` hands the conversation to chat-bricks,
   which uses `token_ids` **verbatim** for the assistant span — never decoded, never compared to
   the rendered text — and text-encodes only the template's structural glue around it, matched
   to the terminator by id. Masks, labels, and the reward position are unchanged. See the
   chat-bricks guide *Train on sampled token ids* for the splice semantics.

The ids are the **full** generation even when the parsed `content` was truncated for tool
extraction (the `ActionAgent` cuts at the first `</action>`). That mirrors verl-agent, which
trains on `batch['responses']` and runs its action regex on a text copy: the environment sees
the extracted action; training sees what was sampled.

### The diagnostic — `token_drift`

`agent.to_verl_dataproto(result, log_token_drift=True)` (off by default) re-tokenizes
up to 64 real rows per step batch with the ids stripped and reports how far the
text path would have drifted. It appears on the per-batch summary line:

```
[StepRollout] training batch: rows=6150 ... token_drift={frac_ids_end_with_eos=0.9980,
    frac_rows_differ=1.0000, frac_rows_with_ids=1.0000, frac_tokens_differ=0.0385, n_sampled=64}
```

- `frac_rows_with_ids` — should be 1.0; anything less means ids aren't reaching training.
- `frac_rows_differ` / `frac_tokens_differ` — the drift you are *no longer* training on.
- `frac_ids_end_with_eos` — the fraction of natural stops (the rest hit `max_tokens`).

In training, turn it on with `+agent.run_config.rollout_config.log_token_drift=true`.
The trainer forwards that setting to conversion for training and validation, and
removes it from a copy of the rollout constructor options, leaving the original
configuration unchanged.

`StepRollout` no longer accepts `log_token_drift` or stores diagnostic state.
For direct Python calls, pass the option to conversion, not to the rollout
constructor or `agent.run(rollout_config=...)`. The standalone
`rollout.conversion.token_drift_stats` calculation remains in use, and its
statistics are in `batch.meta_info["token_drift"]` when the diagnostic is on and
succeeds. Chain and step conversion both support it.

### Scope

- Ids are threaded for the verl backend (the training path). Other backends leave the message
  untouched, and tokenization falls back to text — unchanged behavior.
- `ChainRollout` receives ids too (the hook is shared), and every assistant turn in the
  cumulative history is spliced. For that to be consistent the history text must equal the
  sampled text, so the `ActionAgent` keeps the **full** generated response as the message
  `content` (the action is parsed from a copy cut at the first `</action>`).
  `truncate_history_at_action=True` restores the old cut as an opt-in ablation; note it
  makes history != what was sampled.

## Prompt side: one source of truth for prompt tools

### The problem

Whether a tool's schema is rendered into the prompt (Qwen's `# Tools … <tools>…</tools>` block)
is decided by whatever list of tools reaches the chat template. Generation and training render
the prompt in different places, so they can disagree. The `ActionAgent` is the canonical case:
its prompt text already tells the model to answer with `<action>…</action>` and the tool exists
only to *execute* that action — yet the schema was still being passed to the backend, so every
sampling prompt carried a 164-token tools block (one that even contradicts the `<action>`
instruction) that the training prompt did not.

### What AgentFly does

The agent owns the decision:

```python
class BaseAgent:
    render_tools_in_prompt: bool = True     # constructor arg → agent.init_config.render_tools_in_prompt

    def prompt_tools(self) -> list[dict]:
        """Schemas rendered into the prompt, or [] — the ONLY source for generation AND training."""
```

- `StepRollout`, `ChainRollout`, and `tokenize_trajectories` all call `agent.prompt_tools()`;
  none of them build schemas themselves, so the two paths cannot diverge.
- `ActionAgent` and `ReactAgent` default to `render_tools_in_prompt=False` (their prompt text
  carries the interface). Native tool-calling agents (bash/SWE) keep `True` and now render the
  block at training time too, where previously they sampled with it but trained without.
- Override per agent with `agent.init_config.render_tools_in_prompt=true|false`.

## Prompt side: multi-turn prompts are rendered by the training tokenizer

### The problem

In a multi-turn (chain) rollout the prompt for turn *k* contains the model's own earlier
turns. If the rollout server renders that prompt from the message *text* (the HF
`apply_chat_template` path), those turns are re-tokenized — the response-side drift again,
now inside the prompt — while training splices the sampled ids. The two contexts differ on
every multi-turn row, and the model tends to *copy* the drifted tokenization it is shown
(on Qwen2.5-1.5B, `<th`/`ink` becomes the likely continuation with p≈0.9 once the history
carries it). This collapsed a chain-rollout run within ~35 steps.

### What AgentFly does

`AsyncVerlBackend` renders every prompt with the same chat-bricks call that
`tokenize_trajectories` makes — same template, tokenizer, `prompt_tools()`,
`ignore_tool_calls`, generation prompt — from the trajectory messages as they are (sampled
`token_ids` included), and sends the ids to verl as `prompt_ids`. verl's agent loop executes
those ids and does not re-render the messages. The sampled prompt is therefore the training
row's prefix by construction; `raw_prompt` still travels for verl's bookkeeping and images.

Rows with vision inputs are the exception for now: they keep verl's own render (the backend
prints a one-time notice), because ids and images are not threaded through together yet.

### The guard — `prompt_check`

When enabled, `AsyncVerlBackend` compares the ids it sent with the `prompts` verl actually
sampled from (exact comparison), on the first call and every 200th, a few rows each time. It logs

```
[AsyncVerlBackend] prompt_check: 4/4 rows identical (sampling == training render; assistant turns=[0, 3, 7, 12])
```

or a `PROMPT MISMATCH` line with the turn depths of the checked rows and a decoded diff.
The turn depths show that multi-turn rows are being checked, not only first turns.

The same setting runs a second check, `splice_check`. For the last assistant message that
carries `token_ids`, the current prompt must start with that turn's prompt followed by its
sampled ids, unchanged:

```
[AsyncVerlBackend] splice_check: 4/4 rows continuous (prompt == turn prompt + sampled ids)
```

A `SPLICE DISCONTINUITY` line instead means the template inserts or re-encodes something
around a past reply (for example a re-rendered `<think>` block), which makes updates
off-policy. `prompt_check` cannot see this, because sampling and training share the render.

Both checks are off by default. Turn them on with
`+agent.init_config.backend_config.check_prompt_consistency=true` (log the results) or `=strict`
(raise on a mismatch — use for smoke runs).

The batch summary also reports `valid_frac`, the fraction of format-valid actions
(verl-agent's `valid_action_ratio`), which should converge toward 1.0 within a few steps.

## What to look for on a new run

With both diagnostics on (`+agent.init_config.backend_config.check_prompt_consistency=true`
and `+agent.run_config.rollout_config.log_token_drift=true`):

```
grep -m1 "prompt_check"     <rollout .out>   # 4/4 rows identical
grep -m1 "splice_check"     <rollout .out>   # 4/4 rows continuous
grep -m1 "training batch"   <rollout .out>   # frac_rows_with_ids=1.0000, valid_frac rising
```

and on the tracker: `actor/kl_loss` holding in a narrow band with `actor/ppo_kl` ~1e-3, rather
than climbing step over step.
