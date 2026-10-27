On-Policy Distillation
======================

On-policy distillation (OPD) trains the agent on its **own** rollouts with a dense, per-token
signal from a stronger teacher model. After each rollout, a teacher scores every token the
student sampled; the student is pushed toward tokens the teacher finds more likely than it does.

It uses the ordinary PPO machinery: the teacher-minus-student log-probability becomes the
advantage, and the actor's existing (vanilla) policy loss performs the update. No reward function
is needed for the signal, although the task reward can be added.

## Objective

For every trained position `t` (policy tokens only, per-token reward, no future sum):

```
A_t = action_mask_t · w_t · clamp(log π_teacher(y_t) − log π_student(y_t), −clamp, clamp)
      [+ task_reward_coef · A_GRPO_t]
```

- `log π_student` is the batch's `old_log_probs`. With one optimizer step per batch the
  importance ratio is 1 and clipping never activates, so the vanilla PPO loss reduces to
  `−Σ A_t · log π_student(y_t)` — a policy gradient on the per-token reverse KL.
- `w_t = think_weight` on thinking tokens (a turn's tokens up to and including its first
  `</think>`), 1 elsewhere.
- `A_GRPO` is verl's GRPO outcome advantage from the task reward (grouped by `uid`).

## Pieces

| Piece | Where | Role |
|---|---|---|
| Teacher service | a vLLM deployment + an *endpoints file* | scores token sequences (`prompt_logprobs`) |
| `TeacherClient` | `agentfly.algorithms.teacher` | reads the endpoints file, sends token ids, retries through restarts |
| Teacher step | `agentfly.algorithms.opd.compute_teacher_log_probs` | fills `batch["teacher_log_probs"]` |
| Estimator `opd` | `agentfly.algorithms.opd` | computes advantages (both batch layouts) |
| Trainer hook | `verl/trainer/ppo/ray_trainer.py` | runs the teacher step, logs `teacher/*` and `opd/*` |

### Teacher service and endpoints file

The teacher runs as a separate vLLM server that many runs can share. It publishes a JSON
endpoints file, written atomically and rewritten whenever the service restarts or moves:

```json
{"name": "qwen38_27b", "model": "Qwen/Qwen3.8-27B-FP8", "revision": "017b9c7a...",
 "max_model_len": 96000, "tp_size": 2, "dp_size": 4, "prefix_caching": false,
 "urls": ["http://10.0.4.174:31000"], "status": "ready", "slurm_job": "1212079",
 "written_at": "2026-09-16T22:16:32Z"}
```

`status` is `starting` while the service (re)starts, `ready` while it serves, and `stopped` after
shutdown. Clients connect over the cluster network.

The teacher must use the **same tokenizer/vocabulary** as the student: it scores the student's
token ids directly.

!!! warning "Serve the teacher with prefix caching disabled"
    On hybrid linear-attention models (Qwen3.5 / Qwen3.8), vLLM's prefix caching switches the
    recurrent-state cache to `align` mode, and scoring several long sequences concurrently returned
    corrupted log-probabilities (0.56–0.61 mean per-token error, versus a mean OPD signal of 0.34).
    Use `--no-enable-prefix-caching`. Scoring also needs a small prefill chunk and a low KV-cache
    reservation, because `prompt_logprobs` materializes a full-vocabulary log-softmax per chunk
    (e.g. `--max-num-batched-tokens 4096 --gpu-memory-utilization 0.55`).

### `TeacherClient`

```python
from agentfly.algorithms import TeacherClient

client = TeacherClient("teachers/qwen38_27b.json", expect_model="Qwen/Qwen3.8-27B-FP8")
logprobs, stats = client.score_blocking([[151644, 872, 198], ...])
# logprobs[k][i] = log p(seq_k[i] | seq_k[:i]); position 0 is NaN
```

- Identical sequences are scored once (training batches repeat rows as padding).
- Requests are spread round-robin over the file's URLs, `2 × dp_size` at a time per URL by default.
- Connection errors, timeouts, 5xx responses and a not-ready file are waited out with backoff,
  re-reading the endpoints file, for at most `max_wait_s` of continuous outage; then
  `TeacherUnavailableError`. A call never returns a partial result.
- `expect_model` / `expect_revision` pin the teacher identity.

### Position convention

AgentFly batches are right-padded, and after the trainer's mask shift position `t` refers to token
`t+1` (the same convention as `old_log_probs`). The teacher step follows it:

```
teacher_log_probs[i, t] = log p_teacher(input_ids[i, t+1] | input_ids[i, :t+1])   for t < n − 1
teacher_log_probs[i, t] = 0                                                      at t = n − 1 and on padding
```

vLLM reports `prompt_logprobs[i]` for token `i` itself, so the step shifts by one.

## Configuration

```yaml
algorithm:
  adv_estimator: opd
  teacher:
    endpoints_file: /path/to/teachers/qwen38_27b.json   # setting this enables the teacher step
    expect_model: Qwen/Qwen3.8-27B-FP8                   # optional identity pin
    expect_revision: null
    concurrency: null                                    # default 2 × the deployment's dp_size
    max_wait_s: 1800                                     # longest teacher outage to wait out
  opd:
    clamp: 5.0             # per-token clamp on teacher − student log-prob
    think_weight: 1.0      # weight on thinking tokens (1.0 = no reweighting)
    task_reward_coef: 0.0  # add λ × GRPO outcome advantage (0.0 = pure OPD)
actor_rollout_ref:
  actor:
    use_kl_loss: false     # the teacher anchors the policy
    entropy_coeff: 0
    ppo_epochs: 1
    ppo_mini_batch_size: <rows per step>   # one optimizer step per batch
```

`algorithm.teacher` is independent of verl's `distillation.*` (which starts verl's own teacher
manager); enabling both is an error.

The trainer checks at startup that `adv_estimator=opd` has a teacher, and warns when
`ppo_epochs > 1`, `use_kl_loss=True` or `entropy_coeff ≠ 0`. It also logs
`opd/optimizer_steps` each step and warns once if a batch needs more than one optimizer step (later
steps are off-policy).

The `</think>` token id used for thinking spans is currently fixed to the Qwen3.5 / Qwen3.8 id
(`agentfly.algorithms.opd.THINK_END_TOKEN_ID`).

## Training loop

1. Rollout and conversion to a training batch; the trainer shifts `action_mask` / `rm_scores`.
2. **Teacher step starts** on the trainer's background loop with a snapshot of
   `input_ids` / `attention_mask`, so it runs while reward, `old_log_probs` and reference
   log-probabilities are computed.
3. **Teacher step joins** before the advantage: `teacher_log_probs` is stored in the batch; teacher
   errors (e.g. unavailable past `max_wait_s`) stop training here.
4. `compute_advantage` with the `opd` estimator; actor update.
5. Metrics.

## Metrics

| Metric | Meaning |
|---|---|
| `teacher/score_seconds`, `teacher/tokens`, `teacher/rows`, `teacher/unique_rows`, `teacher/requests`, `teacher/retries` | teacher step cost and health |
| `opd/kl_k1_mean` | mean `old_log_prob − teacher_log_prob` on trained positions (per-token reverse-KL estimate) |
| `opd/kl_k1_think`, `opd/kl_k1_non_think` | the same, split by thinking tokens |
| `opd/think_token_frac` | share of trained positions that are thinking |
| `opd/teacher_prefers_frac` | share of positions where the teacher's log-probability is higher |
| `opd/adv_clamp_frac` | share of positions clamped |
| `opd/adv_abs_mean` | mean absolute advantage |
| `opd/optimizer_steps` | optimizer steps the actor takes on this batch (1 = fully on-policy) |
