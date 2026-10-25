Training Example
==============

Finally, we are ready to train the agent. This page trains Qwen2.5-3B-Instruct with a calculator
tool on GSM8K, on one GPU.

**1. Prepare Training Data**

----------------

The training and validation sets are JSON files holding a list of tasks:

```
[
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "10"
    },
    ...
]
```

The `question` field becomes the user message. Other fields, here `answer`, are passed to the
reward function. See [Data Preparation](data_preparation.md) for the full format.

You don't need to create these files for this example: the script uses
`./data/rlhf/math/gsm8k_train.json` and `gsm8k_test.json`, and `agentfly train` downloads missing
task files from the [AgentFly-Train](https://huggingface.co/datasets/Agent-One/AgentFly-Train)
dataset after asking for confirmation. In a batch job (no terminal to answer the prompt), set
`AF_ASSUME_YES=1` to allow the download.

**2. Create Training Script**

------------------------
Use the prepared training script at `examples/train_scripts/train_example.sh`. Set `WANDB_API_KEY` first if you want Weights & Biases logging:

```bash
export WANDB_API_KEY="your_wandb_key"
```

The script itself:

```bash
--8<-- "examples/train_scripts/train_example.sh"
```

**3. Run Training**

--------------

Run the script from the repository root (the data paths are relative to it):

```bash
bash examples/train_scripts/train_example.sh
```

!!! warning "Shared nodes"
    The example scripts begin with `ray stop` and start a fresh Ray head, which also stops any
    other Ray cluster your user runs on the node. When other jobs share the node, remove the
    `ray stop` / `ray start` lines and run with `RAY_ADDRESS=local`: the trainer then starts a
    private Ray cluster that ends with it (restrict its GPUs with `CUDA_VISIBLE_DEVICES`).
    Container cleanup is already safe to share: `agentfly train` only removes containers whose
    owner process has ended (see [Resources](../features/resources.md#container-lifetime-and-cleanup)).

### Without a wandb account

Log offline and upload later, or log to the console only:

```bash
export WANDB_MODE=offline          # writes runs to ./wandb/offline-run-*
wandb sync wandb/offline-run-*     # later, to upload them
```

or replace `trainer.logger=['console','wandb']` with `trainer.logger=['console']`.

## Key parameters

| Setting in the script | Hydra key | Meaning |
|---|---|---|
| `model` | `agent.init_config.model_name_or_path`, `actor_rollout_ref.model.path` | Base model to train (set both). |
| `train_batch_size` | `data.train_batch_size` | **Tasks** (prompts) per training step. |
| `num_chains` | `agent.run_config.num_chains` | Attempts per task. GRPO compares the attempts of one task, so use at least 2. |
| `ppo_mini_batch_size` | `actor_rollout_ref.actor.ppo_mini_batch_size` | **Trajectory rows** per optimizer step (see below). |
| `ppo_micro_batch_size_per_gpu` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Rows per forward/backward pass on one GPU. Only affects memory, not the result. |
| `max_turns` | `agent.run_config.max_turns` | Model generations per attempt. Leave room for the final answer: the reward grades the model's last message (or an answer tool's output), not a tool result. The example uses 3 (call the calculator, read the result, answer). |
| `max_new_tokens_per_turn` | `agent.run_config.generation_config.max_tokens` | Tokens per generation. |
| `lr` | `actor_rollout_ref.actor.optim.lr` | Learning rate. |
| `total_training_steps` | `trainer.total_training_steps` | Training steps to run. |

### Batch sizes

`data.train_batch_size` counts tasks, but the PPO batch sizes count **trajectory rows**. Each
task produces `num_chains` rows (more with step rollouts or context folding, which record several
rows per attempt). With the example's 64 tasks × 8 chains, a step has 512 rows:

- `ppo_mini_batch_size = train_batch_size × num_chains = 512` gives one optimizer step per
  training step, fully on-policy. A smaller value splits the rows into several optimizer steps.
- Each GPU's share of a mini-batch (`ppo_mini_batch_size / number of GPUs`) must be a multiple of
  `ppo_micro_batch_size_per_gpu`.
- If the number of rows isn't a multiple of `ppo_mini_batch_size`, the trainer adds masked padding
  rows, so a variable number of rows per step is fine.

### Faster validation

The script validates on the whole GSM8K test set (1,319 tasks) before training and every
`trainer.test_freq` steps. For quicker iterations, validate on a subset with
`data.val_max_samples=200`, validate less often (`trainer.test_freq=50`), or skip the first
validation (`trainer.val_before_train=False`).

## Reading the metrics

The console prints one line of metrics per step; wandb shows the same metrics plus AgentFly's
rollout metrics. The ones to watch first:

| Metric | Meaning |
|---|---|
| `critic/score/mean` | Average reward of the step's trajectories. It should rise. |
| `val-core/<val file>/reward/mean@1` | Average reward on the validation set. |
| `agent/rollout/avg_turns`, `agent/rollout/avg_tool_calls`, `agent/rollout/tool_calls/<tool>` | Is the agent using its tools? A drop to 0 tool calls usually means the model learned to answer directly. |
| `agent/rollout/group/frac_all_pass`, `frac_all_fail`, `frac_mixed` | Share of tasks whose attempts all succeed, all fail, or differ. GRPO learns only from mixed groups; if `frac_mixed` is low, the tasks are too easy or too hard for the model. |
| `response_length/mean`, `response_length/clip_ratio` | Generation length, and the share cut off at `max_tokens`. A high clip ratio means `max_new_tokens_per_turn` is too small. |
| `actor/pg_loss`, `actor/kl_loss`, `actor/entropy`, `actor/grad_norm` | Optimizer health. Watch for exploding `grad_norm` or entropy collapsing towards 0. |
| `timing_s/gen`, `timing_s/update_actor`, `timing_s/step` | Seconds spent generating rollouts, updating the model, and in the whole step. |

wandb also stores sample conversations as tables (`agent/rollout/sample_trajectory`, and the
slowest one, `agent/rollout/slowest_message`), which are the quickest way to see what the agent
actually does.
