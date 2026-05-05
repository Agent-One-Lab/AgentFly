# Training Config Reference (Hydra)

AgentFly trains via verl's PPO trainer (`python3 -m agentfly.cli train ...`), which is Hydra-driven. Every key in your training script of the form `namespace.path=value` overrides a node in the underlying Hydra config tree.

The full schema lives upstream in verl: `verl/verl/trainer/config/ppo_trainer.yaml`. This page lists the **keys you actually need to set** in practice, organized by namespace, with notes on what each one controls. For the canonical end-to-end example, see [First Training](first_training.md).

## Namespaces at a Glance

| Namespace | Owns |
|---|---|
| `data.*` | Training/validation file paths and batch sizes |
| `agent.*` | Agent type, tools, reward, generation, rollout shape |
| `algorithm.*` | RL algorithm (advantage estimator, KL coefficient) |
| `actor_rollout_ref.*` | Actor model, rollout engine, reference model (FSDP, vLLM, sequence parallel) |
| `critic.*` | Critic model (only used when `adv_estimator=gae`) |
| `trainer.*` | Logging, checkpointing, GPU/node topology, total steps |

## `data.*` — Data Loading

| Key | Description |
|---|---|
| `data.train_files` | Path to the training JSON file. See [Data Preparation](data_preparation.md). |
| `data.val_files` | Path to the validation JSON file. |
| `data.train_batch_size` | Number of prompts per training step. |
| `data.val_batch_size` | Number of prompts per validation step. |

## `agent.*` — Agent / Rollout

These are the keys defined by AgentFly itself (the rest are inherited from verl).

| Key | Description |
|---|---|
| `agent.use_agent` | `True` to enable the agent rollout path. Always `True` for AgentFly training. |
| `agent.init_config.agent_type` | One of `hf`, `react`, `code`, `gui`, `action`, etc. Picks the agent class via `AutoAgent.from_config`. |
| `agent.init_config.model_name_or_path` | HF model name or local path. |
| `agent.init_config.template` | Chat template name (e.g. `qwen2.5`, `qwen2.5-vl`, `qwen2.5-no-system-tool`). |
| `agent.init_config.tools` | List of tool names available to the agent, e.g. `[calculator]` or `[code_interpreter]`. |
| `agent.init_config.reward_name` | Name of the reward function registered via `@reward(name=...)`. |
| `agent.init_config.backend` | Generation backend during training (typically `async_verl`). |
| `agent.init_config.tool_parser_name` | Optional tool-call parser identifier (e.g. `hermes`). |
| `agent.init_config.max_model_len` | Max sequence length the model engine should support. |
| `agent.max_turns` | Max number of generation/tool turns per chain. |
| `agent.num_chains` | Number of parallel chains (trajectories) per prompt. Higher → more samples for GRPO/RLOO. |
| `agent.train_on_last_turn` | If `True`, only compute loss on the last assistant turn. Useful for stability on long rollouts. |
| `agent.generation_config.max_tokens` | Max tokens generated per turn. |
| `agent.run_config.context_config.resource_backend` | Where containers are allocated: `local` or a cluster backend. |
| `agent.run_config.max_concurrent_chains` | Cap on chains running concurrently (memory protection). `null` for no extra cap. |

Some scripts use `agent.run_config.max_turns` and `agent.run_config.num_chains` — these are equivalent to `agent.max_turns` / `agent.num_chains` and reach the same Hydra node.

## `algorithm.*` — RL Algorithm

| Key | Description |
|---|---|
| `algorithm.adv_estimator` | Advantage estimator: `grpo`, `rloo`, `reinforce_plus_plus`, `remax`, or `gae`. `grpo` is the most common for tool-using agents. |
| `algorithm.kl_ctrl.kl_coef` | KL penalty coefficient against the reference model. Common values: `0.0` to `0.001`. |

## `actor_rollout_ref.*` — Actor / Rollout / Reference

This namespace mirrors verl's grouping: the same set of model-side parameters is shared between the actor (the policy being trained), the rollout worker (which generates trajectories), and the reference model (used for KL).

### Actor (training)

| Key | Description |
|---|---|
| `actor_rollout_ref.actor.optim.lr` | Learning rate. |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | Fraction of training spent on warmup. |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | Mini-batch size for PPO updates within one outer step. |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Per-GPU micro-batch (controls memory). |
| `actor_rollout_ref.actor.use_kl_loss` | If `True`, add a KL term to the loss instead of (or alongside) the reward shaping. |
| `actor_rollout_ref.actor.kl_loss_coef` | Coefficient for the in-loss KL term. |
| `actor_rollout_ref.actor.kl_loss_type` | Form of the KL term: `mse`, `kl`, etc. |
| `actor_rollout_ref.actor.entropy_coeff` | Entropy regularization. Small values (≤ 1e-3) help stabilize tool-calling agents. |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Offload params to CPU. Required for big models on small clusters. |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Offload optimizer state to CPU. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | Sequence-parallel degree (Ulysses). Useful for very long contexts. |

### Model

| Key | Description |
|---|---|
| `actor_rollout_ref.model.path` | Same model path as `agent.init_config.model_name_or_path`. |
| `actor_rollout_ref.model.use_remove_padding` | Whether to pack sequences to remove padding. Disable if it conflicts with your masking logic. |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | Trade compute for memory. |
| `actor_rollout_ref.model.enable_activation_offload` | Stronger memory/compute trade. |

### Rollout (inference during training)

| Key | Description |
|---|---|
| `actor_rollout_ref.rollout.name` | Rollout engine, typically `vllm`. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | Fraction of GPU memory the rollout engine may use. |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | TP degree for the rollout engine. |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | Per-GPU batch when re-computing log-probs over rollout outputs. |

### Reference model

| Key | Description |
|---|---|
| `actor_rollout_ref.ref.fsdp_config.param_offload` | Offload reference params (almost always `True`). |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | Per-GPU batch for reference log-probs. |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size` | Sequence-parallel degree on the reference. |

## `critic.*` — Critic Model

Only relevant when `algorithm.adv_estimator=gae` (GRPO and similar are critic-free).

| Key | Description |
|---|---|
| `critic.model.path` | Critic model path (often the same as the actor). |
| `critic.model.enable_activation_offload` | Memory trade for the critic. |
| `critic.ppo_mini_batch_size` | Critic mini-batch size. |
| `critic.ppo_micro_batch_size_per_gpu` | Critic per-GPU micro-batch. |

## `trainer.*` — Training Loop & Logging

| Key | Description |
|---|---|
| `trainer.project_name` | WandB / logger project. |
| `trainer.experiment_name` | WandB / logger run name. |
| `trainer.logger` | List of loggers, e.g. `['console','wandb']`. |
| `trainer.n_gpus_per_node` | GPUs per node. |
| `trainer.nnodes` | Number of nodes. |
| `trainer.total_training_steps` | Total outer training steps. |
| `trainer.save_freq` | Checkpoint every N steps. |
| `trainer.test_freq` | Run validation every N steps. |
| `trainer.val_before_train` | Run validation once before training begins (sanity check). |
| `trainer.critic_warmup` | Number of critic-only warmup steps (only relevant for GAE). |

## Canonical Example

The WebShop training script wires the above together as a single `agentfly.cli train` invocation. This is a clean Hydra-only example (Ray cluster setup is handled separately by your launcher):

```bash
--8<-- "examples/train_scripts/webshop/train_webshop.sh"
```

## Where to Look When You Need More

- **Full Hydra schema** — `verl/verl/trainer/config/ppo_trainer.yaml`. Everything not listed above is in there with defaults.
- **Generated schemas** — `verl/verl/trainer/config/_generated_ppo_trainer.yaml` (the materialized config tree, useful for grepping).
- **Per-run config dump** — Run with `agent.use_agent=True ... +hydra.job.chdir=False trainer.print_config=True` (when supported) to see the exact resolved config for a run.
- **Other shipped scripts** — `examples/train_scripts/{webshop,alfworld,scienceworld,search,swe,vqa,...}/` show domain-specific overrides (different reward names, tools, batch sizes, sequence-parallel settings).
