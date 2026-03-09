# Where `use_remove_padding` and `ulysses_sequence_parallel_size` Are Handled in Verl

This doc traces the two options through the **adapted** verl code. The training entrypoint is `agentfly.cli train` → `agentfly.verl.trainer.main_ppo`; `src/agentfly/verl` is a symlink to `verl/verl`, so all paths below refer to **`verl/verl/`** in the repo.

---

## 1. Config entrypoints (Hydra)

- **`actor_rollout_ref.model.use_remove_padding`**  
  - Used for both actor and ref (ref gets it in code, see below).
- **`actor_rollout_ref.actor.ulysses_sequence_parallel_size`**  
  - Actor sequence-parallel size.
- **`actor_rollout_ref.ref.ulysses_sequence_parallel_size`**  
  - Ref sequence-parallel size (often set equal to actor; default in yaml: `${oc.select:actor_rollout_ref.actor.ulysses_sequence_parallel_size,1}`).

Actor’s `use_remove_padding` is taken from the **model** in the yaml:

- **`verl/verl/trainer/config/actor/dp_actor.yaml`**  
  - `use_remove_padding: ${oc.select:actor_rollout_ref.model.use_remove_padding,false}`  
  - So setting `actor_rollout_ref.model.use_remove_padding=True` is enough for the actor to see it.

---

## 2. FSDP worker (actor/ref build and actor/ref config)

**File: `verl/verl/workers/fsdp_workers.py`**

- **Ulysses device mesh and SP size**
  - Around 163–169: reads `self.config.actor.get("ulysses_sequence_parallel_size", 1)`, builds `ulysses_device_mesh` with `(dp, sp)` when `ulysses_sequence_parallel_size > 1`, sets `ulysses_sharding_manager`.
  - 236, 244, 261, 266: normalizes `ppo_mini_batch_size` and micro batch sizes by `device_mesh.size() // self.ulysses_sequence_parallel_size`.

- **`use_remove_padding`**
  - 761: `use_remove_padding = self.config.model.get("use_remove_padding", False)`.
  - 784–785: passed into `_build_model_optimizer(..., use_remove_padding=use_remove_padding)` for **actor**.
  - 828: same for **ref** build.
  - 837: **ref** config is updated: `self.config.ref.use_remove_padding = use_remove_padding` so ref policy sees it.

- **Monkey patch (model + Ulysses + remove padding)**
  - 396–402: `apply_monkey_patch(model=actor_module, use_remove_padding=use_remove_padding, ulysses_sp_size=self.ulysses_sequence_parallel_size, ...)` for actor.
  - 1292–1295: same for critic when enabled.
  - 1677–1680: same for reward model when enabled.

- **Context manager for Ulysses**
  - 873, 977, 1017: `with self.ulysses_sharding_manager:` around actor update / rollout log prob / ref log prob so SP group is active.

---

## 3. Actor policy (DP actor) – forward and backward

**File: `verl/verl/workers/actor/dp_actor.py`**

- **Config read**
  - 65: `self.use_remove_padding = self.config.get("use_remove_padding", False)` (actor config; value comes from model via yaml).
  - 72–73: `self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size`, `self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1`.

- **Remove-padding path**
  - 119–168: if `self.use_remove_padding`, unpad inputs and optionally apply Ulysses pad/slice (`ulysses_pad`, `ulysses_pad_and_slice_inputs` with `sp_size=self.ulysses_sequence_parallel_size`).
  - 216–229: if `self.use_ulysses_sp`, gather and unpad outputs (`gather_outputs_and_unpad`).
  - 255: else branch: no rmpad, no Ulysses.

- **Length used for buffers**
  - 347: `max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size`.
  - 417: same idea with `ppo_max_token_len_per_gpu`.

So both options are fully used in the actor’s forward/backward path.

---

## 4. Ref policy (same worker, ref module)

Ref is built in `fsdp_workers.py` with the same `use_remove_padding` and receives `self.config.ref` (with `use_remove_padding` set at 837). Ref is then run as a `DataParallelPPOActor` with that config, so it uses the same logic as the actor in **`dp_actor.py`** (including `use_remove_padding` and `ulysses_sequence_parallel_size` from `self.config`).

Ref’s Ulysses size is set from config **`actor_rollout_ref.ref.ulysses_sequence_parallel_size`** (and defaults from actor in yaml). No separate code path; same `dp_actor` class.

---

## 5. FSDP engine / transformer impl (HF model under FSDP)

**File: `verl/verl/workers/engine/fsdp/transformer_impl.py`**

- **Config**
  - 115: `self.use_remove_padding = self.model_config.use_remove_padding`.
  - 175: `self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size` (engine config gets it from actor’s `fsdp_config` when Ulysses is set on the actor).
  - 177–182: builds `ulysses_device_mesh` when `ulysses_sequence_parallel_size > 1`.

- **Forward**
  - 720, 742–771, 834, 843, 873–876, 980, 983, 998: use_remove_padding and Ulysses pad/slice and gather in the engine’s forward path.

So the FSDP engine also respects both options for the underlying HF model.

---

## 6. Model monkey patch (attention + Ulysses)

**File: `verl/verl/models/transformers/monkey_patch.py`**

- **`apply_monkey_patch(..., use_remove_padding=..., ulysses_sp_size=...)`**
  - 248–253: signature.
  - 273–278: asserts `num_attention_heads` and `num_key_value_heads` divisible by `ulysses_sp_size` when Ulysses is used.
  - Per model type (qwen2_vl, qwen3_vl, llama, etc.): if `use_remove_padding or ulysses_sp_size > 1`, patches attention forward (e.g. Ulysses flash attention).
  - 416–424: generic path (e.g. for Qwen3 text): if `use_remove_padding or ulysses_sp_size > 1`, sets `_flash_attention_forward = _ulysses_flash_attention_forward`.

So both options drive whether the model’s attention is patched for remove-padding and Ulysses.

---

## 7. Ulysses helpers and sharding

- **`verl/verl/utils/ulysses.py`**  
  - `ulysses_pad`, `ulysses_pad_and_slice_inputs`, `gather_outputs_and_unpad`, SP group getters, etc.

- **`verl/verl/workers/sharding_manager/fsdp_ulysses.py`**  
  - `FSDPUlyssesShardingManager`: sets/restores Ulysses SP group around regions that need it.

- **`verl/verl/workers/config/actor.py`**  
  - 256–257: if `ulysses_sequence_parallel_size > 1`, sets `self.fsdp_config.ulysses_sequence_parallel_size`.
  - 263–266: when using FSDP/FSDP2 and Ulysses, requires `model.use_remove_padding` to be True (validation).

---

## 8. Critic (when enabled, e.g. GAE)

**File: `verl/verl/workers/critic/dp_critic.py`**

- 47–48: `self.use_remove_padding = self.config.model.get("use_remove_padding", False)`.
- 50: `self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)`.
- 69–110, 168, 210: same remove-padding and Ulysses slice/gather logic as in the actor.

So if you enable the critic, it will also use both options from its config.

---

## 9. Checklist for your adaptation

- [ ] **Config**: You pass `actor_rollout_ref.model.use_remove_padding=True` and `actor_rollout_ref.actor.ulysses_sequence_parallel_size=N` (and optionally `actor_rollout_ref.ref.ulysses_sequence_parallel_size=N`) from the script or Hydra.
- [ ] **No overrides**: You don’t overwrite or drop `config.model.use_remove_padding` or `config.actor.ulysses_sequence_parallel_size` when building the FSDP worker config.
- [ ] **FSDP workers**: `fsdp_workers.py` still reads `self.config.model.get("use_remove_padding")` and `self.config.actor.get("ulysses_sequence_parallel_size")` and passes them into `_build_model_optimizer` and `apply_monkey_patch`.
- [ ] **Actor/ref**: `dp_actor.py` still reads `self.config.get("use_remove_padding")` and `self.config.ulysses_sequence_parallel_size` and uses the rmpad + Ulysses branches in the forward.
- [ ] **Engine**: `transformer_impl.py` still gets `use_remove_padding` from `model_config` and `ulysses_sequence_parallel_size` from `engine_config` and builds the Ulysses mesh and uses it in forward.
- [ ] **Monkey patch**: `apply_monkey_patch` is still called with the same `use_remove_padding` and `ulysses_sp_size` for actor (and ref via same model build path) so attention is patched for Ulysses/rmpad.

If any of these are changed in your fork, restore or rewire them so the same config flows through and the same code paths run; then the two options will work with your adapted code.
