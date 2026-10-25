# Data Preparation

Training datasets in AgentFly are JSON files containing a list of task dicts. Each dict provides the **prompt** and any **task-specific metadata** the tools and reward function will need. The trainer (verl PPO) reads them via the `data.train_files` and `data.val_files` Hydra keys (see [Hydra Config](hydra_config.md)).

## File Format

A dataset is a single JSON file holding a JSON array:

```json
[
    { "question": "...", "answer": "...", "...": "..." },
    { "question": "...", "answer": "...", "...": "..." }
]
```

Each item must be a dict. There is **no fixed schema** beyond a small set of conventional fields described below — anything extra you put in the dict becomes available to your rewards (by parameter name) and tools (through `context.metadata`).

## Predefined Fields

These keys have special meaning to the rollout layer:

| Field | Required | Purpose |
|---|---|---|
| `question` | usually | Forms the user turn: `[{"role": "user", "content": question}]`. `prompt`, `instruction` or `problem` work too (the first present, in that order, wins). |
| `image` | optional | Attached to that user turn (a path, bytes, base64 string or data URI); images over 1536×1536 pixels are downscaled. |
| `messages` | optional | If supplied, replaces the auto-generated user turn. Use this when you need a pre-built multi-turn prompt or multi-modal content (text + images). |

Training and `agent.run()` turn rows into prompts the same way (`agentfly.agents.utils.messages.row_to_messages`), so you can pass the rows of a training file straight to `agent.run(messages=rows)` to smoke-test it.

Everything else is freeform: any field you add to the dict is kept as the rollout's metadata. Rewards receive it by **parameter name**, and tools read it from `context.metadata` (see *Additional Fields* below). Common conventional names like `answer` are not built-in — they work only because the reward functions in `agentfly.rewards` happen to declare matching parameter names.

## Additional Fields

Beyond the predefined keys, you can add any task-specific fields you need. They are kept as the rollout's metadata (`context.metadata`, and `trajectory.metadata` in the result). A reward receives a field when its signature declares a parameter with the same name; a tool reads it from `context.metadata`.

For example, given:

```json
{
    "question": "Find a rose gold soap dispenser under $30",
    "task_id": 111,
    "price_upper": 30.0
}
```

A reward defined as:

```python
@reward(name="my_webshop_reward")
async def my_webshop_reward(final_response: str, context: Context, task_id: int) -> dict:
    ...
```

will receive `task_id=111` automatically. Tools work differently: their parameters are what the **model** fills in, so dataset fields are not passed by name. A tool that needs one declares a `context` parameter (hidden from the model) and reads it:

```python
@tool(name="search_products")
async def search_products(query: str, context: Context) -> str:
    price_upper = context.metadata.get("price_upper")  # 30.0 for the task above
    ...
```

This is how task-specific configuration (golden moves in chess, environment task ids in ALFWorld, attribute lists in WebShop) reaches the reward and the tool layer without a global registry.

## Examples by Task Type

The dataset shapes shipped under `data/` give a sense of the spread:

### Math (GSM8K)

`data/rlhf/math/gsm8k_train.json` — minimal shape, just `question` and gold answer:

```json
{
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "answer": "72"
}
```

Used with `math_equal_reward_tool`, which reads `final_response` and `answer`.

### QA (HotpotQA)

`data/rlhf/qa/hotpotqa_train_random_8000.json` — same shape as math plus extra metadata that's preserved but unused by the default reward:

```json
{
    "question": "...",
    "answer": "NCAA",
    "type": "bridge",
    "id": 50494
}
```

### ALFWorld

`data/rlhf/alfworld/alfworld_train_tasks.json` (3,553 tasks), with `alfworld_valid_seen_tasks.json` (140) and `alfworld_valid_unseen_tasks.json` (134) for validation. `agentfly train` downloads them from [`Agent-One/AgentFly-Train`](https://huggingface.co/datasets/Agent-One/AgentFly-Train) when missing. Each task names the scenario the env loads:

```json
{
    "question": "Complete this household task: Put the pen on the wooden shelf",
    "answer": "...",
    "messages": [{"role": "user", "content": "..."}],
    "task_id": "trial_T20190907_203141_178191",
    "split": "train",
    "task_description": "Put the pen on the wooden shelf",
    "task_type": "pick_and_place_simple",
    "data_source": "alfworld_train.json"
}
```

`task_id` is consumed by `alfworld_episode_reward` and the alfworld tools through context-based resource sharing.

### ScienceWorld

`data/rlhf/scienceworld/scienceworld_train.json` — task selection fields:

```json
{
    "question": "Your task is to boil water...",
    "task_name": "boil",
    "variation_idx": 0
}
```

### WebShop

`data/rlhf/webshop/webshop_train_verl_agent_small.json` (also `webshop_val_verl_agent_small.json` and the held-out `webshop_heldout_verl_agent_small.json`). Each task carries its own goal and product prices, which the WebShop server scores the purchase against, so a task does not depend on the server's built-in goal list:

```json
{
    "task_id": 0,
    "question": "Find me non slip desks for living room with color: christmasgoo3302, and size: 19.7x31.5in+19.7x63in, and price lower than 50.00 dollars",
    "asin": "B09CQ45ZRB",
    "instruction_text": "Find me non slip desks for living room with ...",
    "price_upper": 50.0,
    "goal_json": "{\"asin\": \"B09CQ45ZRB\", \"attributes\": [\"non slip\", \"living room\"], \"goal_options\": {\"color\": \"christmasgoo3302\", ...}, ...}",
    "prices_json": "{\"B0002J1HOC\": 5.01, ...}",
    "verl_agent_seed": 0,
    "verl_agent_goal_index": 0,
    "verl_agent_draw": "..."
}
```

The tasks replay the goals verl-agent draws for its WebShop environment, so results are comparable with verl-agent; `examples/data_preprocess/prepare_webshop_verl_agent.py` generates them, and the training scripts download them from the `Agent-One/AgentFly-Train` dataset when missing. They are meant for the small-catalog WebShop image (`reasonwang/webshop-env:small`).

### Chess Puzzles

`data/chess/chess_puzzles_train.json` — board position and solution moves:

```json
{
    "question": "You are solving a chess puzzle...",
    "puzzle_id": "00sHx",
    "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "moves": "h5f7",
    "rating": 1000,
    "themes": ["mateIn1", "short"]
}
```

### SWE-Bench

`data/rlhf/os/swe-bench-verified.json` — repo metadata, base commit, and gold patch:

```json
{
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "patch": "diff --git ...",
    "test_patch": "diff --git ..."
}
```

The SWE tools read `image_id` from the rollout context's metadata to pick the task's container, and, when present, `git_commit_hash` (a commit or branch) to check out before the agent's first command. SWE-smith tasks set it to the task's bug branch (the `instance_id`).

## Multi-Modal Data

For vision tasks, supply `messages` directly with mixed content (the rollout will not auto-generate a user turn from `question` if `messages` is present):

```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://example.com/img.jpeg"},
                {"type": "text", "text": "What is in the image?"}
            ]
        }
    ],
    "answer": "a dog"
}
```

See [`docs/examples/image_agent.md`](../examples/image_agent.md) for a full vision-agent example.

## Where Datasets Live

The repo ships small JSONs and download helpers:

```
data/
├── chess/
│   ├── chess_puzzles_train.json
│   └── chess_puzzles_val.json
├── rlhf/
│   ├── alfworld/
│   ├── math/         # GSM8K, MATH-500, ORZ-Math-57k
│   ├── os/           # SWE-Bench, R2E-Gym, SWE-Smith
│   ├── qa/           # HotpotQA, NQ
│   ├── scienceworld/
│   ├── simuphy/
│   └── webshop/
```

Larger artifacts (the Wikipedia corpus and FAISS index used by `async_dense_retrieve`, the WebShop product DB, etc.) are downloaded on demand into `$AF_CACHE_DIR` (default `~/.agentfly/cache`) by `agentfly.utils.download.ensure_tool_data`. Training/eval task files are likewise fetched from the `Agent-One/AgentFly-Train` dataset when missing. Both go through an **approval prompt** that shows the total size first; set `AF_ASSUME_YES=1` to accept non-interactively (CI/batch). See [Configuration](configuration.md) for the full environment-variable surface.

For a brand-new task, the typical workflow is:

1. Convert your source data (HuggingFace dataset, CSV, etc.) into a JSON list of dicts.
2. Make sure `question` is set (or supply `messages` directly).
3. Add any task-specific fields your reward and tools need; **name them to match your function parameters**.
4. Drop the file under `data/rlhf/<task>/` (or anywhere — you point the trainer at it via `data.train_files`).
5. Reference the file paths in your training script's `data.train_files` and `data.val_files` (see [Hydra Config](hydra_config.md)).
