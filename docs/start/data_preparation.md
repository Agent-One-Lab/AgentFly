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

Each item must be a dict. There is **no fixed schema** beyond a small set of conventional fields described below — anything extra you put in the dict becomes available to your tools and rewards.

## Predefined Fields

These keys have special meaning to the rollout layer:

| Field | Required | Purpose |
|---|---|---|
| `question` | usually | Forms the user turn. The rollout wraps it as `[{"role": "user", "content": question}]` if no explicit `messages` is provided. |
| `messages` | optional | If supplied, replaces the auto-generated user turn. Use this when you need a pre-built multi-turn prompt or multi-modal content (text + images). |

Everything else is freeform: any field you add to the dict is forwarded into the rollout context and made available to tools and rewards by **name match** (see *Additional Fields* below). Common conventional names like `answer` are not built-in — they work only because the reward functions in `agentfly.rewards` happen to declare matching parameter names.

## Additional Fields

Beyond the predefined keys, you can add any task-specific fields you need. They flow through the rollout and get auto-injected into your tools and rewards if the function signature declares a parameter with the same name.

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

will receive `task_id=111` automatically. The same applies to tools: any tool whose signature declares `price_upper: float` will get it injected.

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

`data/rlhf/alfworld/alfworld_train_tasks_flat.json` — task ids that the env uses to load the right scenario:

```json
{
    "question": "Complete this household task: Put the pen on the wooden shelf",
    "answer": "...",
    "messages": [{"role": "user", "content": "..."}],
    "task_id": "trial_T20190907_203141_178191",
    "task_description": "Put the pen on the wooden shelf",
    "task_type": "pick_and_place_simple"
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

`data/rlhf/webshop/webshop_goals_train.json` — many task-specific fields, all flowing into the reward:

```json
{
    "question": "i am looking for travel foaming dispenser for hand soap...",
    "task_id": 111,
    "asin": "B085W67P7L",
    "category": "beauty",
    "attributes": ["rose gold"],
    "price_upper": 30.0,
    "goal_options": ["silver pump head"]
}
```

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

The shell tool reads `image_id` and `git_commit_hash` from the rollout context's metadata to pick up the right container per task.

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

Larger artifacts (the Wikipedia corpus and FAISS index used by `asyncdense_retrieve`, the WebShop product DB, etc.) are downloaded on demand into `~/.cache/AgentFly/` by helpers like `agentfly.tools.utils.data.download_tool_data`.

For a brand-new task, the typical workflow is:

1. Convert your source data (HuggingFace dataset, CSV, etc.) into a JSON list of dicts.
2. Make sure `question` is set (or supply `messages` directly).
3. Add any task-specific fields your reward and tools need; **name them to match your function parameters**.
4. Drop the file under `data/rlhf/<task>/` (or anywhere — you point the trainer at it via `data.train_files`).
5. Reference the file paths in your training script's `data.train_files` and `data.val_files` (see [Hydra Config](hydra_config.md)).
