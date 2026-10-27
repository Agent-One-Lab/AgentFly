"""Build WebShop train/val task files that reproduce verl-agent's task sampling.

verl-agent (``examples/gigpo_trainer/run_webshop.sh``) does not use a fixed task list:

- every env worker builds its own ``WebAgentTextEnv(seed=w)``, which regenerates the synthetic
  goals (random price limit per product) and product prices, then shuffles the goals with the
  same seed. Training workers get seeds ``env.seed + j`` for group ``j`` (0-15), validation
  workers ``env.seed + 1000 + i`` (1000-1127);
- the training env draws ``np.random.RandomState(env.seed).choice(range(500, 6910), 16,
  replace=False)`` once per training step and group ``j`` resolves its index on seed ``j``'s
  list; the validation env draws ``RandomState(env.seed + 1000).choice(range(500), 128, ...)``
  twice per evaluation (256 tasks), resolved on seeds 1000-1127.

This script runs verl-agent's own env code (one fresh process per worker seed, exactly as its
Ray workers construct the env) and replays those draws, writing:

- ``webshop_train_verl_agent_small.json``: the exact training tasks, in order — rows
  ``16 * k .. 16 * k + 15`` are step ``k`` (read them with ``data.shuffle=False``);
- ``webshop_val_verl_agent_small.json``: the tasks of verl-agent's first ``--val_evals``
  evaluations (256 each).

Each row carries the full goal and the worker's product prices (``goal_json`` /
``prices_json``), which AgentFly's WebShop server serves per session so the instruction and
the reward match verl-agent's worker. Run inside verl-agent's WebShop environment, e.g.::

    conda run -n verl-agent-webshop python examples/data_preprocess/prepare_webshop_verl_agent.py \\
        --webshop_dir <verl-agent>/agent_system/environments/env_package/webshop/webshop
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

TRAIN_OFFSET = 500  # verl-agent: val goal_idxs = range(500), train = range(500, len(goals))


def build_worker(args):
    """Construct the env the way ``WebshopWorker.__init__`` does and return its goals and prices."""
    webshop_dir, seed = args
    sys.path.append(webshop_dir)
    os.chdir(webshop_dir)
    import gym  # noqa: F401
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the gym envs)
    from web_agent_site.envs import web_agent_text_env
    from web_agent_site.utils import DEFAULT_ATTR_PATH, DEFAULT_FILE_PATH

    # The Lucene index only answers search[...] queries; goals and product prices come from the
    # product files, and the env's RNG never touches the searcher. Stubbing it keeps this script
    # runnable without a pyserini-compatible index (verl-agent's checkout no longer ships one).
    # The train/val rows this produces are byte-identical to the ones verified against verl-agent's
    # live Ray workers, which is what makes the stub safe.
    web_agent_text_env.init_search_engine = lambda num_products=None: None

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": DEFAULT_FILE_PATH.replace("items_shuffle.json", "items_shuffle_1000.json"),
        "attr_path": DEFAULT_ATTR_PATH.replace("items_ins_v2.json", "items_ins_v2_1000.json"),
        "seed": seed,
    }
    env = gym.make("WebAgentTextEnv-v0", **env_kwargs)
    server = env.unwrapped.server
    goals = [
        {k: v for k, v in g.items()} for g in server.goals
    ]
    prices = {asin: float(p) for asin, p in server.product_prices.items()}
    return seed, goals, prices


def goal_row(task_id, goal, prices, seed, goal_index, source):
    return {
        "task_id": task_id,
        "question": goal["instruction_text"],
        "asin": goal["asin"],
        "instruction_text": goal["instruction_text"],
        "price_upper": goal["price_upper"],
        "goal_json": json.dumps(goal, sort_keys=True),
        "prices_json": json.dumps(prices, sort_keys=True),
        "verl_agent_seed": seed,
        "verl_agent_goal_index": int(goal_index),
        "verl_agent_draw": source,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop_dir", required=True, help="verl-agent .../env_package/webshop/webshop")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "../../data/rlhf/webshop"))
    parser.add_argument("--env_seed", type=int, default=0, help="verl-agent env.seed")
    parser.add_argument("--train_batch", type=int, default=16, help="verl-agent data.train_batch_size")
    parser.add_argument("--train_steps", type=int, default=150, help="verl-agent trainer.total_epochs")
    parser.add_argument("--val_batch", type=int, default=128, help="verl-agent data.val_batch_size")
    parser.add_argument("--val_batches_per_eval", type=int, default=2, help="val_data_size * 2 / val_batch_size")
    parser.add_argument("--val_evals", type=int, default=2, help="how many of verl-agent's evaluations to keep")
    parser.add_argument("--heldout_size", type=int, default=512,
                        help="tasks in the held-out file: goals for products that never appear in the training file")
    parser.add_argument("--processes", type=int, default=32)
    args = parser.parse_args()
    webshop_dir = os.path.abspath(args.webshop_dir)

    train_seeds = [args.env_seed + j for j in range(args.train_batch)]
    val_seeds = [args.env_seed + 1000 + i for i in range(args.val_batch)]
    ctx = mp.get_context("spawn")
    # One fresh process per worker seed, like verl-agent's Ray workers (each builds exactly one env).
    # chunksize=1 matters: a second env built in the same process draws different prices.
    with ctx.Pool(args.processes, maxtasksperchild=1) as pool:
        built = pool.map(build_worker, [(webshop_dir, s) for s in train_seeds + val_seeds], chunksize=1)
    goals = {s: g for s, g, _ in built}
    prices = {s: p for s, _, p in built}
    n_goals = len(goals[train_seeds[0]])

    train_rng = np.random.RandomState(args.env_seed)
    train_rows = []
    for step in range(args.train_steps):
        idx = train_rng.choice(range(TRAIN_OFFSET, n_goals), size=args.train_batch, replace=False)
        for j, gi in enumerate(idx):
            s = train_seeds[j]
            train_rows.append(goal_row(len(train_rows), goals[s][gi], prices[s], s, gi, f"train_step_{step}_group_{j}"))

    val_rng = np.random.RandomState(args.env_seed + 1000)
    val_rows = []
    for ev in range(args.val_evals):
        for b in range(args.val_batches_per_eval):
            idx = val_rng.choice(range(TRAIN_OFFSET), size=args.val_batch, replace=False)
            for i, gi in enumerate(idx):
                s = val_seeds[i]
                val_rows.append(goal_row(len(val_rows), goals[s][gi], prices[s], s, gi, f"val_eval_{ev}_batch_{b}_worker_{i}"))

    # Held-out generalization set: same env, same workers, but only goals whose PRODUCT is never
    # bought (or drawn) in training. The training tasks cover 240 of the 1,000 catalogue products,
    # and verl-agent's own validation reuses training products, so this measures what that split
    # cannot: does the policy shop for items it has never been trained on?
    train_asins = {row["asin"] for row in train_rows}
    val_pairs = {(row["verl_agent_seed"], row["verl_agent_goal_index"]) for row in val_rows}
    heldout_rng = np.random.RandomState(args.env_seed + 2000)
    candidates = {  # per worker, the goal indices for unseen products (val workers, as validation)
        s: [gi for gi, g in enumerate(goals[s])
            if g["asin"] not in train_asins and (s, gi) not in val_pairs]
        for s in val_seeds
    }
    heldout_rows, taken = [], set()
    while len(heldout_rows) < args.heldout_size:
        s = val_seeds[len(heldout_rows) % len(val_seeds)]  # spread evenly over the workers
        pool = [gi for gi in candidates[s] if (s, gi) not in taken]
        if not pool:
            raise RuntimeError(f"worker {s} ran out of unseen-product goals")
        gi = int(heldout_rng.choice(pool))
        taken.add((s, gi))
        heldout_rows.append(goal_row(len(heldout_rows), goals[s][gi], prices[s], s, gi, "heldout_unseen_product"))
    print(f"held-out: {len(heldout_rows)} tasks, {len({r['asin'] for r in heldout_rows})} products, "
          f"none of the {len(train_asins)} training products")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("webshop_train_verl_agent_small.json", train_rows),
                       ("webshop_val_verl_agent_small.json", val_rows),
                       ("webshop_heldout_verl_agent_small.json", heldout_rows)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"wrote {len(rows)} tasks -> {path}")


if __name__ == "__main__":
    main()
