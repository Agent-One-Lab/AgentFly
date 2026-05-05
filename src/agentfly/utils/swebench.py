"""
Run an SWE-Bench-style agent over a JSON dataset and save per-sample trajectories and rewards.

Example::

    python -m agentfly.utils.swebench \\
        --data-path data/swe-bench-verified.json \\
        --result-dir runs/swe_eval_001 \\
        --model-name-or-path Qwen/Qwen3.5-4B \\
        --temperature 0.0 \\
        --reward-name r2e_gym_reward

    Local vLLM engine (``async_vllm``) with tensor / data parallel::

        python -m agentfly.cli swebench \\
            --backend async_vllm --tp 2 --dp 1 \\
            --max-model-len 40960 \\
            --data-path data/swe-bench-verified.json \\
            --result-dir runs/swe_eval_001 \\
            --model-name-or-path Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, List

import click
from ..tools import run_shell_command
from ..agents.specialized.swe_agents import BashSWEAgent, Qwen3CoderSWEAgent
from ..agents.specialized.swe_agents.prompts import (
    InstructionSystemPrompt,
    Qwen3CoderToolPrompt,
)
from ..core import ContextConfig
from ..rewards.reward_base import get_reward_from_name
from ..tools import create_file, edit_file, grep_search, read_file, run_python, undo_edit
from .monitor import serialize_for_json


def _normalize_dataset_rows(raw: Any) -> List[dict]:
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            return raw["data"]
        if isinstance(raw.get("instances"), list):
            return raw["instances"]
        if "messages" in raw:
            return [raw]
    if isinstance(raw, list):
        return raw
    raise click.BadParameter(
        "data_path must be JSON: a list of samples, or an object with a 'data' / "
        "'instances' list, or a single message dict with 'messages'."
    )


def _problem_text(row: dict) -> str:
    for key in ("problem_statement", "instruction", "text", "question"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise click.BadParameter(
        "Each sample must include a non-empty string field among: "
        "problem_statement, instruction, text, question (or use pre-built 'messages')."
    )


def _row_to_message_item(row: dict) -> dict:
    """Build a Messages dict: turns under ``messages``, rollout metadata as other keys."""
    if "messages" in row and isinstance(row["messages"], list):
        return dict(row)
    text = _problem_text(row)
    meta = {
        k: v
        for k, v in row.items()
        if k not in ("problem_statement", "instruction", "text", "question")
    }
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ],
        **meta,
    }


def _safe_result_basename(instance_id: str | None, index: int) -> str:
    base = instance_id if instance_id else f"sample_{index:05d}"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or f"sample_{index:05d}"
    return base[:200]


def _file_tool_list():
    return [create_file, read_file, edit_file, grep_search, undo_edit, run_python]


async def _run_agent_async(
    *,
    agent_kind: str,
    model_name_or_path: str,
    template: str | None,
    max_model_len: int | None,
    tools_mode: str,
    reward_name: str,
    backend: str,
    vllm_base_url: str,
    api_key: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    messages: List[dict],
    max_turns: int,
    num_chains: int,
    max_concurrent_chains: int | None,
    temperature: float,
    resource_backend: str,
) -> Any:
    reward_fn = get_reward_from_name(reward_name)
    tools = None
    if tools_mode == "file":
        tools = _file_tool_list()
    elif tools_mode == "bash":
        tools = [run_shell_command]
    else:
        raise click.BadParameter(f"Unknown tools mode: {tools_mode!r}")

    backend_config: dict[str, Any] = {"backend": backend}
    if backend == "client":
        backend_config["base_url"] = vllm_base_url
        backend_config["api_key"] = api_key
    elif backend == "async_vllm":
        backend_config["tensor_parallel_size"] = tensor_parallel_size
        backend_config["data_parallel_size"] = data_parallel_size
        if max_model_len is not None:
            backend_config["max_model_len"] = max_model_len

    common = dict(
        model_name_or_path=model_name_or_path,
        template=template,
        max_model_len=max_model_len,
        tools=tools,
        backend_config=backend_config,
        reward_fn=reward_fn,
        monitors=[],
        streaming="console",
    )

    if agent_kind == "bash":
        agent = BashSWEAgent(
            system_prompt=InstructionSystemPrompt,
            **common,
        )
    elif agent_kind == "qwen3_coder":
        agent = Qwen3CoderSWEAgent(
            system_prompt=Qwen3CoderToolPrompt,
            **common,
        )
    else:
        raise click.BadParameter(f"Unknown agent kind: {agent_kind!r}")

    await agent.run(
        messages=messages,
        max_turns=max_turns,
        num_chains=num_chains,
        generation_config={"temperature": temperature},
        max_concurrent_chains=max_concurrent_chains,
        context_config=ContextConfig(resource_backend=resource_backend),
    )
    return agent


def _save_per_sample_results(
    result_dir: Path,
    agent: Any,
) -> tuple[float, int]:
    result_dir.mkdir(parents=True, exist_ok=True)
    run_result = agent._require_last_run()
    trajectories = run_result.trajectories
    reward_scalars = run_result.rewards
    reward_extras = run_result.reward_extras

    manifest: list[dict[str, Any]] = []
    for idx, traj in enumerate(trajectories):
        # Task identifiers come from the dataset row, which is preserved in
        # Trajectory.metadata after the typed-trajectory refactor.
        iid = traj.metadata.get("instance_id") or traj.metadata.get("uid")
        basename = _safe_result_basename(
            str(iid) if iid is not None else None, idx
        )
        out_path = result_dir / f"{basename}.json"

        r_scalar = (
            reward_scalars[idx] if idx < len(reward_scalars) else traj.reward
        )
        extras_row = {
            k: (v[idx] if idx < len(v) else None) for k, v in reward_extras.items()
        }
        record = {
            "index": idx,
            "instance_id": iid,
            "reward": r_scalar,
            "reward_full": {"reward": traj.reward, "extras": traj.metrics},
            "reward_extras": serialize_for_json(extras_row),
            "trajectory": traj.model_dump(),
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append(
            {
                "file": str(out_path.relative_to(result_dir)),
                "index": idx,
                "instance_id": iid,
                "reward": r_scalar,
            }
        )

    summary_path = result_dir / "run_summary.json"
    n = len(reward_scalars)
    mean_r = float(sum(reward_scalars) / n) if n else 0.0
    summary_path.write_text(
        json.dumps(
            {
                "num_samples": len(trajectories),
                "mean_reward": mean_r,
                "accuracy": mean_r,
                "per_sample": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return mean_r, n


@click.command("swebench")
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON file: list of instances, or {\"data\": [...]} / {\"instances\": [...]}.",
)
@click.option(
    "--result-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write one JSON per sample plus run_summary.json.",
)
@click.option("--model-name-or-path", required=True, help="HF model id or local path.")
@click.option("--template", default=None, help="Optional chat-bricks template name.")
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--max-turns", type=int, default=30, show_default=True)
@click.option("--num-chains", type=int, default=1, show_default=True)
@click.option(
    "--max-concurrent-chains",
    type=int,
    default=None,
    help="Cap concurrent chains across the batch (default: unlimited).",
)
@click.option(
    "--agent",
    type=click.Choice(["bash", "qwen3_coder"], case_sensitive=False),
    default="qwen3_coder",
    show_default=True,
)
@click.option(
    "--tool-set",
    type=click.Choice(["bash", "file"], case_sensitive=False),
    default="bash",
    show_default=True,
    help="bash: bash only; file: Use file workspace tools (for Qwen3 coder).",
)
@click.option(
    "--reward-name",
    default="r2e_gym_reward",
    show_default=True,
    help="Registered reward name (e.g. r2e_gym_reward, swe_reward).",
)
@click.option(
    "--backend",
    type=click.Choice(["client", "async_vllm"], case_sensitive=False),
    default="client",
    show_default=True,
)
@click.option(
    "--vllm-base-url",
    default="http://localhost:8000/v1",
    show_default=True,
    help="OpenAI-compatible base URL when --backend client.",
)
@click.option("--api-key", default="EMPTY", show_default=True)
@click.option(
    "--tensor-parallel-size",
    "--tp",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Tensor parallel size for --backend async_vllm (vLLM AsyncEngineArgs).",
)
@click.option(
    "--data-parallel-size",
    "--dp",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Data parallel size for --backend async_vllm (vLLM AsyncEngineArgs).",
)
@click.option("--max-model-len", type=int, default=None)
@click.option(
    "--resource-backend",
    default="local",
    show_default=True,
    help="Context resource backend (e.g. local, ray).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Run at most this many samples (after shuffle).",
)
@click.option("--seed", type=int, default=None, help="Shuffle dataset with this seed.")
@click.option(
    "--enroot-images-path",
    default=None,
    help="If set, sets ENROOT_IMAGES_PATH for container-backed rewards.",
)
@click.option(
    "--enroot-async/--no-enroot-async",
    default=True,
    help="Set ENROOT_ASYNC=1 (default) or 0.",
)
def main(
    data_path: Path,
    result_dir: Path,
    model_name_or_path: str,
    template: str | None,
    temperature: float,
    max_turns: int,
    num_chains: int,
    max_concurrent_chains: int | None,
    agent: str,
    tool_set: str,
    reward_name: str,
    backend: str,
    vllm_base_url: str,
    api_key: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    max_model_len: int | None,
    resource_backend: str,
    limit: int | None,
    seed: int | None,
    enroot_images_path: str | None,
    enroot_async: bool,
):
    if enroot_images_path:
        os.environ["ENROOT_IMAGES_PATH"] = enroot_images_path
    os.environ["ENROOT_ASYNC"] = "1" if enroot_async else "0"
    if backend != "async_vllm" and (tensor_parallel_size != 1 or data_parallel_size != 1):
        raise click.BadParameter(
            "--tp/--dp only affect --backend async_vllm; use --backend async_vllm "
            "or leave tp/dp at 1."
        )
    if backend == "async_vllm":
        click.echo(
            f"Using async_vllm engine args: tp={tensor_parallel_size}, "
            f"dp={data_parallel_size}, max_model_len={max_model_len}"
        )

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    rows = _normalize_dataset_rows(raw)
    if seed is not None:
        import random

        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit is not None:
        rows = rows[: max(0, limit)]

    messages = [_row_to_message_item(r) for r in rows]
    tools_mode = "file" if tool_set == "file" else "bash"

    rollout_agent = asyncio.run(
        _run_agent_async(
            agent_kind=agent.lower(),
            model_name_or_path=model_name_or_path,
            template=template,
            max_model_len=max_model_len,
            tools_mode=tools_mode,
            reward_name=reward_name,
            backend=backend,
            vllm_base_url=vllm_base_url,
            api_key=api_key,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            messages=messages,
            max_turns=max_turns,
            num_chains=num_chains,
            max_concurrent_chains=max_concurrent_chains,
            temperature=temperature,
            resource_backend=resource_backend,
        )
    )

    accuracy, n = _save_per_sample_results(result_dir, rollout_agent)
    click.echo(
        f"Wrote {len(rollout_agent._require_last_run())} sample(s) under {result_dir.resolve()} "
        f"(see run_summary.json)."
    )
    if n:
        click.echo(f"Final accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%, n={n})")
    else:
        click.echo("Final accuracy: n/a (no samples)")


if __name__ == "__main__":
    main()
