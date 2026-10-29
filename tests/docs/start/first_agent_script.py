"""The complete script shown in docs/start/first_agent.md (run as ``python first_agent_script.py``)."""

# --8<-- [start:full_script]
import asyncio
import json

from agentfly.agents import HFAgent
from agentfly.tools import calculator


async def main():
    agent = HFAgent(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        tools=[calculator],
        template="qwen2.5",
        # vLLM takes 92% of the GPU by default; leave room for other processes.
        backend_config={"backend": "async_vllm", "gpu_memory_utilization": 0.5},
    )

    # --8<-- [start:batch_run]
    # A batch: one dict per task. Extra fields (here "answer") are kept on the trajectory
    # as metadata and passed to the reward function.
    tasks = [
        {"messages": [{"role": "user", "content": "What is 123 * 45?"}], "answer": "5535"},
        {"messages": [{"role": "user", "content": "What is 2 ** 10 - 24?"}], "answer": "1000"},
    ]
    result = await agent.run(messages=tasks, max_turns=3, group_size=2)
    # --8<-- [end:batch_run]

    # --8<-- [start:read_result]
    print(len(result), "trajectories")  # 2 tasks x 2 chains = 4, grouped by task
    for traj in result:
        messages = traj.segments[-1].messages  # the full conversation of this attempt
        print(
            f"task {traj.group_idx} chain {traj.episode_idx}: "
            f"answer={agent.extract_final_response(messages)!r} "
            f"turns={traj.num_turns} finish={traj.finish_reason} tools={traj.tool_call_counts}"
        )
    # --8<-- [end:read_result]

    # --8<-- [start:save_result]
    with open("result.json", "w") as f:
        f.write(result.model_dump_json(indent=2))

    # Or keep only the conversations, one JSON line per trajectory:
    with open("conversations.jsonl", "w") as f:
        for traj in result:
            f.write(json.dumps({"metadata": traj.metadata, "messages": traj.segments[-1].messages}) + "\n")
    # --8<-- [end:save_result]


if __name__ == "__main__":  # required: vLLM starts worker processes that re-import this file
    asyncio.run(main())
# --8<-- [end:full_script]
