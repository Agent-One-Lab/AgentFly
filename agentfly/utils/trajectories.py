"""
This module is used to collect trajectories from the agent.
"""
from typing import Dict, List
import asyncio
import json
import os
from ..agents.llm_backends.backend_configs import ClientConfig
from ..agents import HFAgent
import click

def gather_responses(trajectories: List[Dict]):
    responses = []
    for trajectory in trajectories:
        responses.append({
            "id": trajectory["id"],
            "response": trajectory["messages"][-1]["content"][0]["text"],
        })
    
    return responses

@click.command()
@click.option("--model_name_or_path", type=str, required=True)
@click.option("--api_key", type=str, default="EMPTY")
@click.option("--max_turns", type=int, required=True)
@click.option("--num_chains", type=int, default=1, required=True)
@click.option("--data_file", type=str, required=True)
@click.option("--output_dir", type=str, required=True)
def main(
    model_name_or_path: str,
    api_key: str,
    max_turns: int,
    num_chains: int,
    data_file: str,
    output_dir: str,
):
    async def run_agent():
        with open(data_file, 'r') as f:
            messages = json.load(f)
            
        agent = HFAgent(
            model_name_or_path=model_name_or_path,
            tools=[],
            backend="client",
            backend_config=ClientConfig(
                base_url="http://0.0.0.0:8000/v1",
                api_key=api_key,
                max_new_tokens=30720,
                timeout=2400,
            ),
            local_cache_dir="test_cache",
        )
        await agent.run(
            messages=messages,
            num_chains=num_chains,
            max_turns=max_turns,
        )
        responses = gather_responses(agent.trajectories)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, os.path.basename(model_name_or_path) + '_responses.json'), 'w') as f:
            json.dump(responses, f, indent=2)
    
    asyncio.run(run_agent())

if __name__ == "__main__":
    # """python -m agentfly.utils.trajectories --model_name_or_path Qwen/Qwen2.5-3B-Instruct --api_key EMPTY --max_turns 1 --data_file ../../datasets/viphy_test.json --output_dir data/trajectories/"""
    """python -m agentfly.utils.trajectories --model_name_or_path openai/gpt-oss-20b --api_key EMPTY --max_turns 1 --data_file ../../datasets/viphy_test.json --output_dir data/trajectories/"""
    main()
