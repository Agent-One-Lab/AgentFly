import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import json
from ...utils.timing import Timer
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from termcolor import colored
import numpy as np
from copy import deepcopy
from ...tools.tool_base import Tool, submit_tool_call, submit_tool_calls
from tqdm.asyncio import tqdm_asyncio
from ...utils.monitor import JsonlSink, MetricEvent, Monitor, WandbSink, emit, serialize_for_json
from ... import AGENT_DATA_DIR
import wandb
@dataclass
class Node:
    is_terminal: bool = False
    is_pruned: bool = False
    type: Optional[str] = None
    description: str = ""
    observation: str = ""
    observation_code: Optional[str] = None
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    messages: List[Any] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def print_node(self, process_id: int = 0) -> None:
        if process_id != 0:
            return
        color_converter = {
            "Thought": "red",
            "Action": "blue",
            "Action Input": "cyan",
            "Final Answer": "green",
            "Reflection": "blue"
        }
        color = color_converter.get(self.type, "white")
        print(colored(f"{self.type}: {self.description}", color=color))
        if self.observation:
            obs = (
                self.observation
                if len(self.observation) < 1536
                else f"{self.observation[:1536]}...(len={len(self.observation)})"
            )
            print(colored(f"Observation: {obs}", color="yellow"))

    def to_json(self, use_messages: bool = False) -> dict:
        json_obj = {
            "is_terminal": self.is_terminal,
            "is_pruned": self.is_pruned,
            "depth": self.depth,
            "type": self.type,
            "description": self.description,
            "messages": self.messages if use_messages else []
        }
        if self.observation:
            json_obj["observation"] = self.observation
        if self.observation_code is not None:
            json_obj["observation_code"] = self.observation_code
        return json_obj

    def to_json_recursive(self, use_messages: bool = False) -> dict:
        data = self.to_json(use_messages=use_messages)
        data["children"] = [child.to_json_recursive(use_messages=use_messages) for child in self.children]
        return data
    

class Chain:
    """
    Manages a sequential chain of nodes (chain-of-thought).
    Each node can have at most one child.
    """
    def __init__(self, info):
        self.root: Optional[Node] = None
        self.info: Dict[str, Any] = info

    def add_node(
        self,
        is_terminal: bool = False,
        is_pruned: bool = False,
        type: Optional[str] = None,
        description: str = "",
        observation: str = "",
        observation_code: Optional[str] = None,
        messages: Optional[List[Any]] = None
    ) -> Node:
        messages = messages if messages is not None else []
        new_node = Node(
            is_terminal=is_terminal,
            is_pruned=is_pruned,
            type=type,
            description=description,
            observation=observation,
            observation_code=observation_code,
            messages=messages
        )
        if self.root is None:
            self.root = new_node
        else:
            current = self.root
            while len(current.children) > 0:
                current = current.children[0]
            current.children = [new_node]
            new_node.parent = current
        return new_node

    def to_json(self) -> List[dict]:
        chain_json = []
        node = self.root
        while node:
            chain_json.append(node.to_json())
            if node.children:
                node = node.children[0]
            else:
                break
        return chain_json


class ChainGeneration:
    """
    Basic class for chain-based rollout. It starts multiple chains and runs them asynchronously.
    """
    def __init__(self):
        self.reset()
        self.chains: Dict[str, Chain] = {}
        self.current_nodes: Dict[str, Node] = {}
        self.timer = Timer()
        self.terminal_status = ["terminal", "finish"]
        self.global_step = 0
        self.finished_chains_count = 0
        self.initialize_monitor()
        self.monitor_info = defaultdict(list)

    def reset(self) -> None:
        self.status_code: str = "continue"
        self.query_count: int = 0  # Number of interactions
        self.total_tokens: int = 0
        self.success_count: int = 0
        self.chains = []
        self.current_nodes = {}
    
    @property
    def timing_data(self):
        return self.timer.timing_data
    
    def to_json(self) -> dict:
        return {
            "finish": [chain.status_code == "success" for chain in self.chains],
            "chains": [chain.to_json() for chain in self.chains]
        }

    def initialise_chains(self, msgs_list: List[List[Dict]], info_list: List[Dict], num_chains: int) -> Tuple[Dict[str, Chain], Dict[str, Node]]:
        chains = {}
        start_nodes = {}
        group_ids = [str(uuid.uuid4()) for _ in range(len(msgs_list))]

        for group_idx, (prompt_msgs, info) in enumerate(zip(msgs_list, info_list)):
            group_id = group_ids[group_idx]
            for j in range(num_chains):
                ch = Chain(info | {"group_id": group_id})
                root = ch.add_node(
                    type="Action Input",
                    messages=deepcopy(prompt_msgs)
                )

                cid = str(uuid.uuid4())
                chains[cid] = ch
                start_nodes[cid] = root

        return chains, start_nodes

    def get_messages(self) -> List[Any]:
        messages = []
        for id, node in self.current_nodes.items():
            info = self.chains[id].info
            message_item = {}
            message_item["messages"] = node.messages
            message_item.update(info)
            messages.append(message_item)
        return messages

    def prepare_chain_messages(self, start_messages: Union[List[dict], np.ndarray]):
        if isinstance(start_messages, np.ndarray):
            start_messages_list = start_messages.tolist()
        else:
            start_messages_list = start_messages

        if self.system_prompt is not None and self.system_prompt != "":
            for message in start_messages_list:
                if message["messages"][0]["role"] != "system":
                    message["messages"].insert(0, {"role": "system", "content": self.system_prompt})
        
        example_message = start_messages_list[0]
        if isinstance(example_message, dict):
            assert "messages" in example_message
        
        messages_list = []
        other_info_list = []
        for message in start_messages_list:
            messages_list.append(message["messages"])
            info = {}
            for key, value in message.items():
                if key != "messages":
                    info[key] = value
            other_info_list.append(info)
        
        return messages_list, other_info_list
    
    async def run_async(self,
        max_steps: int,
        start_messages: Union[List[dict], np.ndarray],
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Run the chain-based rollout.
        
        Args:
            max_steps: Maximum number of steps for each chain.
            start_messages: List of messages to start the chains. Each message should be a dict
                with "messages" key containing a list of message dictionaries.
            num_chains: Number of chains to run for each message.
            generation_config: Generation configuration dictionary.
        
        Example:
            .. code-block:: python
            
                start_messages = [
                    {
                        "messages": [{"role": "user", "content": "..."}],
                        "info": {"question": "..."},
                        "answer": "..."
                    },
                    {
                        "messages": [{"role": "user", "content": "..."}],
                        "info": {"question": "..."},
                        "answer": "..."
                    }
                ]
        """
        assert max_steps >= 1, "max_steps must be at least 1."
        Monitor.ensure_started()
        self.reset()
        messages_list, other_info_list = self.prepare_chain_messages(start_messages)
        chains, first_nodes = self.initialise_chains(
            messages_list,
            other_info_list,
            num_chains
        )
        tool_schemas = [tool.schema for tool in self.tools]

        done_q = asyncio.Queue()
        tasks = [
            asyncio.create_task(
                self._run_chain_async(
                    cid,
                    node,
                    chains[cid],
                    tool_schemas,
                    max_steps=max_steps,
                    done_queue=done_q)
                )
                for cid, node in first_nodes.items()
        ]

        # Throttle the number of concurrent chains
        # print([tool.parallel_size for tool in self.tools])

        # minimal_tool_parallel_size = 1
        # sem = asyncio.Semaphore(minimal_tool_parallel_size)
        # async def guarded_run(cid, *args):
        #     async with sem:
        #         return await self._run_chain_async(cid, *args)
        # tasks = [guarded_run(cid, node, chains[cid], max_steps, done_q) for cid, node in first_nodes.items()]
        # await asyncio.gather(*tasks)
        await tqdm_asyncio.gather(*tasks)

        self.chains = {}
        while not done_q.empty():
            cid, chain, node = done_q.get_nowait()
            self.chains[cid] = chain
            self.current_nodes[cid] = node

        self.global_step += 1
        self.monitor_step()

    async def _run_chain_async(self,
        chain_id: str,
        first_node: Node,
        chain: Chain,
        tools: List[Dict],
        max_steps: int,
        done_queue: asyncio.Queue
    ):
        """
        Drives *one* trajectory until it terminates or max_steps is reached.
        Writes (chain_id, chain) to done_queue when finished.
        """
        current_node = first_node
        depth = 0
        final_result = None
        have_set_tools = False

        while not current_node.is_terminal and depth < max_steps:
            newest_messages = current_node.messages
            if not current_node.is_terminal:
                responses = await self.generate_async([current_node.messages], tools=tools, num_return_sequences=1)
                new_msg = self.parse(responses, self.tools)
                new_msg = new_msg[0]
                newest_messages.append(new_msg)
                thought_node = chain.add_node(
                    type="Thought",
                    messages=deepcopy(newest_messages),
                    description=new_msg.get("content", "")
                )
                thought_node.is_terminal = new_msg.get("status", "continue") in self.terminal_status
                current_node = thought_node

            if current_node.messages[-1].get("tool_calls"):
                for tool_call in current_node.messages[-1]["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_input = tool_call["function"]["arguments"]
                    action_node = chain.add_node(
                        type="Action",
                        messages=deepcopy(newest_messages),
                        description=tool_name
                    )
                    if not have_set_tools:
                        await self.set_tools(chain_id, chain.info)
                        have_set_tools = True

                    result = await submit_tool_call(tool_name, tool_input, id=chain_id)
                    final_result = result
                    action_input_node = chain.add_node(
                        type="Action Input",
                        messages=deepcopy(newest_messages),
                        description=result.get("arguments", "")
                    )
                    observation = result["observation"]
                    observation_json = json.dumps({
                        "name": result["name"],
                        "content": observation,
                    }, indent=4)
                    action_input_node.observation = observation_json
                    action_input_node.observation_code = result["status"]
                    newest_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": [{"type": "text", "text": observation_json}],
                    })
                    action_input_node.messages = deepcopy(newest_messages)
                    action_input_node.is_terminal = result["status"] in self.terminal_status
                    current_node = action_input_node
            else:
                # If there is no tool call, we assume the chain is finished
                break

            depth += 1

        if self._reward_fn is not None:
            trajectory = current_node.messages
            final_response = self.extract_final_response(trajectory)
            other_args = {k: v for k, v in chain.info.items() if k not in ['prediction', 'trajectory', 'id']}
            chain.info["reward"] = await self._reward_fn(prediction=final_response, **other_args, trajectory=trajectory, id=chain_id)
        else:
            chain.info["reward"] = None
        await self.release_resources(chain_id)

        await done_queue.put((chain_id, chain, current_node))

        self.finished_chains_count += 1
        self.monitor_chain()

    async def release_resources(self, id: str) -> None:
        for tool in self.tools:
            if isinstance(tool, Tool):
                await tool.release(id=id)
        if self._reward_fn is not None:
            await self._reward_fn.release(id=id)

    async def set_tools(self, id: str, env_args: Dict[str, Any]) -> None:
        for tool in self.tools:
            if isinstance(tool, Tool):
                await tool.set_env(id, env_args)

    def initialize_monitor(self) -> None:
        Monitor.add_sink("jsonl", JsonlSink(f"{AGENT_DATA_DIR}/demo_metrics.jsonl"))
        Monitor.add_sink("wandb", WandbSink(project=self.project_name, run_name=self.run_name))

    def monitor_step(self) -> None:
        messages = self.get_messages()
        avg_turns = 0
        avg_tool_calls = 0
        avg_response_length = 0
        tool_calls_by_name = defaultdict(int)

        for message in messages:
            for msg in message['messages']:
                if msg['role'] == 'assistant':
                    avg_turns += 1
                if msg['role'] == 'tool':
                    avg_tool_calls += 1
                    tool_call_name = json.loads(msg['content'][0]['text'])['name']
                    tool_calls_by_name[tool_call_name] += 1

        avg_turns /= len(messages)
        avg_tool_calls /= len(messages)

        ent = MetricEvent(
            kind="scalar",
            name=f"Agent/rollout/step",
            value=self.global_step,
            x=self.global_step,
            x_name="Agent/rollout/step"
        )
        emit(ent)

        evt = MetricEvent(
            kind="scalar",
            name=f"Agent/rollout/avg_turns",
            value=avg_turns,
            x=self.global_step,
            x_name="Agent/rollout/step"
        )
        emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name=f"Agent/rollout/avg_tool_calls",
            value=avg_tool_calls,
            x=self.global_step,
            x_name="Agent/rollout/step"
        )
        emit(evt)


        for tool_name, tool_call_count in tool_calls_by_name.items():
            evt = MetricEvent(
                kind="scalar",
                name=f"Agent/rollout/tool_calls/{tool_name}",
                value=tool_call_count,
                x=self.global_step,
                x_name="Agent/rollout/step"
            )
            emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name=f"Agent/rollout/step",
            value=self.global_step,
            x=self.global_step,
            x_name="Agent/rollout/step"
        )
        emit(evt)

        sample_message_json = json.dumps(serialize_for_json(messages[0]), indent=2)
        evt = MetricEvent(
            kind="text",
            name="Agent/rollout/sample_message",
            value=sample_message_json,
            x=self.global_step,
            x_name="Agent/rollout/step"
        )
        emit(evt)

        for k, v in self.monitor_info.items():
            if k != "Agent/chains": # We don't log number of chains
                evt = MetricEvent(
                    kind="list",
                    name=k,
                    value=v,
                    x=self.monitor_info['Agent/chains'],
                )
                emit(evt)


    def monitor_chain(self) -> None:
        self.monitor_info['Agent/chains'].append(self.finished_chains_count)
        for tool in self.tools:
            if tool.is_stateful and tool.pool_size > 0:
                self.monitor_info[f"Agent/Tool/{tool.name}/used_env_size"].append(tool.used_env_size)
        
