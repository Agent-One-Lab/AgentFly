import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import json
import time
from ..utils.messages import MessagesList, Messages
from ...utils.timing import Timer
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import uuid
from termcolor import colored
import numpy as np
from copy import deepcopy
from ...tools.tool_base import Tool, submit_tool_call, submit_tool_calls
from tqdm.asyncio import tqdm_asyncio
from ...utils.monitor import JsonlSink, MetricEvent, Monitor, WandbSink, emit, serialize_for_json
from ... import AGENT_DATA_DIR
import wandb
from .streaming_observer import ConsoleStreamObserver, StreamingManager, StreamEvent, StreamEventType

@dataclass
class SimplifiedNode:
    """
    Simplified node that aligns with message types.
    Each node represents a single message turn and only stores its own content.
    """
    role: str  # "user", "assistant", "tool", "system"
    content: Any  # The actual message content
    is_terminal: bool = False
    is_pruned: bool = False
    description: str = ""
    observation: str = ""  # For tool nodes, stores the tool result
    tool_name: Optional[str] = None  # For tool nodes
    tool_call_id: Optional[str] = None  # For tool nodes
    parent: Optional["SimplifiedNode"] = None
    children: List["SimplifiedNode"] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    def print_node(self, process_id: int = 0) -> None:
        if process_id != 0:
            return
        color_converter = {
            "user": "green",
            "assistant": "blue", 
            "tool": "yellow",
            "system": "magenta"
        }
        color = color_converter.get(self.role, "white")
        print(colored(f"{self.role.upper()}: {self.description}", color=color))
        if self.observation:
            obs = (
                self.observation
                if len(self.observation) < 1536
                else f"{self.observation[:1536]}...(len={len(self.observation)})"
            )
            print(colored(f"Observation: {obs}", color="yellow"))

    def to_json(self) -> dict:
        json_obj = {
            "role": self.role,
            "is_terminal": self.is_terminal,
            "is_pruned": self.is_pruned,
            "depth": self.depth,
            "description": self.description,
            "content": self.content
        }
        if self.observation:
            json_obj["observation"] = self.observation
        if self.tool_name:
            json_obj["tool_name"] = self.tool_name
        if self.tool_call_id:
            json_obj["tool_call_id"] = self.tool_call_id
        return json_obj

    def to_json_recursive(self) -> dict:
        data = self.to_json()
        data["children"] = [child.to_json_recursive() for child in self.children]
        return data


class SimplifiedChain:
    """
    Simplified chain that stores nodes aligned with message types.
    Each node represents a single message turn.
    """
    def __init__(self, info):
        self.root: Optional[SimplifiedNode] = None
        self.info: Dict[str, Any] = info
        self.system_message: Optional[SimplifiedNode] = None

    def add_node(
        self,
        role: str,
        content: Any,
        is_terminal: bool = False,
        is_pruned: bool = False,
        description: str = "",
        observation: str = "",
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None
    ) -> SimplifiedNode:
        new_node = SimplifiedNode(
            role=role,
            content=content,
            is_terminal=is_terminal,
            is_pruned=is_pruned,
            description=description,
            observation=observation,
            tool_name=tool_name,
            tool_call_id=tool_call_id
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

    def get_full_messages(self) -> List[Dict[str, Any]]:
        """
        Reconstruct the full message history from the chain nodes.
        """
        messages = []
        node = self.root
        
        # Add system message if it exists
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message.content
            })
        
        # Traverse the chain and collect messages
        while node:
            if node.role == "tool":
                # For tool messages, we need to reconstruct the tool message format
                messages.append({
                    "role": "tool",
                    "tool_call_id": node.tool_call_id,
                    "tool_name": node.tool_name,
                    "content": [{"type": "text", "text": node.observation}]
                })
            else:
                # For user and assistant messages
                messages.append({
                    "role": node.role,
                    "content": node.content
                })
            
            if node.children:
                node = node.children[0]
            else:
                break
        
        return messages

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


class SimplifiedChainRollout:
    """
    Simplified chain-based rollout that uses message-aligned nodes.
    """
    def __init__(self):
        self.reset()
        self.chains: Dict[str, SimplifiedChain] = {}
        self.current_nodes: Dict[str, SimplifiedNode] = {}
        self.timer = Timer()
        self.terminal_status = ["terminal", "finish"]
        self.global_step = 0
        self.finished_chains_count = 0
        self.monitor_info = defaultdict(list)

    def reset(self) -> None:
        self.status_code: str = "continue"
        self.query_count: int = 0
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

    def initialize_chains(self, messages_list: MessagesList, num_chains: int) -> Tuple[Dict[str, SimplifiedChain], Dict[str, SimplifiedNode]]:
        chains = {}
        start_nodes = {}
        group_ids = [str(uuid.uuid4()) for _ in range(len(messages_list))]

        for group_idx, messages in enumerate(messages_list):
            group_id = group_ids[group_idx]
            for j in range(num_chains):
                ch = SimplifiedChain(messages.meta | {"group_id": group_id})
                
                # Extract system message if present
                if messages.messages and messages.messages[0]["role"] == "system":
                    system_content = messages.messages[0]["content"]
                    ch.system_message = SimplifiedNode(
                        role="system",
                        content=system_content,
                        description="System prompt"
                    )
                    user_messages = messages.messages[1:]
                else:
                    user_messages = messages.messages
                
                # Create user node with the initial user message(s)
                user_content = user_messages[0]["content"] if user_messages else ""
                root = ch.add_node(
                    role="user",
                    content=user_content,
                    description="Initial user input"
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
            message_item["messages"] = self.chains[id].get_full_messages()
            message_item.update(info)
            messages.append(message_item)
        return messages

    def validate_run_args(self, max_turns: int, num_chains: int, enable_streaming: bool):
        assert max_turns >= 1, "max_turns must be at least 1."
        assert num_chains >= 1, "num_chains must be at least 1."
        for observer in self.streaming_manager.observers:
            if isinstance(observer, ConsoleStreamObserver) and enable_streaming:
                assert num_chains == 1, "num_chains must be 1 when ConsoleStreamObserver is used."
        
    
    async def run_async(self,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = False,
        streaming_callback: Optional[Callable] = None,
    ):
        """
        Run the simplified chain-based rollout.
        """
        self.validate_run_args(max_turns, num_chains, enable_streaming)
        Monitor.ensure_started()
        self.reset()

        messages_list = MessagesList.from_data(messages)
        chains, first_nodes = self.initialize_chains(
            messages_list,
            num_chains
        )
        tool_schemas = [tool.schema for tool in self.tools]

        done_q = asyncio.Queue()
        tasks = [
            asyncio.create_task(
                    self._run_single_chain(
                        cid,
                        node,
                        chains[cid],
                        tool_schemas,
                        max_turns=max_turns,
                        done_queue=done_q,
                        enable_streaming=enable_streaming
                    )
                )
            for cid, node in first_nodes.items()
        ]

        await tqdm_asyncio.gather(*tasks)

        self.chains = {}
        while not done_q.empty():
            cid, chain, node = done_q.get_nowait()
            self.chains[cid] = chain
            self.current_nodes[cid] = node

        self.global_step += 1
        self.monitor_step()

    async def _run_single_chain(self,
        chain_id: str,
        first_node: SimplifiedNode,
        chain: SimplifiedChain,
        tools: List[Dict],
        max_turns: int,
        done_queue: asyncio.Queue,
        enable_streaming: bool = False
    ):
        """
        Run a single simplified chain.
        """
        current_node = first_node
        depth = 0
        have_set_tools = False

        while not current_node.is_terminal and depth < max_turns:
            # Get current message history for generation
            current_messages = chain.get_full_messages()
            
            if not current_node.is_terminal:
                # Generate assistant response
                assistant_msg = await self._generate_response(
                    current_messages, tools, depth, chain_id, enable_streaming
                )
                
                # Create assistant node
                assistant_node = chain.add_node(
                    role="assistant",
                    content=assistant_msg.get("content", ""),
                    description=assistant_msg.get("content", "")[:100] + "..." if len(assistant_msg.get("content", "")) > 100 else assistant_msg.get("content", ""),
                    is_terminal=assistant_msg.get("status", "continue") in self.terminal_status
                )
                current_node = assistant_node
                
                # Check if the assistant node is terminal
                if current_node.is_terminal:
                    break

            # Handle tool calls
            if assistant_msg.get("tool_calls"):
                for tool_call in assistant_msg["tool_calls"]:
                    result = await self._execute_tool_call(
                        tool_call, chain, chain_id, depth, 
                        have_set_tools, enable_streaming
                    )
                    have_set_tools = True

                    # Create tool node
                    tool_node = chain.add_node(
                        role="tool",
                        content=result.get("arguments", ""),
                        description=f"Tool: {result.get('name', 'unknown')}",
                        observation=result["observation"],
                        tool_name=result.get("name"),
                        tool_call_id=tool_call["id"],
                        is_terminal=result["status"] in self.terminal_status
                    )
                    current_node = tool_node
            else:
                # No tool calls, chain is finished
                break
            
            depth += 1

        # Finalize chain
        await self._finalize_chain(chain_id, chain, current_node, depth)
        await done_queue.put((chain_id, chain, current_node))

        self.finished_chains_count += 1
        self.monitor_chain(trajectory=chain.get_full_messages())

    async def _generate_response(self, current_messages, tools, depth, chain_id, enable_streaming):
        """Generate response with optional streaming support."""
        if enable_streaming:
            # Emit generation start event
            await self.streaming_manager.emit_event(StreamEvent(
                event_type=StreamEventType.LLM_GENERATION_START,
                chain_id=chain_id,
                timestamp=time.time(),
                data={"depth": depth},
                step=depth,
                depth=depth
            ))

            # Check if we have streaming capabilities
            has_streaming = False
            if hasattr(self, 'generate_streaming'):
                has_streaming = True
            elif hasattr(self, 'llm_engine') and hasattr(self.llm_engine, 'generate_streaming'):
                has_streaming = True
                # Create a wrapper to use the LLM engine's streaming
                async def generate_streaming_wrapper(messages_list, **kwargs):
                    async for chunk in self.llm_engine.generate_streaming(messages_list, **kwargs):
                        yield chunk
                self.generate_streaming = generate_streaming_wrapper

            if has_streaming:
                # Collect full response from streaming
                full_response = ""
                async for chunk in self.generate_streaming([current_messages], tools=tools):
                    await self.streaming_manager.emit_event(StreamEvent(
                        event_type=StreamEventType.LLM_GENERATION_CHUNK,
                        chain_id=chain_id,
                        timestamp=time.time(),
                        data={"content": chunk},
                        step=depth,
                        depth=depth
                    ))
                    full_response = chunk
                
                # Emit generation end event
                await self.streaming_manager.emit_event(StreamEvent(
                    event_type=StreamEventType.LLM_GENERATION_END,
                    chain_id=chain_id,
                    timestamp=time.time(),
                    data={"full_response": full_response},
                    step=depth,
                    depth=depth
                ))
                
                # Parse response
                new_msg = self.parse([full_response], tools=self.tools)
                return new_msg[0]
            else:
                # Fallback to non-streaming generation
                responses = await self.generate_async([current_messages], tools=tools, num_return_sequences=1)
                new_msg = self.parse(responses, tools=self.tools)
                
                # Emit a single chunk event for the full response
                full_response = new_msg[0].get("content", "")
                if isinstance(full_response, list) and len(full_response) > 0:
                    if isinstance(full_response[0], dict) and "text" in full_response[0]:
                        full_response = full_response[0]["text"]
                    else:
                        full_response = str(full_response)
                elif not isinstance(full_response, str):
                    full_response = str(full_response)
                
                await self.streaming_manager.emit_event(StreamEvent(
                    event_type=StreamEventType.LLM_GENERATION_CHUNK,
                    chain_id=chain_id,
                    timestamp=time.time(),
                    data={"content": full_response},
                    step=depth,
                    depth=depth
                ))
                
                # Emit generation end event
                await self.streaming_manager.emit_event(StreamEvent(
                    event_type=StreamEventType.LLM_GENERATION_END,
                    chain_id=chain_id,
                    timestamp=time.time(),
                    data={"full_response": full_response},
                    step=depth,
                    depth=depth
                ))
                
                return new_msg[0]
        else:
            # Non-streaming generation
            responses = await self.generate_async([current_messages], tools=tools, num_return_sequences=1)
            new_msg = self.parse(responses, tools=self.tools)
            return new_msg[0]

    async def _execute_tool_call(self, tool_call, chain, chain_id, depth, have_set_tools, enable_streaming):
        """Execute a tool call with optional streaming support."""
        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]
        
        # Set up tools if needed
        if not have_set_tools:
            await self.set_tools(chain_id, chain.info)
            have_set_tools = True

        # Execute tool call
        result = await submit_tool_call(
            tool_name,
            tool_input,
            id=chain_id,
            allowed_tool_names=self.tool_names
        )
        
        if enable_streaming:
            # Emit tool observation event
            await self.streaming_manager.emit_event(StreamEvent(
                event_type=StreamEventType.TOOL_OBSERVATION,
                chain_id=chain_id,
                timestamp=time.time(),
                data={
                    "tool_name": tool_name,
                    "observation": result["observation"],
                    "status": result["status"]
                },
                step=depth,
                depth=depth
            ))

        return result
            

    async def _finalize_chain(self, chain_id, chain, current_node, depth):
        """Finalize the chain with reward calculation and cleanup."""
        if self._reward_fn is not None:
            trajectory = chain.get_full_messages()
            final_response = self.extract_final_response(trajectory)
            other_args = {k: v for k, v in chain.info.items() if k not in ['prediction', 'trajectory', 'id']}
            chain.info["reward"] = await self._reward_fn(prediction=final_response, **other_args, trajectory=trajectory, id=chain_id)
        else:
            chain.info["reward"] = None
            
        await self.release_resources(chain_id)

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
                    tool_call_name = msg['tool_name']
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


    def monitor_chain(self, trajectory) -> None:
        self.monitor_info['Agent/chains'].append(self.finished_chains_count)
        for tool in self.tools:
            if tool.is_stateful and tool.pool_size > 0:
                self.monitor_info[f"Agent/Tool/{tool.name}/used_env_size"].append(tool.used_env_size)
        
        evt = MetricEvent(
            kind="text",
            name="Agent/rollout/trajectory",
            value=json.dumps(serialize_for_json(trajectory), indent=2),
            x=self.global_step,
            x_name="Agent/rollout/step"
        )
        emit(evt)
