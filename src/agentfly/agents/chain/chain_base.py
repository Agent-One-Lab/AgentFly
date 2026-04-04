import asyncio
import inspect
import json
import logging
import random
import sys
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio

from ...core import Context
from ...tools.tool_base import submit_tool_call
from ...utils.monitor import MetricEvent, Monitor, emit, serialize_for_json
from ...utils.timing import Timer
from ...utils.vision import image_to_data_uri
from ..utils.messages import Messages, MessagesList
from agentfly.tools.src.context.tools import fold_messages_with_summarize
from .streaming_observer import ConsoleStreamObserver, StreamEvent, StreamEventType
from .structures import Chain, Node

logger = logging.getLogger(__name__)


class ChainRollout:
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
        self.monitor_info = defaultdict(list)

    def reset(self) -> None:
        self.status_code: str = "continue"
        self.query_count: int = 0  # Number of interactions
        self.total_tokens: int = 0
        self.success_count: int = 0
        self.chains = {}
        self.current_nodes = {}

    @property
    def timing_data(self):
        return self.timer.timing_data

    def to_json(self) -> dict:
        chains_list = list(self.chains.values()) if isinstance(self.chains, dict) else self.chains
        return {
            "finish": [c.info.get("status_code") == "success" for c in chains_list],
            "chains": [c.to_json() for c in chains_list],
        }

    def initialize_chains(
        self, messages_list: MessagesList, num_chains: int
    ) -> Tuple[Dict[str, Chain], Dict[str, Node]]:
        chains = {}
        start_nodes = {}
        group_ids = [str(uuid.uuid4()) for _ in range(len(messages_list))]

        for group_idx, messages in enumerate(messages_list):
            group_id = group_ids[group_idx]
            for j in range(num_chains):
                ch = Chain(
                    messages.meta
                    | {"group_id": group_id, "group_idx": group_idx, "chain_idx": j}
                )
                root = ch.add_node(
                    type="Action Input", messages=deepcopy(messages.messages)
                )

                cid = str(uuid.uuid4())
                chains[cid] = ch
                start_nodes[cid] = root

        return chains, start_nodes

    def get_trajectories(self) -> List[Any]:
        trajectories = []
        # Sort by (group_idx, chain_idx) so trajectories are in deterministic order
        items = list(self.current_nodes.items())
        items.sort(
            key=lambda item: (
                self.chains[item[0]].info.get("group_idx", 0),
                self.chains[item[0]].info.get("chain_idx", 0),
            )
        )
        for id, node in items:
            chain = self.chains[id]
            info = chain.info
            trajectory_item = {}
            # Concatenate all stored history segments for this chain.
            # Histories may contain multiple segments when context tools are used.
            all_segments = []
            for segment in chain.histories:
                all_segments.append(segment)
            trajectory_item["trajectory_segments"] = all_segments
            trajectory_item.update(info)
            trajectories.append(trajectory_item)
        return trajectories

    def validate_run_args(
        self, max_turns: int, num_chains: int, enable_streaming: bool
    ):
        assert max_turns >= 1, "max_turns must be at least 1."
        assert num_chains >= 1, "num_chains must be at least 1."
        for observer in self.streaming_manager.observers:
            if isinstance(observer, ConsoleStreamObserver) and enable_streaming:
                assert num_chains == 1, (
                    "num_chains must be 1 when ConsoleStreamObserver is used."
                )

    async def run_async(
        self,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = False,
    ):
        """
        Run the chain-based rollout with optional streaming support.

        Args:
            max_steps: Maximum number of steps for each chain.
            start_messages: List of messages to start the chains.
            num_chains: Number of chains to run for each message.
            generation_config: Generation configuration dictionary.
            enable_streaming: Whether to enable streaming mode.
            streaming_callback: Optional callback for streaming events.
        """
        self.validate_run_args(max_turns, num_chains, enable_streaming)
        Monitor.ensure_started()
        self.reset()

        messages_list = MessagesList.from_data(messages)
        chains, first_nodes = self.initialize_chains(messages_list, num_chains)
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
                    generation_config=generation_config,
                    done_queue=done_q,
                    enable_streaming=enable_streaming,
                )
            )
            for cid, node in first_nodes.items()
        ]

        await tqdm_asyncio.gather(*tasks, file=sys.stdout)

        self.chains = {}
        while not done_q.empty():
            cid, chain, node = done_q.get_nowait()
            self.chains[cid] = chain
            self.current_nodes[cid] = node

        self.global_step += 1
        self.monitor_step()

    async def _run_single_chain(
        self,
        chain_id: str,
        first_node: Node,
        chain: Chain,
        tools: List[Dict],
        max_turns: int,
        generation_config: Dict[str, Any],
        done_queue: asyncio.Queue,
        enable_streaming: bool = False,
    ):
        """
        Run a single chain with optional streaming support. It supports parallel tool calls (running multiple tool calls for a single turn).
        If there is no tool call, we stop the task immediately.

        Args:
            chain_id: The id of the chain.
            first_node: The first node of the chain.
            chain: The chain object.
            tools: The tools to use for the chain.
            max_turns: The maximum number of turns for the chain.
            generation_config: The generation configuration.
            done_queue: The queue to put the result of the chain.
            enable_streaming: Whether to enable streaming.

        """

        # Build Context from rollout data for tools that need it
        context = Context(
            rollout_id=chain_id,
            group_id=chain.info.get("group_id"),
            metadata=chain.info,
        )

        current_node = first_node
        depth = 0
        have_set_resources = False

        while not current_node.is_terminal and depth < max_turns:
            next_node, have_set_resources, should_continue = await self._run_one_turn(
                chain_id=chain_id,
                chain=chain,
                current_node=current_node,
                depth=depth,
                tools=tools,
                generation_config=generation_config,
                context=context,
                have_set_resources=have_set_resources,
                enable_streaming=enable_streaming,
            )
            if not should_continue:
                current_node = next_node
                break
            current_node = next_node
            depth += 1

        if "finish_reason" not in chain.info:
            chain.info["finish_reason"] = "max_turns"

        # Finalize chain
        await self._finalize_chain(chain_id, chain, current_node, depth, context)

        await done_queue.put((chain_id, chain, current_node))

        self.finished_chains_count += 1
        message_info = chain.info
        self.monitor_chain(trajectory=chain.histories, info=message_info)

    async def _run_one_turn(
        self,
        chain_id: str,
        chain: Chain,
        current_node: Node,
        depth: int,
        tools: List[Dict],
        generation_config: Dict[str, Any],
        context: Context,
        have_set_resources: bool,
        enable_streaming: bool,
    ) -> Tuple[Node, bool, bool]:
        """
        Run one turn: generate response, add thought node, process tool calls.

        Returns:
            (next_node, have_set_resources, should_continue). When should_continue
            is False, the loop should break and next_node is the final node for this chain.
        """
        newest_messages = current_node.messages.copy()
        max_model_len = getattr(self, "max_model_len", None)

        # Terminate without generating if already at or over context length
        if (
            max_model_len is not None
            and current_node.total_token_length >= max_model_len
        ):
            current_node.is_terminal = True
            chain.info["finish_reason"] = "max_model_len"
            return (current_node, have_set_resources, False)

        # Generate response
        new_msg, total_token_length = await self._generate_response(
            chain=chain,
            current_node=current_node,
            tools=tools,
            depth=depth,
            chain_id=chain_id,
            generation_config=generation_config,
            enable_streaming=enable_streaming,
        )

        newest_messages.append(new_msg)
        thought_node = chain.add_node(
            type="Thought",
            messages=newest_messages.copy(),
            description=new_msg.get("content", ""),
        )
        thought_node.total_token_length = (
            total_token_length if total_token_length is not None else 0
        )
        thought_node.is_terminal = (
            new_msg.get("status", "continue") in self.terminal_status
        )
        # Terminate if we hit or exceeded context length after this turn
        if (
            max_model_len is not None
            and thought_node.total_token_length >= max_model_len
        ):
            thought_node.is_terminal = True
            chain.info["finish_reason"] = "max_model_len"
        elif thought_node.is_terminal:
            chain.info["finish_reason"] = "terminal"

        if thought_node.is_terminal:
            return (thought_node, have_set_resources, False)

        # Handle tool calls
        num_parallel_tool_call = 0
        action_input_node = None
        running_total_token_length = thought_node.total_token_length
        if (
            thought_node.messages[-1].get("tool_calls")
            and len(thought_node.messages[-1]["tool_calls"]) > 0
        ):
            for tool_call in thought_node.messages[-1]["tool_calls"]:
                is_valid = self.validate_tool_call(tool_call)
                if not is_valid:
                    logger.debug(f"Invalid tool call: {tool_call}")
                    continue
                logger.debug(f"Valid tool call: {tool_call}")

                result = await self._execute_tool_call(
                    context,
                    tool_call,
                    newest_messages,
                    chain,
                    chain_id,
                    depth,
                    have_set_resources,
                    enable_streaming,
                )
                num_parallel_tool_call += 1
                have_set_resources = True

                action_input_node = chain.add_node(
                    type="Action Input",
                    messages=newest_messages.copy(),
                    description=result.get("arguments", ""),
                )
                observation = result["observation"]
                action_input_node.observation = observation
                action_input_node.observation_code = result["status"]
                observation_token_length = self._get_observation_token_length(
                    observation
                )
                action_input_node.total_token_length = (
                    running_total_token_length + observation_token_length
                )
                running_total_token_length = action_input_node.total_token_length

                # Terminate if context length exceeded after this tool observation
                if (
                    max_model_len is not None
                    and action_input_node.total_token_length >= max_model_len
                ):
                    action_input_node.is_terminal = True
                    chain.info["finish_reason"] = "max_model_len"
                    return (action_input_node, have_set_resources, False)

                new_content = [{"type": "text", "text": observation}]
                if "image" in result:
                    image_base64 = image_to_data_uri(result["image"])
                    new_content.append({"type": "image", "image": image_base64})

                # Apply context folding based on the tool result (currently only summarize).
                updated_messages = self.apply_context_folding(
                    chain=chain,
                    messages=newest_messages,
                    tool_name=tool_call["function"]["name"],
                    observation=observation,
                    tool_call_id=tool_call["id"],
                    tool_result_name=result["name"],
                    new_content=new_content,
                )

                action_input_node.messages = updated_messages
                newest_messages = updated_messages.copy()
                action_input_node.is_terminal = (
                    result["status"] in self.terminal_status
                )

        if num_parallel_tool_call == 0:
            chain.info["finish_reason"] = "no_tool_calls"
            return (thought_node, have_set_resources, False)
        if action_input_node is not None and action_input_node.is_terminal:
            chain.info["finish_reason"] = "terminal"

        return (action_input_node, have_set_resources, True)

    def apply_context_folding(
        self,
        chain: Chain,
        messages: Messages,
        tool_name: str,
        observation: Any,
        tool_call_id: str,
        tool_result_name: str,
        new_content: List[Dict[str, Any]],
    ) -> Messages:
        """
        Apply context folding based on the tool that was just executed.

        Args:
            chain: The current Chain object (used to record full histories).
            messages: The current full Messages object before applying the tool.
            tool_name: Name of the tool that was executed.
            observation: Tool observation returned by the tool.
            tool_call_id: ID of the tool call.
            tool_result_name: Name of the tool result (tool implementation name).
            new_content: Content payload for the tool message.

        Returns:
            Updated Messages object after folding / appending the tool message.
        """
        tool_turn: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "tool_name": tool_result_name,
            "content": new_content,
        }

        # Currently we only support summarize; other tools are no-ops.
        if tool_name == "summarize" and isinstance(observation, str):
            # Before folding, append the full messages to the chain histories
            # so training can still access the original, unfused trajectory.
            chain.histories.append(messages.messages)

            # Delegate folding policy to the tool module to get a folded view
            # of the history up to this point, then append the tool message.
            folded_turns = fold_messages_with_summarize(messages.messages, observation)
            # folded_turns.append(tool_turn)
            meta = messages.meta
            return Messages.from_turns(folded_turns, **meta)

        # Default behavior for non-folding tools: just append the tool message.
        turns = messages.messages + [tool_turn]
        meta = messages.meta
        return Messages.from_turns(turns, **meta)

    def _normalize_full_response_content(self, content: Any) -> str:
        """Convert message content (str or list of content blocks) to a single string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and "text" in content[0]:
                return content[0]["text"]
            return str(content)
        return str(content)

    def _normalize_generate_response(self, responses: Any) -> Dict[str, Any]:
        """Unwrap backend response to a single dict. ClientBackend returns list of dicts."""
        if isinstance(responses, list) and len(responses) == 1:
            return responses[0]
        if isinstance(responses, dict):
            return responses
        return {}

    def _extract_total_length(self, responses: dict) -> Optional[int]:
        """Extract total token length (after chat template) from generate_async response dict."""
        if "total_lengths" not in responses:
            return None
        tl = responses["total_lengths"]
        if tl is None:
            return None
        if hasattr(tl, "tolist"):  # e.g. torch tensor
            tl = tl.tolist()
        if isinstance(tl, list):
            return int(tl[0]) if tl else None
        return int(tl)

    def _get_observation_token_length(self, observation: Any) -> int:
        """Return token length of observation text. Uses self.tokenizer if available."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return 0
        text = observation if isinstance(observation, str) else str(observation)
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        except Exception:
            return 0

    def _prepare_generation_config(
        self,
        generation_config: Optional[Dict[str, Any]],
        current_node: Node,
    ) -> Dict[str, Any]:
        """Set max_tokens to max_model_len - total_token_length when not specified."""
        config = dict(generation_config) if generation_config else {}
        max_model_len = getattr(self, "max_model_len", None)
        if max_model_len is not None and "max_tokens" not in config:
            remaining = max_model_len - current_node.total_token_length
            config["max_tokens"] = max(1, remaining)
        return config

    async def _generate_response(
        self, chain, current_node, tools, depth, chain_id, generation_config, enable_streaming
    ) -> Tuple[Any, Optional[int]]:
        """Generate response with optional streaming support. Returns (message, total_token_length)."""
        effective_config = self._prepare_generation_config(
            generation_config, current_node
        )
        if enable_streaming:
            return await self._generate_response_streaming(
                chain=chain, current_node=current_node, tools=tools, depth=depth, chain_id=chain_id, generation_config=effective_config
            )
        return await self._generate_response_non_streaming(
            chain=chain, current_node=current_node, tools=tools, depth=depth, chain_id=chain_id, generation_config=effective_config
        )

    async def _generate_response_streaming(
        self, chain, current_node, tools, depth, chain_id, generation_config
    ):
        """Generate response with streaming: emit events and optionally stream from LLM."""
        await self.streaming_manager.emit_event(
            StreamEvent(
                event_type=StreamEventType.LLM_GENERATION_START,
                chain_id=chain_id,
                timestamp=time.time(),
                data={"depth": depth},
                step=depth,
                depth=depth,
            )
        )

        has_streaming = False
        if hasattr(self, "generate_streaming"):
            has_streaming = True
        elif hasattr(self, "llm_engine") and hasattr(
            self.llm_engine, "generate_streaming"
        ):
            has_streaming = True
            async def generate_streaming_wrapper(messages_list, **kwargs):
                async for chunk in self.llm_engine.generate_streaming(
                    messages_list, **kwargs
                ):
                    yield chunk
            self.generate_streaming = generate_streaming_wrapper

        if has_streaming:
            full_response = ""
            async for chunk in self.generate_streaming(
                [current_node.messages.messages], tools=tools, **generation_config
            ):
                await self.streaming_manager.emit_event(
                    StreamEvent(
                        event_type=StreamEventType.LLM_GENERATION_CHUNK,
                        chain_id=chain_id,
                        timestamp=time.time(),
                        data={"content": chunk},
                        step=depth,
                        depth=depth,
                    )
                )
                full_response = chunk
            logger.debug(
                f"[ChainRollout._generate_response] full_response: {full_response}"
            )
            await self._emit_generation_end(chain_id, depth, full_response)
            new_msg = self.parse([full_response], [current_node.messages.messages])
            return (new_msg[0], None)

        # Fallback to non-streaming when streaming not available
        raw = await self.generate_async(
            [current_node.messages.messages],
            tools=tools,
            return_dict=True,
            **generation_config,
        )
        responses = self._normalize_generate_response(raw)
        response_texts = responses.get("response_texts") or responses.get(
            "reponse_texts"
        )
        new_msg = self.parse(response_texts, [current_node.messages.messages])
        full_response = self._normalize_full_response_content(
            new_msg[0].get("content", "")
        )
        await self.streaming_manager.emit_event(
            StreamEvent(
                event_type=StreamEventType.LLM_GENERATION_CHUNK,
                chain_id=chain_id,
                timestamp=time.time(),
                data={"content": full_response},
                step=depth,
                depth=depth,
            )
        )
        await self._emit_generation_end(chain_id, depth, full_response)
        total_length = self._extract_total_length(responses)
        return (new_msg[0], total_length)

    async def _emit_generation_end(
        self, chain_id: str, depth: int, full_response: str
    ) -> None:
        """Emit LLM_GENERATION_END streaming event."""
        await self.streaming_manager.emit_event(
            StreamEvent(
                event_type=StreamEventType.LLM_GENERATION_END,
                chain_id=chain_id,
                timestamp=time.time(),
                data={"full_response": full_response},
                step=depth,
                depth=depth,
            )
        )

    async def _generate_response_non_streaming(
        self, chain, current_node, tools, depth, chain_id, generation_config
    ) -> Tuple[Any, Optional[int]]:
        """Generate response without streaming (no events). Returns (message, total_token_length)."""
        raw = await self.generate_async(
            [current_node.messages.messages],
            return_dict=True,
            tools=tools,
            **generation_config,
        )
        responses = self._normalize_generate_response(raw)
        response_texts = responses.get("response_texts")
        new_msg = self.parse(response_texts, [current_node.messages.messages])
        total_length = self._extract_total_length(responses)
        return (new_msg[0], total_length)

    def validate_tool_call(self, tool_call):
        tool_name = tool_call["function"]["name"]
        # TODO: validate tool input
        tool_input = tool_call["function"]["arguments"]  # noqa: F841
        if tool_name not in self.tool_names:
            return False
        return True

    async def _execute_tool_call(
        self,
        context,
        tool_call,
        newest_messages,
        chain,
        chain_id,
        depth,
        have_set_resources,
        enable_streaming,
    ):
        """Execute a tool call with optional streaming support."""
        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]

        # Set up tools if needed (reset resources that may have been acquired at chain start)
        if not have_set_resources:
            env_args = {
                k: context.metadata[k]
                for k in ("task_name", "variation_idx")
                if k in context.metadata
            }
            if env_args:
                await context.reset_resource(scope="rollout", env_args=env_args)
                await context.reset_resource(scope="global", env_args=env_args)
            else:
                await context.reset_resource(scope="rollout")
                await context.reset_resource(scope="global")
            have_set_resources = True

        # Execute tool call
        result = await submit_tool_call(
            tool_name,
            tool_input,
            context=context,
            allowed_tool_names=self.tool_names,
        )

        if enable_streaming:
            # Emit tool observation event
            tool_data = {
                "tool_name": tool_name,
                "observation": result["observation"],
                "status": result["status"],
            }
            if "image" in result:
                tool_data["image"] = result["image"]
            await self.streaming_manager.emit_event(
                StreamEvent(
                    event_type=StreamEventType.TOOL_OBSERVATION,
                    chain_id=chain_id,
                    timestamp=time.time(),
                    data=tool_data,
                    step=depth,
                    depth=depth,
                )
            )

        return result

    async def _finalize_chain(self, chain_id, chain, current_node, depth, context):

        # Always record the final trajectory segment so that histories capture
        chain.histories.append(current_node.messages.messages)

        """Finalize the chain with reward calculation and cleanup."""
        if self._reward_fn is not None:
            # flatten the histories
            full_trajectory = []
            for segment in chain.histories:
                full_trajectory.extend(segment)
            final_response = self.extract_final_response(full_trajectory)

            context.trajectory = full_trajectory
            context.final_response = final_response
            
            other_args = {
                k: v
                for k, v in chain.info.items()
                if k not in ["final_response", "trajectory", "id"]
            }

            # TODO: move the reward calculation to reward module
            reward = self._reward_fn(
                final_response=final_response,
                **other_args,
                trajectory=chain.histories,
                id=chain_id,
                context=context,
            )
            if inspect.iscoroutine(reward):
                reward = await reward

            chain.info["reward"] = reward
        else:
            chain.info["reward"] = None


        # Release global resources so other rollouts can use them
        await context.release_resource(scope="global")

        # Kill rollout-scoped resources
        await context.end_resource(scope="rollout")

    def monitor_step(self) -> None:
        trajectories = self.get_trajectories()
        avg_turns = 0
        avg_tool_calls = 0
        avg_segments = 0
        # avg_response_length = 0
        tool_calls_by_name = defaultdict(int)

        for trajectory in trajectories:
            for segment in trajectory["trajectory_segments"]:
                avg_segments += 1
                for msg in segment:
                    if msg["role"] == "assistant":
                        avg_turns += 1
                        if "tool_calls" in msg:
                            for tool_call in msg["tool_calls"]:
                                tool_call_name = tool_call["function"]["name"]
                                if tool_call_name in ["summarize"]:
                                    tool_calls_by_name["summarize"] += 1

                    if msg["role"] == "tool":
                        avg_tool_calls += 1
                        tool_call_name = msg["tool_name"]
                        tool_calls_by_name[tool_call_name] += 1

        avg_turns /= len(trajectories)
        avg_tool_calls /= len(trajectories)
        avg_segments /= len(trajectories)

        ent = MetricEvent(
            kind="scalar",
            name="agent/rollout/avg_segments",
            value=avg_segments,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(ent)

        ent = MetricEvent(
            kind="scalar",
            name="agent/rollout/step",
            value=self.global_step,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(ent)

        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/avg_turns",
            value=avg_turns,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/avg_tool_calls",
            value=avg_tool_calls,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        for tool_name, tool_call_count in tool_calls_by_name.items():
            evt = MetricEvent(
                kind="scalar",
                name=f"agent/rollout/tool_calls/{tool_name}",
                value=tool_call_count/len(trajectories),
                x=self.global_step,
                x_name="agent/rollout/step",
            )
            emit(evt)

        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/step",
            value=self.global_step,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        sample_trajectory_json = json.dumps(
            serialize_for_json(random.choice(trajectories)), indent=2
        )
        evt = MetricEvent(
            kind="text",
            name="agent/rollout/sample_trajectory",
            value=sample_trajectory_json,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        for k, v in self.monitor_info.items():
            if k != "agent/chains":  # We don't log number of chains
                evt = MetricEvent(
                    kind="list",
                    name=k,
                    value=v,
                    x=self.monitor_info["agent/chains"],
                )
                emit(evt)

        reward_values, other_values = self.rewards
        avg_reward = sum(reward_values) / len(reward_values)
        evt = MetricEvent(
            kind="scalar",
            name="agent/rollout/reward",
            value=avg_reward,
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)
        for key, value in other_values.items():
            if isinstance(value[0], float) or isinstance(value[0], int):
                avg_value = sum(value) / len(value)
                evt = MetricEvent(
                    kind="scalar",
                    name=f"agent/rollout/{key}",
                    value=avg_value,
                )
                emit(evt)

    def monitor_chain(self, trajectory, info) -> None:
        self.monitor_info["agent/chains"].append(self.finished_chains_count)

        # We only log the trajectory to local jsonl file, for wandb much bandwidth is needed
        evt = MetricEvent(
            sinks=["jsonl"],
            kind="text",
            name="agent/rollout/trajectory",
            value=json.dumps(serialize_for_json(trajectory), indent=2),
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)

        evt = MetricEvent(
            sinks=["jsonl"],
            kind="text",
            name="agent/rollout/info",
            value=json.dumps(serialize_for_json(info), indent=2),
            x=self.global_step,
            x_name="agent/rollout/step",
        )
        emit(evt)
