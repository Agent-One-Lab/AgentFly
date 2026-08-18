import asyncio
import logging
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from ...core import Context, ContextConfig
from ...rewards.reward_base import calculate_reward
from ...rewards.types import RewardResult
from ..types import Trajectory
from ...tools.tool_base import InvalidToolCallError, submit_tool_call
from ...utils.monitor import Monitor
from ...utils.timing import Timer
from ...utils.vision import image_to_data_uri
from ..utils.messages import Messages, MessagesList
from agentfly.tools.src.context.tools import fold_messages_with_summarize
from .events import (
    ChainEnded,
    ChainStarted,
    FinishReason,
    MessageProduced,
    RolloutError,
    ToolObserved,
)
from .generation import generate_response
from .metrics import RolloutMetrics
from .structures import Chain, Node
from .tokens import estimate_chat_prompt_tokens, observation_token_length


# Extra tokens charged per tool observation for the chat-template prefix that wraps it
# (e.g. ``<|im_start|>user<tool_response>`` ... ``</tool_response><|im_end|>``). Approximate;
# used only for the max_model_len budget check, not for the actual prompt rendering.
TOOL_OBS_PREFIX_TOKENS = 15


@dataclass(frozen=True)
class TurnResult:
    """Internal control value yielded by :meth:`ChainRollout._run_turn` as its last
    item — carries the loop-control outcome that the old method returned. Not a
    ``RolloutEvent``; :meth:`_run_chain` consumes it and does not relay it."""

    next_node: Node
    have_set_resources: bool
    should_continue: bool


@dataclass(frozen=True)
class ToolLoopResult:
    """Internal control value yielded by :meth:`ChainRollout._run_tools` as its last
    item — summarizes the tool-call loop's outcome for :meth:`_run_turn`. Not a
    ``RolloutEvent``.

    ``last_node`` is the final ``Action Input`` node (None if no valid tool ran).
    ``terminated`` is True when the loop stopped early on ``max_model_len`` (finish_reason
    is already set); the caller then ends the chain on ``last_node``.
    """

    last_node: Optional[Node]
    num_tool_calls: int
    have_set_resources: bool
    terminated: bool


logger = logging.getLogger(__name__)


class ChainRollout:
    """
    Basic class for chain-based rollout. It starts multiple chains and runs them asynchronously.

    Host contract: this is a mixin — the concrete agent it is mixed into
    (``BaseAgent`` and subclasses) must provide the model/tooling state and the
    template-method hooks the loop calls. That required surface is declared as
    :class:`~agentfly.agents.rollout.host.RolloutHost`; the ``TYPE_CHECKING`` block
    below mirrors it so static analysis can resolve ``self.<member>`` when this file is
    read in isolation. It has no runtime effect.
    """

    if TYPE_CHECKING:
        # Provided by the host (the concrete agent), not by ChainRollout itself.
        # Mirrors rollout.host.RolloutHost — see that module for the rationale.
        tools: List[Any]
        tool_names: List[str]
        tokenizer: Any
        processor: Any
        template: Any
        max_model_len: Optional[int]
        skills: Optional[List[Any]]
        system_prompt: Optional[str]
        default_tool_call_source: str
        _reward_fn: Optional[Callable[..., Any]]

        async def generate_async(self, messages_list: List[List[Dict]], **kwargs) -> Any: ...

        def parse(
            self,
            responses: List[str],
            tool_calls: Optional[List] = None,
            context: Any = None,
            **kwargs,
        ) -> List[Dict]: ...

        def extract_final_response(self, trajectory: List[Dict]) -> Any: ...

    def __init__(self):
        self.reset()
        self.chains: Dict[str, Chain] = {}
        self.current_nodes: Dict[str, Node] = {}
        self.timer = Timer()
        self.terminal_status = ["terminal", "finish"]
        self.global_step = 0
        self.finished_chains_count = 0
        self.metrics = RolloutMetrics()
        self.chain_rollout_seconds: Dict[str, float] = {}

    def reset(self) -> None:
        self.chains = {}
        self.current_nodes = {}
        self.chain_rollout_seconds = {}

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
                cid = str(uuid.uuid4())
                ch = Chain(
                    messages.meta
                    | {
                        "chain_id": cid,
                        "group_id": group_id,
                        "group_idx": group_idx,
                        "chain_idx": j,
                    }
                )
                root = ch.add_node(
                    type="Action Input", messages=deepcopy(messages.messages)
                )

                chains[cid] = ch
                start_nodes[cid] = root

        return chains, start_nodes

    # Reserved chain.info keys that map to typed Trajectory fields
    # rather than going into the free-form `metadata` dict.
    _RESERVED_INFO_KEYS = frozenset({
        "chain_id",
        "group_id",
        "chain_idx",
        "group_idx",
        "finish_reason",
        "rollout_time_sec",
        "reward",
    })

    def get_trajectories(self) -> List[Trajectory]:
        """Build :class:`Trajectory` objects for every active chain.

        Maps ``chain.info`` keys into the typed Trajectory fields:

        - ``reward`` (a ``RewardResult`` produced by ``calculate_reward``) →
          ``Trajectory.reward`` (float) + ``Trajectory.metrics`` (extras)
        - ``finish_reason``, ``rollout_time_sec``, ``chain_id``,
          ``group_id``, ``chain_idx``, ``group_idx`` → first-class fields
        - everything else (dataset-row passthrough fields like ``answer``,
          ``task_id``, etc.) → ``Trajectory.metadata``
        """
        trajectories: List[Trajectory] = []
        # Sort by (group_idx, chain_idx) so trajectories are in deterministic order
        items = list(self.current_nodes.items())
        items.sort(
            key=lambda item: (
                self.chains[item[0]].info.get("group_idx", 0),
                self.chains[item[0]].info.get("chain_idx", 0),
            )
        )
        for chain_id, node in items:
            chain = self.chains[chain_id]
            info = chain.info

            # Histories may contain multiple segments when context tools fold.
            segments = list(chain.histories)

            # Unwrap reward into the typed (reward, metrics) split.
            raw_reward = info.get("reward")
            if isinstance(raw_reward, RewardResult):
                reward_value = raw_reward.reward
                metrics = dict(raw_reward.extras)
            elif isinstance(raw_reward, dict):
                # Legacy back-compat: a dict reward stored directly without
                # going through calculate_reward (rare).
                reward_value = raw_reward.get("reward")
                metrics = {k: v for k, v in raw_reward.items() if k != "reward"}
            elif isinstance(raw_reward, (int, float)):
                reward_value = float(raw_reward)
                metrics = {}
            else:
                # None when no reward function was configured.
                reward_value = None
                metrics = {}

            # Free-form bag: everything in chain.info that isn't a reserved key.
            metadata = {
                k: v for k, v in info.items() if k not in self._RESERVED_INFO_KEYS
            }

            trajectories.append(
                Trajectory(
                    segments=segments,
                    reward=reward_value,
                    metrics=metrics,
                    finish_reason=info.get("finish_reason"),
                    rollout_time_sec=info.get("rollout_time_sec"),
                    chain_id=info.get("chain_id"),
                    group_id=info.get("group_id"),
                    chain_idx=info.get("chain_idx"),
                    group_idx=info.get("group_idx"),
                    metadata=metadata,
                )
            )
        return trajectories

    def validate_run_args(
        self,
        max_turns: int,
        num_chains: int,
        max_concurrent_chains: Optional[int],
    ):
        assert max_turns >= 1, "max_turns must be at least 1."
        assert num_chains >= 1, "num_chains must be at least 1."
        if max_concurrent_chains is not None:
            assert (
                max_concurrent_chains >= 1
            ), "max_concurrent_chains must be at least 1 when set."

    async def run_async(
        self,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        max_concurrent_chains: Optional[int] = None,
        context_config: Optional[ContextConfig] = None,
    ):
        """
        Run the chain-based rollout with optional streaming support.

        Args:
            max_steps: Maximum number of steps for each chain.
            start_messages: List of messages to start the chains.
            num_chains: Number of chains to run for each message.
            max_concurrent_chains: Maximum number of chains to execute concurrently
                across the whole rollout. If None, all chains are scheduled at once.
            generation_config: Generation configuration dictionary.
            context_config: Optional :class:`~agentfly.core.context_config.ContextConfig` for tool resources (backend).
        """
        # Drain the rollout event stream. Training ignores the events themselves; the
        # final state (``self.chains`` / ``self.current_nodes``) is populated by each
        # chain as it ends. ``run()`` builds the RunResult from that state afterwards.
        async for _ev in self._run_chains(
            messages,
            max_turns=max_turns,
            num_chains=num_chains,
            generation_config=generation_config,
            max_concurrent_chains=max_concurrent_chains,
            context_config=context_config,
        ):
            pass

    async def _run_chains(
        self,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        max_concurrent_chains: Optional[int] = None,
        context_config: Optional[ContextConfig] = None,
        show_progress: bool = True,
    ):
        """Run all chains concurrently and yield their merged :mod:`.events` stream.

        This is the single shared rollout path: the per-chain generator :meth:`_run_chain` is the loop,
        and this method fans the per-chain event streams into one ``chain_id``-tagged
        stream via a queue. Each chain populates ``self.chains`` / ``self.current_nodes``
        as it ends; this method just relays events and, when drained, advances the
        global step and emits step metrics.

        Args:
            show_progress: render a ``tqdm`` bar over chain completion. Disable when a
                consumer owns the terminal (e.g. the Textual console).
        """
        self.validate_run_args(max_turns, num_chains, max_concurrent_chains)
        Monitor.ensure_started()
        self.reset()

        messages_list = MessagesList.from_data(messages)
        chains, first_nodes = self.initialize_chains(messages_list, num_chains)
        tool_schemas = [tool.schema for tool in self.tools]

        q: asyncio.Queue = asyncio.Queue()
        sem: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(int(max_concurrent_chains))
            if max_concurrent_chains is not None
            else None
        )
        done = object()  # sentinel pushed once every chain task has settled

        async def drive(cid: str, node: Node) -> None:
            if sem is not None:
                async with sem:
                    async for ev in self._run_chain(
                        cid, node, chains[cid], tool_schemas,
                        max_turns, generation_config, context_config,
                    ):
                        await q.put(ev)
            else:
                async for ev in self._run_chain(
                    cid, node, chains[cid], tool_schemas,
                    max_turns, generation_config, context_config,
                ):
                    await q.put(ev)

        tasks = [
            asyncio.create_task(drive(cid, node))
            for cid, node in first_nodes.items()
        ]

        async def _close():
            # return_exceptions so one failing chain doesn't prevent the sentinel;
            # the exceptions are re-raised below after the queue is fully drained.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await q.put(done)
            return results

        closer = asyncio.create_task(_close())

        pbar = tqdm(total=len(tasks), file=sys.stdout) if show_progress else None
        try:
            while True:
                ev = await q.get()
                if ev is done:
                    break
                if pbar is not None and isinstance(ev, ChainEnded):
                    pbar.update(1)
                yield ev
        finally:
            if pbar is not None:
                pbar.close()

        results = await closer
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # Fail loud (as the previous gather-based path did): surface the first
            # chain error rather than silently dropping trajectories.
            raise errors[0]

        self.global_step += 1
        self.metrics.record_step(
            global_step=self.global_step,
            trajectories=self.get_trajectories(),
            chain_rollout_seconds=self.chain_rollout_seconds,
            current_nodes=self.current_nodes,
            chains=self.chains,
        )

    async def _run_chain(
        self,
        chain_id: str,
        first_node: Node,
        chain: Chain,
        tools: List[Dict],
        max_turns: int,
        generation_config: Dict[str, Any],
        context_config: Optional[ContextConfig] = None,
    ):
        """Run a single chain as an async generator, yielding :mod:`.events`.

        The per-chain analog of Claude Code's ``queryLoop``: one chain, one loop, events
        ``yield``ed inline. Supports parallel tool calls in a turn; stops immediately when
        a turn produces no tool call. Populates ``self.chains`` / ``self.current_nodes``
        on completion (the fan-in in :meth:`_run_chains` reads final state from there, not
        from the events).
        """
        chain_started_at = time.monotonic()

        # Build Context from rollout data for tools that need it
        context = Context(
            rollout_id=chain_id,
            group_id=chain.info.get("group_id"),
            metadata=chain.info,
            context_config=context_config,
        )

        yield ChainStarted(chain_id=chain_id, group_id=chain.info.get("group_id"))

        current_node = first_node
        depth = 0
        have_set_resources = False

        # Optional per-chain hook: env-source-of-truth agents (e.g. WebShop) may reset the
        # env with task_id and inject the env's instruction as the first user turn before
        # generation, so the instruction the agent reads matches the goal it is graded on.
        # The hook self-guards (no-op unless the task actually uses that env).
        prepare_first_node = getattr(self, "prepare_first_node", None)
        if prepare_first_node is not None:
            try:
                await prepare_first_node(context, current_node)
            except Exception as e:
                logger.warning(
                    f"prepare_first_node hook failed for chain {chain_id}: {e}"
                )

        while not current_node.is_terminal and depth < max_turns:
            turn_result: Optional[TurnResult] = None
            async for item in self._run_turn(
                chain_id=chain_id,
                chain=chain,
                current_node=current_node,
                depth=depth,
                tools=tools,
                generation_config=generation_config,
                context=context,
                have_set_resources=have_set_resources,
            ):
                if isinstance(item, TurnResult):
                    turn_result = item
                else:
                    yield item  # RolloutEvent — relay to consumer
            have_set_resources = turn_result.have_set_resources
            if not turn_result.should_continue:
                current_node = turn_result.next_node
                break
            current_node = turn_result.next_node
            depth += 1

        if "finish_reason" not in chain.info:
            chain.info["finish_reason"] = FinishReason.MAX_TURNS

        # Finalize chain
        await self._finalize_chain(chain_id, chain, current_node, depth, context)

        chain_elapsed = time.monotonic() - chain_started_at
        self.chain_rollout_seconds[chain_id] = chain_elapsed
        chain.info["rollout_time_sec"] = chain_elapsed

        # This chain owns its final state; _run_chains reads from here, not from events.
        self.chains[chain_id] = chain
        self.current_nodes[chain_id] = current_node

        self.finished_chains_count += 1
        self.metrics.record_chain(
            global_step=self.global_step,
            finished_chains_count=self.finished_chains_count,
            trajectory=chain.histories,
            info=chain.info,
        )

        yield ChainEnded(
            chain_id=chain_id, finish_reason=chain.info.get("finish_reason")
        )

    async def _run_turn(
        self,
        chain_id: str,
        chain: Chain,
        current_node: Node,
        depth: int,
        tools: List[Dict],
        generation_config: Dict[str, Any],
        context: Context,
        have_set_resources: bool,
    ):
        """
        Run one turn: generate response, add thought node, process tool calls.

        Async generator. Yields :mod:`.events` (``MessageProduced`` after generation,
        ``ToolObserved`` per tool result) and, as its **last** item, exactly one
        :class:`TurnResult` carrying ``(next_node, have_set_resources, should_continue)``
        — the control value the method used to return. :meth:`_run_chain` relays the
        events and consumes the ``TurnResult``. When ``should_continue`` is False the
        chain loop breaks and ``next_node`` is the final node for this chain.

        The turn reads as a sequence of steps: context-length pre-check → generate →
        emit message → terminal check → run tool calls (:meth:`_run_tools`) →
        decide outcome.
        """
        max_model_len = getattr(self, "max_model_len", None)

        # 1. Pre-check: stop before generating if already at/over the context limit.
        if (
            max_model_len is not None
            and current_node.total_token_length >= max_model_len
        ):
            current_node.is_terminal = True
            chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
            yield TurnResult(current_node, have_set_resources, False)
            return

        # 2. Generate the assistant message and record it as a thought node.
        # Agents may mutate current_node.messages (e.g. append a user nudge). Must run
        # before copying newest_messages so trajectories and tool folding see the same
        # history the LLM was given (generate_async reads current_node.messages).
        self._maybe_append_context_trigger_user_message(current_node)
        newest_messages = current_node.messages.copy()
        new_msg, total_token_length = await generate_response(
            self,
            chain=chain,
            current_node=current_node,
            tools=tools,
            depth=depth,
            chain_id=chain_id,
            generation_config=generation_config,
            context=context,
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
        yield MessageProduced(
            chain_id=chain_id,
            depth=depth,
            message=new_msg,
            total_token_length=total_token_length,
        )

        # 3. Terminal checks on the generated message (status / context length).
        thought_node.is_terminal = (
            new_msg.get("status", "continue") in self.terminal_status
        )
        if (
            max_model_len is not None
            and thought_node.total_token_length >= max_model_len
        ):
            thought_node.is_terminal = True
            chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
        elif thought_node.is_terminal:
            chain.info["finish_reason"] = FinishReason.TERMINAL
        if thought_node.is_terminal:
            yield TurnResult(thought_node, have_set_resources, False)
            return

        # 4. Execute tool calls. _run_tools yields ToolObserved per call and, as
        # its last item, a ToolLoopResult summarizing the loop.
        loop_result: Optional[ToolLoopResult] = None
        async for item in self._run_tools(
            chain_id=chain_id,
            chain=chain,
            depth=depth,
            tools=tools,
            context=context,
            thought_node=thought_node,
            newest_messages=newest_messages,
            have_set_resources=have_set_resources,
        ):
            if isinstance(item, ToolLoopResult):
                loop_result = item
            else:
                yield item  # ToolObserved — relay
        have_set_resources = loop_result.have_set_resources

        # 5. Decide the turn outcome.
        if loop_result.terminated:  # context length hit mid tool loop (finish_reason set)
            yield TurnResult(loop_result.last_node, have_set_resources, False)
            return
        if loop_result.num_tool_calls == 0:
            chain.info["finish_reason"] = FinishReason.NO_TOOL_CALLS
            yield TurnResult(thought_node, have_set_resources, False)
            return
        if loop_result.last_node is not None and loop_result.last_node.is_terminal:
            chain.info["finish_reason"] = FinishReason.TERMINAL
        yield TurnResult(loop_result.last_node, have_set_resources, True)

    async def _run_tools(
        self,
        *,
        chain_id: str,
        chain: Chain,
        depth: int,
        tools: List[Dict],
        context: Context,
        thought_node: Node,
        newest_messages: Messages,
        have_set_resources: bool,
    ):
        """Execute the assistant message's tool calls (supports multiple per turn).

        Async generator: yields a ``ToolObserved`` per executed call and, as its **last**
        item, exactly one :class:`ToolLoopResult`. Owns per-tool observation-node
        bookkeeping, token accounting, context folding, and the mid-loop ``max_model_len``
        termination.

        Invalid tool calls (hallucinated tool name / non-JSON arguments / unknown
        argument names) are governed by ``self.invalid_tool_call_as_observation``: when
        True (default) the call is passed to ``submit_tool_call``, which returns a
        ``status="error"`` observation with an informative hint (what's available /
        expected) — so it counts as a tool call and the chain continues with that hint
        fed back to the model. When False an ``InvalidToolCallError`` is raised
        (fail-fast). This is distinct from an exception raised inside a tool's body,
        which is governed by ``TOOL_ERROR_AS_OBSERVATION``.
        """
        max_model_len = getattr(self, "max_model_len", None)
        invalid_as_observation = getattr(
            self, "invalid_tool_call_as_observation", True
        )
        num_tool_calls = 0
        action_input_node: Optional[Node] = None
        running_total_token_length = thought_node.total_token_length

        for tool_call in thought_node.messages[-1].get("tool_calls") or []:
            if not self.validate_tool_call(tool_call):
                if not invalid_as_observation:
                    # Fail-fast: an invalid tool call aborts the rollout.
                    raise InvalidToolCallError(f"Invalid tool call: {tool_call!r}")
                # Otherwise fall through: submit_tool_call turns it into a status="error"
                # observation with an informative hint; it counts as a tool call so the
                # chain continues and the model can recover next turn.
                logger.debug(f"Invalid tool call as observation: {tool_call}")
            else:
                logger.debug(f"Valid tool call: {tool_call}")

            result = await self._execute_tool_call(
                context,
                tool_call,
                newest_messages,
                chain,
                chain_id,
                depth,
                have_set_resources,
            )
            num_tool_calls += 1
            have_set_resources = True

            action_input_node = chain.add_node(
                type="Action Input",
                messages=newest_messages.copy(),
                description=result.get("arguments", ""),
            )
            observation = result["observation"]
            yield ToolObserved(
                chain_id=chain_id,
                depth=depth,
                tool_name=tool_call["function"]["name"],
                observation=observation,
                status=result["status"],
                image=result.get("image"),
            )
            action_input_node.observation = observation
            action_input_node.observation_code = result["status"]
            obs_token_length = observation_token_length(self, observation)
            action_input_node.total_token_length = (
                running_total_token_length
                + obs_token_length
                + TOOL_OBS_PREFIX_TOKENS
            )
            running_total_token_length = action_input_node.total_token_length

            # Terminate if context length exceeded after this tool observation
            if (
                max_model_len is not None
                and action_input_node.total_token_length >= max_model_len
            ):
                action_input_node.is_terminal = True
                chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
                yield ToolLoopResult(
                    action_input_node, num_tool_calls, have_set_resources, terminated=True
                )
                return

            new_content = [{"type": "text", "text": observation}]
            if "image" in result:
                image_base64 = image_to_data_uri(result["image"])
                new_content.append({"type": "image", "image": image_base64})

            # Apply context folding based on the tool result (currently only summarize).
            updated_messages, did_fold_context = self.apply_context_folding(
                chain=chain,
                messages=newest_messages,
                tool_name=tool_call["function"]["name"],
                observation=observation,
                tool_call_id=tool_call["id"],
                tool_result_name=result["name"],
                new_content=new_content,
                context=context,
            )

            action_input_node.messages = updated_messages
            newest_messages = updated_messages.copy()
            if did_fold_context:
                # Folded history is shorter; re-measure prompt tokens for the new
                # messages (same path as first generation) instead of keeping the
                # pre-fold cumulative total.
                try:
                    folded_prompt_tokens = estimate_chat_prompt_tokens(
                        self, action_input_node, tools=tools
                    )
                except ValueError:
                    folded_prompt_tokens = 0
                action_input_node.total_token_length = folded_prompt_tokens
                running_total_token_length = folded_prompt_tokens
            action_input_node.is_terminal = result["status"] in self.terminal_status

        yield ToolLoopResult(
            action_input_node, num_tool_calls, have_set_resources, terminated=False
        )

    def apply_context_folding(
        self,
        chain: Chain,
        messages: Messages,
        tool_name: str,
        observation: Any,
        tool_call_id: str,
        tool_result_name: str,
        new_content: List[Dict[str, Any]],
        context: Context,
    ) -> Tuple[Messages, bool]:
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
            (updated Messages, did_fold) where ``did_fold`` is True when history was
            replaced by a summarize fold (caller should refresh cumulative token counts).
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
            # Default behavior: keep only the fresh summary (do not stack summaries).
            folded_turns = fold_messages_with_summarize(
                messages.messages, observation, keep_previous_summary=False
            )
            context.trajectory_format = "segmented"
            meta = messages.meta
            return Messages.from_turns(folded_turns, **meta), True

        # Default behavior for non-folding tools: just append the tool message.
        turns = messages.messages + [tool_turn]
        meta = messages.meta
        return Messages.from_turns(turns, **meta), False

    def _maybe_append_context_trigger_user_message(self, current_node: Node) -> None:
        """Hook for subclasses to append a user turn before the next LLM call. No-op by default."""
        return

    def _skills_payload(self) -> Optional[List[Dict[str, str]]]:
        """Skill list as JSON-serializable dicts for ``chat_template_kwargs``.

        This is the *template* path: chat-bricks renders the skills slot from
        these dicts. It is mutually exclusive with the agent-layer ``{skills}``
        substitution in ``_preprocess_messages`` — when the system prompt carries
        a ``{skills}`` slot the agent already renders the block into the system
        message, so we return None here to avoid injecting skills twice.

        Returns None when no skills are configured (so the backend can skip the
        ``extra_body`` field entirely) or when the agent renders skills itself.
        """
        skills = getattr(self, "skills", None)
        if not skills:
            return None
        system_prompt = getattr(self, "system_prompt", None)
        if system_prompt and "{skills}" in system_prompt:
            return None
        return [{"name": s.name, "description": s.description} for s in skills]

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
    ):
        """Execute a tool call."""
        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]

        # Set up tools if needed (reset resources that may have been acquired at chain start)
        if not have_set_resources:
            env_args = {
                k: context.metadata[k]
                for k in ("task_name", "variation_idx")
                if k in context.metadata
            }
            # We have moved reset to the tool call
            # if env_args:
            #     await context.reset_resource(scope="rollout", env_args=env_args)
            #     await context.reset_resource(scope="global", env_args=env_args)
            # else:
            #     await context.reset_resource(scope="rollout")
            #     await context.reset_resource(scope="global")
            # have_set_resources = True

        # Execute tool call
        result = await submit_tool_call(
            tool_name,
            tool_input,
            context=context,
            allowed_tool_names=self.tool_names,
            invalid_call_as_observation=getattr(
                self, "invalid_tool_call_as_observation", True
            ),
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

            if context.trajectory_format == "flat":
                context.trajectory = chain.histories[0]
            else:
                context.trajectory = chain.histories
            
            context.final_response = final_response
            
            other_args = {
                k: v
                for k, v in chain.info.items()
                if k not in ["final_response", "trajectory", "id"]
            }

            reward = await calculate_reward(
                self._reward_fn,
                final_response=final_response,
                **other_args,
                trajectory=context.trajectory,
                id=chain_id,
                context=context,
            )
            chain.info["reward"] = reward
        else:
            chain.info["reward"] = None


        # Release global resources so other rollouts can use them
        await context.release_resource(scope="global")

        # Kill rollout-scoped resources
        await context.end_resource(scope="rollout")
