"""Rollout strategy base class.

A ``Rollout`` is *how* an agent completes a task (chain / step / …). It is a
per-``run`` strategy object: the agent (the :class:`~agentfly.agents.rollout.agent.RolloutAgent`)
passes itself as ``agent`` and the rollout drives the loop, reaching agent-specific
behavior through ``agent`` and reusing the shared machinery defined here.

Layering:
- **L1** agent hooks (``parse``/``validate_tool_call``/``execute_tool_call``/…) live on
  the agent and are reached via ``agent``.
- **L2** shared machinery (this class, concrete), used by every strategy:
  - loop-control policy: ``resolve_tool_control``, ``tool_finish_reason``, …;
  - one episode's lifecycle: ``make_context`` → ``run_first_step_hook`` → (the
    strategy's turn loop) → ``compute_reward`` → ``release_resources`` →
    ``make_trajectory``;
  - running many episodes: ``validate_run_args``, ``make_specs``, ``run_episodes``.

An *episode* is one attempt at one task; the attempts at one task form a *group*
(``group_id``).
- **L3** the thin strategy (a subclass): ``run`` and its turn loop, returning a
  ``RunResult``. Training conversion and tokenization live outside the rollout
  interface; inference-only rollouts need no training methods.
"""

import asyncio
import inspect
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from ...core import Context
from ...rewards.reward_base import calculate_reward
from ...rewards.types import RewardResult
from ...tools.tool_base import InvalidToolCallError
from ...tools.types import CONTROL, CONTROL_CONTINUE, CONTROL_END, ToolResult
from ..types import RunResult, Segment, Trajectory
from ..utils.action_format import message_text
from ..utils.messages import Messages, MessagesList
from .events import FinishReason
from .loop.progress import progress_bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode to roll out: a task's messages and the episode's ids.

    ``episode_idx`` is the episode's index among the ``num_chains`` attempts at its
    task; ``group_idx`` is the task's index in the batch. Episodes of one task share
    ``group_id``.
    """

    episode_id: str
    group_id: str
    group_idx: int
    episode_idx: int
    messages: Messages

    @property
    def info(self) -> Dict[str, Any]:
        """A fresh dict: the task's metadata plus this episode's ids.

        The ids use the public ``Trajectory`` / reward-function names (``chain_id`` /
        ``chain_idx`` for ``episode_id`` / ``episode_idx``).
        """
        return self.messages.meta | {
            "chain_id": self.episode_id,
            "group_id": self.group_id,
            "group_idx": self.group_idx,
            "chain_idx": self.episode_idx,
        }


def unwrap_reward(raw_reward: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Split a stored reward into ``(reward value, reward metrics)``.

    ``raw_reward`` is what :func:`calculate_reward` returned (a ``RewardResult``, or
    ``None`` when no reward function is configured). A plain number or a dict with a
    ``reward`` key is also accepted (rewards stored without ``calculate_reward``).
    """
    if isinstance(raw_reward, RewardResult):
        return raw_reward.reward, dict(raw_reward.metrics)
    if isinstance(raw_reward, dict):
        return raw_reward.get("reward"), {k: v for k, v in raw_reward.items() if k != "reward"}
    if isinstance(raw_reward, (int, float)):
        return float(raw_reward), {}
    return None, {}

class Rollout(ABC):
    """Base rollout strategy: shared machinery (L2) + the abstract L3 interface.

    Concrete strategies (:class:`~agentfly.agents.rollout.strategies.chain_rollout.ChainRollout`, …) subclass
    this, implement :meth:`run`, and reuse the concrete methods
    here. Agent-provided state/hooks are reached through ``agent`` (a
    :class:`~agentfly.agents.rollout.agent.RolloutAgent`).
    """

    # Runtime-owned keys, excluded from task metadata. Execution counts are
    # derived from records, never trusted from incoming task metadata.
    RESERVED_INFO_KEYS = frozenset(
        {
            "chain_id",
            "group_id",
            "chain_idx",
            "group_idx",
            "finish_reason",
            "rollout_time_sec",
            "num_turns",
            "tool_call_counts",
            "reward",
            "tool_metrics",
        }
    )

    # Parsed-message ``status`` values that mean the MODEL explicitly ended the episode
    # (an "end task" sentinel, or a parser-supplied status). Distinct from the
    # continue/end/raise control vocabulary; the loop maps a match to ``end``.
    TERMINAL_MESSAGE_STATUS = frozenset({"terminal", "finish"})

    # Whether generated assistant messages carry the sampled ``token_ids``; set per run
    # from ``run(use_token_ids=...)``.
    use_token_ids: bool = True

    # ---- L2: loop-control policy (the single authority for continue/end/raise) ----

    def __init__(
        self,
        *,
        on_no_tool_call: str = CONTROL_END,
        on_invalid_tool_call: str = CONTROL_END,
        on_tool_error: str = CONTROL_CONTINUE,
    ):
        """Store the run's loop-control policy.

        Each knob maps one *condition where the model didn't produce a clean, executed
        action* to a control (``continue`` / ``end`` / ``raise``):

        - ``on_no_tool_call`` — the turn produced no tool call at all (default ``end``).
        - ``on_invalid_tool_call`` — a malformed call: unknown name / bad args (default ``end``).
        - ``on_tool_error`` — an exception raised inside a tool body (default ``continue``:
          the error text is fed back as an observation so the model can recover).

        A tool that ran cleanly continues; a tool may override any of this by writing an
        explicit ``control`` on its result. Hard limits (context length, max turns) always
        end and are not governed here.
        """
        for label, val in (
            ("on_no_tool_call", on_no_tool_call),
            ("on_invalid_tool_call", on_invalid_tool_call),
            ("on_tool_error", on_tool_error),
        ):
            if val not in CONTROL:
                raise ValueError(f"{label} must be one of {sorted(CONTROL)}, got {val!r}")
        self.on_no_tool_call = on_no_tool_call
        self.on_invalid_tool_call = on_invalid_tool_call
        self.on_tool_error = on_tool_error
        # Metrics x-axis; see resolve_global_step. A strategy instance is constructed per
        # run, so this counter only ever carries one run's step unless a caller supplies
        # its own — which is why callers that have a step counter should pass it.
        self.global_step = 0

    def resolve_global_step(self, global_step: Optional[int] = None) -> int:
        """Fix this run's metrics x value, *before* any metric is recorded.

        A caller that owns a step counter (the trainer's ``global_steps``) passes it and
        it is used verbatim, so the agent's curves share the trainer's axis, validation
        does not advance the training axis, and a resumed run continues from its
        checkpoint's step instead of restarting at 1. ``None`` (standalone use: eval
        scripts, notebooks) advances this instance's own counter instead.

        Resolving up front — rather than incrementing once the rollout has drained —
        also puts the per-chain records and the end-of-run summaries on the same x.
        """
        if global_step is None:
            self.global_step += 1
        else:
            self.global_step = int(global_step)
        return self.global_step

    def resolve_tool_control(self, result) -> str:
        """The loop control for a turn that produced a tool ``result``.

        Precedence: the tool's explicit ``control`` wins (the tool opted to steer); else
        the ``status`` *diagnosis* routes to the run policy — ``"invalid"`` →
        ``on_invalid_tool_call``, ``"error"`` → ``on_tool_error``; else a clean result
        continues.
        """
        if result.control is not None:
            return result.control
        if result.status == "invalid":
            return self.on_invalid_tool_call
        if result.status == "error":
            return self.on_tool_error
        return CONTROL_CONTINUE

    @staticmethod
    def tool_finish_reason(result: ToolResult) -> FinishReason:
        """Describe a tool stop after its control has already resolved to ``end``.

        Explicit tool control takes diagnostic precedence over status, just as
        it does in ``resolve_tool_control``. This helper does not decide whether
        the loop should stop; callers must resolve and handle control first.
        """
        if result.control == CONTROL_END:
            return FinishReason.TOOL_CONTROL_END
        if result.status == "invalid":
            return FinishReason.INVALID_END
        if result.status == "error":
            return FinishReason.TOOL_ERROR_END
        return FinishReason.TOOL_END

    # ---- shared helpers (used by StepRollout and ChainRollout) ------------------------

    @classmethod
    def projected_tool_call(cls, agent, msg: Dict) -> Optional[Dict]:
        """verl-agent's always-step projection: a turn with no extractable ``<action>``
        still steps the env, with the last 30 characters of the (lowercased) raw output as
        a garbage fallback action (the env answers "nothing happens"; the episode goes on).
        The turn stays format-invalid via ``action_format_valid``. Returns None if the
        agent has no action tool to project onto.
        """
        import json

        tool_name = getattr(agent, "action_tool_name", None) or (
            agent.tools[0].name if getattr(agent, "tools", None) else None
        )
        if tool_name is None:
            return None
        fallback = message_text(msg).lower()[-30:]
        return {
            "id": "call_projected",
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps({"action": fallback})},
        }

    def message_ends_episode(self, message: Dict) -> bool:
        """True when the MODEL explicitly ended via its parsed message ``status`` (a
        sentinel / parser status) — checked only for a turn that *did* carry a tool call
        (a tool-less turn is governed by ``on_no_tool_call`` instead)."""
        return message.get("status", "continue") in self.TERMINAL_MESSAGE_STATUS

    @staticmethod
    def raise_for_result(result) -> None:
        """Fail-fast for a ``raise`` control: an invalid call raises
        :class:`InvalidToolCallError`; any other error raises ``RuntimeError``. The
        result's observation carries the informative message."""
        if result.status == "invalid":
            raise InvalidToolCallError(result.observation)
        raise RuntimeError(f"Tool {result.name!r} error: {result.observation}")

    # ---- L2: running many episodes ----

    @staticmethod
    def validate_run_args(max_turns: int, num_chains: int, max_concurrent_chains: Optional[int]) -> None:
        assert max_turns >= 1, "max_turns must be at least 1."
        assert num_chains >= 1, "num_chains must be at least 1."
        if max_concurrent_chains is not None:
            assert max_concurrent_chains >= 1, "max_concurrent_chains must be at least 1 when set."

    @staticmethod
    def make_specs(messages: Any, num_chains: int) -> List[EpisodeSpec]:
        """One :class:`EpisodeSpec` per (task, attempt), in ``(group_idx, episode_idx)``
        order. Attempts at one task share a ``group_id``; every episode gets its own
        ``episode_id``."""
        specs: List[EpisodeSpec] = []
        for group_idx, task_messages in enumerate(MessagesList.from_data(messages)):
            group_id = str(uuid.uuid4())
            for episode_idx in range(num_chains):
                specs.append(
                    EpisodeSpec(
                        episode_id=str(uuid.uuid4()),
                        group_id=group_id,
                        group_idx=group_idx,
                        episode_idx=episode_idx,
                        messages=task_messages,
                    )
                )
        return specs

    @staticmethod
    async def run_episodes(
        specs: List[EpisodeSpec],
        run_episode: Callable[[EpisodeSpec], Union[AsyncIterator[Any], Any]],
        *,
        max_concurrent_chains: Optional[int] = None,
        show_progress: bool = True,
    ) -> AsyncIterator[Any]:
        """Run ``run_episode(spec)`` for every spec concurrently and yield their events.

        ``run_episode(spec)`` is either an async generator, whose items are yielded here
        as one merged stream, or a coroutine (no events). Results are kept by the
        strategy, not returned here. At most ``max_concurrent_chains`` run at once (no
        limit when ``None``). The progress bar counts finished episodes. Every episode
        runs to the end even if another fails; the first error (in spec order) is raised
        once the stream is drained.
        """
        queue: asyncio.Queue = asyncio.Queue()
        sem = asyncio.Semaphore(int(max_concurrent_chains)) if max_concurrent_chains is not None else None
        finished = object()  # one per episode, for the progress bar
        done = object()  # once every episode has settled

        async def run_spec(spec: EpisodeSpec) -> None:
            result = run_episode(spec)
            if inspect.isasyncgen(result):
                async for event in result:
                    await queue.put(event)
            else:
                await result

        async def drive(spec: EpisodeSpec) -> None:
            try:
                if sem is None:
                    await run_spec(spec)
                else:
                    async with sem:
                        await run_spec(spec)
            finally:
                await queue.put(finished)

        tasks = [asyncio.create_task(drive(spec)) for spec in specs]

        async def close() -> List[Any]:
            # return_exceptions so one failing episode doesn't prevent the sentinel;
            # the exceptions are re-raised below after the queue is fully drained.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(done)
            return results

        closer = asyncio.create_task(close())
        pbar = progress_bar(total=len(tasks)) if show_progress else None
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if item is finished:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                yield item
        finally:
            if pbar is not None:
                pbar.close()

        results = await closer
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # Fail loud: surface the first episode error rather than silently
            # dropping episodes.
            raise errors[0]

    # ---- L2: one episode's lifecycle ----

    @staticmethod
    def make_context(
        *,
        episode_id: str,
        group_id: Optional[str],
        metadata: Dict[str, Any],
        context_config: Optional[Any] = None,
    ) -> Context:
        """The per-episode :class:`Context` (resources and metadata for tools)."""
        return Context(
            rollout_id=episode_id,
            group_id=group_id,
            metadata=metadata,
            context_config=context_config,
        )

    @staticmethod
    async def run_first_step_hook(agent, context: Context, first_step, episode_id: str) -> None:
        """Run the agent's optional ``prepare_first_step(context, first_step)`` hook.

        Env-source-of-truth agents (e.g. WebShop) use it to reset the env and inject the
        env's instruction as the first user turn, so the instruction the agent reads
        matches the goal it is graded on. A failing hook is logged, not raised.
        """
        hook = getattr(agent, "prepare_first_step", None)
        if hook is None:
            return
        try:
            await hook(context, first_step)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"prepare_first_step hook failed for episode {episode_id}: {e}")

    @staticmethod
    async def compute_reward(
        agent,
        context: Context,
        *,
        episode_id: str,
        trajectory: Any,
        final_messages: List[Dict],
        reward_args: Dict[str, Any],
    ) -> Optional[RewardResult]:
        """The trajectory's outcome reward, or ``None`` when the agent has no reward function.

        ``trajectory`` is what the reward function receives (and ``context.trajectory``
        is set to); ``final_messages`` is the flat message list the final response is
        read from. ``reward_args`` (task metadata) are passed as extra keyword arguments,
        except the reserved ``final_response`` / ``trajectory`` / ``id``.
        """
        if agent.reward_fn is None:
            return None
        final_response = agent.extract_final_response(final_messages)
        context.trajectory = trajectory
        context.final_response = final_response
        extra = {k: v for k, v in reward_args.items() if k not in ("final_response", "trajectory", "id")}
        return await calculate_reward(
            agent.reward_fn,
            final_response=final_response,
            **extra,
            trajectory=trajectory,
            id=episode_id,
            context=context,
        )

    @staticmethod
    async def release_resources(context: Context) -> None:
        """Release global resources so other rollouts can use them, then end the
        rollout-scoped ones."""
        await context.release_resource(scope="global")
        await context.end_resource(scope="rollout")

    def make_trajectory(
        self,
        *,
        info: Dict[str, Any],
        segments: List[Segment],
        reward: Any,
        tool_metrics: Optional[Dict[str, Any]],
        finish_reason: Optional[str],
        rollout_time_sec: Optional[float],
        num_turns: int,
        tool_call_counts: Dict[str, int],
        steps: List[Any],
    ) -> Trajectory:
        """Build the :class:`Trajectory` for one finished episode.

        - ``reward`` (as stored by :meth:`compute_reward`) → ``Trajectory.reward`` and
          ``Trajectory.metrics``; ``tool_metrics`` are merged into the metrics after it.
        - ``info`` gives the ids (``chain_id``, ``group_id``, ``chain_idx``,
          ``group_idx``); its other non-reserved keys (dataset-row fields like
          ``answer``, ``task_id``) become ``Trajectory.metadata``.
        """
        reward_value, metrics = unwrap_reward(reward)
        metrics.update(tool_metrics or {})
        return Trajectory(
            segments=segments,
            reward=reward_value,
            metrics=metrics,
            finish_reason=finish_reason,
            rollout_time_sec=rollout_time_sec,
            num_turns=num_turns,
            tool_call_counts=tool_call_counts,
            chain_id=info.get("chain_id"),
            group_id=info.get("group_id"),
            chain_idx=info.get("chain_idx"),
            group_idx=info.get("group_idx"),
            metadata={k: v for k, v in info.items() if k not in self.RESERVED_INFO_KEYS},
            steps=steps,
        )

    # ---- L3: the thin, strategy-specific interface ----

    @abstractmethod
    async def run(
        self,
        agent,
        messages: List[Dict],
        max_turns: int,
        generation_config: Optional[Dict[str, Any]] = None,
        context_config: Optional[Any] = None,
        global_step: Optional[int] = None,
        use_token_ids: bool = True,
        **kwargs,
    ) -> RunResult:
        """Drive the rollout for a batch of tasks and return a :class:`RunResult`.

        ``global_step`` is the caller's step counter for the metrics x-axis; see
        :meth:`resolve_global_step`. ``use_token_ids``: carry the backend's sampled token
        ids onto generated assistant messages (see :func:`generate_response`); a strategy
        stores it as ``self.use_token_ids`` for the run.
        """
        ...
