import asyncio
import logging
import os
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ....core import Context, ContextConfig
from ....tools.types import CONTROL_CONTINUE, CONTROL_END, CONTROL_RAISE
from ....utils.monitor import Monitor
from ...types import RunResult
from ...utils.messages import MessagesList
from ..base import (
    NO_TOOL_CALL_MESSAGE_MINISWE,
    Rollout,
    ToolLoopResult,
    TOKEN_SAFETY_MARGIN,
    TOOL_OBS_PREFIX_TOKENS,
)
from ..events import (
    ChainEnded,
    ChainStarted,
    FinishReason,
    MessageProduced,
)
from ..loop.generation import generate_response, prepare_generation_config
from ..loop.metrics import RolloutMetrics
from ..structures import Chain, Step
from ..loop.tokens import estimate_chat_prompt_tokens, observation_token_length


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """Internal control value yielded by :meth:`ChainRollout._run_turn` as its last
    item — carries the loop-control outcome that the old method returned. Not a
    ``RolloutEvent``; :meth:`_run_chain` consumes it and does not relay it."""

    next_step: Step
    have_set_resources: bool
    should_continue: bool


class ChainRollout(Rollout):
    """
    Chain-based rollout strategy (L3). It starts multiple chains and runs them
    asynchronously, reusing the shared machinery on :class:`Rollout` (L2) and reaching
    agent-specific behavior through ``agent`` (a
    :class:`~agentfly.agents.rollout.agent.RolloutAgent`). The transient run-state
    (``self.chains`` / ``self.current_steps`` / ``self.agent``) is rebuilt each
    :meth:`run` call, so a strategy instance is reusable across runs.
    """

    def __init__(
        self,
        *,
        on_no_tool_call: str = CONTROL_END,
        on_invalid_tool_call: str = CONTROL_END,
        on_tool_error: str = CONTROL_CONTINUE,
        max_consecutive_no_tool_calls: int = 3,
        no_tool_call_message: Optional[str] = NO_TOOL_CALL_MESSAGE_MINISWE,
        project_actions: bool = False,
    ):
        super().__init__(
            on_no_tool_call=on_no_tool_call,
            on_invalid_tool_call=on_invalid_tool_call,
            on_tool_error=on_tool_error,
            max_consecutive_no_tool_calls=max_consecutive_no_tool_calls,
            no_tool_call_message=no_tool_call_message,
        )
        # verl-agent-faithful ALWAYS-STEP rollout (same semantics as StepRollout): a turn
        # with no extractable ``<action>`` steps the env with the last-30-chars garbage
        # fallback (format-invalid, penalized) instead of ending, and the episode ends
        # ONLY on env done or max_turns. Keeps the two rollouts' termination identical
        # so step-vs-chain comparisons don't confound rollout type with early exits.
        self.project_actions = project_actions
        self.reset()
        self.chains: Dict[str, Chain] = {}
        self.current_steps: Dict[str, Step] = {}
        self.metrics = RolloutMetrics()
        self.agent = None

    def reset(self) -> None:
        self.chains = {}
        self.current_steps = {}

    def to_json(self) -> dict:
        chains_list = list(self.chains.values()) if isinstance(self.chains, dict) else self.chains
        return {
            "finish": [c.info.get("status_code") == "success" for c in chains_list],
            "chains": [c.to_json() for c in chains_list],
        }

    def initialize_chains(
        self, messages_list: MessagesList, num_chains: int
    ) -> Tuple[Dict[str, Chain], Dict[str, Step]]:
        chains = {}
        start_steps = {}
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
                root = ch.add_step(
                    type="Action Input", messages=deepcopy(messages.messages)
                )

                chains[cid] = ch
                start_steps[cid] = root

        return chains, start_steps

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

    async def run(
        self,
        agent,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        max_concurrent_chains: Optional[int] = None,
        context_config: Optional[ContextConfig] = None,
        global_step: Optional[int] = None,
        **kwargs,
    ) -> RunResult:
        """Drive the chain-based rollout and return its :class:`RunResult`.

        ``agent`` (the agent) is stashed for the run so the internal loop can reach
        agent state/hooks; the transient run-state is rebuilt each call.
        ``global_step`` sets the metrics x-axis (see ``Rollout.resolve_global_step``).
        """
        self.agent = agent
        # Drain the rollout event stream. Training ignores the events themselves; the
        # final state (``self.chains`` / ``self.current_steps``) is populated by each
        # chain as it ends. We build the RunResult from that state afterwards.
        async for _ev in self._run_chains(
            messages,
            max_turns=max_turns,
            num_chains=num_chains,
            generation_config=generation_config,
            max_concurrent_chains=max_concurrent_chains,
            context_config=context_config,
            global_step=global_step,
        ):
            pass
        return RunResult(trajectories=self.build_trajectories(agent), rollout="chain")

    async def _run_chains(
        self,
        messages: List[Dict],
        max_turns: int,
        num_chains: int,
        generation_config: Optional[Dict[str, Any]] = None,
        max_concurrent_chains: Optional[int] = None,
        context_config: Optional[ContextConfig] = None,
        global_step: Optional[int] = None,
        show_progress: bool = True,
    ):
        """Run all chains concurrently and yield their merged :mod:`.events` stream.

        This is the single shared rollout path: the per-chain generator :meth:`_run_chain` is the loop,
        and this method fans the per-chain event streams into one ``chain_id``-tagged
        stream via a queue. Each chain populates ``self.chains`` / ``self.current_steps``
        as it ends; this method just relays events and, when drained, emits step metrics.
        The metrics x is resolved before any chain starts so per-chain records and the
        end-of-run summaries share one step.

        Args:
            show_progress: render a ``tqdm`` bar over chain completion. Disable when a
                consumer owns the terminal (e.g. the Textual console).
        """
        self.validate_run_args(max_turns, num_chains, max_concurrent_chains)
        Monitor.ensure_started()
        self.reset()
        step = self.resolve_global_step(global_step)

        messages_list = MessagesList.from_data(messages)
        chains, first_steps = self.initialize_chains(messages_list, num_chains)
        # Prompt tools come from the agent's single source of truth (also used by
        # tokenize_trajectories), so generation and training render one prompt.
        tool_schemas = self.agent.prompt_tools()

        q: asyncio.Queue = asyncio.Queue()
        sem: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(int(max_concurrent_chains))
            if max_concurrent_chains is not None
            else None
        )
        done = object()  # sentinel pushed once every chain task has settled

        async def drive(cid: str, step: Step) -> None:
            if sem is not None:
                async with sem:
                    async for ev in self._run_chain(
                        cid, step, chains[cid], tool_schemas,
                        max_turns, generation_config, context_config,
                    ):
                        await q.put(ev)
            else:
                async for ev in self._run_chain(
                    cid, step, chains[cid], tool_schemas,
                    max_turns, generation_config, context_config,
                ):
                    await q.put(ev)

        tasks = [
            asyncio.create_task(drive(cid, step))
            for cid, step in first_steps.items()
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

        self.metrics.record_step(
            global_step=step,
            trajectories=self.build_trajectories(self.agent),
        )

    async def _run_chain(
        self,
        chain_id: str,
        first_step: Step,
        chain: Chain,
        tools: List[Dict],
        max_turns: int,
        generation_config: Dict[str, Any],
        context_config: Optional[ContextConfig] = None,
    ):
        """Run a single chain as an async generator, yielding :mod:`.events`.

        The per-chain analog of Claude Code's ``queryLoop``: one chain, one loop, events
        ``yield``ed inline. Supports parallel tool calls in a turn; stops immediately when
        a turn produces no tool call. Populates ``self.chains`` / ``self.current_steps``
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

        current_step = first_step
        depth = 0
        have_set_resources = False

        # Optional per-chain hook: env-source-of-truth agents (e.g. WebShop) may reset the
        # env with task_id and inject the env's instruction as the first user turn before
        # generation, so the instruction the agent reads matches the goal it is graded on.
        # The hook self-guards (no-op unless the task actually uses that env).
        prepare_first_step = getattr(self.agent, "prepare_first_step", None)
        if prepare_first_step is not None:
            try:
                await prepare_first_step(context, current_step)
            except Exception as e:
                logger.warning(
                    f"prepare_first_step hook failed for chain {chain_id}: {e}"
                )

        while not current_step.is_terminal and depth < max_turns:
            turn_result: Optional[TurnResult] = None
            async for item in self._run_turn(
                chain_id=chain_id,
                chain=chain,
                current_step=current_step,
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
                current_step = turn_result.next_step
                break
            current_step = turn_result.next_step
            depth += 1

        if "finish_reason" not in chain.info:
            chain.info["finish_reason"] = FinishReason.MAX_TURNS

        # Finalize chain (reward + cleanup) — shared L2.
        await self.finalize_chain(self.agent, chain_id, chain, current_step, depth, context)

        chain_elapsed = time.monotonic() - chain_started_at
        chain.info["rollout_time_sec"] = chain_elapsed

        # This chain owns its final state; _run_chains reads from here, not from events.
        self.chains[chain_id] = chain
        self.current_steps[chain_id] = current_step

        self.metrics.record_chain(
            global_step=self.global_step,
            trajectory=chain.histories,
            info=chain.info,
        )

        yield ChainEnded(
            chain_id=chain_id, finish_reason=chain.info.get("finish_reason")
        )

    def _generation_filled_budget(
        self, new_msg: Dict[str, Any], current_step: Step, tools, generation_config
    ) -> bool:
        """True when the sampled response used its whole per-turn token budget.

        The training backend reports no per-turn stop reason, so a cut-off turn is
        inferred from the sampled ``token_ids`` reaching the effective ``max_tokens``
        that :func:`prepare_generation_config` resolved for this step. Unknown when
        either side is missing (then ``False``).
        """
        ids = new_msg.get("token_ids") if isinstance(new_msg, dict) else None
        if not ids:
            return False
        try:
            budget = prepare_generation_config(
                self.agent, generation_config, current_step, tools=tools
            ).get("max_tokens")
            budget = int(budget) if budget is not None else None
        except Exception:  # noqa: BLE001 — diagnostics must never break the loop
            budget = None
        return budget is not None and len(ids) >= budget

    async def _run_turn(
        self,
        chain_id: str,
        chain: Chain,
        current_step: Step,
        depth: int,
        tools: List[Dict],
        generation_config: Dict[str, Any],
        context: Context,
        have_set_resources: bool,
    ):
        """
        Run one turn: generate response, add thought step, process tool calls.

        Async generator. Yields :mod:`.events` (``MessageProduced`` after generation,
        ``ToolObserved`` per tool result) and, as its **last** item, exactly one
        :class:`TurnResult` carrying ``(next_step, have_set_resources, should_continue)``
        — the control value the method used to return. :meth:`_run_chain` relays the
        events and consumes the ``TurnResult``. When ``should_continue`` is False the
        chain loop breaks and ``next_step`` is the final step for this chain.

        The turn reads as a sequence of steps: context-length pre-check → generate →
        project actions → record/emit message → terminal check → run tool calls
        (:meth:`Rollout.execute_turn_tools`) → decide outcome.
        """
        max_model_len = getattr(self.agent, "max_model_len", None)

        # --- SKILLRL_TOKEN_DEBUG: per-turn drift probe. The pre-check below trusts
        # the INCREMENTAL running estimate (running_total + obs_raw + TOOL_OBS_PREFIX
        # per turn). This probe re-tokenizes the real prompt EXACTLY (uncapped) and
        # logs incremental-vs-exact so we can catch the turn where the incremental
        # estimate under-counts what vLLM will actually tokenize (the 40968>40960
        # crash). Localized to the danger zone so the O(n) recompute cost is bounded;
        # off unless the env var is set. ---
        # Gate on DEPTH (not on the incremental estimate we're auditing) and use
        # print(flush) — agentfly's rollout logger is not captured in the job
        # stdout, but verl's step prints are, so print is what actually surfaces.
        if os.environ.get("SKILLRL_TOKEN_DEBUG") and max_model_len is not None:
            _min_depth = int(os.environ.get("SKILLRL_TOKEN_DEBUG_MINDEPTH", "6"))
            if depth >= _min_depth:
                _incr = current_step.total_token_length
                try:
                    _exact = estimate_chat_prompt_tokens(
                        self.agent, current_step, tools=tools, cap=False)
                except Exception:  # noqa: BLE001
                    _exact = -1
                _margin = int(os.environ.get("SKILLRL_TOKEN_DEBUG_MARGIN", "6000"))
                # Only surface turns that matter: near/over the ceiling by EITHER
                # measure, or a large incremental-vs-exact divergence.
                if (_exact >= max_model_len - _margin
                        or _incr >= max_model_len - _margin
                        or (_exact >= 0 and abs(_exact - _incr) > 100)):
                    print(
                        f"[TOKDBG] chain={chain_id} depth={depth} incr={_incr} "
                        f"exact={_exact} drift={(_exact - _incr) if _exact >= 0 else 'NA'} "
                        f"max={max_model_len} guard_fires={_incr >= max_model_len} "
                        f"exact_over={(_exact > max_model_len) if _exact >= 0 else 'NA'} "
                        f"slips_through={_exact >= 0 and _exact > max_model_len and _incr < max_model_len}",
                        flush=True,
                    )

        # 1. Pre-check: stop before generating if already at/over the context limit.
        if (
            max_model_len is not None
            and current_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN
        ):
            current_step.is_terminal = True
            chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
            yield TurnResult(current_step, have_set_resources, False)
            return

        # 2. Generate the assistant message and record it as a thought step.
        # Agents may mutate current_step.messages (e.g. append a user nudge). Must run
        # before copying newest_messages so trajectories and tool folding see the same
        # history the LLM was given (generate_async reads current_step.messages).
        self.agent.maybe_append_context_trigger_user_message(current_step)
        newest_messages = current_step.messages.copy()
        try:
            new_msg, total_token_length = await generate_response(
                self.agent,
                chain=chain,
                current_step=current_step,
                tools=tools,
                depth=depth,
                chain_id=chain_id,
                generation_config=generation_config,
                context=context,
            )
        except Exception as e:  # noqa: BLE001
            # Backstop for an over-length prompt reaching vLLM: it RAISES
            # (ValueError "Prompt length ... exceeds the model's maximum context
            # length" / VLLMValidationError "max_tokens must be at least 1, got
            # 0"), and unhandled that kills the entire run. The TOKEN_SAFETY_MARGIN
            # guard should prevent ever getting here, but if a residual drift slips
            # through, abort THIS chain gracefully instead of propagating.
            _m = str(e)
            if ("maximum context length" in _m
                    or "Prompt length" in _m
                    or "max_tokens must be at least 1" in _m):
                print(
                    f"[TOKDBG] OVER-LENGTH at generation (chain={chain_id} "
                    f"depth={depth}) — aborting chain, not raising. incr="
                    f"{current_step.total_token_length} max={max_model_len} :: {_m[:180]}",
                    flush=True,
                )
                current_step.is_terminal = True
                chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
                yield TurnResult(current_step, have_set_resources, False)
                return
            raise

        # Resolve projection before append()/copy() snapshot the message:
        # execute_turn_tools reads tool_calls from the recorded thought step.
        tool_calls = new_msg.get("tool_calls") or []
        if not tool_calls and self.project_actions:
            projected = self._projected_tool_call(self.agent, new_msg)
            if projected is not None:
                tool_calls = [projected]
                new_msg["tool_calls"] = tool_calls

        newest_messages.append(new_msg)
        thought_step = chain.add_step(
            type="Thought",
            messages=newest_messages.copy(),
            description=new_msg.get("content", ""),
        )
        thought_step.total_token_length = (
            total_token_length if total_token_length is not None else 0
        )
        yield MessageProduced(
            chain_id=chain_id,
            depth=depth,
            message=new_msg,
            total_token_length=total_token_length,
        )

        # Per-turn budget hit (mini-swe's finish_reason == "length"): counted for
        # every generation so the batch summary can report looping/runaway
        # thinking, and reused below to pick the length-aware nudge.
        filled_budget = self._generation_filled_budget(
            new_msg, current_step, tools, generation_config
        )
        if filled_budget:
            chain.info["capped_turns"] = chain.info.get("capped_turns", 0) + 1

        # 3. Terminal / control checks on the generated message.
        if tool_calls:
            # A tool-carrying turn resets mini-swe's consecutive format-error streak.
            chain.info["no_tool_call_streak"] = 0
        if (
            max_model_len is not None
            and thought_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN
        ):
            # Hard limit: context length exceeded — always end.
            thought_step.is_terminal = True
            chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
        elif not tool_calls:
            # #1 no tool call at all → the run's on_no_tool_call policy.
            control = self.on_no_tool_call
            if control == CONTROL_RAISE:
                raise RuntimeError(
                    "Model produced no tool call and on_no_tool_call='raise': "
                    f"{new_msg.get('content', '')!r}"
                )
            thought_step.is_terminal = control == CONTROL_END
            if thought_step.is_terminal:
                chain.info["finish_reason"] = FinishReason.NO_TOOL_CALLS
            else:
                # 'continue' (mini-swe-agent format-error semantics): count the
                # miss; the streak limit ends the chain like mini-swe's
                # RepeatedFormatError. Any tool-carrying turn resets the streak.
                streak = chain.info.get("no_tool_call_streak", 0) + 1
                chain.info["no_tool_call_streak"] = streak
                chain.info["no_tool_call_turns"] = chain.info.get("no_tool_call_turns", 0) + 1
                limit = self.max_consecutive_no_tool_calls
                if limit and streak >= limit:
                    thought_step.is_terminal = True
                    chain.info["finish_reason"] = FinishReason.REPEATED_NO_TOOL_CALLS
        elif not self.project_actions and self.message_ends_episode(new_msg):
            # The model explicitly ended (a sentinel / parser status) on a turn that
            # carried a tool call — end without executing it (existing behavior). Under
            # ``project_actions`` verl-agent has no such sentinel: it always steps and ends
            # only on env done / max_turns, so this early end is skipped (as in StepRollout).
            thought_step.is_terminal = True
            chain.info["finish_reason"] = FinishReason.TERMINAL

        if thought_step.is_terminal:
            yield TurnResult(thought_step, have_set_resources, False)
            return
        if not tool_calls:
            # on_no_tool_call == 'continue': nothing to execute. Append the nudge as
            # a user turn (mini-swe's format_error_template) so the next generation
            # is conditioned on it; it is a user message, so training masks it out
            # like any observation. Token accounting mirrors a tool observation.
            nudge = self.render_no_tool_call_message(truncated=filled_budget)
            if nudge is None:
                yield TurnResult(thought_step, have_set_resources, True)
                return
            chain.info["no_tool_call_nudges"] = chain.info.get("no_tool_call_nudges", 0) + 1
            nudge_msg = {"role": "user", "content": [{"type": "text", "text": nudge}]}
            nudge_step = chain.add_step(
                type="Format Error",
                messages=[*newest_messages, nudge_msg],
                description=nudge,
            )
            nudge_step.total_token_length = (
                thought_step.total_token_length
                + observation_token_length(self.agent, nudge)
                + TOOL_OBS_PREFIX_TOKENS
            )
            if (
                max_model_len is not None
                and nudge_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN
            ):
                nudge_step.is_terminal = True
                chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
                yield TurnResult(nudge_step, have_set_resources, False)
                return
            yield TurnResult(nudge_step, have_set_resources, True)
            return

        # 4. Execute tool calls (shared L2). execute_turn_tools yields ToolObserved per
        # call and, as its last item, a ToolLoopResult summarizing the loop.
        loop_result: Optional[ToolLoopResult] = None
        async for item in self.execute_turn_tools(
            self.agent,
            chain_id=chain_id,
            chain=chain,
            depth=depth,
            tools=tools,
            context=context,
            thought_step=thought_step,
            newest_messages=newest_messages,
            have_set_resources=have_set_resources,
        ):
            if isinstance(item, ToolLoopResult):
                loop_result = item
            else:
                yield item  # ToolObserved — relay
        have_set_resources = loop_result.have_set_resources

        # 5. Decide the turn outcome. (tool_calls was non-empty, so a step was produced.)
        if loop_result.terminated:  # context length hit mid tool loop (finish_reason set)
            yield TurnResult(loop_result.last_step, have_set_resources, False)
            return
        if loop_result.last_step is not None and loop_result.last_step.is_terminal:
            # A tool ended the episode (explicit control='end' or an error routed to 'end').
            chain.info["finish_reason"] = self.tool_finish_reason(loop_result.last_step.tool_result)
            yield TurnResult(loop_result.last_step, have_set_resources, False)
            return
        yield TurnResult(loop_result.last_step, have_set_resources, True)
