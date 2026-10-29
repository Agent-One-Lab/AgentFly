"""Chain rollout: one growing conversation per trajectory.

:class:`ChainRollout` runs many chains concurrently. Each turn generates a response, parses
and executes its tool calls, and appends the results to the same conversation, so the model is
trained on all of its turns in one sequence. Loop control (when a chain continues, ends or
raises) follows the rollout's run policy.
"""

import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agentfly.tools.src.context.tools import fold_messages_with_summarize

from ....core import Context, ContextConfig
from ....tools.types import CONTROL_CONTINUE, CONTROL_END, CONTROL_RAISE
from ....utils.monitor import Monitor
from ....utils.vision import image_to_data_uri
from ...types import RunResult, Segment, Trajectory
from ...utils.messages import Messages
from ..base import EpisodeSpec, Rollout
from ..events import (
    ChainEnded,
    ChainStarted,
    FinishReason,
    MessageProduced,
    ToolObserved,
)
from ..loop.generation import generate_response, prepare_generation_config
from ..loop.metrics import RolloutMetrics, aggregate_tool_metrics, count_tool_calls
from ..loop.tokens import estimate_chat_prompt_tokens, observation_token_length
from ..structures import Chain, Step

# Extra tokens charged per tool observation for the chat-template prefix that wraps it
# (e.g. ``<|im_start|>user<tool_response>`` ... ``</tool_response><|im_end|>``). Approximate;
# used only for the max_model_len budget check, not for the actual prompt rendering.
TOOL_OBS_PREFIX_TOKENS = 15
# Headroom reserved below max_model_len for the context guard. The incremental
# per-turn token estimate has a small residual drift (a few tokens: single-render
# wrapper markers + generation suffix vs the flat TOOL_OBS_PREFIX_TOKENS), which
# at the exact ceiling can let an over-length prompt slip to vLLM — which RAISES
# and kills the whole run. Clipping the guard this far below max_model_len absorbs
# that drift. Env-overridable.
TOKEN_SAFETY_MARGIN = int(os.environ.get("AGENTFLY_TOKEN_SAFETY_MARGIN", "512"))

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolLoopResult:
    """Internal control value yielded by :meth:`Rollout.execute_turn_tools` as its
    last item — summarizes the tool-call loop's outcome for the strategy's turn loop.
    Not a ``RolloutEvent``.

    ``last_step`` is the final ``Action Input`` step (None if no valid tool ran).
    ``terminated`` is True when the loop stopped early on ``max_model_len`` (finish_reason
    is already set); the caller then ends the chain on ``last_step``.
    """

    last_step: Optional[Step]
    num_tool_calls: int
    have_set_resources: bool
    terminated: bool


# ---- mini-swe-agent format-error nudges (``config/mini.yaml`` ``format_error_template``,
# mini-swe-agent 2.4.6, rendered for the "no tool calls" case). The same text appears in
# the miniswe SFT trajectories, so it is on-distribution for models trained on them. ----
NO_TOOL_CALL_MESSAGE_MINISWE = "miniswe"
MINISWE_NO_TOOL_CALL_TEXT = (
    "Tool call error:\n\n<error>\nNo tool calls found in the response. Every response "
    "MUST include at least one tool call.\n</error>\n\nHere is general guidance on how "
    "to submit correct toolcalls:\n\nEvery response needs to use the 'bash' tool at "
    "least once to execute commands.\n\nCall the bash tool with your command as the "
    'argument:\n- Tool: bash\n- Arguments: {"command": "your_command_here"}\n\nIf '
    "you want to end the task, please issue the following command: `echo "
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command."
)
MINISWE_NO_TOOL_CALL_TRUNCATED_TEXT = (
    "Your previous response reached the output token limit (finish_reason=length) "
    "before you produced a tool call, so it was cut off. Respond more concisely and "
    "finish with exactly one bash tool call. If you need to think more, do so briefly."
)


@dataclass(frozen=True)
class TurnResult:
    """Internal control value yielded by :meth:`ChainRollout.run_turn` as its last
    item — carries the loop-control outcome that the old method returned. Not a
    ``RolloutEvent``; :meth:`ChainRollout.run_episode` consumes it and does not relay it."""

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
        """See :meth:`Rollout.__init__` for the loop-control policy.

        Under ``on_no_tool_call="continue"`` the loop mirrors mini-swe-agent's format-error
        handling: a tool-less turn gets a corrective **user** message (the nudge) and the
        model is queried again; ``max_consecutive_no_tool_calls`` tool-less turns in a row
        end the chain (``0`` = unlimited), and any turn that carries a tool call resets
        the streak. ``no_tool_call_message`` selects the nudge text: ``"miniswe"``
        renders mini-swe-agent's own ``format_error_template`` (length-aware: a turn cut
        off at the per-turn token cap gets the "reached the output token limit" variant),
        any other string is used verbatim, and ``None`` continues silently (no nudge).
        """
        super().__init__(
            on_no_tool_call=on_no_tool_call,
            on_invalid_tool_call=on_invalid_tool_call,
            on_tool_error=on_tool_error,
        )
        try:
            max_consecutive_no_tool_calls = int(max_consecutive_no_tool_calls)
        except (TypeError, ValueError):
            raise ValueError(
                f"max_consecutive_no_tool_calls must be a non-negative int, got {max_consecutive_no_tool_calls!r}"
            )
        if max_consecutive_no_tool_calls < 0:
            raise ValueError(
                f"max_consecutive_no_tool_calls must be a non-negative int, got {max_consecutive_no_tool_calls!r}"
            )
        if no_tool_call_message is not None and not isinstance(no_tool_call_message, str):
            raise ValueError(f"no_tool_call_message must be a str or None, got {no_tool_call_message!r}")
        self.max_consecutive_no_tool_calls = max_consecutive_no_tool_calls
        self.no_tool_call_message = no_tool_call_message or None
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

    def initialize_chains(self, specs: List[EpisodeSpec]) -> Tuple[Dict[str, Chain], Dict[str, Step]]:
        """A :class:`Chain` (whose ``info`` is the spec's task metadata and ids) and its
        root step for each spec, keyed by ``episode_id`` (the chain id)."""
        chains = {}
        start_steps = {}
        for spec in specs:
            ch = Chain(spec.info)
            root = ch.add_step(type="Action Input", messages=deepcopy(spec.messages.messages))
            chains[spec.episode_id] = ch
            start_steps[spec.episode_id] = root
        return chains, start_steps

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
        use_token_ids: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> RunResult:
        """Drive the chain-based rollout and return its :class:`RunResult`.

        ``agent`` (the agent) is stashed for the run so the internal loop can reach
        agent state/hooks; the transient run-state is rebuilt each call.
        ``global_step`` sets the metrics x-axis (see ``Rollout.resolve_global_step``).
        ``use_token_ids`` see ``Rollout.run``. ``show_progress`` renders a progress bar
        over finished chains.
        """
        self.agent = agent
        self.use_token_ids = bool(use_token_ids)
        # Drain the rollout event stream. Training ignores the events themselves; the
        # final state (``self.chains`` / ``self.current_steps``) is populated by each
        # chain as it ends. We build the RunResult from that state afterwards.
        async for _ev in self.stream(
            messages,
            max_turns=max_turns,
            num_chains=num_chains,
            generation_config=generation_config,
            max_concurrent_chains=max_concurrent_chains,
            context_config=context_config,
            global_step=global_step,
            show_progress=show_progress,
        ):
            pass
        return RunResult(trajectories=self.build_trajectories(agent), rollout="chain")

    async def stream(
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

        This is the single shared rollout path: the per-chain generator :meth:`run_episode` is the loop,
        and :meth:`Rollout.run_episodes` fans the per-chain event streams into one ``chain_id``-tagged
        stream. Each chain populates ``self.chains`` / ``self.current_steps``
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

        specs = self.make_specs(messages, num_chains)
        chains, first_steps = self.initialize_chains(specs)
        # Prompt tools come from the agent's single source of truth (also used by
        # tokenize_trajectories), so generation and training render one prompt.
        tool_schemas = self.agent.prompt_tools()

        def run_episode(spec: EpisodeSpec):
            return self.run_episode(
                spec.episode_id,
                first_steps[spec.episode_id],
                chains[spec.episode_id],
                tool_schemas,
                max_turns,
                generation_config,
                context_config,
            )

        async for ev in self.run_episodes(
            specs,
            run_episode,
            max_concurrent_chains=max_concurrent_chains,
            show_progress=show_progress,
        ):
            yield ev

        self.metrics.record_step(
            global_step=step,
            trajectories=self.build_trajectories(self.agent),
        )

    async def run_episode(
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
        on completion (the fan-in in :meth:`stream` reads final state from there, not
        from the events).
        """
        started_at = time.monotonic()

        # Build Context from rollout data for tools that need it
        context = self.make_context(
            episode_id=chain_id,
            group_id=chain.info.get("group_id"),
            metadata=chain.info,
            context_config=context_config,
        )

        yield ChainStarted(chain_id=chain_id, group_id=chain.info.get("group_id"))

        current_step = first_step
        depth = 0
        have_set_resources = False

        await self.run_first_step_hook(self.agent, context, current_step, chain_id)

        while not current_step.is_terminal and depth < max_turns:
            turn_result: Optional[TurnResult] = None
            async for item in self.run_turn(
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
        # Loop-only state; chain.info becomes the trajectory's metadata.
        chain.info.pop("no_tool_call_streak", None)

        # Finalize chain (reward + cleanup).
        await self.finalize_chain(self.agent, chain_id, chain, current_step, depth, context)

        chain.info["rollout_time_sec"] = time.monotonic() - started_at

        # This chain owns its final state; stream reads from here, not from events.
        self.chains[chain_id] = chain
        self.current_steps[chain_id] = current_step

        self.metrics.record_chain(
            global_step=self.global_step,
            trajectory=chain.histories,
            info=chain.info,
        )

        yield ChainEnded(chain_id=chain_id, finish_reason=chain.info.get("finish_reason"))

    def generation_filled_budget(self, new_msg: Dict[str, Any], current_step: Step, tools, generation_config) -> bool:
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
            budget = prepare_generation_config(self.agent, generation_config, current_step, tools=tools).get(
                "max_tokens"
            )
            budget = int(budget) if budget is not None else None
        except Exception:  # noqa: BLE001 — diagnostics must never break the loop
            budget = None
        return budget is not None and len(ids) >= budget

    async def run_turn(
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
        — the control value the method used to return. :meth:`run_episode` relays the
        events and consumes the ``TurnResult``. When ``should_continue`` is False the
        chain loop breaks and ``next_step`` is the final step for this chain.

        The turn reads as a sequence of steps: context-length pre-check → generate →
        project actions → record/emit message → terminal check → run tool calls
        (:meth:`execute_turn_tools`) → decide outcome.
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
            min_depth = int(os.environ.get("SKILLRL_TOKEN_DEBUG_MINDEPTH", "6"))
            if depth >= min_depth:
                incr = current_step.total_token_length
                try:
                    exact = estimate_chat_prompt_tokens(self.agent, current_step, tools=tools, cap=False)
                except Exception:  # noqa: BLE001
                    exact = -1
                margin = int(os.environ.get("SKILLRL_TOKEN_DEBUG_MARGIN", "6000"))
                # Only surface turns that matter: near/over the ceiling by EITHER
                # measure, or a large incremental-vs-exact divergence.
                if (
                    exact >= max_model_len - margin
                    or incr >= max_model_len - margin
                    or (exact >= 0 and abs(exact - incr) > 100)
                ):
                    logger.info(
                        f"[TOKDBG] chain={chain_id} depth={depth} incr={incr} "
                        f"exact={exact} drift={(exact - incr) if exact >= 0 else 'NA'} "
                        f"max={max_model_len} guard_fires={incr >= max_model_len} "
                        f"exact_over={(exact > max_model_len) if exact >= 0 else 'NA'} "
                        f"slips_through={exact >= 0 and exact > max_model_len and incr < max_model_len}"
                    )

        # 1. Pre-check: stop before generating if already at/over the context limit.
        if max_model_len is not None and current_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN:
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
                use_token_ids=self.use_token_ids,
            )
        except Exception as e:  # noqa: BLE001
            # Backstop for an over-length prompt reaching vLLM: it RAISES
            # (ValueError "Prompt length ... exceeds the model's maximum context
            # length" / VLLMValidationError "max_tokens must be at least 1, got
            # 0"), and unhandled that kills the entire run. The TOKEN_SAFETY_MARGIN
            # guard should prevent ever getting here, but if a residual drift slips
            # through, abort THIS chain gracefully instead of propagating.
            error_text = str(e)
            if (
                "maximum context length" in error_text
                or "Prompt length" in error_text
                or "max_tokens must be at least 1" in error_text
            ):
                logger.warning(
                    f"[TOKDBG] OVER-LENGTH at generation (chain={chain_id} "
                    f"depth={depth}) — aborting chain, not raising. incr="
                    f"{current_step.total_token_length} max={max_model_len} :: {error_text[:180]}"
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
            projected = self.projected_tool_call(self.agent, new_msg)
            if projected is not None:
                tool_calls = [projected]
                new_msg["tool_calls"] = tool_calls

        newest_messages.append(new_msg)
        thought_step = chain.add_step(
            type="Thought",
            messages=newest_messages.copy(),
            description=new_msg.get("content", ""),
        )
        thought_step.total_token_length = total_token_length if total_token_length is not None else 0
        yield MessageProduced(
            chain_id=chain_id,
            depth=depth,
            message=new_msg,
            total_token_length=total_token_length,
        )

        # Per-turn budget hit (mini-swe's finish_reason == "length"): counted for
        # every generation so the batch summary can report looping/runaway
        # thinking, and reused below to pick the length-aware nudge.
        filled_budget = self.generation_filled_budget(new_msg, current_step, tools, generation_config)
        if filled_budget:
            chain.info["capped_turns"] = chain.info.get("capped_turns", 0) + 1

        # 3. Terminal / control checks on the generated message.
        if tool_calls:
            # A tool-carrying turn resets mini-swe's consecutive format-error streak.
            chain.info["no_tool_call_streak"] = 0
        if max_model_len is not None and thought_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN:
            # Hard limit: context length exceeded — always end.
            thought_step.is_terminal = True
            chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
        elif not tool_calls:
            # #1 no tool call at all → the run's on_no_tool_call policy.
            control = self.on_no_tool_call
            if control == CONTROL_RAISE:
                raise RuntimeError(
                    f"Model produced no tool call and on_no_tool_call='raise': {new_msg.get('content', '')!r}"
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
                thought_step.total_token_length + observation_token_length(self.agent, nudge) + TOOL_OBS_PREFIX_TOKENS
            )
            if max_model_len is not None and nudge_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN:
                nudge_step.is_terminal = True
                chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
                yield TurnResult(nudge_step, have_set_resources, False)
                return
            yield TurnResult(nudge_step, have_set_resources, True)
            return

        # 4. Execute tool calls. execute_turn_tools yields ToolObserved per
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

    # ---- moved from Rollout: used only by the chain loop ----

    def render_no_tool_call_message(self, *, truncated: bool = False) -> Optional[str]:
        """The nudge text for a tool-less turn, or ``None`` when nudging is disabled.

        ``truncated`` marks a turn whose generation filled the per-turn token budget
        (mini-swe's ``finish_reason == "length"`` case), which selects the length-aware
        variant of the mini-swe template.
        """
        if self.no_tool_call_message is None:
            return None
        if self.no_tool_call_message != NO_TOOL_CALL_MESSAGE_MINISWE:
            return self.no_tool_call_message
        if truncated:
            return MINISWE_NO_TOOL_CALL_TRUNCATED_TEXT
        return MINISWE_NO_TOOL_CALL_TEXT

    async def execute_turn_tools(
        self,
        agent,
        *,
        chain_id: str,
        chain: Chain,
        depth: int,
        tools: List[Dict],
        context: Context,
        thought_step: Step,
        newest_messages: Messages,
        have_set_resources: bool,
    ):
        """Execute the assistant message's tool calls (supports multiple per turn).

        Async generator: yields a ``ToolObserved`` per executed call and, as its **last**
        item, exactly one :class:`ToolLoopResult`. Owns per-tool observation-step
        bookkeeping, token accounting, context folding, and the mid-loop ``max_model_len``
        termination.

        Each tool result is resolved to a loop control via :meth:`resolve_tool_control`
        (the tool's explicit ``control`` wins; else the ``status`` diagnosis routes to the
        run policy). A malformed call (``status="invalid"``) → ``on_invalid_tool_call``; a
        tool-body exception (``status="error"``) → ``on_tool_error``; a clean result
        continues. ``raise`` fails fast; ``end`` marks the step terminal and stops the loop.
        """
        max_model_len = getattr(agent, "max_model_len", None)
        num_tool_calls = 0
        action_input_step: Optional[Step] = None
        running_total_token_length = thought_step.total_token_length

        for tool_call in thought_step.messages[-1].get("tool_calls") or []:
            # A malformed call comes back as a ``status="invalid"`` result (submit_tool_call
            # no longer raises); ``resolve_tool_control`` below routes it via the run policy.
            result = await agent.execute_tool_call(
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

            # The per-turn tool result is carried on the Step it produced
            # (``action_input_step.tool_result``) — the single per-turn source for
            # metric-averaging (logging, in finalize_chain) and feature harvesting
            # (Trajectory.steps). One Step per tool call (== one turn for action agents).
            action_input_step = chain.add_step(
                type="Action Input",
                messages=newest_messages.copy(),
                description=result.arguments,
                tool_result=result,
            )
            observation = result.observation
            yield ToolObserved(
                chain_id=chain_id,
                depth=depth,
                tool_name=tool_call["function"]["name"],
                observation=observation,
                status=result.status,
                image=result.image,
            )
            obs_token_length = observation_token_length(agent, observation)
            action_input_step.total_token_length = (
                running_total_token_length + obs_token_length + TOOL_OBS_PREFIX_TOKENS
            )
            running_total_token_length = action_input_step.total_token_length

            # Terminate if context length exceeded after this tool observation
            if (
                max_model_len is not None
                and action_input_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN
            ):
                action_input_step.is_terminal = True
                chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
                yield ToolLoopResult(action_input_step, num_tool_calls, have_set_resources, terminated=True)
                return

            new_content = [{"type": "text", "text": observation}]
            if result.image:
                image_base64 = image_to_data_uri(result.image)
                new_content.append({"type": "image", "image": image_base64})

            # Apply context folding based on the tool result (currently only summarize).
            updated_messages, did_fold_context = self.apply_context_folding(
                agent,
                chain=chain,
                messages=newest_messages,
                tool_name=tool_call["function"]["name"],
                observation=observation,
                tool_call_id=tool_call["id"],
                tool_result_name=result.name,
                new_content=new_content,
                context=context,
            )

            action_input_step.messages = updated_messages
            newest_messages = updated_messages.copy()
            if did_fold_context:
                # Folded history is shorter; re-measure prompt tokens for the new
                # messages (same path as first generation) instead of keeping the
                # pre-fold cumulative total.
                try:
                    folded_prompt_tokens = estimate_chat_prompt_tokens(agent, action_input_step, tools=tools)
                except ValueError:
                    folded_prompt_tokens = 0
                action_input_step.total_token_length = folded_prompt_tokens
                running_total_token_length = folded_prompt_tokens

            # Resolve this tool result to a loop control (tool's explicit ``control`` wins;
            # else ``status`` routes to the run policy). ``raise`` aborts; ``end`` marks the
            # step terminal and stops the tool loop; ``continue`` runs the next call / turn.
            control = self.resolve_tool_control(result)
            if control == CONTROL_RAISE:
                self.raise_for_result(result)
            action_input_step.is_terminal = control == CONTROL_END
            if action_input_step.is_terminal:
                break

        yield ToolLoopResult(action_input_step, num_tool_calls, have_set_resources, terminated=False)

    def apply_context_folding(
        self,
        agent,
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
            folded_turns = fold_messages_with_summarize(messages.messages, observation, keep_previous_summary=False)
            context.trajectory_format = "segmented"
            meta = messages.meta
            return Messages.from_turns(folded_turns, **meta), True

        # Default behavior for non-folding tools: just append the tool message.
        turns = messages.messages + [tool_turn]
        meta = messages.meta
        return Messages.from_turns(turns, **meta), False

    async def finalize_chain(self, agent, chain_id, chain, current_step, depth, context):
        """Finalize the chain with reward calculation and cleanup."""
        # Always record the final trajectory segment so that histories capture
        chain.histories.append(current_step.messages.messages)

        # The reward sees the whole conversation (every segment when context folding
        # split it); a flat trajectory passes its single segment.
        full_trajectory = [message for segment in chain.histories for message in segment]
        chain.info["reward"] = await self.compute_reward(
            agent,
            context,
            episode_id=chain_id,
            trajectory=chain.histories[0] if context.trajectory_format == "flat" else chain.histories,
            final_messages=full_trajectory,
            reward_args=chain.info,
        )

        # Both strategies aggregate actual ToolResults, independent of Step layout.
        chain.info["tool_metrics"] = aggregate_tool_metrics(step.tool_result for step in chain.steps())

        await self.release_resources(context)

    def build_trajectories(self, agent) -> List[Trajectory]:
        """Build :class:`Trajectory` objects for every finished chain, in
        ``(group_idx, chain_idx)`` order.

        Reads ``chain.info``: ``reward`` and ``tool_metrics`` (set by
        :meth:`finalize_chain`), ``finish_reason``, ``rollout_time_sec``, the ids, and
        the task metadata (see :meth:`Rollout.make_trajectory`). Execution counts come
        from the full chain's records, before filtering the internal
        ``Trajectory.steps`` channel to tool-result steps.
        """
        items = list(self.current_steps.items())
        items.sort(
            key=lambda item: (
                self.chains[item[0]].info.get("group_idx", 0),
                self.chains[item[0]].info.get("chain_idx", 0),
            )
        )
        trajectories: List[Trajectory] = []
        for chain_id, _ in items:
            chain = self.chains[chain_id]
            info = chain.info
            chain_steps = chain.steps()
            trajectories.append(
                self.make_trajectory(
                    info=info,
                    # Histories may contain multiple segments when context tools fold.
                    segments=[Segment(messages=messages) for messages in chain.histories],
                    reward=info.get("reward"),
                    tool_metrics=info.get("tool_metrics"),
                    finish_reason=info.get("finish_reason"),
                    rollout_time_sec=info.get("rollout_time_sec"),
                    num_turns=sum(s.type == "Thought" for s in chain_steps),
                    tool_call_counts=count_tool_calls(s.tool_result for s in chain_steps),
                    steps=[s for s in chain_steps if s.tool_result is not None],
                )
            )
        return trajectories
