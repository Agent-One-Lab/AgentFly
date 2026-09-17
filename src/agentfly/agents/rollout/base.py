"""Rollout strategy base class.

A ``Rollout`` is *how* an agent completes a task (chain / step / …). It is a
per-``run`` strategy object: the agent (the :class:`~agentfly.agents.rollout.agent.RolloutAgent`)
passes itself as ``agent`` and the rollout drives the loop, reaching agent-specific
behavior through ``agent`` and reusing the shared machinery defined here.

Layering:
- **L1** agent hooks (``parse``/``validate_tool_call``/``execute_tool_call``/…) live on
  the agent and are reached via ``agent``.
- **L2** shared machinery (this class, concrete): ``execute_turn_tools``,
  ``finalize_chain``, ``build_trajectories``, ``apply_context_folding`` — reused
  by rollout strategies.
- **L3** the thin strategy (a subclass): ``run`` (control flow), returning a
  ``RunResult``. Training conversion and tokenization live outside the rollout
  interface; inference-only rollouts need no training methods.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ...core import Context
from ...rewards.reward_base import calculate_reward
from ...rewards.types import RewardResult
from ...tools.tool_base import InvalidToolCallError
from ...tools.types import CONTROL, CONTROL_CONTINUE, CONTROL_END, CONTROL_RAISE, ToolResult
from ...utils.vision import image_to_data_uri
from ..types import RunResult, Segment, Trajectory
from ..utils.action_format import action_format_valid, message_text
from ..utils.messages import Messages
from agentfly.tools.src.context.tools import fold_messages_with_summarize
from .events import FinishReason, ToolObserved
from .structures import Chain, Step
from .loop.metrics import aggregate_tool_metrics, count_tool_calls
from .loop.tokens import estimate_chat_prompt_tokens, observation_token_length


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
import os
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
    "argument:\n- Tool: bash\n- Arguments: {\"command\": \"your_command_here\"}\n\nIf "
    "you want to end the task, please issue the following command: `echo "
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command."
)
MINISWE_NO_TOOL_CALL_TRUNCATED_TEXT = (
    "Your previous response reached the output token limit (finish_reason=length) "
    "before you produced a tool call, so it was cut off. Respond more concisely and "
    "finish with exactly one bash tool call. If you need to think more, do so briefly."
)

class Rollout(ABC):
    """Base rollout strategy: shared machinery (L2) + the abstract L3 interface.

    Concrete strategies (:class:`~agentfly.agents.rollout.strategies.chain_rollout.ChainRollout`, …) subclass
    this, implement :meth:`run`, and reuse the concrete methods
    here. Agent-provided state/hooks are reached through ``agent`` (a
    :class:`~agentfly.agents.rollout.agent.RolloutAgent`).
    """

    # Runtime-owned keys, excluded from task metadata. Execution counts are
    # derived from records, never trusted from incoming task metadata.
    _RESERVED_INFO_KEYS = frozenset({
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
    })

    # Parsed-message ``status`` values that mean the MODEL explicitly ended the episode
    # (an "end task" sentinel, or a parser-supplied status). Distinct from the
    # continue/end/raise control vocabulary; the loop maps a match to ``end``.
    TERMINAL_MESSAGE_STATUS = frozenset({"terminal", "finish"})

    # ---- L2: loop-control policy (the single authority for continue/end/raise) ----

    def __init__(
        self,
        *,
        on_no_tool_call: str = CONTROL_END,
        on_invalid_tool_call: str = CONTROL_END,
        on_tool_error: str = CONTROL_CONTINUE,
        max_consecutive_no_tool_calls: int = 3,
        no_tool_call_message: Optional[str] = NO_TOOL_CALL_MESSAGE_MINISWE,
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

        Under ``on_no_tool_call="continue"`` the loop mirrors mini-swe-agent's format-error
        handling: a tool-less turn gets a corrective **user** message (the nudge) and the
        model is queried again; ``max_consecutive_no_tool_calls`` tool-less turns in a row
        end the chain (``0`` = unlimited), and any turn that carries a tool call resets
        the streak. ``no_tool_call_message`` selects the nudge text: ``"miniswe"``
        renders mini-swe-agent's own ``format_error_template`` (length-aware: a turn cut
        off at the per-turn token cap gets the "reached the output token limit" variant),
        any other string is used verbatim, and ``None`` continues silently (no nudge).
        """
        for label, val in (
            ("on_no_tool_call", on_no_tool_call),
            ("on_invalid_tool_call", on_invalid_tool_call),
            ("on_tool_error", on_tool_error),
        ):
            if val not in CONTROL:
                raise ValueError(
                    f"{label} must be one of {sorted(CONTROL)}, got {val!r}"
                )
        self.on_no_tool_call = on_no_tool_call
        self.on_invalid_tool_call = on_invalid_tool_call
        self.on_tool_error = on_tool_error
        try:
            max_consecutive_no_tool_calls = int(max_consecutive_no_tool_calls)
        except (TypeError, ValueError):
            raise ValueError(
                "max_consecutive_no_tool_calls must be a non-negative int, "
                f"got {max_consecutive_no_tool_calls!r}"
            )
        if max_consecutive_no_tool_calls < 0:
            raise ValueError(
                "max_consecutive_no_tool_calls must be a non-negative int, "
                f"got {max_consecutive_no_tool_calls!r}"
            )
        if no_tool_call_message is not None and not isinstance(no_tool_call_message, str):
            raise ValueError(
                f"no_tool_call_message must be a str or None, got {no_tool_call_message!r}"
            )
        self.max_consecutive_no_tool_calls = max_consecutive_no_tool_calls
        self.no_tool_call_message = no_tool_call_message or None
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

    _message_text = staticmethod(message_text)
    _action_format_valid = staticmethod(action_format_valid)

    @classmethod
    def _projected_tool_call(cls, agent, msg: Dict) -> Optional[Dict]:
        """verl-agent's always-step projection: a turn with no extractable ``<action>``
        still steps the env, with the last 30 characters of the (lowercased) raw output as
        a garbage fallback action (the env answers "nothing happens"; the episode goes on).
        The turn stays format-invalid via ``_action_format_valid``. Returns None if the
        agent has no action tool to project onto.
        """
        import json

        tool_name = getattr(agent, "action_tool_name", None) or (
            agent.tools[0].name if getattr(agent, "tools", None) else None
        )
        if tool_name is None:
            return None
        fallback = cls._message_text(msg).lower()[-30:]
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

    # ---- L2: shared machinery (reused by every rollout strategy) ----

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
                running_total_token_length
                + obs_token_length
                + TOOL_OBS_PREFIX_TOKENS
            )
            running_total_token_length = action_input_step.total_token_length

            # Terminate if context length exceeded after this tool observation
            if (
                max_model_len is not None
                and action_input_step.total_token_length >= max_model_len - TOKEN_SAFETY_MARGIN
            ):
                action_input_step.is_terminal = True
                chain.info["finish_reason"] = FinishReason.MAX_MODEL_LEN
                yield ToolLoopResult(
                    action_input_step, num_tool_calls, have_set_resources, terminated=True
                )
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
                    folded_prompt_tokens = estimate_chat_prompt_tokens(
                        agent, action_input_step, tools=tools
                    )
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

        yield ToolLoopResult(
            action_input_step, num_tool_calls, have_set_resources, terminated=False
        )

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

    async def finalize_chain(self, agent, chain_id, chain, current_step, depth, context):
        """Finalize the chain with reward calculation and cleanup."""
        # Always record the final trajectory segment so that histories capture
        chain.histories.append(current_step.messages.messages)

        if agent._reward_fn is not None:
            # flatten the histories
            full_trajectory = []
            for segment in chain.histories:
                full_trajectory.extend(segment)
            final_response = agent.extract_final_response(full_trajectory)

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
                agent._reward_fn,
                final_response=final_response,
                **other_args,
                trajectory=context.trajectory,
                id=chain_id,
                context=context,
            )
            chain.info["reward"] = reward
        else:
            chain.info["reward"] = None

        # Both strategies aggregate actual ToolResults, independent of Step layout.
        chain.info["tool_metrics"] = aggregate_tool_metrics(
            step.tool_result for step in chain.steps()
        )

        # Release global resources so other rollouts can use them
        await context.release_resource(scope="global")

        # Kill rollout-scoped resources
        await context.end_resource(scope="rollout")

    def build_trajectories(self, agent) -> List[Trajectory]:
        """Build :class:`Trajectory` objects for every active chain.

        Maps ``chain.info`` keys into the typed Trajectory fields:

        - ``reward`` (a ``RewardResult`` produced by ``calculate_reward``) →
          ``Trajectory.reward`` (float) + ``Trajectory.metrics`` (extras)
        - ``tool_metrics`` (per-chain averages finalized in ``finalize_chain``)
          → merged into ``Trajectory.metrics`` alongside the reward metrics
        - ``finish_reason``, ``rollout_time_sec``, ``chain_id``,
          ``group_id``, ``chain_idx``, ``group_idx`` → first-class fields
        - everything else (dataset-row passthrough fields like ``answer``,
          ``task_id``, etc.) → ``Trajectory.metadata``

        Execution counts come from the full chain's records, before filtering the
        internal ``Trajectory.steps`` channel to tool-result steps.
        """
        trajectories: List[Trajectory] = []
        # Sort by (group_idx, chain_idx) so trajectories are in deterministic order
        items = list(self.current_steps.items())
        items.sort(
            key=lambda item: (
                self.chains[item[0]].info.get("group_idx", 0),
                self.chains[item[0]].info.get("chain_idx", 0),
            )
        )
        for chain_id, step in items:
            chain = self.chains[chain_id]
            info = chain.info
            chain_steps = chain.steps()

            # Histories may contain multiple segments when context tools fold.
            segments = [Segment(messages=messages) for messages in chain.histories]

            # Unwrap reward into the typed (reward, metrics) split.
            raw_reward = info.get("reward")
            if isinstance(raw_reward, RewardResult):
                reward_value = raw_reward.reward
                metrics = dict(raw_reward.metrics)
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

            # Tool metrics were finalized per-chain in finalize_chain; read them
            # the same way reward metrics are read above.
            metrics.update(info.get("tool_metrics") or {})

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
                    num_turns=sum(s.type == "Thought" for s in chain_steps),
                    tool_call_counts=count_tool_calls(s.tool_result for s in chain_steps),
                    chain_id=info.get("chain_id"),
                    group_id=info.get("group_id"),
                    chain_idx=info.get("chain_idx"),
                    group_idx=info.get("group_idx"),
                    metadata=metadata,
                    steps=[s for s in chain_steps if s.tool_result is not None],
                )
            )
        return trajectories

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
        **kwargs,
    ) -> RunResult:
        """Drive the rollout for a batch of tasks and return a :class:`RunResult`.

        ``global_step`` is the caller's step counter for the metrics x-axis; see
        :meth:`resolve_global_step`.
        """
        ...
