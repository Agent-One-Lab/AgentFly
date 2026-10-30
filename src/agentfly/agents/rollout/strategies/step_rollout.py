"""Single-turn / bounded-history rollout (verl-agent style), as a :class:`Rollout`.

``StepRollout`` records **one segment per generation** with a **bounded** per-step prompt
(last N ``(obs, action)`` turns), the natural substrate for step-level RL (GiGPO):
the grouping ``anchor`` equals the state the policy conditioned on, so step credit is
well-posed.

Unlike verl-agent (batch-synchronous stepping), this mirrors :class:`ChainRollout`:
each trajectory runs its **own async coroutine concurrently**, generation and
``execute_tool_call`` are async, and the loop is async. Step grouping is by
``anchor_obs`` (not step index), so async progression across trajectories does not
affect the algorithm. It reuses the agent L1 hooks (``generate_async`` / ``parse`` /
``execute_tool_call`` / ``prepare_first_step``), the ``Context`` lifecycle, and
``calculate_reward``.

Like :class:`ChainRollout`, it emits the shared :class:`~agentfly.agents.types.Trajectory`
(one per rollout) carrying per-turn :class:`~agentfly.agents.rollout.structures.Step`
records. Each generation produces a :class:`~agentfly.agents.types.Segment`;
the standalone training converter reads these segments and uses the internal
``trajectory.steps`` records for the corresponding environment signals.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ....tools.types import CONTROL_CONTINUE, CONTROL_END, CONTROL_RAISE
from ...types import Segment
from ...utils.action_format import message_text
from ...utils.messages import Messages
from ..base import Rollout
from ..events import FinishReason, MessageProduced, ToolObserved
from ..loop.generation import generate_response
from ..sessions import LoopState, SessionResult
from ..structures import Step
from .prompt_builders import resolve_prompt_builder

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class WindowState(LoopState):
    """The turn loop's state for one session; :meth:`StepRollout.run_loop` updates it.

    ``current`` is the state the model acts on next and ``history`` the earlier
    ``(obs, action)`` turns (oldest first); the prompt builder renders a bounded window
    of them. ``full_messages`` is the whole conversation (for the outcome reward) and
    ``steps`` holds one :class:`Step` per generation.
    """

    current: Dict[str, Any]
    task_description: str
    full_messages: List[Dict]
    history: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    finish_reason: Optional[FinishReason] = None


class StepRollout(Rollout):
    name = "step"

    def __init__(
        self,
        history_length: int = 2,
        prompt_builder: Any = "chat_window",
        max_prompt_length: int = 2048,
        *,
        on_no_tool_call: str = CONTROL_END,
        on_invalid_tool_call: str = CONTROL_END,
        on_tool_error: str = CONTROL_CONTINUE,
        project_actions: bool = False,
        **kwargs,
    ):
        # Keep **kwargs so log_token_drift gets its migration-specific error.
        if "log_token_drift" in kwargs:
            raise TypeError(
                "log_token_drift is a conversion option, not a StepRollout option. "
                "Pass it to agent.to_verl_dataproto(result, log_token_drift=...)."
            )
        if kwargs:
            unknown_options = ", ".join(repr(name) for name in sorted(kwargs))
            raise TypeError(f"Unexpected StepRollout constructor option(s): {unknown_options}")
        super().__init__(
            on_no_tool_call=on_no_tool_call,
            on_invalid_tool_call=on_invalid_tool_call,
            on_tool_error=on_tool_error,
        )
        # verl-agent-faithful ALWAYS-STEP rollout (``alfworld_projection`` + ``env_manager``):
        # every turn steps the env — a turn with no extractable ``<action>`` falls back to the
        # last-30-chars of the raw output (garbage) and steps anyway (format-invalid, recorded
        # via is_action_valid) — and the episode ends ONLY on env done (won/lost) or max_turns,
        # never on a malformed / no-tool-call turn. This removes the early-exit lever and keeps
        # the rollout distribution grounded, matching verl-agent's GRPO/GiGPO rollout.
        self.project_actions = project_actions
        self.history_length = history_length
        self.max_prompt_length = max_prompt_length
        self.prompt_builder = resolve_prompt_builder(prompt_builder, max_prompt_length=max_prompt_length)

    # ---- the strategy contract ----

    def new_state(self, agent, messages: List[Dict], *, episode_id: str, info: Dict[str, Any]) -> WindowState:
        """A window whose current observation is the last user message of ``messages``
        (the whole of ``messages`` starts the full conversation)."""
        task_description = info.get("task_description", "")
        return WindowState(
            episode_id=episode_id,
            info=info,
            current={
                "raw_obs": "",
                "obs": self.last_user_text(messages),
                "admissible": [],
                "task_description": task_description,
            },
            task_description=task_description,
            full_messages=list(messages),
        )

    def after_first_step_hook(self, state: WindowState, context) -> None:
        """Take the env's start from what the first-step hook recorded.

        Env-source-of-truth agents (WebShop, ALFWorld) record the initial observation and
        admissible actions in ``context.metadata`` for the prompt builders. The task
        description is the env's templated goal when there is one (WebShop records it;
        for ALFWorld it is verl-agent's ``extract_task``: the text after "Your task is
        to: " in the initial observation), not the dataset's human-annotated one;
        verl-agent uses ``goal_desc_human_anns_prob=0.0`` (templated goals).
        """
        initial_obs = context.metadata.get("_flat_initial_obs", "")
        task_description = (
            context.metadata.get("_flat_task_description")
            or self.extract_task(str(initial_obs))
            or state.task_description
        )
        state.current["raw_obs"] = initial_obs
        state.current["admissible"] = context.metadata.get("_flat_initial_admissible", [])
        state.current["task_description"] = task_description
        state.task_description = task_description

    def transcript(self, state: WindowState) -> List[List[Dict]]:
        """The full (unwindowed) conversation."""
        return [state.full_messages]

    def segments(self, state: WindowState) -> List[Segment]:
        """One segment per generation (its bounded prompt and response), the units the
        step-level training converter reads; one empty segment if nothing was generated."""
        if not state.steps:
            return [Segment(messages=[])]
        return [Segment(messages=step.messages.messages) for step in state.steps]

    def records(self, state: WindowState) -> List[Step]:
        """One step per generation; ``tool_result`` is set when an action ran."""
        return state.steps

    def result(self, state: WindowState) -> SessionResult:
        return self.session_result(state, finish_reason=state.finish_reason, turns=len(state.steps))

    async def run_loop(
        self,
        *,
        agent,
        context,
        state: WindowState,
        tools: List[Dict],
        max_turns: int,
        generation_config: Optional[Dict[str, Any]],
    ):
        """Run up to ``max_turns`` turns of ``agent`` on ``state``; yields
        ``MessageProduced`` / ``ToolObserved``.

        Only the turns: no context set-up, first-step hook, reward or resource release.
        ``state`` is updated in place and ``state.finish_reason`` is set when the loop
        stops (``max_turns`` unless a turn ended it).
        """
        for _ in range(max_turns):
            depth = state.depth
            state.depth += 1
            prompt_messages = self.prompt_builder.build(
                agent,
                history=state.history,
                current=state.current,
                history_length=self.history_length,
                step_index=depth,
            )
            # Reuse the chain generation helper (return_dict / tools / tool_call_source /
            # skills / trajectory_segments / raw-field overlay). ``chain``/``depth``/
            # ``episode_id`` are unused by it; pass None / depth / episode_id.
            gen_step = Step(
                type="Action Input",
                messages=Messages.from_turns(prompt_messages),
            )
            action_msg, total_token_length = await generate_response(
                agent,
                None,
                gen_step,
                tools,
                depth,
                state.episode_id,
                generation_config,
                context,
                use_token_ids=self.use_token_ids,
            )
            state.full_messages.append(action_msg)

            # Record the generation into the trajectory the moment it is produced — the same
            # way ChainRollout adds every generated message as a step before it decides whether
            # to run a tool. ``messages`` = the bounded prompt this turn ending in the action;
            # ``tool_result`` is attached below only if an action actually executes. A terminal
            # / no-action turn keeps ``tool_result=None`` (a generation-only step), so no part
            # of the trajectory is ever lost and every trajectory has >=1 step.
            gen_step.messages = Messages.from_turns(list(prompt_messages) + [action_msg])
            state.steps.append(gen_step)
            yield MessageProduced(
                episode_id=state.episode_id,
                depth=depth,
                message=action_msg,
                total_token_length=total_token_length,
            )

            # Control decisions run through the SAME base machinery as ChainRollout.
            tool_calls = action_msg.get("tool_calls") or []
            if not tool_calls and self.project_actions:
                # verl-agent projection: no ``<action>`` block was extracted → step the env
                # anyway with the last-30-chars garbage fallback (see Rollout.projected_tool_call).
                # The turn is format-invalid (recorded via is_action_valid); we do NOT end here.
                projected = self.projected_tool_call(agent, action_msg)
                if projected is not None:
                    tool_calls = [projected]
            if not tool_calls:
                # #1 no tool call at all → the run's on_no_tool_call policy.
                control = self.on_no_tool_call
                if control == CONTROL_RAISE:
                    raise RuntimeError(
                        f"Model produced no tool call and on_no_tool_call='raise': {message_text(action_msg)!r}"
                    )
                if control == CONTROL_END:
                    state.finish_reason = FinishReason.NO_TOOL_CALLS
                    break
                # 'continue': no tool ran this turn (no observation to advance on); loop
                # to the next generation. The generation is already recorded as a step.
                continue
            if not self.project_actions and self.message_ends_episode(action_msg):
                # Explicit model end (sentinel / parser status) on a turn that carried a
                # tool call — end without executing it (mirrors ChainRollout). Under
                # ``project_actions`` verl-agent has no such sentinel: it always steps and
                # ends only on env done / max_turns, so this early end is skipped.
                state.finish_reason = FinishReason.TERMINAL
                break

            tool_call = tool_calls[0]
            result = await agent.execute_tool_call(
                context,
                tool_call,
                Messages.from_turns(prompt_messages),
                None,
                state.episode_id,
                depth,
                state.have_set_resources,
            )
            state.have_set_resources = True

            # Attach the env signals (obs / anchor / step_reward / invalid) to the step that
            # already holds this turn's generation. The pre-action grouping anchor is derived
            # by right-shifting these anchors during training conversion.
            gen_step.tool_result = result

            obs_text = result.observation or ""
            state.full_messages.append({"role": "tool", "content": obs_text})
            yield ToolObserved(
                episode_id=state.episode_id,
                depth=depth,
                tool_name=tool_call["function"]["name"],
                observation=result.observation,
                status=result.status,
                image=result.image,
            )

            # Resolve the tool result to a control: the tool's explicit ``control`` wins
            # (e.g. an env tool that reached a terminal state writes ``control="end"``),
            # else the ``status`` diagnosis routes to on_invalid_tool_call / on_tool_error,
            # else a clean result continues. The action ran and its observation is recorded,
            # so this terminal step keeps its ``tool_result``.
            control = self.resolve_tool_control(result)
            if control == CONTROL_RAISE:
                self.raise_for_result(result)
            if control == CONTROL_END:
                state.finish_reason = self.tool_finish_reason(result)
                break

            # Record the just-acted step into history (pre-action obs + the action taken),
            # then advance ``current`` to the resulting state.
            state.history.append(
                {
                    "raw_obs": state.current["raw_obs"],
                    "obs": state.current["obs"],
                    "response": message_text(action_msg),  # full text (chat window)
                    "action": self.extract_action(tool_call),  # clean action (flat builder)
                    "admissible": state.current["admissible"],
                }
            )
            state.current = {
                "raw_obs": result.anchor if result.anchor is not None else obs_text,
                "obs": obs_text,
                "admissible": (result.metrics or {}).get("admissible_actions", []),
                "task_description": state.task_description,
            }
        else:
            state.finish_reason = FinishReason.MAX_TURNS

    # ---- helpers ----------------------------------------------------------
    # ``message_text`` / ``action_format_valid`` (utils.action_format) and
    # ``Rollout.projected_tool_call`` give StepRollout and ChainRollout one definition of
    # validity and projection.

    @staticmethod
    def extract_task(obs: str) -> str:
        """The env's templated goal — verl-agent's ``extract_task``: the text after
        ``"Your task is to: "`` in the (initial) observation. Empty if not present (a
        non-ALFWorld env), so the caller falls back to the dataset task_description.
        """
        marker = "Your task is to: "
        i = obs.find(marker)
        return obs[i + len(marker) :].strip() if i != -1 else ""

    @staticmethod
    def extract_action(tool_call) -> str:
        """The clean action string the tool was invoked with (the ``<action>`` body).

        For env action tools the arguments are ``{"action": "<cmd>"}``; used to render the
        flat verl-agent history (``Action n: '<cmd>'``), which stores the extracted action,
        not the full model response.
        """
        try:
            args = (tool_call or {}).get("function", {}).get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, dict):
                return str(args.get("action") or next(iter(args.values()), ""))
            return str(args or "")
        except Exception:  # noqa: BLE001
            return ""

    def last_user_text(self, messages) -> str:
        turns = messages.messages if hasattr(messages, "messages") else list(messages)
        for msg in reversed(turns):
            if msg.get("role") == "user":
                return message_text(msg)
        # fall back to the last message's text
        return message_text(turns[-1]) if turns else ""
