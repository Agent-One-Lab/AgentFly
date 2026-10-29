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
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from ....core.context_config import ContextConfig
from ....tools.types import CONTROL_CONTINUE, CONTROL_END, CONTROL_RAISE
from ....utils.monitor import Monitor
from ...types import RunResult, Segment, Trajectory
from ...utils.messages import Messages
from ..base import Rollout, TrajectorySpec
from ..events import FinishReason
from ..loop.generation import generate_response
from ..loop.metrics import RolloutMetrics, aggregate_tool_metrics, count_tool_calls
from ..structures import Step
from .prompt_builders import resolve_prompt_builder

logger = logging.getLogger(__name__)


class StepRollout(Rollout):
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
        self.agent = None
        # Rollout metric reporting (mirrors ChainRollout): emit agent/rollout/* to the
        # monitor once the whole rollout is drained.
        self.metrics = RolloutMetrics()

    # ---- L3: control flow -------------------------------------------------

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
        """Roll out ``num_chains`` trajectories per task, async-concurrently.

        Returns a :class:`RunResult` of one :class:`~agentfly.agents.types.Trajectory`
        per rollout. Conversations live in ``segments``; runtime ``steps`` provide
        the corresponding environment signals. Trajectories come back in
        ``(group_idx, chain_idx)`` order (the spec order). ``global_step`` sets the
        metrics x-axis (see ``Rollout.resolve_global_step``). ``use_token_ids`` see
        ``Rollout.run``. ``show_progress`` renders a progress bar over finished trajectories.
        """
        self.validate_run_args(max_turns, num_chains, max_concurrent_chains)
        self.agent = agent
        self.use_token_ids = bool(use_token_ids)
        Monitor.ensure_started()
        step = self.resolve_global_step(global_step)

        specs = self.make_specs(messages, num_chains)
        results: Dict[str, Trajectory] = {}

        async def run_one(spec: TrajectorySpec) -> None:
            results[spec.chain_id] = await self._run_trajectory(
                agent,
                spec,
                max_turns=max_turns,
                generation_config=generation_config,
                context_config=context_config,
            )

        async for _ev in self.run_all(
            specs,
            run_one,
            max_concurrent_chains=max_concurrent_chains,
            show_progress=show_progress,
        ):
            pass

        trajectories = [results[spec.chain_id] for spec in specs]
        # Shared end-of-rollout reporting reads the completed trajectories, including
        # episode durations for timing summaries and slowest-trajectory logging.
        self.metrics.record_step(
            global_step=step,
            trajectories=trajectories,
        )
        return RunResult(trajectories=trajectories, rollout="step")

    async def _run_trajectory(
        self,
        agent,
        spec: TrajectorySpec,
        *,
        max_turns: int,
        generation_config: Optional[Dict[str, Any]],
        context_config: Optional[ContextConfig],
    ) -> Trajectory:
        """One trajectory's async step-loop. Emits one :class:`Step` per env step and
        returns the per-rollout :class:`Trajectory` (reward = episode outcome)."""
        # Match chain timing: setup through cleanup, excluding semaphore queueing.
        rollout_started_at = time.monotonic()
        traj_uid = spec.chain_id
        task_messages = spec.messages
        # Task metadata and the group ids; the chain id is not passed to the context
        # metadata or the reward function (unlike ChainRollout).
        meta = {k: v for k, v in spec.info.items() if k != "chain_id"}
        context = self.new_context(
            chain_id=traj_uid,
            group_id=spec.group_id,
            metadata=meta,
            context_config=context_config,
        )

        # Seed the initial observation from the env (env-source-of-truth agents reset
        # the env and inject the first obs). We reuse the same hook as ChainRollout;
        # the initial obs text is read back from the seeded first step.
        first_step = Step(
            type="Action Input",
            messages=Messages.from_turns(deepcopy(task_messages.messages)),
        )
        await self.run_first_step_hook(agent, context, first_step, traj_uid)
        # Structured pre-action context for the prompt builder. ``current`` is the step the
        # model is about to act on; ``history`` is the prior steps (oldest first). Both carry
        # raw + bundled obs so either builder (chat window / flat verl-agent) can render.
        # task_description = the ENV's templated goal (verl-agent's extract_task: the text
        # after "Your task is to: " in the initial obs), NOT the dataset's human-annotated
        # task_description — verl-agent uses goal_desc_human_anns_prob=0.0 (templated goals).
        initial_obs = context.metadata.get("_flat_initial_obs", "")
        task_description = (
            context.metadata.get("_flat_task_description")  # set by the env hook (WebShop)
            or self._extract_task(str(initial_obs))  # ALFWorld: "Your task is to: …"
            or meta.get("task_description", "")
        )
        current: Dict[str, Any] = {
            "raw_obs": context.metadata.get("_flat_initial_obs", ""),
            "obs": self._last_user_text(first_step.messages),
            "admissible": context.metadata.get("_flat_initial_admissible", []),
            "task_description": task_description,
        }
        history: List[Dict[str, Any]] = []
        full_messages: List[Dict] = list(first_step.messages.messages)  # for the outcome reward
        steps: List[Step] = []
        finish_reason: Optional[FinishReason] = None
        have_set_resources = False
        # Tool schemas, exactly as ChainRollout builds them — so per-step generation
        # goes through the *same* path (tools / tool_call_source / skills / parse with
        # trajectory_segments). StepRollout must differ from ChainRollout only in
        # history length + per-step layout, never in how it talks to the model.
        # Prompt tools come from the agent's single source of truth (also used by
        # tokenize_trajectories), so generation and training render one prompt.
        tools = agent.prompt_tools()

        for step in range(max_turns):
            prompt_messages = self.prompt_builder.build(
                agent,
                history=history,
                current=current,
                history_length=self.history_length,
                step_index=step,
            )
            # Reuse the chain generation helper (return_dict / tools / tool_call_source /
            # skills / trajectory_segments / raw-field overlay). ``chain``/``depth``/
            # ``chain_id`` are unused by it; pass None / step / traj_uid.
            gen_step = Step(
                type="Action Input",
                messages=Messages.from_turns(prompt_messages),
            )
            action_msg, _ = await generate_response(
                agent,
                None,
                gen_step,
                tools,
                step,
                traj_uid,
                generation_config,
                context,
                use_token_ids=self.use_token_ids,
            )
            full_messages.append(action_msg)

            # Record the generation into the trajectory the moment it is produced — the same
            # way ChainRollout adds every generated message as a step before it decides whether
            # to run a tool. ``messages`` = the bounded prompt this turn ending in the action;
            # ``tool_result`` is attached below only if an action actually executes. A terminal
            # / no-action turn keeps ``tool_result=None`` (a generation-only step), so no part
            # of the trajectory is ever lost and every trajectory has >=1 step.
            gen_step.messages = Messages.from_turns(list(prompt_messages) + [action_msg])
            steps.append(gen_step)

            # Control decisions run through the SAME base machinery as ChainRollout.
            tool_calls = action_msg.get("tool_calls") or []
            if not tool_calls and self.project_actions:
                # verl-agent projection: no ``<action>`` block was extracted → step the env
                # anyway with the last-30-chars garbage fallback (see Rollout._projected_tool_call).
                # The turn is format-invalid (recorded via is_action_valid); we do NOT end here.
                projected = self._projected_tool_call(agent, action_msg)
                if projected is not None:
                    tool_calls = [projected]
            if not tool_calls:
                # #1 no tool call at all → the run's on_no_tool_call policy.
                control = self.on_no_tool_call
                if control == CONTROL_RAISE:
                    raise RuntimeError(
                        f"Model produced no tool call and on_no_tool_call='raise': {self._message_text(action_msg)!r}"
                    )
                if control == CONTROL_END:
                    finish_reason = FinishReason.NO_TOOL_CALLS
                    break
                # 'continue': no tool ran this turn (no observation to advance on); loop
                # to the next generation. The generation is already recorded as a step.
                continue
            if not self.project_actions and self.message_ends_episode(action_msg):
                # Explicit model end (sentinel / parser status) on a turn that carried a
                # tool call — end without executing it (mirrors ChainRollout). Under
                # ``project_actions`` verl-agent has no such sentinel: it always steps and
                # ends only on env done / max_turns, so this early end is skipped.
                finish_reason = FinishReason.TERMINAL
                break

            tool_call = tool_calls[0]
            result = await agent.execute_tool_call(
                context,
                tool_call,
                Messages.from_turns(prompt_messages),
                None,
                traj_uid,
                step,
                have_set_resources,
            )
            have_set_resources = True

            # Attach the env signals (obs / anchor / step_reward / invalid) to the step that
            # already holds this turn's generation. The pre-action grouping anchor is derived
            # by right-shifting these anchors during training conversion.
            gen_step.tool_result = result

            obs_text = result.observation or ""
            full_messages.append({"role": "tool", "content": obs_text})

            # Resolve the tool result to a control: the tool's explicit ``control`` wins
            # (e.g. an env tool that reached a terminal state writes ``control="end"``),
            # else the ``status`` diagnosis routes to on_invalid_tool_call / on_tool_error,
            # else a clean result continues. The action ran and its observation is recorded,
            # so this terminal step keeps its ``tool_result``.
            control = self.resolve_tool_control(result)
            if control == CONTROL_RAISE:
                self.raise_for_result(result)
            if control == CONTROL_END:
                finish_reason = self.tool_finish_reason(result)
                break

            # Record the just-acted step into history (pre-action obs + the action taken),
            # then advance ``current`` to the resulting state.
            history.append(
                {
                    "raw_obs": current["raw_obs"],
                    "obs": current["obs"],
                    "response": self._message_text(action_msg),  # full text (chat window)
                    "action": self._extract_action(tool_call),  # clean action (flat builder)
                    "admissible": current["admissible"],
                }
            )
            current = {
                "raw_obs": result.anchor if result.anchor is not None else obs_text,
                "obs": obs_text,
                "admissible": (result.metrics or {}).get("admissible_actions", []),
                "task_description": task_description,
            }
        else:
            finish_reason = FinishReason.MAX_TURNS

        # Episode outcome on the full (unwindowed) conversation.
        reward = await self.compute_reward(
            agent,
            context,
            chain_id=traj_uid,
            trajectory=full_messages,
            final_messages=full_messages,
            reward_args=meta,
        )
        tool_metrics = aggregate_tool_metrics(step.tool_result for step in steps)

        await self.release_resources(context)
        rollout_time_sec = time.monotonic() - rollout_started_at

        # One segment PER GENERATION (each the bounded prompt + response) — so
        # inspection of segments shows the per-step
        # structure used by the standalone training converter, not a chain-like blob.
        # (The episode reward above used ``full_messages``; segments are the training
        # units.)
        return self.make_trajectory(
            info=spec.info,
            segments=([Segment(messages=s.messages.messages) for s in steps] if steps else [Segment(messages=[])]),
            reward=reward,
            tool_metrics=tool_metrics,
            finish_reason=finish_reason,
            rollout_time_sec=rollout_time_sec,
            num_turns=len(steps),
            tool_call_counts=count_tool_calls(s.tool_result for s in steps),
            steps=steps,
        )

    # ---- helpers ----------------------------------------------------------
    # ``_message_text`` / ``_action_format_valid`` / ``_projected_tool_call`` live on the
    # base ``Rollout`` so StepRollout and ChainRollout share one definition of validity
    # and projection.

    @staticmethod
    def _extract_task(obs: str) -> str:
        """The env's templated goal — verl-agent's ``extract_task``: the text after
        ``"Your task is to: "`` in the (initial) observation. Empty if not present (a
        non-ALFWorld env), so the caller falls back to the dataset task_description.
        """
        marker = "Your task is to: "
        i = obs.find(marker)
        return obs[i + len(marker) :].strip() if i != -1 else ""

    @staticmethod
    def _extract_action(tool_call) -> str:
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

    def _last_user_text(self, messages) -> str:
        turns = messages.messages if hasattr(messages, "messages") else list(messages)
        for msg in reversed(turns):
            if msg.get("role") == "user":
                return self._message_text(msg)
        # fall back to the last message's text
        return self._message_text(turns[-1]) if turns else ""
