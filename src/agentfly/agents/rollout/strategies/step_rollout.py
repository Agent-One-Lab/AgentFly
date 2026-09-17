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
import asyncio
import json
import logging
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from ....core import Context
from ....core.context_config import ContextConfig
from ....rewards.reward_base import calculate_reward
from ....tools.types import CONTROL_CONTINUE, CONTROL_END, CONTROL_RAISE
from ...types import RunResult, Segment, Trajectory
from ...utils.messages import Messages, MessagesList
from ....utils.monitor import Monitor
from ..base import Rollout, TOKEN_SAFETY_MARGIN
from ..events import FinishReason
from ..loop.generation import generate_response
from ..loop.metrics import RolloutMetrics, aggregate_tool_metrics, count_tool_calls
from .prompt_builders import resolve_prompt_builder
from ..structures import Step

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
        self.prompt_builder = resolve_prompt_builder(
            prompt_builder, max_prompt_length=max_prompt_length
        )
        self.agent = None
        # Rollout metric reporting (mirrors ChainRollout): emit agent/rollout/* to the
        # monitor once the whole rollout is drained.
        self.metrics = RolloutMetrics()

    # ---- L3: control flow -------------------------------------------------

    def validate_run_args(self, max_turns, num_chains, max_concurrent_chains):
        assert max_turns >= 1, "max_turns must be at least 1."
        assert num_chains >= 1, "num_chains must be at least 1."
        if max_concurrent_chains is not None:
            assert max_concurrent_chains >= 1, "max_concurrent_chains must be >= 1 when set."

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
        """Roll out ``num_chains`` trajectories per task, async-concurrently.

        Returns a :class:`RunResult` of one :class:`~agentfly.agents.types.Trajectory`
        per rollout. Conversations live in ``segments``; runtime ``steps`` provide
        the corresponding environment signals. Trajectories come back in
        ``(group_idx, chain_idx)`` order (the spec order). ``global_step`` sets the
        metrics x-axis (see ``Rollout.resolve_global_step``).
        """
        self.validate_run_args(max_turns, num_chains, max_concurrent_chains)
        self.agent = agent
        Monitor.ensure_started()
        step = self.resolve_global_step(global_step)

        messages_list = MessagesList.from_data(messages)
        # Build the (task x num_chains) trajectory specs, sharing a group uid per task.
        specs: List[Tuple[str, str, int, int, Any, Dict]] = []
        for group_idx, task_messages in enumerate(messages_list):
            uid = str(uuid.uuid4())
            for chain_idx in range(num_chains):
                traj_uid = str(uuid.uuid4())
                meta = task_messages.meta | {
                    "group_id": uid,
                    "group_idx": group_idx,
                    "chain_idx": chain_idx,
                }
                specs.append((traj_uid, uid, group_idx, chain_idx, task_messages, meta))

        sem = (
            asyncio.Semaphore(int(max_concurrent_chains))
            if max_concurrent_chains is not None
            else None
        )

        async def drive(spec) -> Trajectory:
            traj_uid, uid, group_idx, chain_idx, task_messages, meta = spec
            if sem is not None:
                async with sem:
                    return await self._run_trajectory(
                        agent, traj_uid, uid, group_idx, chain_idx, task_messages,
                        meta, max_turns, generation_config, context_config,
                    )
            return await self._run_trajectory(
                agent, traj_uid, uid, group_idx, chain_idx, task_messages,
                meta, max_turns, generation_config, context_config,
            )

        results = await asyncio.gather(*(drive(s) for s in specs), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]

        trajectories = list(results)
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
        traj_uid,
        uid,
        group_idx,
        chain_idx,
        task_messages,
        meta,
        max_turns,
        generation_config,
        context_config,
    ) -> Trajectory:
        """One trajectory's async step-loop. Emits one :class:`Step` per env step and
        returns the per-rollout :class:`Trajectory` (reward = episode outcome)."""
        # Match chain timing: setup through cleanup, excluding semaphore queueing.
        rollout_started_at = time.monotonic()
        context = Context(
            rollout_id=traj_uid,
            group_id=uid,
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
        prepare_first_step = getattr(agent, "prepare_first_step", None)
        if prepare_first_step is not None:
            try:
                await prepare_first_step(context, first_step)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"prepare_first_step failed for {traj_uid}: {e}")
        # Structured pre-action context for the prompt builder. ``current`` is the step the
        # model is about to act on; ``history`` is the prior steps (oldest first). Both carry
        # raw + bundled obs so either builder (chat window / flat verl-agent) can render.
        # task_description = the ENV's templated goal (verl-agent's extract_task: the text
        # after "Your task is to: " in the initial obs), NOT the dataset's human-annotated
        # task_description — verl-agent uses goal_desc_human_anns_prob=0.0 (templated goals).
        initial_obs = context.metadata.get("_flat_initial_obs", "")
        task_description = (
            context.metadata.get("_flat_task_description")      # set by the env hook (WebShop)
            or self._extract_task(str(initial_obs))              # ALFWorld: "Your task is to: …"
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
                agent, None, gen_step, tools, step, traj_uid,
                generation_config, context,
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
                        "Model produced no tool call and on_no_tool_call='raise': "
                        f"{self._message_text(action_msg)!r}"
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
                context, tool_call, Messages.from_turns(prompt_messages),
                None, traj_uid, step, have_set_resources,
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
            history.append({
                "raw_obs": current["raw_obs"],
                "obs": current["obs"],
                "response": self._message_text(action_msg),   # full text (chat window)
                "action": self._extract_action(tool_call),     # clean action (flat builder)
                "admissible": current["admissible"],
            })
            current = {
                "raw_obs": result.anchor if result.anchor is not None else obs_text,
                "obs": obs_text,
                "admissible": (result.metrics or {}).get("admissible_actions", []),
                "task_description": task_description,
            }
        else:
            finish_reason = FinishReason.MAX_TURNS

        # Episode outcome (reuse the L2 reward path). Scale 1.0.
        outcome: Optional[float] = None
        reward_extras: Dict[str, Any] = {}
        if agent._reward_fn is not None:
            context.trajectory = full_messages
            final_response = agent.extract_final_response(full_messages)
            context.final_response = final_response
            reward = await calculate_reward(
                agent._reward_fn,
                final_response=final_response,
                trajectory=full_messages,
                id=traj_uid,
                context=context,
                **{k: v for k, v in meta.items() if k not in ("final_response", "trajectory", "id")},
            )
            outcome = float(getattr(reward, "reward", reward) or 0.0)
            reward_extras = dict(getattr(reward, "metrics", {}) or {})

        # Match chain's merge precedence: tool aggregates own the tool/* namespace.
        reward_extras.update(aggregate_tool_metrics(step.tool_result for step in steps))

        await context.release_resource(scope="global")
        await context.end_resource(scope="rollout")
        rollout_time_sec = time.monotonic() - rollout_started_at

        # One segment PER GENERATION (each the bounded prompt + response) — so
        # inspection of segments shows the per-step
        # structure used by the standalone training converter, not a chain-like blob.
        # (The episode reward above used ``full_messages``; segments are the training
        # units.)
        return Trajectory(
            segments=(
                [Segment(messages=s.messages.messages) for s in steps]
                if steps else [Segment(messages=[])]
            ),
            reward=outcome,
            metrics=reward_extras,
            group_id=uid,
            chain_id=traj_uid,
            group_idx=group_idx,
            chain_idx=chain_idx,
            finish_reason=finish_reason,
            rollout_time_sec=rollout_time_sec,
            num_turns=len(steps),
            tool_call_counts=count_tool_calls(s.tool_result for s in steps),
            # Keep task data, not context scratch state or duplicated typed fields.
            metadata={k: v for k, v in meta.items() if k not in self._RESERVED_INFO_KEYS},
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
        return obs[i + len(marker):].strip() if i != -1 else ""

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
