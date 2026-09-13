"""Step-prompt builders for :class:`~agentfly.agents.rollout.strategies.step_rollout.StepRollout`.

A ``StepPromptBuilder`` renders **one** step's model input from the bounded recent
history plus the current (pre-action) observation. It is the pluggable seam that lets
the single-turn rollout use different prompt formats:

- :class:`ChatWindowPromptBuilder` keeps AgentFly's native chat format and simply
  truncates to the last N turns — so a ``StepRollout`` differs from a ``ChainRollout``
  only in *history length* + *per-step layout* (a clean controlled comparison).
- :class:`FlatAlfworldPromptBuilder` reproduces **verl-agent's** ALFWorld prompt exactly:
  a single flat user message rendered from ``ALFWORLD_TEMPLATE`` (no system role), for a
  faithful reproduction of their GiGPO training.

Both consume the same structured per-step inputs (``history`` / ``current`` dicts) so the
builder is the only thing that changes between prompt formats. ``build`` returns a list of
chat-message dicts (the chat builder returns a real conversation; the flat builder returns
a single ``[{"role": "user", "content": ...}]``).

Builders are resolved by the same name-or-path idiom as rollouts/tools/rewards.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ....utils.references import resolve_reference


# ---- verl-agent ALFWorld templates (verbatim; agent_system/environments/prompts/alfworld.py) --
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


# ---- verl-agent WebShop templates (verbatim; agent_system/environments/prompts/webshop.py) ---
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


class StepPromptBuilder(ABC):
    """Render the model input for a single step of a bounded-history rollout.

    ``history`` and ``current`` are structured per-step dicts. A ``history`` entry (oldest
    first) carries ``{"raw_obs", "obs", "response", "action", "admissible"}``; ``current``
    carries ``{"raw_obs", "obs", "admissible", "task_description"}`` for the pre-action step
    now. ``raw_obs`` is the undecorated env observation; ``obs`` is the LLM-facing bundled
    string (obs + admissible menu); ``response`` is the full assistant text; ``action`` is
    the clean extracted action. A builder reads only the fields it needs (the chat window
    uses ``obs``/``response``; the flat ALFWorld builder uses ``raw_obs``/``action``).
    """

    @abstractmethod
    def build(
        self,
        agent,
        *,
        history: List[Dict[str, Any]],
        current: Dict[str, Any],
        history_length: int,
        step_index: int = 0,
    ) -> List[Dict[str, Any]]:
        ...


class ChatWindowPromptBuilder(StepPromptBuilder):
    """AgentFly-native bounded chat window.

    ``[system] + last-N (user=obs, assistant=action) turns + [user=current_obs]``, where
    ``obs`` is the bundled observation. Drops the oldest ``(obs, action)`` pairs until the
    rendered prompt fits ``max_prompt_length`` (best-effort; uses the agent's tokenizer when
    available, else a rough char estimate).
    """

    def __init__(self, max_prompt_length: int = 2048):
        self.max_prompt_length = max_prompt_length

    def _messages(self, agent, history, current, history_length):
        msgs: List[Dict[str, Any]] = []
        system_prompt = getattr(agent, "system_prompt", None)
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        window = history[-history_length:] if history_length and history_length > 0 else []
        for h in window:
            msgs.append({"role": "user", "content": h["obs"]})
            msgs.append({"role": "assistant", "content": h["response"]})
        msgs.append({"role": "user", "content": current["obs"]})
        return msgs

    def build(self, agent, *, history, current, history_length, step_index=0):
        n = history_length
        msgs = self._messages(agent, history, current, n)
        while n > 0 and self._too_long(agent, msgs):
            n -= 1
            msgs = self._messages(agent, history, current, n)
        return msgs

    def _prompt_token_count(self, agent, msgs) -> int:
        tokenizer = getattr(agent, "tokenizer", None)
        if tokenizer is not None:
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                return len(tokenizer(text)["input_ids"])
            except Exception:  # noqa: BLE001
                pass
        return sum(len(str(m.get("content", ""))) for m in msgs) // 4

    def _too_long(self, agent, msgs) -> bool:
        if not self.max_prompt_length:
            return False
        return self._prompt_token_count(agent, msgs) > self.max_prompt_length


def _normalize_obs(obs: Any) -> str:
    """Return the plain observation text.

    AgentFly's ALFWorld env sometimes yields the observation as a 1-tuple ``(text,)``
    (so ``str(obs)`` is ``"('text',)"``); verl-agent uses the bare string. Unwrap it so the
    rendered prompt matches byte-for-byte.
    """
    if isinstance(obs, (tuple, list)) and len(obs) == 1:
        return str(obs[0])
    return str(obs)


class FlatAlfworldPromptBuilder(StepPromptBuilder):
    """verl-agent's ALFWorld prompt, reproduced exactly.

    One flat user message from ``ALFWORLD_TEMPLATE`` (``_NO_HIS`` at step 0). No system role.
    History renders as ``[Observation {n}: '{raw_obs}', Action {n}: '{action}']`` (1-based
    absolute step numbers), admissible actions as newline-joined ``'cmd'`` minus ``help``.
    """

    def __init__(self, max_prompt_length: int = 2048):
        self.max_prompt_length = max_prompt_length

    @staticmethod
    def _fmt_admissible(admissible) -> str:
        # verl-agent: "\n ".join(f"'{s}'" for s in admissible if s != 'help')
        return "\n ".join(f"'{s}'" for s in (admissible or []) if s != "help")

    def build(self, agent, *, history, current, history_length, step_index=0):
        window = history[-history_length:] if history_length and history_length > 0 else []
        cur_obs = _normalize_obs(current.get("raw_obs", ""))
        admissible = self._fmt_admissible(current.get("admissible"))

        if not window:
            text = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=cur_obs,
                admissible_actions=admissible,
            )
        else:
            # Absolute 1-based step numbers, matching verl-agent's SimpleMemory.fetch:
            # window[j] is step (step_index - len(window) + j), rendered +1.
            start = step_index - len(window)
            lines = [
                "[Observation {n}: '{obs}', Action {n}: '{act}']".format(
                    n=start + j + 1,
                    obs=_normalize_obs(h.get("raw_obs", "")),
                    act=h.get("action", ""),
                )
                for j, h in enumerate(window)
            ]
            text = ALFWORLD_TEMPLATE.format(
                task_description=current.get("task_description", ""),
                step_count=step_index,
                history_length=len(window),
                action_history="\n".join(lines),
                current_step=step_index + 1,
                current_observation=cur_obs,
                admissible_actions=admissible,
            )
        return [{"role": "user", "content": text}]



def _history_lines(window, step_index: int) -> str:
    """verl-agent's ``SimpleMemory.fetch`` rendering: one ``[Observation n: '…', Action n: '…']``
    line per remembered step, absolute 1-based step numbers, newline-joined."""
    start = step_index - len(window)
    return "\n".join(
        "[Observation {n}: '{obs}', Action {n}: '{act}']".format(
            n=start + j + 1, obs=h.get("raw_obs", ""), act=h.get("action", "")
        )
        for j, h in enumerate(window)
    )


class FlatWebshopPromptBuilder(StepPromptBuilder):
    """verl-agent's WebShop prompt, reproduced exactly.

    One flat user message from ``WEBSHOP_TEMPLATE`` (``_NO_HIS`` at step 0). No system role.
    ``current_observation`` / history observations are the verl-agent-formatted page (the
    tool's ``anchor``: quoted ``[SEP]`` parts after the instruction); ``available_actions``
    is the raw ``admissible_actions`` list rendered one quoted action per line.
    """

    def __init__(self, max_prompt_length: int = 4096):
        self.max_prompt_length = max_prompt_length

    @staticmethod
    def _fmt_available(actions) -> str:
        # verl-agent: "\n".join(f"'{s}'," for s in available_actions)
        return "\n".join(f"'{a}'," for a in (actions or []))

    def build(self, agent, *, history, current, history_length, step_index=0):
        window = history[-history_length:] if history_length and history_length > 0 else []
        cur_obs = current.get("raw_obs", "")
        available = self._fmt_available(current.get("admissible"))
        if not window:
            text = WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=current.get("task_description", ""),
                current_observation=cur_obs,
                available_actions=available,
            )
        else:
            text = WEBSHOP_TEMPLATE.format(
                task_description=current.get("task_description", ""),
                step_count=step_index,
                history_length=len(window),
                action_history=_history_lines(window, step_index),
                current_step=step_index + 1,
                current_observation=cur_obs,
                available_actions=available,
            )
        return [{"role": "user", "content": text}]


_PROMPT_BUILDER_REGISTRY = {
    "chat_window": ChatWindowPromptBuilder,
    "alfworld_flat": FlatAlfworldPromptBuilder,
    "webshop_flat": FlatWebshopPromptBuilder,
}


def register_prompt_builder(name: str, builder_cls) -> None:
    """Register a step-prompt builder class under ``name`` (case-insensitive)."""
    _PROMPT_BUILDER_REGISTRY[name.lower()] = builder_cls


def resolve_prompt_builder(spec, **kwargs) -> StepPromptBuilder:
    """Resolve ``spec`` into a ``StepPromptBuilder`` instance.

    - a ``StepPromptBuilder`` instance passes through unchanged;
    - a string is a registered name (``"chat_window"`` / ``"alfworld_flat"``) OR an import
      reference (``"pkg.mod:MyBuilder"``) resolved via ``references.resolve_reference`` and
      constructed with ``**kwargs`` (e.g. ``max_prompt_length``).
    """
    if isinstance(spec, StepPromptBuilder):
        return spec
    builder_cls = resolve_reference(spec, _PROMPT_BUILDER_REGISTRY, "prompt_builder")
    return builder_cls(**kwargs)
