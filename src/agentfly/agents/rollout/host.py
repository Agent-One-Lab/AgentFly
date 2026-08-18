"""The contract between the rollout loop and its agent.

``ChainRollout`` is a mixin: ``BaseAgent(ChainRollout, ABC)`` provides the model/tooling
state and the template-method hooks the loop calls (``generate_async``, ``parse``, …),
and subclasses override those hooks. That contract used to be implicit — the loop just
read ``self.tokenizer`` / ``self.generate_async`` / … with no declared interface, so you
couldn't tell from ``chain.py`` what a host must supply.

``RolloutHost`` makes it explicit and statically checkable. It is a typing artifact only:
the loop stays a mixin (no runtime change), but the host's required surface is now named
in one place. If we ever switch from inheritance to composition (injecting a host instead
of subclassing), this Protocol is already the interface to inject against.

Members are exactly what ``ChainRollout`` reads from ``self`` but does **not** define
itself. Hooks the loop *defines with a default and subclasses override*
(``validate_tool_call``, ``_execute_tool_call``, ``_maybe_append_context_trigger_user_message``,
``_skills_payload``) are **not** here — they belong to the loop, not the host.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class RolloutHost(Protocol):
    # --- model / tooling state the loop reads ---
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

    # --- template-method hooks the loop calls (subclasses override) ---
    async def generate_async(self, messages_list: List[List[Dict]], **kwargs) -> Any:
        """Produce model responses for a batch of message lists."""
        ...

    def parse(
        self,
        responses: List[str],
        tool_calls: Optional[List] = None,
        context: Any = None,
        **kwargs,
    ) -> List[Dict]:
        """Turn raw response text (+ optional backend tool calls) into assistant messages."""
        ...

    def extract_final_response(self, trajectory: List[Dict]) -> Any:
        """Extract the final answer from a flattened trajectory (used for reward)."""
        ...
