import logging
from typing import Optional

from ..agent_base import BaseAgent

logger = logging.getLogger(__file__)


class HFAgent(BaseAgent):
    def __init__(
        self,
        model_name_or_path: str,
        tool_parser_name: Optional[str] = "hermes",
        **kwargs,
    ):
        super().__init__(
            model_name_or_path, tool_parser_name=tool_parser_name, **kwargs
        )

    # def parse(self, responses: List[str], **kwargs) -> List[Dict]:
    #     logger.debug(f"[HFAgent] Responses: {responses}")
    #     new_messages_list = []
    #     for response in responses:
    #         tool_calls = extract_tool_calls(response)

    #         formatted_tool_calls = []
    #         if len(tool_calls) == 1:
    #             tool_call = tool_calls[0]
    #             try:
    #                 tool_call = json.loads(tool_call)
    #                 # {"name": "...", "arguments": "..."}
    #                 if "name" in tool_call and "arguments" in tool_call:
    #                     name = tool_call["name"]
    #                     arguments = tool_call["arguments"]

    #                 formatted_tool_calls.append({
    #                     "id": None,
    #                     "type": "function",
    #                     "function": {
    #                         "name": name,
    #                         "arguments": arguments,
    #                     }
    #                 })
    #             except:
    #                 pass
    #         message = {
    #             "role": "assistant",
    #             "content": [{"type": "text", "text": response}],
    #             "tool_calls": formatted_tool_calls,
    #             "loss": True
    #         }
    #         new_messages_list.append(message)
    #     return new_messages_list
