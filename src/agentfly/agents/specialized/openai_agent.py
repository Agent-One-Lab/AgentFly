import asyncio
import json
from typing import Any, Dict, List, Union

from openai import OpenAI

from ...tools import answer_qa
from ...tools.decorator import tool
from ..agent_base import BaseAgent
from ..llm_backends.backend_configs import ClientConfig


class OpenAIAgent(BaseAgent):
    def __init__(self, api_key="", base_url="https://api.openai.com/v1", **kwargs):
        assert api_key is not None and api_key != "", "API key is required"
        backend = kwargs.get("backend", "client")
        assert backend == "client", "OpenAI agent only supports client backend"
        kwargs["backend"] = backend

        # Create client-specific configuration
        client_config = ClientConfig(
            api_key=api_key,
            base_url=base_url,
            max_requests_per_minute=kwargs.get("max_requests_per_minute", 100),
            timeout=kwargs.get("timeout", 600),
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 1.0),
        )
        kwargs["backend_config"] = client_config

        # Initialize the base class
        super(OpenAIAgent, self).__init__(**kwargs)

        model_name_or_path = kwargs.get("model_name_or_path", "gpt-3.5-turbo")
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.model = model_name_or_path

        # For OpenAI models, we don't need a tokenizer for the LLM engine, but we still need one for trajectory processing
        if self.backend == "client":
            self.tokenizer = None
            self.processor = None

    async def generate_async(self, messages_list_or_inputs: List[List[Dict]], **args):
        responses = await super().generate_async(
            messages_list_or_inputs, tool_choice="auto", **args
        )
        return responses

    def parse(
        self, responses: Union[Dict[str, List], List[str]], tools: List[Any], **args
    ) -> List[Dict]:
        """
        Parse responses into the correct message format.

        Args:
            responses: List of response strings from the LLM.
            tools: List of tools available to the agent.
            **args: Additional arguments for parsing.

        Returns:
            List of assistant messages in the correct format.
        """

        new_messages = []
        if isinstance(responses, dict):
            for response, tool_calls in zip(
                responses["response_texts"], responses["tool_calls"]
            ):
                new_message = {"role": "assistant"}

                new_message["content"] = response
                if len(tool_calls) > 0:
                    tool_calls = tool_calls[:1]
                    new_message["tool_calls"] = tool_calls

                new_messages.append(new_message)
        elif isinstance(responses, list):
            for content in responses:
                new_message = {"role": "assistant", "content": content}
                new_messages.append(new_message)
        else:
            raise ValueError(f"Invalid responses type: {type(responses)}")

        return new_messages

    # @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))
    # def chat_completion_request(
    #     self,
    #     messages,
    #     tools=None,
    #     tool_choice=None,
    #     model=None,
    #     stop=None,
    #     client=None,
    #     **args
    # ):
    #     if model is None:
    #         model = self.model
    #     if client is None:
    #         client = self.client

    #     json_data = {
    #         "model": model,
    #         "messages": messages,
    #         **args
    #     }
    #     if stop is not None:
    #         json_data.update({"stop": stop})
    #     if tools is not None:
    #         json_data.update({"tools": tools})
    #     if tool_choice is not None:
    #         json_data.update({"tool_choice": tool_choice})

    #     try:
    #         # We use chat completion API
    #         openai_response = client.chat.completions.create(**json_data)
    #         json_data = openai_response.dict()
    #         return json_data
    #     except Exception as e:
    #         print(f"Unable to generate ChatCompletion response: {e}")
    #         raise e


@tool()
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """
    Get the current weather in a given location
    """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


if __name__ == "__main__":
    agent = OpenAIAgent(
        model_name_or_path="gpt-4o-mini",
        api_key="OpenAI API Key",
        tools=[get_current_weather, answer_qa],
    )
    messages = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
                }
            ]
        }
    ]
    asyncio.run(agent.run_async(max_steps=5, start_messages=messages, num_chains=1))
    trajectories = agent.trajectories
    print(f"""Trajectory: {trajectories[0]["messages"]}""")
