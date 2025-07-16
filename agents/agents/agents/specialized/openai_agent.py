import os
from typing import Any, Dict, List
from openai import OpenAI, AzureOpenAI
import httpx

from ...tools.tool_base import tool
from ..agent_base import BaseAgent
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import json
from multiprocess import Pool

class OpenAIAgent(BaseAgent):
    def __init__(
            self,
            api_key="",
            **kwargs
        ):
        wrapper = kwargs.get("wrapper", False)
        if not wrapper:
            kwargs["wrapper"] = True
        template = kwargs.get("template", "openai")
        kwargs["template"] = template
        super(OpenAIAgent, self).__init__(**kwargs)
        model_name_or_path = kwargs.get("model_name_or_path", "gpt-3.5-turbo")
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.model = model_name_or_path

    def parse(self, messages_list: List[List[Dict]], tools: List[Any], **args):
        # OpenAI use 'n' to specify the number of return sequences
        num_return_sequences = args.get("num_return_sequences", 1)
        tool_schemas = [tool.schema for tool in tools]
        del args["num_return_sequences"]
        args["n"] = num_return_sequences

        def process_message(message_data):
            message, api_key, model, tool_schemas, args = message_data
            client = OpenAI(api_key=api_key)
            try:
                json_data = {
                    "model": model,
                    "messages": message,
                    **args
                }
                if tool_schemas is not None:
                    json_data.update({"tools": tool_schemas})

                openai_response = client.chat.completions.create(**json_data)
                result = openai_response.dict()
                new_message = result['choices'][0]['message']
                new_message["loss"] = True
                return new_message
            except Exception as e:
                print(f"Parsing Exception: {repr(e)}. Try again.")
                return {
                    "role": "assistant",
                    "content": result,
                    "tool_calls": [],
                    "loss": True
                }

        # Prepare data for each message
        message_data_list = [
            (message, self.api_key, self.model, tool_schemas, args)
            for message in messages_list
        ]

        pool = Pool()
        results = pool.map(process_message, message_data_list)
        pool.close()
        pool.join()

        return results
    
    
    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(
        self,
        messages,
        tools=None,
        tool_choice=None,
        model=None,
        stop=None,
        client=None,
        **args
    ):
        if model is None:
            model = self.model
        if client is None:
            client = self.client
            
        json_data = {
            "model": model,
            "messages": messages,
            **args
        }
        if stop is not None:
            json_data.update({"stop": stop})
        if tools is not None:
            json_data.update({"tools": tools})
        if tool_choice is not None:
            json_data.update({"tool_choice": tool_choice})

        try:
            # We use chat completion API
            openai_response = client.chat.completions.create(**json_data)
            json_data = openai_response.dict()
            return json_data
        except Exception as e:
            print(f"Unable to generate ChatCompletion response: {e}")
            raise e


@tool()
def get_current_weather(location: str, unit: str="fahrenheit"):
    """
    Get the current weather in a given location
    """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


if __name__ == "__main__":
    agent = OpenAIAgent(
        model_name_or_path="gpt-3.5-turbo",
        api_key="",
        tools=[get_current_weather]
    )
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    agent.run(
        max_steps=3,
        start_messages=messages,
        num_chains=1
    )
    trajectories = agent.trajectories
    print(trajectories[0]["messages"])

    
