import fcntl
import json
import os
import http.client
import yaml
from ....__init__ import AGENT_CONFIG_DIR
from ....envs.redis_env import RedisEnv
import timeout_decorator
import requests
import redis
from ...tool_base import tool
from ....utils.logging import get_logger
import time
import httpx

# LOGGER = get_logger("agents/tools/data/search", "google_search", level="INFO")
# HIT_MISS = {"hit": 0, "miss": 0}

# class RedisClient:
#     def __init__(self, host="localhost", port=6379, db=0):
#         self.redis_client = redis.Redis(host=host, port=port, db=db)

#     def get(self, key):
#         return self.redis_client.get(key).decode("utf-8")

#     def set(self, key, value):
#         self.redis_client.set(key, value)

#     def exists(self, key):
#         return self.redis_client.exists(key)

#     def close(self):
#         self.redis_client.close()

# REDIS_CLIENT = RedisClient()

# current_dir = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(current_dir, "..", "..", "..", "..", "agents", "tools", "configs", "search.yaml")
config_path = os.path.join(AGENT_CONFIG_DIR, "search.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    SERPER_API_KEY = config["SERPER_API_KEY"]
    GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
    CUSTOM_SEARCH_ENGINE_ID = config["CUSTOM_SEARCH_ENGINE_ID"]


@timeout_decorator.timeout(10)  # 10 seconds timeout
def req(query, n=10):
    global SERPER_API_KEY

    # This is not actually asynchronous, but we leave it because serper api's rate limitation
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": query,
        "num": n
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))


async def req_async(query, n=10):
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": n
    }
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        return response.json()


@tool(name="google_search", description="Get relevant information from Google Search.", stateful=True, env_cls=RedisEnv, pool_size=1)
async def google_search_serper(query, env: RedisEnv):
    """
    Get relevant information from Google Search. Implemented by Serper API.
    Args:
        query (str): The query to search for.
    Returns:
        str: The context from the search results based on snippets.
    """
    if env._exists(query):
        result = await env.step(query)
        return result
    else:
        result = None
        for i in range(1):
            try:
                n = 10
                result = await req_async(query, n)
            except Exception as e:
                time.sleep(2)
                result = str(e)
                continue
        if result is None:
            raise Exception("Google Search Error: No result found")

        if isinstance(result, dict):
            if "answerBox" in result and result["answerBox"]["answer"] != "":
                context = f"{result['answerBox']['answer']}"
            elif "answerBox" in result and result["answerBox"]["snippet"] != "":
                context = f"{result['answerBox']['snippet']}"
            else:
                results = result.get("organic", [])[:3] # Choose top 3 result
                snippets = [f"{i+1}. {x['title']} {x['snippet']}" for i, x in enumerate(results)]
                context = "\n".join(snippets)

            env._set(query, context)
            return context
        else:
            return result