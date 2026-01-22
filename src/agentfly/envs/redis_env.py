import subprocess
import redis
import socket
import asyncio
import os
from .env_base import BaseEnv

global_redis_env = None


class RedisEnv(BaseEnv):
    def __init__(self, host: str = "localhost", port: int = None, db: int = 0):
        self.process = None
        self.host = host
        self.port = port  # If None, a free port will be detected
        self.db = db
        self.client = None

    def _find_free_port(self):
        """Find a free port on the system."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    async def start(self):
        global AGENT_DATA_DIR
        global AGENT_CONFIG_DIR
        # If port is None, find a free port
        if self.port is None:
            self.port = self._find_free_port()

        config_path = os.path.join(AGENT_CONFIG_DIR, "redis", "redis.conf")
        data_path = os.path.join(AGENT_DATA_DIR, "redis")
        os.makedirs(data_path, exist_ok=True)
        print(
            f"Starting Redis server at {self.host}:{self.port} with config {config_path}"
        )
        print(f"Using data directory: {data_path}")

        self.process = subprocess.Popen(
            [
                "redis-server",
                config_path,
                "--port",
                str(self.port),
                "--dir",
                data_path,
                "--daemonize",
                "no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        await asyncio.sleep(1)

        # Try to connect with retries
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
                # Test the connection
                self.client.ping()
                print(f"Successfully connected to Redis at {self.host}:{self.port}")
                break
            except redis.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(
                        f"Failed to connect to Redis (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # Check if process is still running
                    if self.process.poll() is not None:
                        stderr = self.process.stderr.read()
                        raise RuntimeError(f"Redis server failed to start: {stderr}")
                    raise RuntimeError(
                        f"Failed to connect to Redis after {max_retries} attempts: {e}"
                    )

    def _exists(self, key):
        return self.client.exists(key)

    def _set(self, key, value):
        self.client.set(key, value)

    def _get(self, key):
        return self.client.get(key).decode("utf-8")

    async def reset(self, env_args: dict | None = None):
        pass

    async def step(self, action: str):
        """
        args:
            action (str): The key to query the redis server
        """
        if self._exists(action):
            return self._get(action)
        else:
            return "No result in the database."

    async def aclose(self):
        global global_redis_env
        if self == global_redis_env:
            print("Skip global redis env close")
        else:
            if self.process:
                self.process.terminate()
                self.process = None
            if self.client:
                self.client.close()
                self.client = None

    def close(self):
        if self.process:
            self.process.terminate()
            self.process = None
        if self.client:
            self.client.close()
            self.client = None

    @staticmethod
    async def acquire():
        """
        We use the same redis env for all tools
        """
        global global_redis_env
        if global_redis_env is None:
            global_redis_env = RedisEnv(port=None)  # Use automatic port detection
            await global_redis_env.start()
        return global_redis_env
