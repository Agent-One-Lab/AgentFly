import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .env_base import BaseEnv, SupportsDocker


class ALFWorldEnv(BaseEnv, SupportsDocker):
    """
    ALFWorld environment that runs in a Docker container with HTTP interface.
    Supports both text-only and visual modalities (starting with text-only).
    Now supports task selection by task_id.
    """

    def __init__(
        self,
        image: str = "bitalov/alfworld-http-env-3:latest",
        runtime: str = "runc",
        cpu: int = 2,
        mem: str = "2g",
        start_timeout: float = 120.0,
        max_episodes: int = 50,
        host_ip: str = "127.0.0.1",
        container_port: int = 8000,
        config_path: str = None,
        train_eval: str = "train",
        batch_size: int = 1,
    ):
        """
        Initialize the ALFWorldEnv environment.
        """
        super().__init__()
        self.image = image
        self.runtime = runtime
        self.cpu = cpu
        self.mem = mem
        self.start_timeout = start_timeout
        self.max_episodes = max_episodes
        self.container_port = container_port
        self.host_ip = host_ip
        self.config_path = config_path
        self.train_eval = train_eval
        self.batch_size = batch_size

        self._client: httpx.AsyncClient | None = None
        self._episodes = 0
        self._current_env = None
        self._current_info = None
        self._current_obs = None

    async def start(self) -> None:
        """
        Start the ALFWorld environment container.
        """
        await self._docker_start(
            image=self.image,
            runtime=self.runtime,
            cpu_count=self.cpu,
            mem_limit=self.mem,
            ports={f"{self.container_port}/tcp": None},
            read_only=False,  # ALFWorld needs to write temporary files
            cap_drop=["ALL"],
            pids_limit=256,
            environment={
                "ALFWORLD_DATA": "/root/.cache/alfworld",
                "TRAIN_EVAL": self.train_eval,
                "BATCH_SIZE": str(self.batch_size),
            },
        )

        await self._connect()
        await self._wait_ready()

    async def _wait_ready(self) -> None:
        """Poll /health until the server is ready."""
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            try:
                r = await self._client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)

        logs = self.get_container_logs()
        raise RuntimeError(
            f"ALFWorld environment did not become ready within {self.start_timeout}s.\n{logs}"
        )

    async def _connect(self):
        """
        Finds the mapped host port and connects the HTTP client.

        After the container starts, Docker maps the container's internal port to a
        dynamic port on the host. This method inspects the container's network
        settings to find this host port and then initializes the `httpx.AsyncClient`
        to communicate with it.

        Raises:
            RuntimeError: If the port mapping cannot be found after the timeout.
        """
        deadline = time.time() + self.start_timeout
        host_port = None

        while time.time() < deadline:
            ports = self._container.attrs["NetworkSettings"]["Ports"]
            binding = ports.get(f"{self.container_port}/tcp")
            if binding:
                host_port = binding[0]["HostPort"]
                break
            await asyncio.sleep(0.1)
            self._container.reload()

        if host_port is None:
            logs = self._container.logs().decode()
            raise RuntimeError(f"Port mapping not found. Logs:\n{logs}")

        base_url = f"http://{self.host_ip}:{host_port}"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    def _extract_goal_from_observation(self, observation: str) -> str:
        """Extract the goal from the observation string."""
        try:
            obs_str = str(observation)

            # Look for the task description pattern
            if "Your task is to: " in obs_str:
                # Extract the part after "Your task is to: "
                goal_part = obs_str.split("Your task is to: ")[1]

                # Extract until the first period (end of sentence)
                if "." in goal_part:
                    goal = goal_part.split(".")[0].strip()
                else:
                    # Fallback: take until newline
                    goal = goal_part.split("\n")[0].strip()

                # Clean up any trailing quotes or punctuation
                goal = goal.rstrip("'\".,)")

                return goal

        except Exception as e:
            print(f"Failed to extract goal from observation: {e}")

    async def get_available_tasks(
        self, split: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of available tasks for a given split."""
        if split is None:
            split = self.train_eval

        try:
            resp = await self._client.get("/available_tasks", params={"split": split})
            data = resp.json()
            if resp.status_code == 200:
                return data.get("tasks", [])
            else:
                return []
        except Exception:
            return []

    async def reset(
        self, env_args=None, split: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resets the environment to start a new episode.

        This method can start a random episode from a specified split or a
        specific episode identified by its `task_id`.

        Args:
            env_args (Optional[Dict[str, Any]]): A dictionary of arguments. To load a specific task,
                         provide `{'task_id': 'trial_T...'}`. If None, a
                         default example task is used.
            split (Optional[str]): The data split to use. If None, defaults to `self.train_eval`.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the initial observation string and an info dictionary.

        Raises:
            RuntimeError: If the reset request fails.
        """
        if env_args is None:
            env_args = {"task_id": "trial_T20190907_212755_456877"}

        task_id = env_args.get("task_id", None)

        if task_id is None:
            task_id = "trial_T20190909_013611_626994"

        if split is None:
            split = self.train_eval

        try:
            # Prepare reset request data
            reset_data = {"split": split}
            if task_id is not None:
                reset_data["task_id"] = task_id

            resp = await self._client.post("/reset", json=reset_data)
            data = resp.json()
            if resp.status_code == 200:
                self._episodes += 1
                self._current_obs = data["observation"]
                self._current_info = data.get("info", {})
                return data["observation"], self._current_info
            else:
                raise RuntimeError(f"Reset failed: {data}")
        except (asyncio.TimeoutError, httpx.TransportError):
            # Environment is wedged - restart container
            await self.aclose()
            await self.start()
            return await self.reset(task_id=task_id, split=split)

    async def get_info(self) -> Dict[str, Any]:
        """Get current environment information."""
        if self._current_info:
            # Try to get goal from cached info (should be there after reset)
            goal = (
                self._current_info.get("goal")
                or self._current_info.get("task_description")
                or self._current_info.get("task_desc")
                or self._current_info.get("description")
                or ""
            )

            # If still no goal, try to extract from stored observation
            if not goal and hasattr(self, "_current_obs") and self._current_obs:
                goal = self._extract_goal_from_observation(self._current_obs)
                # Store it for future use
                if goal:
                    self._current_info["goal"] = goal
            task_info = self._current_info.get(
                "extra.gamefile", self._current_info.get("task", "unknown")
            )
            task = (
                task_info[0] if isinstance(task_info, list) and task_info else task_info
            )
            return {
                "task": task,
                "goal": goal,
                "won": self._current_info.get("won", False),
                "lost": self._current_info.get("lost", False),
                "admissible_commands_count": len(
                    self._current_info.get("admissible_commands", [])
                ),
            }

        # Fallback to HTTP request if no cached info
        try:
            resp = await self._client.get("/info")
            resp.raise_for_status()
            return resp.json().get("info", {})
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"Error getting info: {e}")
            return {}

    @staticmethod
    async def acquire():
        """Create and initialize a new ALFWorld environment."""
        env = ALFWorldEnv()
        await env.start()
        return env

    async def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Takes a single step in the environment by executing an action.

        Args:
            action (str): The text command to execute in the environment.

        Returns:
            Tuple[str, float, bool, Dict[str, Any]]: A tuple of (observation, reward, done, info).
                - observation (str): The new observation from the environment.
                - reward (float): The reward received for the action.
                - done (bool): Whether the episode has ended.
                - info (dict): A dictionary with auxiliary information.
        """
        try:
            resp = await self._client.post("/step", json={"action": action})
            data = resp.json()

            if resp.status_code == 200:
                obs = data["observation"]
                reward = data.get("reward", 0.0)
                done = data.get("done", False)
                info = data.get("info", {})

                # Store current info for later access
                if (
                    self._current_info
                    and "goal" in self._current_info
                    and "goal" not in info
                ):
                    info["goal"] = self._current_info["goal"]
                self._current_info = info
                self._current_obs = obs
                return obs, reward, done, info
            else:
                error_msg = data.get("detail", "Unknown error")
                return f"Error: {error_msg}", 0.0, False, {}

        except Exception as e:
            return f"Error executing action: {str(e)}", 0.0, False, {}

    async def get_admissible_commands(self) -> List[str]:
        """
        Gets the list of admissible (valid) commands for the current state.

        Returns:
            List[str]: A list of valid action strings. Returns an empty list on failure.
        """
        try:
            resp = await self._client.get("/admissible_commands")
            data = resp.json()
            if resp.status_code == 200:
                return data.get("commands", [])
            else:
                return []
        except Exception:
            return []

    async def get_inventory(self) -> str:
        """Get the current inventory."""
        try:
            resp = await self._client.get("/inventory")
            data = resp.json()
            if resp.status_code == 200:
                return data.get("inventory", "")
            else:
                return ""
        except Exception:
            return ""

    async def aclose(self) -> None:
        """
        Asynchronously stops the container and closes the HTTP client.
        """
        await self._docker_stop()
        if self._client:
            await self._client.aclose()
            self._client = None

    def close(self) -> None:
        """
        Synchronously stops and removes the Docker container.
        """
        if self._container:
            self._container.kill()
            self._container = None
