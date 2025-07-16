import asyncio, time, httpx, docker, random
from typing import Any, Mapping, Tuple
from .env_base import BaseEnv, SupportsDocker
from typing import Union

class ScienceWorldEnv(BaseEnv, SupportsDocker):
    """
    ScienceWorld environment wrapper that provides a Docker-based interface to the ScienceWorld simulator.
    
    This class manages a Docker container running the ScienceWorld environment and provides
    an async interface for interacting with science experiments. The container is configured
    with security restrictions and resource limits for safe execution.
    
    Attributes:
        image (str): Docker image name for the ScienceWorld environment
        runtime (str): Docker runtime to use (default: "runc")
        cpu (int): Number of CPU cores allocated to the container
        mem (str): Memory limit for the container (e.g., "2g")
        start_timeout (float): Maximum time to wait for container startup
        max_episodes (int): Maximum number of episodes allowed
        container_port (int): Port inside the container (default: 2700)
        host_ip (str): Host IP for port binding (default: "127.0.0.1")
        score (float): Current score for the active episode
        _client (httpx.AsyncClient): HTTP client for communicating with the container
    
    Example:
        ```python
        env = ScienceWorldEnv()
        await env.start()
        await env.reset({'task_name': 'boil', 'variation_idx': 0})
        observation = await env.step("look around")
        await env.aclose()
        ```
    """
    def __init__(
        self,
        image: str = "rifoag/scienceworld-env:latest",
        runtime: str = "runc",
        cpu: int = 2,
        mem: str = "2g",
        start_timeout: float = 10.0,
        max_episodes: int = 100,
        host_ip: str = "127.0.0.1",
        container_port: int = 2700,
    ):
        """
        Initialize the ScienceWorld environment.
        
        Args:
            image (str): Docker image name for the ScienceWorld environment
            runtime (str): Docker runtime to use for container execution
            cpu (int): Number of CPU cores to allocate to the container
            mem (str): Memory limit for the container (e.g., "2g", "512m")
            start_timeout (float): Maximum time in seconds to wait for container startup
            max_episodes (int): Maximum number of episodes allowed per session
            host_ip (str): Host IP address for port binding (defaults to localhost)
            container_port (int): Port number inside the container to expose
        """
        super().__init__()
        self.image, self.runtime = image, runtime
        self.cpu, self.mem = cpu, mem
        self.start_timeout, self.max_episodes = start_timeout, max_episodes
        self.container_port = container_port
        self.host_ip = host_ip
        self._client: httpx.AsyncClient | None = None

        # self.is_completed = False
        self.score = 0

    async def _wait_ready(self) -> None:
        """
        Poll the container's health endpoint until it responds with 200 OK or timeout.
        
        This method continuously checks the /health endpoint of the ScienceWorld
        container until it becomes ready or the start_timeout is exceeded.
        
        Raises:
            RuntimeError: If the container doesn't become ready within the timeout period
        """
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            try:
                r = await self._client.get("/health")   # FastAPI route in the image
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)

        # Last-ditch diagnostics
        logs = self.get_container_logs()
        raise RuntimeError(
            f"Sandbox did not become ready within {self.start_timeout}s.\n{logs}"
        )
    
    async def start(self) -> None:
        """
        Start the ScienceWorld Docker container and establish connection.
        
        This method:
        1. Starts a Docker container with the specified ScienceWorld image
        2. Configures security restrictions (read-only filesystem, dropped capabilities)
        3. Maps the container port to a random available host port
        4. Establishes HTTP connection to the container
        5. Waits for the container to become ready
        6. Loads the initial task (boil task with variation 0)
        
        Raises:
            RuntimeError: If container startup fails or health check times out
        """
        # Ask Docker to map container 8000/tcp ⇒ random free host port
        await self._docker_start(
            image=self.image,
            runtime=self.runtime,
            cpu_count=self.cpu,
            mem_limit=self.mem,
            # bridge is the default; omit network_mode entirely if you like
            ports={f"{self.container_port}/tcp": None},
            # ← None lets Docker choose a host port
            read_only=True,
            cap_drop=["ALL"],
            pids_limit=256,
        )

        await self._connect()
        await self._wait_ready()
        r = await self._client.get(f"/load?task_name=boil&variation_idx=0")

    async def reset(self, env_args=None) -> str:
        """
        Reset the environment to start a new episode with specified task parameters.
        
        Args:
            env_args (dict, optional): Dictionary containing task configuration. Defaults to None. Used during training.
                - task_name (str): Name of the science task (default: 'boil')
                - variation_idx (int): Task variation index (default: 0)
        
        Returns:
            str: Initial observation from the reset environment
        
        Example:
            ```python
            await env.reset({'task_name': 'measure', 'variation_idx': 1})
            ```
        """
        env_args = env_args or {}
        task_name = env_args.get('task_name', 'boil')
        variation_idx = env_args.get('variation_idx', 0)
        r = await self._client.get(f"/load?task_name={task_name}&variation_idx={variation_idx}")
        await self._client.get("/reset")
        r = r.json()
        # self.is_completed = False
        self.score = 0

    async def step(self, action: str) -> Union[str, dict]:
        """
        Execute an action in the ScienceWorld environment.
        
        Args:
            action (str): The action to perform. Can be either:
                - A regular ScienceWorld action (e.g., "look around", "open door to kitchen")
                - "get_reward" to retrieve current reward without performing an action
                - accept other string, but it will be treated as ineffective action
        
        Returns:
            Union[str, dict]: Either:
                - str: The observation text if a regular action is performed
                - dict: A dictionary containing 'observation' and 'reward' if action is "get_reward"
        
        Note:
            When action is "get_reward", returns a simplified response with
            current task completion status and accumulated score.
        """
        if action == "get_reward":
            return {
                "observation": 'Task completed' if self.score >= 1 else 'Task not completed',
                "reward": self.score
            }
        else:
            r  = await self._client.get(f"/step?action={action}")
            r = r.json()
            current_reward = r['info']['score']/100
            self.score = max(self.score, current_reward)
            # self.is_completed = r['done'] # Commented out since there are three conditions for completion: actually complete, reach maximum turns, and got negative score (whic is not clear yet what negative score means, possibly just error in the data annotaiton)
            return r['observation']

    async def aclose(self) -> None:
        """
        Asynchronously close the environment and clean up resources.
        
        This method:
        1. Stops the Docker container
        2. Closes the HTTP client connection
        3. Cleans up any remaining resources
        
        Should be called when done using the environment to prevent resource leaks.
        """
        await self._docker_stop()
        if self._client:
            await self._client.aclose()
            self._client = None

    def close(self) -> None:
        """
        Synchronously close the environment and force-kill the container.
        
        This is a synchronous alternative to aclose() that immediately
        terminates the Docker container without graceful shutdown.
        
        Note:
            Prefer aclose() for normal cleanup as it allows graceful shutdown.
        """
        if self._container:
            self._container.kill()
            self._container = None

    async def _connect(self):
        """
        Discover the host port mapping and establish HTTP connection to the container.
        
        This method:
        1. Waits for Docker to assign a host port for the container
        2. Creates an httpx.AsyncClient targeting the mapped port
        3. Sets up the base URL for all subsequent API calls
        
        Raises:
            RuntimeError: If port mapping is not found within the timeout period
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
        self._client = httpx.AsyncClient(base_url=base_url, timeout=20.0)

    @staticmethod
    async def acquire():
        """
        Static factory method to create and initialize a ScienceWorld environment.
        
        This is a convenience method that creates a new ScienceWorldEnv instance,
        starts it, and resets it to the default state in one call.
        
        Returns:
            ScienceWorldEnv: A fully initialized and ready-to-use environment instance
        
        Example:
            ```python
            env = await ScienceWorldEnv.acquire()
            # Use env for experiments
            await env.aclose()
            ```
        """
        env = ScienceWorldEnv()
        await env.start()
        await env.reset()
        return env
