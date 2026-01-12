import asyncio, time, httpx, docker, random
from typing import Any, Mapping, Tuple
from .env_base import BaseEnv, SupportsDocker


class PythonSandboxEnv(BaseEnv, SupportsDocker):
    """
    Untrusted Python snippet executor over HTTP (localhost TCP).
    Safe-ish when the container still runs with gVisor/Kata
    and the port is only bound to 127.0.0.1.
    """
    def __init__(
        self,
        image: str = "reasonwang/python-http-env:latest",
        runtime: str = "runc",
        cpu: int = 2,
        mem: str = "2g",
        start_timeout: float = 60.0,
        max_episodes: int = 30,
        host_ip: str = "127.0.0.1",
        container_port: int = 8000,
    ):
        """
        Initialize the PythonSandboxEnv.
        """
        super().__init__()
        self.image, self.runtime = image, runtime
        self.cpu, self.mem = cpu, mem
        self.start_timeout, self.max_episodes = start_timeout, max_episodes
        self.container_port = container_port
        self.host_ip = host_ip
        self._client: httpx.AsyncClient | None = None
        self._episodes = 0

    async def _wait_ready(self) -> None:
        """Poll /health until the server answers 200 OK or we time out."""
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
        Start the Docker container and wait for it to be ready.
        """
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

    async def reset(self, env_args=None) -> str:
        """
        Reset the environment by clearing the global state.
        """
        try:
            await asyncio.wait_for(self.step("globals().clear()"), timeout=20.0)
        except (asyncio.TimeoutError, httpx.TransportError):
            # interpreter is wedged – restart container
            await self.aclose()
            await self.start()

        return ""

    async def step(self, code: str) -> Tuple[str, float, bool, Mapping]:
        """
        Execute the code in the environment and return the output from stdout or stderr.

        Args:
            code (str): The code to execute.

        Returns:
            str: The output from stdout or stderr.
        """

        self._episodes += 1
        resp = await self._client.post("/exec", json={"code": code})
        data = resp.json()
        if "output" in data:
            obs   = data["output"]
            return obs
        elif "detail" in data:
            return data["detail"]
        else:
            raise RuntimeError(f"Unknown response: {data}")

    async def aclose(self) -> None:
        """
        Close the environment by stopping the Docker container and closing the HTTP client.
        """
        await self._docker_stop()
        if self._client:
            await self._client.aclose()
            self._client = None

    def close(self) -> None:
        if self._container:
            self._container.kill()
            self._container = None

    async def _connect(self):
        # Discover which host port Docker chose and open an httpx client targeting http://127.0.0.1:<host_port>.
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
        Create a new PythonSandboxEnv instance and start it.
        """
        env = PythonSandboxEnv()
        await env.start()
        await env.reset()
        return env
