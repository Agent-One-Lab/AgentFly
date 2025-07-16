import asyncio
import grpc, pathlib, tempfile, time
from uuid import uuid4
from typing import Any, Tuple, Mapping

from env_base import BaseEnv, SupportsDocker
from code_env_pb2 import Command, ResetRequest
from code_env_pb2_grpc import CodeEnvStub

class OSShellEnv(BaseEnv, SupportsDocker):
    """
    OS-sandbox environment.

    • Isolates each episode in a gVisor container (runtime=runsc).
    • Implements BaseEnv's async start / reset / step / close.
    """
    def __init__(
        self,
        image: str = "code-env:latest",
        cpu_limit: str = "2",
        mem_limit: str = "4g",
        runtime: str = "runsc",          # gVisor for syscall filtering
        max_episodes: int = 50,
        start_timeout: float = 15.0,
    ):
        self.image, self.cpu_limit, self.mem_limit = image, cpu_limit, mem_limit
        self.runtime, self.start_timeout = runtime, start_timeout
        self.max_episodes = max_episodes
        self._episodes = 0
        self._uds_dir = pathlib.Path(tempfile.mkdtemp(prefix="envsock_"))
        self._stub: CodeEnvStub | None = None

    # ---------- BaseEnv interface ----------
    async def start(self) -> None:
        await self._docker_start(
            image=self.image,
            runtime=self.runtime,
            cpus=self.cpu_limit,
            mem_limit=self.mem_limit,
            network_mode="none",
            volumes={str(self._uds_dir): {"bind": "/tmp/envsock", "mode": "rw"}},
            environment={"CODE_ENV_RPC": "/tmp/envsock/rpc.sock"},
        )
        await self._connect_grpc()

    async def reset(self) -> Any:
        """
        Wipe user workspace OR recycle container every `max_episodes`.
        Returns the initial observation (here: empty string).
        """
        if self._episodes >= self.max_episodes:
            await self.close()
            await self.start()
            self._episodes = 0
        else:
            await self._stub.Reset(ResetRequest())
        self._episodes += 1
        return ""

    async def step(self, action: str) -> Tuple[str, float, bool, Mapping]:
        """
        `action` is a shell command.  We stream it to gRPC and capture output.
        """
        call = self._stub.Exec(iter([Command(line=action)]))
        chunks = [chunk.chunk async for chunk in call]
        obs = b"".join(chunks).decode()
        done = False                       # single-step tasks never terminate
        return obs, 0.0, done, {}

    async def close(self) -> None:
        await self._docker_stop()
        self._stub = None

    # ---------- helper ----------
    async def _connect_grpc(self):
        sock = self._uds_dir / "rpc.sock"
        deadline = time.time() + self.start_timeout
        while not sock.exists():
            if time.time() > deadline:
                raise RuntimeError("gRPC socket never appeared")
            await asyncio.sleep(0.1)
        chan = grpc.aio.insecure_channel(f"unix://{sock}")
        await chan.channel_ready()
        self._stub = CodeEnvStub(chan)