# app.py  ─ parent process (lives in your Enroot container)
import asyncio
import os
import signal
import sys
from subprocess import PIPE

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MAX_WALL = 10  # wall-clock seconds per snippet
CHILD_MEM = 1 * 2**30  # 1 GiB  (match cgroup limit outside!)
CHILD_CPU = 100  # 100 % of a core

app = FastAPI()


class Script(BaseModel):
    code: str


async def _run(src: str) -> str:
    # 1️⃣  spawn the helper; we feed code via STDIN
    cmd = [sys.executable, "-u", "snippet_runner.py"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        preexec_fn=os.setsid,  # own process-group for SIGKILL
    )
    proc.stdin.write(src.encode())
    await proc.stdin.drain()
    proc.stdin.close()  # EOF for the child

    # 2️⃣  enforce wall-clock timeout from the parent side
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=MAX_WALL)
    except asyncio.TimeoutError:
        os.killpg(proc.pid, signal.SIGKILL)
        raise HTTPException(408, "Execution timed out")

    # 3️⃣  return result or propagate sandbox error
    if proc.returncode != 0:
        detail = (stderr or b"").decode()[:8_192]
        raise HTTPException(400, detail)
    return stdout.decode()[:16_384]  # truncate overly chatty output


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/exec")
async def exec_code(s: Script):
    return {"output": await _run(s.code)}
