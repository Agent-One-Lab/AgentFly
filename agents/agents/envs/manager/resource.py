import asyncio
from tqdm import tqdm
GLOBAL_ENVS = set()

async def cleanup_resources():
    """Cleanup all resources."""
    await asyncio.gather(*[env.close() for env in GLOBAL_ENVS])


def run_async_cleanup():
    """Run a function asynchronously and cleanup all resources."""
    loop = asyncio.new_event_loop()
    loop.run_until_complete(cleanup_resources())
    loop.close()


def cleanup_envs():
    print("Cleaning up environments...")
    for env in tqdm(GLOBAL_ENVS):
        env.close()

import atexit, signal

atexit.register(cleanup_envs)
for sig in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(sig, cleanup_envs)
