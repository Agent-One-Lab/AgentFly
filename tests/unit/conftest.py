# conftest.py
import asyncio, pytest
from agentfly.resources.runner import LocalRunner


@pytest.fixture(scope="session")  # ONE loop for the whole test-session
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()



@pytest.fixture(scope="session")
def local_runner():
    runner = LocalRunner()
    yield runner