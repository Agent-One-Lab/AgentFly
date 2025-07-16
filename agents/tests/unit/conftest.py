# conftest.py
import asyncio, pytest

@pytest.fixture(scope="session")      # ONE loop for the whole test-session
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
