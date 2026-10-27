# conftest.py
import asyncio, socket, pytest
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

LOCAL_SERVER = ("localhost", 8000)


def _local_server_up(host_port=LOCAL_SERVER, timeout=1.0) -> bool:
    try:
        with socket.create_connection(host_port, timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip ``local_server`` tests when no model server listens on localhost:8000.

    Those tests call an OpenAI-compatible server the user starts beforehand (e.g. ``vllm
    serve``); without one, the client retries indefinitely and the test hangs."""
    marked = [item for item in items if item.get_closest_marker("local_server")]
    if marked and not _local_server_up():
        skip = pytest.mark.skip(reason="needs a model server on localhost:8000 (e.g. `vllm serve`)")
        for item in marked:
            item.add_marker(skip)
