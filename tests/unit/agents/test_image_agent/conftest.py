import gc

import pytest


def _release(agent):
    """Free the GPU memory an ImageEditingAgent holds, so later tests in the same process can
    start their own engines: the vLLM engine (its process outlives the agent otherwise) and the
    image-edit pipeline the tool loads on first use."""
    engine = getattr(agent.llm_engine, "llm_engine", None)
    if engine is not None and hasattr(engine, "shutdown"):
        engine.shutdown()
    agent._qwen_image_edit_tool._pipeline = None
    gc.collect()
    import torch

    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def release_image_agent():
    """``release_image_agent(agent)``: free an image agent's GPU memory (see ``_release``)."""
    return _release
