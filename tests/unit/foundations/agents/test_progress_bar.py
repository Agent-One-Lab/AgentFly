"""The rollout progress bar: plain tqdm outside Ray, Ray's tqdm_ray inside a Ray worker."""

import pytest
from tqdm import tqdm

from agentfly.agents.rollout.loop.progress import in_ray_worker, progress_bar


def test_plain_bar_outside_ray():
    assert not in_ray_worker()
    bar = progress_bar(total=3)
    try:
        assert isinstance(bar, tqdm)
    finally:
        bar.close()


def test_ray_bar_inside_a_ray_actor():
    ray = pytest.importorskip("ray")
    ray.init(address="local", include_dashboard=False, num_cpus=1, log_to_driver=False)
    try:
        assert not in_ray_worker()  # the driver draws plain bars itself

        @ray.remote
        class Runner:
            def bar_type(self):
                from agentfly.agents.rollout.loop.progress import in_ray_worker, progress_bar

                bar = progress_bar(total=2)
                bar.update(2)
                bar.close()
                return in_ray_worker(), type(bar).__module__

        assert ray.get(Runner.remote().bar_type.remote()) == (True, "ray.experimental.tqdm_ray")
    finally:
        ray.shutdown()
