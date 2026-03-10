"""
Compute reward from a raw Docker container and instance dataset entry.

Install the project first (e.g. `pip install -e .` or `uv sync`), then:

  from reward_from_container import setup_container_for_reward, reward_from_container

  # If you get "No such file or directory" for /run_tests.sh, call setup first:
  setup_container_for_reward(container, dataset="r2e")  # or "swebench", "swesmith"

  reward = reward_from_container(container, ds, dataset="swebench", timeout=300)
  # or
  reward, test_output = reward_from_container(
      container, ds, dataset="r2e", timeout=300, get_test_output=True
  )

Supported dataset values: "swebench", "swebench_verified", "swesmith", "r2e".
For "r2e", the container usually has the test script at /root/run_tests.sh; setup
creates /run_tests.sh so reward_from_container can run it. For "swesmith", setup
requires ds to build the test script.
"""

import json
import re
import concurrent.futures

from r2egym.agenthub.trajectory.swebench_utils import get_logs_eval, make_test_spec
from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from swebench.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    ResolvedStatus,
    TESTS_ERROR,
    TESTS_TIMEOUT,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import get_eval_type

# Match R2E-Gym Docker runtime PATH for /run_tests.sh
DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Default test script path per dataset (before setup). After setup_container_for_reward, use /run_tests.sh.
# R2E Dockerfiles (aiohttp, pillow, etc.) copy run_tests.sh to /testbed/run_tests.sh; WORKDIR is /testbed.
_TEST_CMD_BY_DATASET = {
    "swebench": "/run_tests.sh",
    "swebench_verified": "/run_tests.sh",
    "swesmith": "/run_tests.sh",
    "r2e": "bash /testbed/run_tests.sh",
}


def setup_container_for_reward(
    container,
    dataset: str,
    ds: dict | None = None,
    setup_timeout: int = 60,
) -> None:
    """
    Ensure the container has a runnable /run_tests.sh so reward_from_container can run tests.

    Call this once after starting the container and before reward_from_container.
    - **r2e**: Creates /run_tests.sh as a wrapper that runs `bash /root/run_tests.sh`
      (or `bash /testbed/run_tests.sh` if /root/run_tests.sh is missing).
    - **swebench / swebench_verified**: Runs chmod +x /run_tests.sh if it exists.
    - **swesmith**: Builds /run_tests.sh from ds (requires ds with FAIL_TO_PASS etc.)
      using the same logic as DockerRuntime.setup_env_swesmith.

    Args:
        container: Docker SDK container object.
        dataset: One of "swebench", "swebench_verified", "swesmith", "r2e".
        ds: Required for dataset "swesmith"; optional otherwise.
        setup_timeout: Timeout (seconds) applied to individual setup shell commands.
    """

    if dataset not in ("swebench", "swebench_verified", "swesmith", "r2e"):
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            "Expected one of: swebench, swebench_verified, swesmith, r2e"
        )

    def exec_run(cmd: list | str, workdir: str = "/testbed", timeout: int | None = None):
        """Run a short setup command inside the container with a hard timeout."""
        if timeout is None:
            timeout = setup_timeout
        if isinstance(cmd, str):
            inner_cmd = ["/bin/sh", "-c", cmd]
        else:
            inner_cmd = cmd
        full_cmd = ["timeout", str(timeout), *inner_cmd]
        r = container.exec_run(
            cmd=full_cmd,
            workdir=workdir,
            stdout=True,
            stderr=True,
            environment={"PATH": DOCKER_PATH},
        )
        return r.output.decode("utf-8", errors="replace") if r.output else ""

    if dataset == "r2e":
        # R2E images have r2e_tests at /r2e_tests; run_tests.sh runs from /testbed and expects r2e_tests there.
        # DockerRuntime.setup_env() does: mv /r2e_tests /root/r2e_tests; ln -s /root/r2e_tests /testbed/r2e_tests.
        # Without that, create a symlink so /testbed/r2e_tests exists (point to /r2e_tests if still there).
        exec_run(
            "[ -d /r2e_tests ] && ln -sf /r2e_tests /testbed/r2e_tests || true",
            workdir="/",
        )
        exec_run(
            "[ -d /root/r2e_tests ] && ln -sf /root/r2e_tests /testbed/r2e_tests || true",
            workdir="/",
        )
        # Create /run_tests.sh that delegates to the real script. R2E Dockerfiles put it at /testbed/run_tests.sh.
        exec_run("echo '#!/bin/sh' > /run_tests.sh", workdir="/")
        exec_run(
            "echo 'if [ -f /testbed/run_tests.sh ]; then cd /testbed && exec bash /testbed/run_tests.sh \"$@\"; fi' >> /run_tests.sh",
            workdir="/",
        )
        exec_run(
            "echo 'if [ -f /root/run_tests.sh ]; then exec bash /root/run_tests.sh \"$@\"; fi' >> /run_tests.sh",
            workdir="/",
        )
        exec_run("chmod +x /run_tests.sh", workdir="/")
        return

    if dataset in ("swebench", "swebench_verified"):
        exec_run("test -f /run_tests.sh && chmod +x /run_tests.sh")
        return

    if dataset == "swesmith":
        if ds is None:
            raise ValueError("setup_container_for_reward(..., dataset='swesmith') requires ds")
        import base64
        from r2egym.swesmith.utils import get_test_command

        test_command, _ = get_test_command(ds)
        content = "\n".join([
            "#!/bin/bash",
            "set -uxo pipefail",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd testbed/",
            ": '>>>>> Start Test Output'",
            test_command,
            ": '>>>>> End Test Output'",
        ]) + "\n"
        b64 = base64.b64encode(content.encode()).decode()
        exec_run(f"echo '{b64}' | base64 -d > /run_tests.sh && chmod +x /run_tests.sh", workdir="/")
        return


def _ensure_r2e_tests_in_testbed(container, timeout: int = 60) -> None:
    """Ensure /testbed/r2e_tests exists (symlink). run_tests.sh runs from /testbed and expects r2e_tests."""
    container.exec_run(
        cmd=[
            "timeout",
            str(timeout),
            "/bin/sh",
            "-c",
            "([ -d /r2e_tests ] && ln -sf /r2e_tests /testbed/r2e_tests) || "
            "([ -d /root/r2e_tests ] && ln -sf /root/r2e_tests /testbed/r2e_tests) || true",
        ],
        workdir="/",
        stdout=True,
        stderr=True,
        environment={"PATH": DOCKER_PATH},
    )


def _run_tests_in_container(
    container,
    timeout: int = 300,
    dataset: str | None = None,
) -> str:
    """Run the test script in container and return raw output. Uses dataset to pick command if /run_tests.sh is missing."""
    if dataset == "r2e":
        # Ensure r2e_tests symlink exists; cap this small helper at at most 60s even
        # if the main test timeout is larger.
        _ensure_r2e_tests_in_testbed(container, timeout=min(timeout, 60))
    # Prefer /run_tests.sh if it exists (after setup), else use dataset-specific command
    if dataset and dataset in _TEST_CMD_BY_DATASET:
        test_cmd = _TEST_CMD_BY_DATASET[dataset]
    else:
        test_cmd = "/run_tests.sh"
    command = f"timeout {timeout} {test_cmd}"
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    timed_out = False
    try:
        future = executor.submit(
            container.exec_run,
            cmd=["/bin/sh", "-c", command],
            workdir="/testbed",
            stdout=True,
            stderr=True,
            environment={"PATH": DOCKER_PATH},
        )
        exec_result = future.result(timeout=timeout + 5)
        out = exec_result.output.decode("utf-8", errors="replace")
        return re.sub(r"\x1b\[[0-9;]*m|\r", "", out)
    except concurrent.futures.TimeoutError:
        timed_out = True
        # Do not wait for the stuck exec_run thread; otherwise shutdown(wait=True)
        # would block indefinitely when the container does not respond.
        executor.shutdown(wait=False)
        return TESTS_TIMEOUT
    except Exception:
        timed_out = True
        executor.shutdown(wait=False)
        return TESTS_ERROR
    finally:
        if not timed_out:
            executor.shutdown(wait=True)


def _reward_swebench(ds: dict, out: str, get_test_output: bool):
    """SWE-Bench / SWE-Bench Verified: grade via FAIL_TO_PASS and PASS_TO_PASS."""
    test_spec = make_test_spec(ds)
    eval_status_map, found = get_logs_eval(test_spec, out)
    if not found:
        reward = 0.0
    else:
        eval_ref = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(test_spec)
        )
        reward = 1.0 if get_resolution_status(report) == ResolvedStatus.FULL.value else 0.0
    if get_test_output:
        return reward, out
    return reward


def _reward_swesmith(ds: dict, out: str, get_test_output: bool):
    """SWE-Smith: parse pytest-style log and check FAIL_TO_PASS / PASS_TO_PASS."""
    repo = ds.get("repo", ds.get("repo_name", ""))
    parse = parse_log_fn(repo)(out)
    if not parse:
        reward = 0.0
    else:
        fail2pass = [".".join(line.split("::")[1:]) for line in ds.get("FAIL_TO_PASS", [])]
        pass2pass = [".".join(line.split("::")[1:]) for line in ds.get("PASS_TO_PASS", [])]
        reward = 1.0
        for test_name in fail2pass:
            matching_key = next((k for k in parse.keys() if test_name in k), None)
            if matching_key is None or parse.get(matching_key) != "PASSED":
                reward = 0.0
                break
        if reward == 1.0:
            for test_name in pass2pass:
                matching_key = next((k for k in parse.keys() if test_name in k), None)
                if matching_key is None or parse.get(matching_key) != "PASSED":
                    reward = 0.0
                    break
    if get_test_output:
        return reward, out
    return reward


def _reward_r2e(
    container,
    ds: dict,
    out: str,
    get_test_output: bool,
    timeout: int = 60,
):
    """R2E-Gym-Lite / r2e-edits: parse log and compare to expected_output_json."""
    repo_name = ds.get("repo_name", ds.get("repo", ""))
    parse = parse_log_fn(repo_name)(out)
    parse = decolor_dict_keys(parse)
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}

    try:
        expected_json = ds.get("expected_output_json")
        if not expected_json:
            # Try reading from container (R2E dockers may write it here). Use a short timeout so
            # a stuck filesystem or container cannot hang reward evaluation indefinitely.
            result = container.exec_run(
                cmd=[
                    "timeout",
                    str(min(timeout, 60)),
                    "/bin/sh",
                    "-c",
                    "cat /root/expected_test_output.json 2>/dev/null || "
                    "cat /testbed/expected_test_output.json 2>/dev/null || echo '{}'",
                ],
                workdir="/testbed",
                stdout=True,
                stderr=True,
                environment={"PATH": DOCKER_PATH},
            )
            expected_json = (
                result.output.decode("utf-8", errors="replace")
                if result.output
                else "{}"
            )
        if isinstance(expected_json, dict):
            expected = expected_json
        else:
            expected = json.loads(expected_json)
    except (json.JSONDecodeError, KeyError, TypeError):
        reward = 0.0
        if get_test_output:
            return reward, out
        return reward

    expected = decolor_dict_keys(expected)
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

    if len(parse) != len(expected):
        reward = 0.0
    else:
        match = True
        for k in parse:
            if not k:
                continue
            if k not in expected or parse[k] != expected[k]:
                match = False
                break
        reward = 1.0 if match else 0.0

    if get_test_output:
        return reward, out
    return reward


def reward_from_container(
    container,
    ds: dict,
    dataset: str,
    timeout: int = 300,
    get_test_output: bool = False,
):
    """
    Compute reward from a raw Docker container and instance dataset entry.

    Use when you have a Docker SDK container and the instance dict `ds` from the
    benchmark. The `dataset` argument selects the grading method.

    Args:
        container: Docker SDK container object (has .exec_run()).
        ds: Dataset entry for this instance (fields depend on dataset).
        dataset: Which reward logic to use. One of:
            "swebench" or "swebench_verified" -> SWE-Bench grading (FAIL_TO_PASS / PASS_TO_PASS).
            "swesmith" -> SWE-Smith grading (pytest log + FAIL_TO_PASS / PASS_TO_PASS).
            "r2e" -> R2E-Gym-Lite / r2e-edits (parse log vs expected_output_json).
        timeout: Timeout in seconds for /run_tests.sh.
        get_test_output: If True, return (reward, raw_test_output); else return reward only.

    Returns:
        float: reward 0.0 or 1.0; or tuple (reward, test_output) if get_test_output=True.
    """
    if dataset not in ("swebench", "swebench_verified", "swesmith", "r2e"):
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            "Expected one of: swebench, swebench_verified, swesmith, r2e"
        )

    out = _run_tests_in_container(container, timeout=timeout, dataset=dataset)

    if dataset in ("swebench", "swebench_verified"):
        return _reward_swebench(ds, out, get_test_output)
    if dataset == "swesmith":
        return _reward_swesmith(ds, out, get_test_output)
    return _reward_r2e(container, ds, out, get_test_output, timeout=timeout)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compute reward from a running container.")
    parser.add_argument("container_id", help="Docker container ID or name")
    parser.add_argument("ds_path", help="Path to a JSON file containing the instance dataset entry (one dict)")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["swebench", "swebench_verified", "swesmith", "r2e"],
        help="Dataset type for reward calculation",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for /run_tests.sh (seconds)")
    parser.add_argument("--get-output", action="store_true", help="Print raw test output")
    args = parser.parse_args()

    import docker

    with open(args.ds_path) as f:
        ds_entry = json.load(f)
    client = docker.from_env()
    container = client.containers.get(args.container_id)
    result = reward_from_container(
        container,
        ds_entry,
        dataset=args.dataset,
        timeout=args.timeout,
        get_test_output=args.get_output,
    )
    if args.get_output:
        reward, out = result
        print("Reward:", reward)
        print("Test output (last 2000 chars):")
        print(out[-2000:] if len(out) > 2000 else out)
    else:
        print("Reward:", result)
