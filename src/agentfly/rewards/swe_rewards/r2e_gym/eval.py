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
import asyncio
from types import SimpleNamespace

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
        # R2E-Gym's DockerRuntime.setup_env() does:
        #   mv /r2e_tests /root/r2e_tests
        #   ln -s /root/r2e_tests /testbed/r2e_tests
        # to avoid path/symlink issues when the test root is referenced via multiple roots.
        exec_run(
            # If /root/r2e_tests is already a symlink (from a previous broken setup),
            # remove it so the subsequent mv/link results in a stable single-root tree.
            "[ -L /root/r2e_tests ] && rm -f /root/r2e_tests || true",
            workdir="/",
        )
        exec_run(
            # Make this idempotent: if /root/r2e_tests already exists, mv would create
            # /root/r2e_tests/r2e_tests (recursive nesting). Instead, drop the source.
            "[ -d /r2e_tests ] && [ ! -d /root/r2e_tests ] && mv /r2e_tests /root/r2e_tests || true",
            workdir="/",
        )
        exec_run(
            # If the destination already exists, remove the source to prevent recursion
            # and to ensure later /testbed/r2e_tests links are stable.
            "[ -d /r2e_tests ] && [ -d /root/r2e_tests ] && rm -rf /r2e_tests || true",
            workdir="/",
        )
        exec_run(
            "[ -d /root/r2e_tests ] && ln -sf /root/r2e_tests /testbed/r2e_tests || true",
            workdir="/",
        )
        exec_run(
            # Harden against accidental nested move behavior across repeated calls:
            # if /root/r2e_tests already contains a child named r2e_tests, remove it
            # to prevent pytest from traversing the same tests via multiple roots.
            "[ -e /root/r2e_tests/r2e_tests ] && rm -rf /root/r2e_tests/r2e_tests || true",
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
            # If /testbed/r2e_tests is already a symlink to a directory, `ln -sf SOURCE DEST`
            # may treat DEST as a directory (because it follows symlinks) and create
            # /testbed/r2e_tests/r2e_tests instead of replacing the symlink.
            # Avoid this by removing DEST first.
            "desired=''; "
            "[ -d /root/r2e_tests ] && desired=/root/r2e_tests || true; "
            "[ -z \"$desired\" ] && [ -d /r2e_tests ] && desired=/r2e_tests || true; "
            "[ -z \"$desired\" ] && exit 0; "
            "rm -rf /testbed/r2e_tests || true; "
            "ln -s \"$desired\" /testbed/r2e_tests || true",
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
    from swebench.harness.constants import TESTS_ERROR, TESTS_TIMEOUT

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


def _instance_fail_pass_lists(instance: dict) -> tuple[list, list]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS like swebench make_test_spec (list or JSON string)."""

    def _from_json_or_obj(key: str):
        if key not in instance:
            return []
        v = instance[key]
        if isinstance(v, str):
            return json.loads(v)
        return v

    return _from_json_or_obj("FAIL_TO_PASS"), _from_json_or_obj("PASS_TO_PASS")


def _resolve_swebench_log_parser(name: str):
    """
    ``install_config['log_parser']`` is often a function name (e.g. parse_log_pytest).
    Older swebench builds only register those under MAP_REPO_TO_PARSER on newer releases;
    fall back to attributes on ``swebench.harness.log_parsers.python``.
    """
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.log_parsers import python as swebench_log_parsers_python

    if name in MAP_REPO_TO_PARSER:
        return MAP_REPO_TO_PARSER[name]
    parser = getattr(swebench_log_parsers_python, name, None)
    if parser is not None:
        return parser
    raise KeyError(
        f"Unknown SWE-bench log parser {name!r}: not in MAP_REPO_TO_PARSER and "
        "not found on swebench.harness.log_parsers.python"
    )


def _swebench_logs_eval_string(test_spec, content: str) -> tuple[dict, bool]:
    """
    Build a test status map from raw log text (same idea as swebench grading.get_logs_eval,
    but accepts a string instead of a file path).

    Supports ``test_spec.install_config`` (SWE-rebench / inline specs). R2E-Gym's
    ``get_logs_eval`` does not, and always indexes ``MAP_REPO_VERSION_TO_SPECS[repo]``.
    """
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        END_TEST_OUTPUT,
        MAP_REPO_VERSION_TO_SPECS,
        RESET_FAILED,
        START_TEST_OUTPUT,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    )
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    repo = test_spec.repo
    version = test_spec.version
    if getattr(test_spec, "install_config", None) is not None:
        log_parser = _resolve_swebench_log_parser(test_spec.install_config["log_parser"])
        test_cmd = test_spec.install_config["test_cmd"]
    else:
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    bad_codes = list(
        filter(
            lambda x: x in content,
            [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT],
        )
    )
    if bad_codes:
        return {}, False

    if START_TEST_OUTPUT in content and END_TEST_OUTPUT in content:
        content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    elif test_cmd and test_cmd in content:
        # Raw pytest / agent container logs (no harness markers): match r2e_gym split behavior.
        content = content.split(test_cmd)[-1]

    return log_parser(content, test_spec), True


def _reward_swebench(ds: dict, out: str, get_test_output: bool):
    """SWE-Bench / SWE-Bench Verified: grade via FAIL_TO_PASS and PASS_TO_PASS."""
    from r2egym.agenthub.trajectory.swebench_utils import get_logs_eval, make_test_spec
    from swebench.harness.constants import (
        FAIL_TO_PASS,
        KEY_INSTANCE_ID,
        PASS_TO_PASS,
        ResolvedStatus,
    )
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers import get_eval_type

    # R2E-Gym's make_test_spec ignores install_config and always uses MAP_REPO_VERSION_TO_SPECS.
    # PyPI swebench often has no install_config on TestSpec either; use a minimal namespace so
    # SWE-rebench rows work without upgrading swebench.
    if ds.get("install_config") is not None:
        f2p, p2p = _instance_fail_pass_lists(ds)
        test_spec = SimpleNamespace(
            repo=ds["repo"],
            version=ds.get("version"),
            instance_id=ds[KEY_INSTANCE_ID],
            FAIL_TO_PASS=f2p,
            PASS_TO_PASS=p2p,
            install_config=ds["install_config"],
        )
        eval_status_map, found = _swebench_logs_eval_string(test_spec, out)
    else:
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
    from r2egym.repo_analysis.execution_log_parser import parse_log_fn

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
    from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_fn

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


async def _exec_run_any(container, **kwargs):
    """Use native async exec when available; fallback to thread-wrapped sync exec."""
    if hasattr(container, "exec_run_async"):
        return await container.exec_run_async(**kwargs)
    return await asyncio.to_thread(container.exec_run, **kwargs)


async def setup_container_for_reward_async(
    container,
    dataset: str,
    ds: dict | None = None,
    setup_timeout: int = 60,
) -> None:
    if dataset not in ("swebench", "swebench_verified", "swesmith", "r2e"):
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            "Expected one of: swebench, swebench_verified, swesmith, r2e"
        )

    async def exec_run(cmd: list | str, workdir: str = "/testbed", timeout: int | None = None):
        if timeout is None:
            timeout = setup_timeout
        inner_cmd = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else cmd
        full_cmd = ["timeout", str(timeout), *inner_cmd]
        return await _exec_run_any(
            container,
            cmd=full_cmd,
            workdir=workdir,
            stdout=True,
            stderr=True,
            environment={"PATH": DOCKER_PATH},
        )

    if dataset == "r2e":
        await exec_run("[ -L /root/r2e_tests ] && rm -f /root/r2e_tests || true", workdir="/")
        await exec_run(
            "[ -d /r2e_tests ] && [ ! -d /root/r2e_tests ] && mv /r2e_tests /root/r2e_tests || true",
            workdir="/",
        )
        await exec_run(
            "[ -d /r2e_tests ] && [ -d /root/r2e_tests ] && rm -rf /r2e_tests || true",
            workdir="/",
        )
        await exec_run(
            "[ -d /root/r2e_tests ] && ln -sf /root/r2e_tests /testbed/r2e_tests || true",
            workdir="/",
        )
        await exec_run(
            "[ -e /root/r2e_tests/r2e_tests ] && rm -rf /root/r2e_tests/r2e_tests || true",
            workdir="/",
        )
        await exec_run("echo '#!/bin/sh' > /run_tests.sh", workdir="/")
        await exec_run(
            "echo 'if [ -f /testbed/run_tests.sh ]; then cd /testbed && exec bash /testbed/run_tests.sh \"$@\"; fi' >> /run_tests.sh",
            workdir="/",
        )
        await exec_run(
            "echo 'if [ -f /root/run_tests.sh ]; then exec bash /root/run_tests.sh \"$@\"; fi' >> /run_tests.sh",
            workdir="/",
        )
        await exec_run("chmod +x /run_tests.sh", workdir="/")
        return

    if dataset in ("swebench", "swebench_verified"):
        # First, try existing script
        check = await exec_run("test -f /run_tests.sh && echo YES || echo NO")
        if "YES" in check:
            await exec_run("chmod +x /run_tests.sh")
            return

        # Build from instance spec (SWE-Bench harness)
        if ds is None:
            raise ValueError("swebench_verified setup needs ds to build /run_tests.sh")

        import base64
        from swebench.harness.test_spec.test_spec import make_test_spec

        test_spec = make_test_spec(ds)
        content = test_spec.eval_script  # full bash script with FAIL_TO_PASS etc.
        b64 = base64.b64encode(content.encode()).decode()
        await exec_run(
            f"echo '{b64}' | base64 -d > /run_tests.sh && chmod +x /run_tests.sh",
            workdir="/",
        )
        return

    if dataset == "swesmith":
        if ds is None:
            raise ValueError("setup_container_for_reward_async(..., dataset='swesmith') requires ds")
        import base64
        from r2egym.swesmith.utils import get_test_command

        test_command, _ = get_test_command(ds)
        content = "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                "source /opt/miniconda3/bin/activate",
                "conda activate testbed",
                "cd testbed/",
                ": '>>>>> Start Test Output'",
                test_command,
                ": '>>>>> End Test Output'",
            ]
        ) + "\n"
        b64 = base64.b64encode(content.encode()).decode()
        await exec_run(
            f"echo '{b64}' | base64 -d > /run_tests.sh && chmod +x /run_tests.sh",
            workdir="/",
        )


async def _ensure_r2e_tests_in_testbed_async(container, timeout: int = 60) -> None:
    await _exec_run_any(
        container,
        cmd=[
            "timeout",
            str(timeout),
            "/bin/sh",
            "-c",
            "desired=''; "
            "[ -d /root/r2e_tests ] && desired=/root/r2e_tests || true; "
            "[ -z \"$desired\" ] && [ -d /r2e_tests ] && desired=/r2e_tests || true; "
            "[ -z \"$desired\" ] && exit 0; "
            "rm -rf /testbed/r2e_tests || true; "
            "ln -s \"$desired\" /testbed/r2e_tests || true",
        ],
        workdir="/",
        stdout=True,
        stderr=True,
        environment={"PATH": DOCKER_PATH},
    )


async def _run_tests_in_container_async(
    container,
    timeout: int = 300,
    dataset: str | None = None,
) -> str:
    from swebench.harness.constants import TESTS_ERROR, TESTS_TIMEOUT

    if dataset == "r2e":
        await _ensure_r2e_tests_in_testbed_async(container, timeout=min(timeout, 60))
    test_cmd = _TEST_CMD_BY_DATASET.get(dataset, "/run_tests.sh") if dataset else "/run_tests.sh"
    command = f"timeout {timeout} {test_cmd}"
    # Host-side exec timeout (enroot-py / Ray actor): guest `timeout` does not bound the
    # host `enroot exec` process. Pass timeout= so RayContainerResource.raw_exec_run applies
    # a host guard; keep slack above guest timeout to avoid racing the outer asyncio.wait_for.
    host_exec_cap = float(timeout) + 30.0
    try:
        exec_result = await asyncio.wait_for(
            _exec_run_any(
                container,
                cmd=["/bin/sh", "-c", command],
                workdir="/testbed",
                stdout=True,
                stderr=True,
                environment={"PATH": DOCKER_PATH},
                timeout=host_exec_cap,
            ),
            timeout=host_exec_cap,
        )
        out = exec_result.output.decode("utf-8", errors="replace")
        return re.sub(r"\x1b\[[0-9;]*m|\r", "", out)
    except asyncio.TimeoutError:
        return TESTS_TIMEOUT
    except Exception:
        return TESTS_ERROR


async def _reward_r2e_async(
    container,
    ds: dict,
    out: str,
    get_test_output: bool,
    timeout: int = 60,
):
    from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys, parse_log_fn

    repo_name = ds.get("repo_name", ds.get("repo", ""))
    parse = parse_log_fn(repo_name)(out)
    parse = decolor_dict_keys(parse)
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
    try:
        expected_json = ds.get("expected_output_json")
        if not expected_json:
            result = await _exec_run_any(
                container,
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
            expected_json = result.output.decode("utf-8", errors="replace") if result.output else "{}"
        expected = expected_json if isinstance(expected_json, dict) else json.loads(expected_json)
    except (json.JSONDecodeError, KeyError, TypeError):
        return (0.0, out) if get_test_output else 0.0

    expected = decolor_dict_keys(expected)
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}
    if len(parse) != len(expected):
        reward = 0.0
    else:
        reward = 1.0 if all((not k) or (k in expected and parse[k] == expected[k]) for k in parse) else 0.0
    return (reward, out) if get_test_output else reward


async def reward_from_container_async(
    container,
    ds: dict,
    dataset: str,
    timeout: int = 300,
    get_test_output: bool = False,
):
    if dataset not in ("swebench", "swebench_verified", "swesmith", "r2e"):
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            "Expected one of: swebench, swebench_verified, swesmith, r2e"
        )

    out = await _run_tests_in_container_async(container, timeout=timeout, dataset=dataset)
    if dataset in ("swebench", "swebench_verified"):
        return _reward_swebench(ds, out, get_test_output)
    if dataset == "swesmith":
        return _reward_swesmith(ds, out, get_test_output)
    return await _reward_r2e_async(container, ds, out, get_test_output, timeout=timeout)


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
