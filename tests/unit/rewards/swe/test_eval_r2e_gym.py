import pytest
import os
import contextlib
# Set ENROOT_IMAGES_PATH in your shell before running this test; see docs/examples/swe.md.
os.environ.setdefault("ENROOT_IMAGES_PATH", "/path/to/enroot/images/r2e-gym-lite")
os.environ["ENROOT_ASYNC"] = "1"
import enroot
from agentfly.rewards.swe_rewards.r2e_gym.eval import (
    reward_from_container,
    setup_container_for_reward,
    _ensure_r2e_tests_in_testbed,
)


SAMPLE = {
    "repo_name": "aiohttp",
    "docker_image": "namanjain12/aiohttp_final:f0d74880deec8fcd982bce639c93c5e130d41198",
    "git_commit_hash": "f0d74880deec8fcd982bce639c93c5e130d41198",
    "expected_output_json": {
        "TestUrlDispatcher.test_add_invalid_path": "PASSED",
        "TestUrlDispatcher.test_add_route_invalid_method": "PASSED",
        "TestUrlDispatcher.test_add_route_not_started_with_slash": "PASSED",
        "TestUrlDispatcher.test_add_route_root": "PASSED",
        "TestUrlDispatcher.test_add_route_simple": "PASSED",
        "TestUrlDispatcher.test_add_route_with_re": "PASSED",
        "TestUrlDispatcher.test_add_route_with_re_and_slashes": "PASSED",
        "TestUrlDispatcher.test_add_route_with_re_including_slashes": "PASSED",
        "TestUrlDispatcher.test_add_route_with_re_not_match": "PASSED",
        "TestUrlDispatcher.test_add_static": "PASSED",
        "TestUrlDispatcher.test_add_url_escaping": "PASSED",
        "TestUrlDispatcher.test_add_url_invalid1": "PASSED",
        "TestUrlDispatcher.test_add_url_invalid2": "PASSED",
        "TestUrlDispatcher.test_add_url_invalid3": "PASSED",
        "TestUrlDispatcher.test_add_url_invalid4": "PASSED",
        "TestUrlDispatcher.test_add_with_matchdict": "PASSED",
        "TestUrlDispatcher.test_add_with_name": "PASSED",
        "TestUrlDispatcher.test_add_with_tailing_slash": "PASSED",
        "TestUrlDispatcher.test_any_method": "PASSED",
        "TestUrlDispatcher.test_contains": "PASSED",
        "TestUrlDispatcher.test_custom_expect_handler_dynamic": "PASSED",
        "TestUrlDispatcher.test_custom_expect_handler_plain": "PASSED",
        "TestUrlDispatcher.test_default_expect_handler": "PASSED",
        "TestUrlDispatcher.test_double_add_url_with_the_same_name": "PASSED",
        "TestUrlDispatcher.test_dynamic_match_non_ascii": "PASSED",
        "TestUrlDispatcher.test_dynamic_match_two_part2": "PASSED",
        "TestUrlDispatcher.test_dynamic_match_unquoted_path": "PASSED",
        "TestUrlDispatcher.test_dynamic_match_with_static_part": "PASSED",
        "TestUrlDispatcher.test_dynamic_not_match": "PASSED",
        "TestUrlDispatcher.test_dynamic_repr": "PASSED",
        "TestUrlDispatcher.test_dynamic_with_trailing_slash": "PASSED",
        "TestUrlDispatcher.test_expect_handler_non_coroutine": "PASSED",
        "TestUrlDispatcher.test_iter": "PASSED",
        "TestUrlDispatcher.test_len": "PASSED",
        "TestUrlDispatcher.test_match_second_result_in_table": "PASSED",
        "TestUrlDispatcher.test_not_allowed_repr": "PASSED",
        "TestUrlDispatcher.test_not_found_repr": "PASSED",
        "TestUrlDispatcher.test_plain_not_match": "PASSED",
        "TestUrlDispatcher.test_plain_repr": "PASSED",
        "TestUrlDispatcher.test_raise_method_not_allowed": "PASSED",
        "TestUrlDispatcher.test_raise_method_not_found": "PASSED",
        "TestUrlDispatcher.test_register_route": "PASSED",
        "TestUrlDispatcher.test_register_route_checks": "PASSED",
        "TestUrlDispatcher.test_regular_match_info": "PASSED",
        "TestUrlDispatcher.test_route_dynamic": "PASSED",
        "TestUrlDispatcher.test_route_dynamic_with_regex": "PASSED",
        "TestUrlDispatcher.test_route_dynamic_with_regex_spec": "PASSED",
        "TestUrlDispatcher.test_route_dynamic_with_regex_spec_and_trailing_slash": "PASSED",
        "TestUrlDispatcher.test_route_plain": "PASSED",
        "TestUrlDispatcher.test_route_unknown_route_name": "PASSED",
        "TestUrlDispatcher.test_route_with_qs": "PASSED",
        "TestUrlDispatcher.test_routes_abc": "PASSED",
        "TestUrlDispatcher.test_routes_view_contains": "PASSED",
        "TestUrlDispatcher.test_routes_view_iter": "PASSED",
        "TestUrlDispatcher.test_routes_view_len": "PASSED",
        "TestUrlDispatcher.test_static_adds_slash": "PASSED",
        "TestUrlDispatcher.test_static_dont_add_trailing_slash": "PASSED",
        "TestUrlDispatcher.test_static_handle_again": "PASSED",
        "TestUrlDispatcher.test_static_handle_eof": "PASSED",
        "TestUrlDispatcher.test_static_handle_exception": "PASSED",
        "TestUrlDispatcher.test_static_not_match": "PASSED",
        "TestUrlDispatcher.test_static_repr": "PASSED",
        "TestUrlDispatcher.test_system_route": "PASSED",
        "TestUrlDispatcher.test_add_route_with_invalid_re": "FAILED"
    }

}

def test_eval_r2e_gym():

    client = enroot.from_env()
    container = client.containers.create(
        image=SAMPLE["docker_image"]
    )
    container.start()

    setup_container_for_reward(container, dataset="r2e")

    # container.exec_run(f"git checkout {SAMPLE['git_commit_hash']}")

    reward, out = reward_from_container(
        container,
        SAMPLE,
        dataset="r2e",
        timeout=120,
        get_test_output=True,
    )
    print("Output:", out)
    print("Reward:", reward)

    container.kill()


def test_r2e_tests_symlink_is_not_recursive():
    """
    Regression test for r2e pytest collection blow-ups caused by recursive
    nesting like r2e_tests/r2e_tests/.../test_*.py.
    """
    client = enroot.from_env()
    container = client.containers.create(
        image=SAMPLE["docker_image"]
    )
    container.start()
    try:
        def _check_paths(label: str) -> str:
            res = container.exec_run(
                cmd=[
                    "/bin/sh",
                    "-c",
                    "set -eu; "
                    f'echo "== {label} =="; '
                    'echo "[/testbed] /testbed => $(ls -ld /testbed 2>/dev/null || true)"; '
                    'echo "[/testbed] /testbed/r2e_tests => $(ls -ld /testbed/r2e_tests 2>/dev/null || true)"; '
                    'echo "[/testbed] nested => $(test -e /testbed/r2e_tests/r2e_tests && echo YES || echo NO)"; '
                    'echo "[/root] /root => $(ls -ld /root 2>/dev/null || true)"; '
                    'echo "[/root] /root/r2e_tests => $(ls -ld /root/r2e_tests 2>/dev/null || true)"; '
                    'echo "[/root] nested => $(test -e /root/r2e_tests/r2e_tests && echo YES || echo NO)"; '
                    'echo "[/r2e_tests] /r2e_tests => $(ls -ld /r2e_tests 2>/dev/null || true)"; '
                    'echo "[/r2e_tests] nested => $(test -e /r2e_tests/r2e_tests && echo YES || echo NO)"; '
                ],
                workdir="/",
                stdout=True,
                stderr=True,
            )
            return res.output.decode("utf-8", errors="replace") if getattr(res, "output", None) else ""

        before = _check_paths("BEFORE setup")

        # Apply setup more than once to catch idempotency issues / accumulation.
        setup_container_for_reward(container, dataset="r2e")
        after1 = _check_paths("AFTER setup x1")

        setup_container_for_reward(container, dataset="r2e")
        after2 = _check_paths("AFTER setup x2")

        # The real reward flow also ensures /testbed/r2e_tests right before running
        # the test command. Check that phase too.
        _ensure_r2e_tests_in_testbed(container, timeout=60)
        after_ensure = _check_paths("AFTER ensure")

        # Regression assertions: we should not have nested r2e_tests under any of the
        # potential roots. Recursive pytest collection typically indicates one of:
        #   /testbed/r2e_tests/r2e_tests
        #   /root/r2e_tests/r2e_tests
        #   /r2e_tests/r2e_tests
        assert "[/testbed] nested => NO" in after1, (
            "Expected /testbed/r2e_tests/r2e_tests to NOT exist after setup.\n"
            f"{before}\n{after1}\n{after2}"
        )
        assert "[/root] nested => NO" in after1, (
            "Expected /root/r2e_tests/r2e_tests to NOT exist after setup.\n"
            f"{before}\n{after1}\n{after2}"
        )
        assert "[/r2e_tests] nested => NO" in after1, (
            "Expected /r2e_tests/r2e_tests to NOT exist after setup.\n"
            f"{before}\n{after1}\n{after2}"
        )
        assert "[/testbed] nested => NO" in after_ensure, (
            "Expected /testbed/r2e_tests/r2e_tests to NOT exist after ensure.\n"
            f"{before}\n{after1}\n{after2}\n{after_ensure}"
        )
        assert "[/root] nested => NO" in after_ensure, (
            "Expected /root/r2e_tests/r2e_tests to NOT exist after ensure.\n"
            f"{before}\n{after1}\n{after2}\n{after_ensure}"
        )
        assert "[/r2e_tests] nested => NO" in after_ensure, (
            "Expected /r2e_tests/r2e_tests to NOT exist after ensure.\n"
            f"{before}\n{after1}\n{after2}\n{after_ensure}"
        )
    finally:
        container.kill()
