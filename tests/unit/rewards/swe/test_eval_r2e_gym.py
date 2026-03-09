import pytest
import os
import contextlib
os.environ["ENROOT_IMAGES_PATH"] = "/mnt/weka/home/renxi.wang/Agent-One-Lab/enroot-py/data/images/r2e-gym-lite"
os.environ["ENROOT_ASYNC"] = "1"
import enroot
from agentfly.rewards.swe_rewards.r2e_gym.eval import reward_from_container, setup_container_for_reward





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
