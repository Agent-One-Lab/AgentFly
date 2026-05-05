"""
Integration-style check for SWE-Bench Verified reward via ``reward_from_container``.

``test_patch`` must match the HuggingFace row exactly: ``make_test_spec`` parses it
with ``unidiff``; a truncated or hand-edited patch raises ``UnidiffParseError``.
"""

from __future__ import annotations

import pytest

SAMPLE = {
    "docker_image": "epoch-research/swe-bench.eval.x86_64.astropy__astropy-12907",
    "instance": {
        "instance_id": "astropy__astropy-12907",
        "repo": "astropy/astropy",
        "version": "4.3",
        "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
        "test_patch": """
diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py
--- a/astropy/modeling/tests/test_separable.py
+++ b/astropy/modeling/tests/test_separable.py
@@ -28,6 +28,13 @@
 p1 = models.Polynomial1D(1, name='p1')
 
 
+cm_4d_expected = (np.array([False, False, True, True]),
+                  np.array([[True,  True,  False, False],
+                            [True,  True,  False, False],
+                            [False, False, True,  False],
+                            [False, False, False, True]]))
+
+
 compound_models = {
     'cm1': (map3 & sh1 | rot & sh1 | sh1 & sh2 & sh1,
             (np.array([False, False, True]),
@@ -52,7 +59,17 @@
     'cm7': (map2 | p2 & sh1,
             (np.array([False, True]),
              np.array([[True, False], [False, True]]))
-            )
+            ),
+    'cm8': (rot & (sh1 & sh2), cm_4d_expected),
+    'cm9': (rot & sh1 & sh2, cm_4d_expected),
+    'cm10': ((rot & sh1) & sh2, cm_4d_expected),
+    'cm11': (rot & sh1 & (scl1 & scl2),
+             (np.array([False, False, True, True, True]),
+              np.array([[True,  True,  False, False, False],
+                        [True,  True,  False, False, False],
+                        [False, False, True,  False, False],
+                        [False, False, False, True,  False],
+                        [False, False, False, False, True]]))),
 }
 
 
""",
        "FAIL_TO_PASS": ["astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]"],
        "PASS_TO_PASS": ["astropy/modeling/tests/test_separable.py::test_coord_matrix", "astropy/modeling/tests/test_separable.py::test_cdot", "astropy/modeling/tests/test_separable.py::test_cstack", "astropy/modeling/tests/test_separable.py::test_arith_oper", "astropy/modeling/tests/test_separable.py::test_separable[compound_model0-result0]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model1-result1]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model2-result2]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model3-result3]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model4-result4]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model5-result5]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model7-result7]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model8-result8]", "astropy/modeling/tests/test_separable.py::test_custom_model_separable"],
    },
}


def _sample_ready() -> bool:
    img = SAMPLE.get("docker_image", "")
    inst = SAMPLE.get("instance") or {}
    iid = inst.get("instance_id", "")
    return bool(img and img != "FILL_ME" and iid and iid != "FILL_ME")


def test_swe_bench_verified_reward_path():
    if not _sample_ready():
        pytest.skip("Fill SAMPLE when running locally.")
    pytest.importorskip("r2egym")
    pytest.importorskip("swebench")
    import enroot
    from agentfly.rewards.swe_rewards.r2e_gym.eval import (
        reward_from_container,
        setup_container_for_reward,
    )
    client = enroot.from_env()
    container = client.containers.run(SAMPLE["docker_image"], detach=True, remove=False)
    try:
        setup_container_for_reward(
            container, dataset="swebench_verified", ds=SAMPLE["instance"]
        )
        reward, out = reward_from_container(
            container,
            SAMPLE["instance"],
            dataset="swebench_verified",
            timeout=600,
            get_test_output=True,
        )
        assert reward in (0.0, 1.0), f"unexpected reward: {reward!r}"
        assert isinstance(out, str)
        print("test output (truncated):", out[:2000])
        print("reward:", reward)
    finally:
        container.stop(timeout=30)
