"""Lazy public exports must remain visible to static documentation tooling."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(scope="module")
def static_modules():
    # Griffe is supplied by the optional docs dependencies, not the runtime API.
    griffe = pytest.importorskip("griffe")
    loader = griffe.GriffeLoader(
        search_paths=[Path(__file__).resolve().parents[3] / "src"],
        allow_inspection=False,
    )
    loader.load("agentfly")
    return loader.modules_collection


@pytest.mark.parametrize(
    ("package_name", "unavailable_exports"),
    [
        ("agentfly.rewards", set()),
        # This pre-existing broken export is outside the lazy-discovery fix.
        ("agentfly.tools", {"google_search_serper"}),
    ],
)
def test_public_exports_have_static_declarations(
    static_modules, package_name, unavailable_exports
):
    package = static_modules[package_name]
    assert package.exports
    for name in sorted(set(package.exports) - unavailable_exports):
        # Check every public export so adding one without a static declaration
        # fails here, even if no documentation page references it yet.
        assert name in package.members, f"{package_name}.{name} is not declared"


@pytest.mark.parametrize(
    ("public_path", "definition_path"),
    [
        (
            "agentfly.rewards.math_equal_reward",
            "agentfly.rewards.math_reward.math_equal_reward",
        ),
        (
            "agentfly.rewards.qa_f1_reward",
            "agentfly.rewards.qa_reward.qa_f1_reward",
        ),
        (
            "agentfly.tools.code_interpreter",
            "agentfly.tools.src.code.tools.code_interpreter",
        ),
    ],
)
def test_documented_lazy_exports_resolve_statically(
    static_modules, public_path, definition_path
):
    assert static_modules[public_path].final_target.path == definition_path


@pytest.mark.parametrize(
    ("package_name", "export_name"),
    [
        ("agentfly.rewards", "math_equal_reward"),
        ("agentfly.tools", "code_interpreter"),
    ],
)
def test_public_exports_remain_lazy_at_runtime(tmp_path, package_name, export_name):
    # A fresh interpreter prevents earlier tests' imports from masking eager
    # implementation loading. Attribute access exercises the real lazy loader.
    code = textwrap.dedent(
        """
        import importlib
        import sys

        package_name, export_name = sys.argv[1:]
        package = importlib.import_module(package_name)
        for module_name in ("agentfly.rewards.impls", "agentfly.tools.impls"):
            assert module_name not in sys.modules, module_name
        assert export_name not in vars(package)

        value = getattr(package, export_name)
        impls = sys.modules[f"{package_name}.impls"]
        assert value is getattr(impls, export_name)
        assert callable(value)
        assert vars(package)[export_name] is value
        assert getattr(package, export_name) is value
        """
    )
    env = {
        **os.environ,
        "AF_HOME": str(tmp_path),
        "AF_CACHE_DIR": str(tmp_path / "cache"),
        "AF_CONFIG_DIR": str(tmp_path / "config"),
        "AF_DATA_DIR": str(tmp_path / "data"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, package_name, export_name],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
