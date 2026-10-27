"""``agentfly deploy``: help text, chat-template files, and the vllm serve command it runs."""

import pytest
from click.testing import CliRunner

from agentfly.utils import deploy


@pytest.fixture
def ran(monkeypatch, tmp_path):
    """Capture the command instead of starting vLLM; keep template files in tmp_path."""
    monkeypatch.setattr(deploy, "AF_CACHE_DIR", str(tmp_path))
    commands = []

    def fake_execv(path, argv):  # os.execv never returns; vLLM's exit code becomes ours
        commands.append(" ".join(argv))
        raise SystemExit(3)

    monkeypatch.setattr(deploy.os, "execv", fake_execv)
    return commands


def test_help_names_the_command_and_describes_options():
    result = CliRunner().invoke(deploy.main, ["--help"], prog_name="agentfly deploy")
    assert result.exit_code == 0
    assert "Usage: agentfly deploy" in result.output
    text = " ".join(result.output.split())  # help is line-wrapped
    assert "enables tool calling" in text and "Tensor-parallel size" in text


def test_model_is_required(ran):
    result = CliRunner().invoke(deploy.main, [])
    assert result.exit_code != 0 and "--model-name-or-path" in result.output and ran == []


def test_command_and_exit_code(ran):
    result = CliRunner().invoke(deploy.main, ["--model-name-or-path", "Qwen/Qwen2.5-3B-Instruct", "--tool-call-parser", "hermes"])
    assert result.exit_code == 3  # vLLM's exit code is passed through
    assert "vllm serve Qwen/Qwen2.5-3B-Instruct" in ran[0]
    assert "--enable-auto-tool-choice --tool-call-parser hermes" in ran[0]
    assert "serve Qwen/Qwen2.5-3B-Instruct" in result.output  # the command is printed


def test_no_tool_calling_by_default(ran):
    CliRunner().invoke(deploy.main, ["--model-name-or-path", "m"])
    assert "--tool-call-parser" not in ran[0]


def test_each_template_gets_its_own_file(ran, tmp_path):
    CliRunner().invoke(deploy.main, ["--model-name-or-path", "m", "--template", "qwen2.5"])
    CliRunner().invoke(deploy.main, ["--model-name-or-path", "m", "--template", "llama-3.2"])
    files = sorted(p.name for p in (tmp_path / "templates").iterdir())
    assert files == ["llama-3.2.jinja", "qwen2.5.jinja"]
    assert f"--chat-template {tmp_path}/templates/qwen2.5.jinja" in ran[0]
    assert (tmp_path / "templates" / "qwen2.5.jinja").read_text() != (tmp_path / "templates" / "llama-3.2.jinja").read_text()


def test_json_option_is_one_argument(monkeypatch, tmp_path):
    # No shell: a JSON value with spaces reaches vLLM as a single argument.
    monkeypatch.setattr(deploy, "AF_CACHE_DIR", str(tmp_path))
    argv = deploy.vllm_serve_command("m", None, 1, 1, 1, 0.8, None, 8000, None, '{"temperature": 0.6}')
    assert argv[argv.index("--override-generation-config") + 1] == '{"temperature": 0.6}'
