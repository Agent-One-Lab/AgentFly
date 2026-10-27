"""``agentfly deploy``: start a vLLM OpenAI-compatible server for a model (prints the command, then becomes it).

Serves a model with an optional chat template, tool-call parser and tensor/pipeline/data
parallelism, e.g. ``python -m agentfly.utils.deploy --model-name-or-path Qwen/Qwen2.5-3B-Instruct``.
"""

import os
import shlex
import shutil
import sys
from typing import List

import click
from chat_bricks import get_template

from .. import AF_CACHE_DIR


def write_chat_template(template: str) -> str:
    """Write ``template``'s Jinja to ``$AF_CACHE_DIR/templates/<template>.jinja`` and return the path.

    One file per template name, written atomically, so concurrent deploys with different
    templates never overwrite each other's file.
    """
    folder = os.path.join(AF_CACHE_DIR, "templates")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{template}.jinja")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(get_template(template).jinja_template())
    os.replace(tmp, path)
    return path


def vllm_serve_command(
    model_name_or_path,
    template,
    tp,
    pp,
    dp,
    gpu_memory_utilization,
    tool_call_parser,
    port,
    allowed_local_media_path,
    override_generation_config,
) -> List[str]:
    """The ``vllm serve`` command line (argv) for these options."""
    command = [
        _vllm_executable(),
        "serve",
        model_name_or_path,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--pipeline-parallel-size", str(pp),
        "--data-parallel-size", str(dp),
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]  # fmt: skip
    if template is not None:
        command += ["--chat-template", write_chat_template(template)]
    if tool_call_parser:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
    if allowed_local_media_path:
        command += ["--allowed-local-media-path", allowed_local_media_path]
    if override_generation_config:
        command += ["--override-generation-config", override_generation_config]
    return command


def _vllm_executable() -> str:
    """The ``vllm`` script of the running Python environment, else ``vllm`` on PATH."""
    local = os.path.join(os.path.dirname(sys.executable), "vllm")
    if os.access(local, os.X_OK):
        return local
    return shutil.which("vllm") or "vllm"


@click.command()
@click.option("--model-name-or-path", required=True, help="Hugging Face model id or local path.")
@click.option(
    "--template",
    default=None,
    help="chat-bricks template name; its Jinja is passed to vllm serve as --chat-template. "
    "Default: the model's own chat template.",
)
@click.option("--tp", type=int, default=1, show_default=True, help="Tensor-parallel size.")
@click.option("--pp", type=int, default=1, show_default=True, help="Pipeline-parallel size.")
@click.option("--dp", type=int, default=1, show_default=True, help="Data-parallel size.")
@click.option(
    "--gpu-memory-utilization", type=float, default=0.8, show_default=True, help="Fraction of GPU memory vLLM may use."
)
@click.option(
    "--tool-call-parser",
    type=str,
    default=None,
    help="vLLM tool-call parser (e.g. hermes for Qwen2.5, qwen3_coder for Qwen3-Coder); enables tool calling. "
    "Default: no tool calling.",
)
@click.option("--port", type=int, default=8000, show_default=True, help="Port of the OpenAI-compatible server.")
@click.option(
    "--allowed-local-media-path", type=str, default=None, help="Directory vision models may read local images from."
)
@click.option(
    "--override-generation-config",
    type=str,
    default=None,
    help="JSON of default sampling settings, e.g. '{\"temperature\": 0.6}'.",
)
def main(
    model_name_or_path,
    template,
    tp,
    pp,
    dp,
    gpu_memory_utilization,
    tool_call_parser,
    port,
    allowed_local_media_path,
    override_generation_config,
):
    """Start a vLLM OpenAI-compatible server for a model (prints the vllm serve command, then runs it)."""
    command = vllm_serve_command(
        model_name_or_path,
        template,
        tp,
        pp,
        dp,
        gpu_memory_utilization,
        tool_call_parser,
        port,
        allowed_local_media_path,
        override_generation_config,
    )
    print(shlex.join(command), flush=True)  # flushed so it shows up in log files before vLLM's output
    # Become the vLLM process: signals (kill, Ctrl+C, a scheduler's SIGTERM) reach vLLM itself
    # and its exit code is ours, instead of vLLM outliving a killed parent.
    try:
        os.execv(command[0], command)
    except FileNotFoundError:
        sys.exit("vllm is not installed in this environment (it comes with agentfly-rl's dependencies).")


if __name__ == "__main__":
    main()
