

import os
from ..templates import get_template
from .. import AGENT_DATA_DIR
import click


def vllm_serve(model_name_or_path, template, tp, pp, dp, gpu_memory_utilization):
    port = 8000
    
    
    if template is None:
        template_option = ""
    else:
        jinja_template = get_template(template).jinja_template()
        if not os.path.exists(f"{AGENT_DATA_DIR}/cache"):
            os.makedirs(f"{AGENT_DATA_DIR}/cache")
        with open(f"{AGENT_DATA_DIR}/cache/jinja_template.jinja", "w") as f:
            f.write(jinja_template)
        template_option = f"--chat-template {AGENT_DATA_DIR}/cache/jinja_template.jinja"
    # command = f"vllm serve {model_name_or_path} --chat-template {AGENT_DATA_DIR}/cache/jinja_template.jinja --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --data-parallel-size {dp} --port {port} --enable-auto-tool-choice --tool-call-parser hermes --expand-tools-even-if-tool-choice-none"
    command = f"""vllm serve {model_name_or_path} \
{template_option} \
--tensor-parallel-size {tp} \
--pipeline-parallel-size {pp} \
--data-parallel-size {dp} --port {port} \
--gpu-memory-utilization {gpu_memory_utilization} \
--enable-auto-tool-choice --tool-call-parser hermes"""

    print(command)
    os.system(command)



@click.command()
@click.option("--model_name_or_path")
@click.option("--template", default=None)
@click.option("--tp", type=int, default=1)
@click.option("--pp", type=int, default=1)
@click.option("--dp", type=int, default=1)
@click.option("--gpu_memory_utilization", type=float, default=0.8)
def main(model_name_or_path, template, tp, pp, dp, gpu_memory_utilization):
    vllm_serve(model_name_or_path, template, tp, pp, dp, gpu_memory_utilization)


if __name__=="__main__":
    "python -m agentfly.utils.deploy --model_name_or_path Qwen/Qwen2.5-3B-Instruct --template qwen2.5 --tp 2 --dp 2"
    "python -m agentfly.utils.deploy --model_name_or_path openai/gpt-oss-20b --tp 1 --dp 1"
    "python -m agentfly.utils.deploy --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tp 1 --dp 1"
    main()