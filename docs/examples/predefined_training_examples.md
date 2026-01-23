# 🚀 Predefined Training Examples

This document provides a comprehensive overview of all the predefined training examples available in the `verl/run_agents/` folder. Each example has been tested and configured for specific agent types and tasks.

## 📋 Training Examples Overview

| Example Name | Checkpoint | Model | Agent Type | Dataset | Tools | Reward Function | Max Steps | Training Steps | Batch Size | Learning Rate | Advantage Estimator |
|--------------|------------|-------|------------|---------|-------|-----------------|-----------|----------------|------------|---------------|-------------------|
| **[🖥️ GUI Agent](verl/examples/run_agents/run_gui_agent.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | GUI | GUI R1 Train/Test | pyautogui_code_generator | gui_reward | 4 | 200 | 64 | 4e-7 | GRPO |
| **[🔍 VLM QA Agent](verl/examples/run_agents/run_vlm_qa.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | React | InfoSeek Train/Val | asyncdense_retrieve, answer_qa | infoseek_reward | 5 | 200 | 128 | 5e-7 | Reinforce++ |
| **[💻 Code Agent](verl/examples/run_agents/run_code_agent.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Code | Orz Math 57K Train | code_interpreter | math_reward_tool | 8 | 200 | 64 | 5e-7 | GRPO |
| **[🛒 Webshop Agent](verl/examples/run_agents/run_webshop_agent.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | React | Webshop Goals Train/Val | webshop_browser | webshop_reward | 8 | 200 | 128 | 4e-7 | GRPO |
| **[🔎 Search Agent](verl/examples/run_agents/run_react_agent_search.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | React | HotpotQA Train | asyncdense_retrieve, answer_qa | qa_f1_reward_format | 4 | 100 | 128 | 5e-7 | Reinforce++ |
| **[🧪 Science World Agent](verl/examples/run_agents/run_science_world.slurm)** | [checkpoint](placeholder_link) | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | React | ScienceWorld Train/Val | scienceworld_explorer | scienceworld_reward | 20 | 100 | 128 | 4e-7 | Reinforce++ |
| **[🏠 ALFWorld Agent](verl/examples/run_agents/run_react_agent_alfworld.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | React | ALFWorld Train/Val | alfworld_step, alfworld_get_admissible_commands, alfworld_get_task_objective | alfworld_episode_reward | 10 | 150 | 64 | 1e-6 | Reinforce++ |
| **[📚 Retrieve Agent](verl/examples/run_agents/run_react_agent_search.sh)** | [checkpoint](placeholder_link) | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | React | HotpotQA Train | asyncdense_retrieve, answer_qa | qa_f1_reward_format | 4 | 100 | 128 | 5e-7 | Reinforce++ |

## ⚙️ Detailed Configurations
For detailed configurations, please refer to the training scripts.

## Using Trained Agents

The trained agents can be easily tested and used through the provided test examples. Each agent type has a corresponding test function that demonstrates its capabilities.

### Available Agent Demos

| Agent Type | Test Function | Command |
|------------|---------------|---------|
| **Code Agent** | `test_code_agent` | `pytest agentfly/tests/docs/examples/test_react_agents.py::test_code_agent -s` |
| **VLM QA Agent** | `test_react_vqa_agent` | `pytest agentfly/tests/docs/examples/test_react_agents.py::test_react_vqa_agent -s` |
| **VLM Retrieval Agent** | `test_react_vqa_retrieval_agent` | `pytest agentfly/tests/docs/examples/test_react_agents.py::test_react_vqa_retrieval_agent -s` |
| **Science World Agent** | `test_react_scienceworld_agent` | `pytest agentfly/tests/docs/examples/test_react_agents.py::test_react_scienceworld_agent -s` |
| **Webshop Agent** | `test_react_webshop_agent` | `pytest agentfly/tests/docs/examples/test_react_agents.py::test_react_webshop_agent -s` |

### Running the Demos

To run any of the agent demos, use the pytest command with the `-s` flag to see the streaming output:

```bash
# Example: Run the Code Agent demo
pytest agentfly/tests/docs/examples/test_react_agents.py::test_code_agent -s

# Example: Run the VLM QA Agent demo
pytest agentfly/tests/docs/examples/test_react_agents.py::test_react_vqa_agent -s
```

### Demo Descriptions

- **Code Agent**: Solves mathematical problems using code interpretation
- **VLM QA Agent**: Answers questions about images using vision-language understanding
- **VLM Retrieval Agent**: Retrieves relevant information and answers questions about images
- **Science World Agent**: Performs scientific reasoning tasks in a virtual environment
- **Webshop Agent**: Navigates and shops in an e-commerce environment

Each demo will show the agent's reasoning process, tool usage, and final results in real-time.

## Training Curves

Training curves and metrics are logged to WandB for each experiment.
<p align="center">
  <a href="https://wandb.ai/AgentRL/Open">
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg" width="40" alt="W&B Logo"/><br/>
    <b> 🦋 AgentFly </b><br/>
    <i> 📊 Curves & Logs</i><br/>
  </a>
</p>


<a href="https://wandb.ai/AgentRL/Open" target="_blank"
   style="
     display: inline-flex;
     align-items: center;
     padding: 8px 14px;
     border-radius: 6px;
     background-color: #f6f8fa;
     text-decoration: none;
     font-family: Arial, sans-serif;
     font-size: 14px;
     font-weight: 500;
     color: #333;
     box-shadow: 0 2px 4px rgba(0,0,0,0.1);
     transition: background 0.2s, transform 0.2s;
   "
   onmouseover="this.style.background='#e9ecef'; this.style.transform='translateY(-1px)';"
   onmouseout="this.style.background='#f6f8fa'; this.style.transform='none';">
  <img src="https://wandb.ai/logo.svg" alt="WandB Logo"
       style="height: 20px; margin-right: 8px;" />
  View Training Curves on WandB
</a>
