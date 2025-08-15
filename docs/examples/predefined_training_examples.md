# Predefined Training Examples

This document provides a comprehensive overview of all the predefined training examples available in the `verl/examples/run_agents/` folder. Each example has been tested and configured for specific agent types and tasks.

## Training Examples Overview

| Example Name | Model | Agent Type | Dataset | Tools | Reward Function | Max Steps | Training Steps | Batch Size | Learning Rate | Advantage Estimator |
|--------------|-------|------------|---------|-------|-----------------|-----------|----------------|------------|---------------|-------------------|
| **GUI Agent** | Qwen2.5-VL-Instruct | GUI | GUI R1 Train/Test | pyautogui_code_generator | gui_reward | 4 | 200 | 64 | 4e-7 | GRPO |
| **VLM QA Agent** | Qwen2.5-VL-Instruct | React | InfoSeek Train/Val | asyncdense_retrieve, answer_qa | infoseek_reward | 6 | 200 | 128 | 5e-7 | Reinforce++ |
| **Code Agent** | Qwen2.5-Instruct | Code | Orz Math 57K Train | code_interpreter | math_reward_tool | 8 | 200 | 64 | 5e-7 | GRPO |
| **Webshop Agent** | Qwen2.5-Instruct | React | Webshop Goals Train/Val | webshop_browser | webshop_reward | 8 | 200 | 128 | 4e-7 | GRPO |
| **Search Agent** | Qwen2.5-Instruct | React | HotpotQA Train | google_search, answer | qa_f1_reward | 4 | 200 | 128 | 5e-7 | Reinforce++ |
| **Science World Agent** | Qwen2.5-Instruct | React | ScienceWorld Train/Val | scienceworld_explorer | scienceworld_reward | 20 | 200 | 128 | 4e-7 | Reinforce++ |
| **ALFWorld Agent** | Qwen2.5-Instruct | React | ALFWorld Train/Val | alfworld_step, alfworld_get_admissible_commands, alfworld_get_task_objective | alfworld_episode_reward | 10 | 150 | 64 | 1e-6 | Reinforce++ |
| **Retrieve Agent** | Qwen2.5-Instruct | React | HotpotQA Train | asyncdense_retrieve, answer_qa | qa_f1_reward_format | 4 | 100 | 128 | 5e-7 | Reinforce++ |

## Detailed Configurations
For detailed configurations, please refer to the training scripts.

## Training Curves

Training curves and metrics are logged to WandB for each experiment.

