# Predefined Rewards

Every built-in reward, by the name you pass as `reward_name` (or import from `agentfly.rewards`).
The task data row's fields reach the reward as keyword arguments by name (see
[Reward System](../../features/reward_system.md)).

| Reward | Task | Value |
|---|---|---|
| `math_equal_reward` | Math | 1 if `final_response` equals `answer` symbolically, else 0. |
| `math_equal_reward_tool` | Math with tools | 1 correct, 0.1 wrong, 0 if no tool was called. The answer must be in `\boxed{}` or the whole response. |
| `math_equal_reward_think` | Math with tools | Like `math_equal_reward_tool`, and every assistant turn must contain `<think>...</think>`. |
| `math_reward_thought_with_tool` | Math with tools | Like `math_equal_reward_tool`, and every assistant turn must start with "Thought". |
| `math_string_equal_reward_tool` | Math with tools | Like `math_equal_reward_tool`, comparing the last number in the response as a string. |
| `qa_f1_reward` | QA | F1 of `final_response` against `answer`. |
| `qa_f1_reward_tool` | QA with tools | F1, or 0 unless the trajectory has at least two tool results. |
| `qa_em_reward` | QA (Search-R1) | Exact match of the `<answer>` content against `golden_answers`. |
| `qa_em_format_reward` | QA (Search-R1) | 1 correct, 0.1 wrong, 0 if the format is off (`<answer>` tags, a tool call, `<think>` in earlier turns). |
| `qa_em_reward_tool_call` | QA (Search-R1) | Exact match of the answer tool's argument (or the response) against `golden_answers`. |
| `ok_vqa_reward` | Visual QA | Best F1 against any of `answers`. |
| `infoseek_reward` | Visual QA | 0.8 × best F1, plus 0.2 if the trajectory has at least two tool results. |
| `webshop_reward` | WebShop | 10 if the purchase fully matches the goal, else 0. |
| `webshop_score_reward` | WebShop | The environment's purchase score in [0, 1]. |
| `alfworld_reward` | ALFWorld | 10 if the task was completed, else 0. |
| `scienceworld_reward` | ScienceWorld | 10 if the task was completed, else 0. |
| `chess_puzzle_reward` | Chess puzzles | Dense: move quality from Stockfish, plus a bonus for solving. |
| `chess_puzzle_reward_simple` | Chess puzzles | 1 if the puzzle was solved, else 0. |
| `gui_reward` | GUI grounding | Action accuracy against `gt_action` / `gt_bbox` / `gt_input_text`. |
| `r2e_gym_reward` | SWE (R2E-Gym, SWE-bench, SWE-smith) | 1 if the task's tests pass in the rollout's container, else 0. |
| `swe_reward` | SWE-smith | 1 if the agent's patch resolves the task, else 0. |
| `vlm_as_judge_pass_reward` | SimuScene | 1 if every VLM judge passes the rendered video, else 0. |
| `vlm_as_judge_pass_reward_multi_model` | SimuScene | 1 if a majority of judges pass, else 0.06 if the video opens (0.04 if it was only saved); 0 without a video. |
| `vlm_as_judge_ratio_reward_multi_model` | SimuScene | Fraction of questions the judges' majority vote marks true. |
| `llm_as_judge_pass_reward_multi_model` | SimuScene | Text-only ablation: judges read the code instead of a video. |

Rewards that read an environment (`webshop_*`, `alfworld_reward`, `scienceworld_reward`,
`chess_*`, the SWE rewards) use the same resource as the task's tools, through the rollout's
`context`. The ones below render their docstrings.

::: agentfly.rewards.qa_f1_reward

::: agentfly.rewards.qa_f1_reward_tool

::: agentfly.rewards.math_equal_reward

::: agentfly.rewards.math_equal_reward_tool

::: agentfly.rewards.webshop_rewards.webshop_reward

::: agentfly.rewards.webshop_rewards.webshop_score_reward

::: agentfly.rewards.alfworld_reward

::: agentfly.rewards.scienceworld_rewards.scienceworld_reward
