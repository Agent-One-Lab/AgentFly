"""Verbatim mini-swe-agent prompt constants for :class:`MinisweAgent`.

GENERATED FILE — do not edit by hand. Regenerated from real miniswe SFT
trajectory records by SkillsScrape's ``scripts/rl/gen_miniswe_prompts.py``
so the rendered prompts match what the SFT'd model saw byte-for-byte
(majority variant over 300/300 sampled records).

``INSTANCE_BOILERPLATE`` contains ``__SYSTEM_INFORMATION__`` where the
container's system information (uname) goes; MinisweAgent substitutes the
actual value at rollout time.
"""

SYSTEM_BASE = "You are a helpful assistant that can interact with a computer."

# Header + per-skill block wording of the CURRENT collection/eval config
# (trajectories/configs/mini_swe_skill.yaml + harbor_agent._fill). Early
# single-skill SFT records used a singular variant ("A Skill is available
# for this task:"); the current stack renders the plural header for any
# skill count, so that is what RL and eval use.
SKILL_HEADER = (
    "Skills are available for this task. Consider every listed skill equally and\n"
    "use whichever are relevant:"
)

INSTANCE_PREFIX = "Please solve this issue: "
# The template's own joiner between the task text and the boilerplate.
# The task text is inserted VERBATIM (instruction.md usually ends with a
# newline, which is why records mostly show three consecutive newlines here).
INSTANCE_SEPARATOR = "\n\n"
SYSTEM_INFORMATION_SENTINEL = "__SYSTEM_INFORMATION__"

INSTANCE_BOILERPLATE = "You can execute bash commands and edit files to implement the necessary changes.\n\n## Recommended Workflow\n\nThis workflow should be done step-by-step so that you can iterate on your changes and any possible problems.\n\n1. Analyze the codebase by finding and reading relevant files\n2. Create a script to reproduce the issue\n3. Edit the source code to resolve the issue\n4. Verify your fix works by running your script again\n5. Test edge cases to ensure your fix is robust\n6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>\n\n## Command Execution Rules\n\nYou are operating in an environment where\n\n1. You issue at least one command\n2. The system executes the command(s) in a subshell\n3. You see the result(s)\n4. You write your next command(s)\n\nEach response should include:\n\n1. **Reasoning text** where you explain your analysis and plan\n2. At least one tool call with your command\n\n**CRITICAL REQUIREMENTS:**\n\n- Your response SHOULD include reasoning text explaining what you're doing\n- Your response MUST include AT LEAST ONE bash tool call\n- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.\n- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files\n- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>\n\nExample of a CORRECT response:\n<example_response>\nI need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.\n\n[Makes bash tool call with {\"command\": \"ls -la\"} as arguments]\n</example_response>\n\n<system_information>\n__SYSTEM_INFORMATION__\n</system_information>\n\n## Useful command examples\n\n### Create a new file:\n\n```bash\ncat <<'EOF' > newfile.py\nimport numpy as np\nhello = \"world\"\nprint(hello)\nEOF\n```\n\n### Edit files with sed:```bash\n# Replace all occurrences\nsed -i 's/old_string/new_string/g' filename.py\n\n# Replace only first occurrence\nsed -i 's/old_string/new_string/' filename.py\n\n# Replace first occurrence on line 1\nsed -i '1s/old_string/new_string/' filename.py\n\n# Replace all occurrences in lines 1-10\nsed -i '1,10s/old_string/new_string/g' filename.py\n```\n\n### View file content:\n\n```bash\n# View specific lines with numbers\nnl -ba filename.py | sed -n '10,20p'\n```\n\n### Any other command you want to run\n\n```bash\nanything\n```"
