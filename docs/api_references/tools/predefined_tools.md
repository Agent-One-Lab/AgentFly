# Predefined Tools

The following are predefined tools that can be used directly with agents.

## Code Interpreter

::: agentfly.tools.code_interpreter


## Calculator

::: agentfly.tools.calculator

## ALFWorld

::: agentfly.tools.alfworld_step

## ScienceWorld

::: agentfly.tools.scienceworld_explorer

## SWE (container workspace)

Tools for SWE-bench style rollouts: they are **stateful** and expect rollout `Context` with `context.metadata["image_id"]` set to the task container image. File tools mount `file_manager.py` into the container and delegate reads and edits to that helper. `run_shell_command` runs in the container with working directory `/testbed`. On first shell acquisition for a command, if `context.metadata` includes `git_commit_hash`, the shell tool runs `git fetch` and `git checkout` for that commit under `/testbed` (for example swe-smith style tasks).

### File workspace

Defined in `agentfly.tools.src.file.tools` (also re-exported from `agentfly.tools`).

::: agentfly.tools.read_file

::: agentfly.tools.grep_search

::: agentfly.tools.create_file

::: agentfly.tools.edit_file

::: agentfly.tools.undo_edit

::: agentfly.tools.run_python

### Shell

::: agentfly.tools.run_shell_command

## Retrieval

::: agentfly.tools.async_dense_retrieve

