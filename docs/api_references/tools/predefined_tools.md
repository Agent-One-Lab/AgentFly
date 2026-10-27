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

Tools for SWE-bench style rollouts: they are **stateful** and expect rollout `Context` with `context.metadata["image_id"]` set to the task container image. File tools copy `file_manager.py` into the container on first use and delegate reads and edits to that helper. `run_shell_command` runs in the container with working directory `/testbed` (or `context.metadata["workdir"]`). If `context.metadata` includes `git_commit_hash`, the first SWE tool call of a rollout (shell, file tools or MinisweAgent's `bash`) checks it out in the repository, running `git fetch` only if the checkout fails; if it still fails, the tool returns an error instead of running on the wrong code. SWE-smith tasks need this: their images start on `main` and each task's bug lives on a branch named after the task, which `swe-smith-py.json` stores in `git_commit_hash`.

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

