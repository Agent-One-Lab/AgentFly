# Implementation of GUI-Agent within AgentFly

## Overview

I have implemented the gui-agent @/mnt/weka/home/yongxin.wang/workspace/AgentFly on AgentFly codebase, but there is some training issues. Now please implement gui-agent referring to @/mnt/weka/home/yongxin.wang/workspace/AgentFly, and solve the issues.

## Current Tasks

### Completed

✅ **GUI Agent Implementation** - See: /tasks/001-gui-agent.md
- Implemented GUI agent class extending BaseAgent
- Created UI action parser for processing GUI actions
- Implemented GUI reward function with format and accuracy scoring
- Integrated with VERL training framework
- Fixed coordinate parsing and normalization issues
- Added comprehensive test suite


## Development Workflow

1. **Task Planning**

- Study the existing codebase and understand the current state
- Update `ROADMAP.md` to include the new task
- Priority tasks should be inserted after the last completed task

2. **Task Creation**

- Study the existing codebase and understand the current state
- Create a new task file in the `/tasks` directory
- Name format: `XXX-description.md` (e.g., `001-db.md`)
- Include high-level specifications, relevant files, acceptance criteria, and implementation steps
- Refer to last completed task in the `/tasks` directory for examples. For example, if the current task is `012`, refer to `011` and `010` for examples.
- Note that these examples are completed tasks, so the content reflects the final state of completed tasks (checked boxes and summary of changes). For the new task, the document should contain empty boxes and no summary of changes. Refer to `000-sample.md` as the sample for initial state.

3. **Task Implementation**

- Follow the specifications in the task file
- Implement features and functionality
- Update step progress within the task file after each step
- Stop after completing each step and wait for further instructions

4. **Roadmap Updates**

- Mark completed tasks with ✅ in the roadmap
- Add reference to the task file (e.g., `See: /tasks/001-db.md`)