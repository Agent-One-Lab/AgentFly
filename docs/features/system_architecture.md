## System Architecture

AgentFly organizes agentic reinforcement learning into a **four-layer system** that separates user-facing logic from rollout orchestration and low-level resources:

- **Agent Layer**: Exposes the main interfaces to users. It abstracts:
  - **Agents**: classes that implement the interaction loop (e.g., `ReactAgent`, `HFAgent`).
  - **Tools**: callable interfaces to external systems (functions, APIs, environments).
  - **Rewards**: task-specific reward functions used during training.
  Agentic RL at this layer is decomposed into *defining agents, tools, and rewards*.

- **Rollout Layer**: Builds and runs the **agent loop** (multi-turn, tool-using trajectories).
  It calls the model, invokes tools, collects observations, and packages trajectories and rewards
  to be consumed by the RL trainer (e.g., Verl PPO / GRPO).

- **Context Layer**: Acts as the glue between rollout and resources.
  It:
  - Tracks rollout IDs, task metadata, and auxiliary fields.
  - Injects contextual information (e.g., gold answers or extra fields) into tools and rewards.
  - Manages which resources (containers, model engines, etc.) are bound to which trajectory IDs.

- **Resource Layer**: Implements scalable, asynchronous **resource management**.
  It manages pools of resource units such as:
  - Docker/enroot containers for environments (code interpreter, ALFWorld, WebShop, ScienceWorld, Chess, etc.).
  - Model engines (e.g., async vLLM) used for generation.
  Resources are allocated, reused, and released asynchronously, enabling high-throughput, multi-turn rollouts.

This architecture is what allows AgentFly to support:

- Multi-turn, tool-rich rollouts with masking-based multi-turn RL.
- Asynchronous **generation → tool calling → reward calculation** pipelines.
- Scaling to many concurrent environments simply by increasing pool sizes in tool/reward definitions.

