# 🪽 AgentFly

Training scalable LLM agents with RL (multi-turn, asynchronous tools/rewards, multimodal)

!!! note
    Parts of this documentation are AI-generated. We aim to keep it accurate, but if you find a discrepancy with the code, the code is the source of truth — please open an issue or PR.

![Overall Structure](https://agent-one-lab.github.io/assets/agentfly/overview.png)

<!-- ![GitHub](https://img.shields.io/github/stars/Agent-One-Lab/AgentFly?style=for-the-badge&logo=github&color=a2d2ff) -->

<!-- <div style="text-align: center; margin: 2rem 0;">
  <div class="shown_logo" role="img" aria-label="AgentFly wing logo">
    <span class="glyph">🪽</span>
  </div>
</div> -->


## Resources

=== "AgentFly Paper 📜"

    <div class="grid cards" markdown>

    -   :material-file-document-outline: __AgentFly: Extensible and Scalable Reinforcement Learning for LLM Agents__

        ---

        Methods to build LLM agents have evolved from prompt engineering and supervised finetuning to agentic reinforcement learning (agentic RL). AgentFly is an agentic RL framework that tackles bottlenecks in environment interaction, reward calculation, and large-scale training through a four-layer design: an agent layer for defining agents, tools, and rewards; a rollout layer that drives agent loops and collects trajectories; a context layer that injects task metadata and coordinates resources; and a resource layer that manages low-level execution backends such as containers and model engines. With a suite of prebuilt tools and environments (including search, code, and interactive environments), AgentFly enables scalable training of multi-turn, tool-using LLM agents across diverse tasks.

        [:octicons-arrow-right-24: Read Paper](https://arxiv.org/abs/2507.14897)
    </div>

=== "GitHub Repo 💻"

    <div class="grid cards" markdown>

    -   :fontawesome-brands-github: __GitHub Repository__

        ---

        Code repository in GitHub.

        [:octicons-arrow-right-24: Explore Code](https://github.com/Agent-One-Lab/AgentFly)
    </div>


=== "Weights & Biases 📈"

    <div class="grid cards" markdown>

    -   :material-chart-line: __WandB__

        ---

        The training curves, parameters, rewards, and trajectories.

        [:octicons-arrow-right-24: Training](https://wandb.ai/AgentRL/Open)
    </div>

=== "Models 🤗"

    <div class="grid cards" markdown>

    -   :material-robot: __HuggingFace__

        ---

        Check out the models on Hugging Face. Agent for code interpreter, retrieval, ScienceWorld, WebShop, etc.

        [:octicons-arrow-right-24: Explore Model](https://huggingface.co/collections/Agent-One/agentfly-6882061c6cf08537cb66c12b)
    </div>

=== "Tutorials 📚"


    <div class="grid cards" markdown>

    -   :material-book-open-variant: __Tutorials__

        ---

        Check out the tutorials on how to build agents, tools, rewards, and start training.


        [:octicons-arrow-right-24: Read More](start/first_agent.md)
    </div>

## Welcome to join our community!

<div class="grid cards community-cards" markdown>

-   :material-wechat:{ .lg .middle } __WeChat Group__

    Scan to join WeChat group.

    ---

    ![WeChat QR Code](https://agent-one-lab.github.io/assets/agentfly/wechat.jpg)

-   :material-message:{ .lg .middle } __Discord__

    Join our [Discord](https://discord.gg/ekrKVg8Y) community.

    ---

    ![Discord QR Code](https://agent-one-lab.github.io/assets/agentfly/discord.png)

</div>
