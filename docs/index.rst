🪽AgentFly
=====================

.. grid:: 12
    :gutter: 0

    .. grid-item::
        :columns: 10
        :class: sd-fs-4

        Training scalable LLM agents with RL (multi-turn, asynchronous tools/rewards, multimodal)

        .. raw:: html

            <img src="https://img.shields.io/github/stars/Agent-One-Lab/AgentFly?style=for-the-badge&logo=github&color=a2d2ff" alt="GitHub" width="20%" style="display:block;" />

    .. grid-item::
        :columns: 2

        .. raw:: html

            <div class="shown_logo" role="img" aria-label="AgentFly wing logo">
            <span class="glyph">🪽</span>
            </div>

            <style>
            :root{
                --size: 140px;
                --gold: #D4AF37;          /* primary gold */
            }

            .shown_logo{
                width: var(--size);
                aspect-ratio: 1;
                border-radius: 50%;
                display: grid;
                place-items: center;
                border: 0.1px solid var(--gold);

                /* soft, matte depth */
                box-shadow:
                0 3px 14px rgba(212,175,55,0.35);              /* outer golden glow */
                position: relative;
                background:
                radial-gradient(circle at 30% 25%, rgba(255,255,255,.12), transparent 40%),
                radial-gradient(circle at 70% 75%, rgba(255,255,255,.06), transparent 50%);
            }

            /* inner rings & subtle bevel */
            .shown_logo::after{
                content: "";
                position: absolute; inset: 0;
                border-radius: 50%;
                box-shadow:
                inset 0 0 0 1.5px rgba(255,255,255,0.18),        /* crisp inner ring */
                inset 0 12px 24px rgba(255,255,255,0.06),      /* top highlight */
                inset 0 -12px 28px rgba(0,0,0,0.28);           /* inner shadow */
                pointer-events: none;
            }

            .glyph{
                font-size: 100px;   /* try 108–116px if you want fuller */
                line-height: 1;
                /* tiny optical nudge to counter emoji side bearing */
                transform: translateX(1px) translateY(1px);
                filter: drop-shadow(0 2px 2px rgba(0,0,0,.25));
                font-family: "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",system-ui,sans-serif;
            }

            /* small hover delight (optional) */
            .shown_logo { transition: transform .24s ease; }
            .shown_logo:hover { transform: translateY(-1px) scale(1.02); }
            </style>


.. grid:: 3
    :gutter: 2

    .. grid-item-card::
        :link: https://arxiv.org/abs/2507.14897
        :class-header: bg-light

        AgentFly Paper 📜
        ^^^

        Explore the full paper, including the design inspiration, technical deatils, and training curves.

    .. grid-item-card::
        :link: https://github.com/Agent-One-Lab/AgentFly
        :class-header: bg-light

        GitHub Repo 💻
        ^^^

        Code repository in GitHub. 

    .. grid-item-card::
        :link: https://wandb.ai/AgentRL/Open
        :class-header: bg-light

        Weights & Biases 📈
        ^^^

        The training curves, parameters, rewards, and trajectories.

.. grid:: 3
    :gutter: 2

    .. grid-item-card::
        :link: https://huggingface.co/collections/Agent-One/agentfly-6882061c6cf08537cb66c12b
        :class-header: bg-light

        Models 🤗
        ^^^

        Check out the models on Hugging Face. Agent for code interpreter, retrieval, ScienceWorld, WebShop, etc.

    .. grid-item-card::
        :link: start/first_agent
        :class-header: bg-light

        Tutorials 📚
        ^^^

        Check out the tutorials on how to build agents, tools, rewards, and start training. 



Welcome to join our community!
--------------------------------

.. grid:: 3
    :gutter: 2

    .. grid-item-card::
        :img-top: ../assets/images/wechat.jpg
        :class-card: sd-text-center

        Scan to join WeChat group.


    .. grid-item-card::
        :img-top: ../assets/images/discord.png
        :class-card: sd-text-center

        `Join our Discord <https://discord.gg/ekrKVg8Y>`_


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Quick Start

    start/installation
    start/first_agent
    start/first_tool_reward
    start/first_training

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Features & Concepts

    features/agent_rollout
    features/tool_system
    features/reward_system
    features/environments
    features/chat_template/index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API Reference

    api_references/agents/index
    api_references/tools/index
    api_references/rewards/index
    api_references/environments/index
    api_references/chat_template/index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Examples

    examples/predefined_training_examples
