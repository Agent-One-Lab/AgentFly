Training Example
==============

This guide walks through the steps to train an agent using AgentFly.

1. Prepare Training Data
-----------------------
First, prepare your training and validation datasets in JSON format. The datasets should follow this structure:

.. code-block:: json

    [
        {
            "id": "0",
            "question": "$P(x)$ is a polynomial of degree $3n$ such that\n\\begin{eqnarray*} P(0) = P(3) = \\cdots &=& P(3n) = 2, \\\\ P(1) = P(4) = \\cdots &=& P(3n-2) = 1, \\\\ P(2) = P(5) = \\cdots &=& P(3n-1) = 0, \\quad\\text{ and }\\\\ && P(3n+1) = 730.\\end{eqnarray*}\nDetermine $n$.",
            "answer": "n = 4"
        },
        {
            "id": "1",
            "question": "Diameter $AB$ of a circle has length a $2$-digit integer (base ten). Reversing the digits gives the length of the perpendicular chord $CD$. The distance from their intersection point $H$ to the center $O$ is a positive rational number. Determine the length of $AB$.",
            "answer": "65"
        },
        ...
    ]

Save your training data and validation data.

2. Create Training Script 
------------------------
Create a training script (e.g., ``train.sh``) with the following configuration:

.. code-block:: bash

    export WANDB_API_KEY="your_wandb_key"  # For logging to Weights & Biases
    export VLLM_USE_V1=1

    # Ray cluster configuration
    ray stop
    ray start --head --node-ip-address="$(hostname --ip-address)" --port=6379 --num-cpus 192 --num-gpus 8

    # Training parameters
    model="Qwen/Qwen2.5-3B-Instruct"
    template="qwen-chat"
    lr=5e-7
    length=512
    batch_size=128
    num_chains=4
    kl_coef=0.001
    train_dataset="train"
    adv_estimator="grpo"

    # Agent configuration
    agent_type="code"
    tools="[code_interpreter]"
    reward_name="math_reward_format"
    entropy_coeff=0.001
    kl_loss_type="mse"
    max_steps=4
    agent_backend="async_verl"
    project_name="AgentRL"
    total_training_steps=200

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$adv_estimator \
        data.train_files=./data/train.json \
        data.val_files=./data/val.json \
        data.train_batch_size=$batch_size \
        agent.agent_type=$agent_type \
        agent.tools=$tools \
        agent.template=$template \
        agent.model_name_or_path=$model \
        agent.max_steps=${max_steps} \
        agent.backend=${agent_backend} \
        agent.reward_name=$reward_name \
        agent.num_chains=$num_chains \
        agent.use_agent=True \
        actor_rollout_ref.actor.optim.lr=$lr \
        actor_rollout_ref.model.path=${model} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.total_training_steps=$total_training_steps

3. Run Training
--------------
Execute the training script:

.. code-block:: bash

    bash train.sh

The training progress will be logged to Weights & Biases if configured. You can monitor metrics like reward, loss, and KL divergence during training.

Key parameters to consider:

- ``model``: Base model to fine-tune
- ``batch_size``: Training batch size
- ``lr``: Learning rate
- ``num_chains``: Number of interaction chains per sample
- ``max_steps``: Maximum steps per interaction chain
- ``total_training_steps``: Total number of training steps
