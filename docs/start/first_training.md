Training Example
==============

Finally, we are ready to train the agent.

**1. Prepare Training Data**

----------------

We show an example of training on GSM8K dataset. First, prepare your training and validation datasets in JSON format. The datasets should follow this structure:

```

[
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "10"
    },
    ...
]
```

We use `question` filed to put task queries, and these "questions" will be used to form input messages. While other fileds, in our case, "answer" will be given to the reward function.

**2. Create Training Script**

------------------------
Use the prepared training script at `examples/train_scripts/train_example.sh`. Set `WANDB_API_KEY` first if you want Weights & Biases logging:

```bash
export WANDB_API_KEY="your_wandb_key"
```

The script itself:

```bash
--8<-- "examples/train_scripts/train_example.sh"
```

**3. Run Training**

--------------

Execute the training script. This training script run agent RL in a single node with one GPU. We have wrapped up everything, including tools, rewards, and training data. Run the following command to start training.

```
cd verl
bash run_agents/train_example.sh
```

The training progress will be logged to Weights & Biases if configured. You can monitor metrics like reward, loss, and KL divergence during training.

Key parameters to consider:

- ``model``: Base model to fine-tune
- ``batch_size``: Training batch size
- ``lr``: Learning rate
- ``num_chains``: Number of interaction chains per sample
- ``max_turns``: Maximum turns per interaction chain (set via ``agent.run_config.max_turns``)
- ``total_training_steps``: Total number of training steps
