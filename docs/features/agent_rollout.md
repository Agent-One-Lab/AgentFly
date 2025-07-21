Agent Rollout
==============
Author: Renxi Wang

Note: The findings discussed here are based on preliminary observations and have not been rigorously validated through controlled experiments.

## Rollout for LLM
In reinforcement learning, rollout refers to the stage where the LLM generates the responses to queries. The queries and responses will be concatenated later in the training stage, to calculate the advantages and update the LLM. To maximize the trainig efficiency, the rollout stage is a balance between exploration and exploitation. 

Traditionaly, this is achieved by setting a proper temperature (e.g. 1.0). A higher temperature encourage the model to generate more diverse responses (explore more), while a lower temperature limits the model to generate more accurate responses.

## Rollout for Agent
However, things become more complicated for agents: The rollout for agents is naturally multi-turn, where agents first generate responses, call tools, then get the observation from the tool. The rollout is not just affected by previous generated content, but will also be diverged by tool observations. This makes agent rollout more complex and flexible. How do we set the rollout strategy, to maximize the training efficiency?

Unfortunately, there are not many studies explored the agent reinforcement learning, not to say the best rollout strategy. We provide some initial thoughts on how we can do agent rollout for reinforcement learning.

### Chain-Search
Intuitively, for each query, we generate one response, and call the tool, append the observation... Repeating this process will form a chain-like agent interaction trajectory. This is what AgentFly adopts in the rollout stage. Although intuitive and simple, as there are more and more turns, the rollout trajectories will diverge more and more, possibly making the training unstable. 


### Tree-Search
In contrast to chain-search, tree-search generates multiple responses in each turn, trying to explore more action space for each query and obtain tree-based trajectories. Compared to single chain, tree-search will more likely to obtain both successful and failed trajectories, therefore better for RL learning.

### Filtering
Some studies show that some trajectories with specific patterns will lead to failture of agent reinforcement learning, even if their rewards are given accurately. SimpleTIR found a pattern, which they call void turns that "contain fragmented code or repetitive sentences and are often triggered by the premature generation of an eos token" [1].

### Single-Turn v.s. Multi-Turn
We found that training with shorter turns are more stable then training with longer turns, as shown in our report. This is also validated in other studies like GiGPO [1] and SimpleTIR. Another proposal is to convert multi-turn rollout into single-turn, which have the following benifits: (1) Memory efficient: we can only include the tool call and observations in history. Advantages, losses are only calculated in current turn response. (2) Stable: Training on single turn is much more stable than multi-turn.

### Our strategy
Currently, we adopt chain-based rollout. For each query, we initially generate *n* responses. Therefore we maintain *n* chains. For all generations, we use a fixed temperature. We will explore more on this and we also welcome your contributions!


[1] SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning

[2] GiGPO: Group-in-Group Policy Optimization for LLM Agent Training


## Asynchronous Implementation

To make the full rollout pipeline asynchronous, we have to make the generation and tool calling asynchronous.

For generation, we directly wrap the verl's asynchronous rollout worker.

For tool calling, we define each function to be asynchronous. For tools that require environments, we also ensure the envionment's methods to be asynchronous, which is detailed in our tool system.