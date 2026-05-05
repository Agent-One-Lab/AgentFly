# Define Tools & Reward Functions

We have shown how to build an agent, to further customize the training, we need to define tools and reward functions.

**Tool Definition**

Define a tools is simple and easy in AgentFly framework. You simply write a function, and then decorate it with `@tool`. The following example shows the calculator tool we used previously. You can also inherit a `BaseTool` class to define the tool with more flexibility and complexity (refer to Features section).

```python
--8<-- "tests/docs/start/quick_example.py:tool_def"
```

Now we have the tool, we can then define the reward function, which also simply use a `@reward` decorator. The following example shows a reward by extracting the last number of a text and compare it with the golden asnwer. The return of the reward function is a float number representing the reward, or a dictionary containing "reward" as a key.

```python
--8<-- "tests/docs/start/quick_example.py:reward_def"
```
Note that in this reward function, we use the trajectory to count how many tools the agent has called. If the agent called at least one, we give it the basic format reward (0.1), then if it further gets the answer correct, it gets the full reward (1.0).
Now we can use the agent with the reward function we just defined.

```python
--8<-- "tests/docs/start/quick_example.py:agent_with_reward"
```

Then we can run the agent and get rewards:

```python
--8<-- "tests/docs/start/quick_example.py:run_with_answer"
```

Now we can get the trajectories and rewards with following code:
```python
--8<-- "tests/docs/start/quick_example.py:agent_results"
```
