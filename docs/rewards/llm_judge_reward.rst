.. _llm_judge_reward:

LLM-as-Judge Reward
===================

.. currentmodule:: agents.agents.rewards.llm_as_judge.llm_as_judge_reward

.. autofunction:: llm_as_judge_reward

Function Signature
------------------

.. code-block:: python

    def llm_as_judge_reward(prediction: str, golden_answer: str, llm_backend) -> Dict[str, float]

Description
-----------

Uses a language model as an expert judge to evaluate the quality and correctness of agent responses against expected answers.

**Parameters:**
    - **prediction** (str): The agent's prediction or response to evaluate
    - **golden_answer** (str): The ground truth or expected answer
    - **llm_backend**: LLM backend instance configured for evaluation

**Returns:**
    Dict[str, float]: Dictionary containing:
        - **reward** (float): Evaluated score from 0.0 to 1.0
        - **raw_score** (float): Original score before clamping

**Decorator Configuration:**
    - **name**: "llm_as_judge_reward"
    - **llm_config**: 
        - **backend_type**: "transformers"
        - **model_name**: "Qwen/Qwen2.5-0.5B" (configurable via LLM_JUDGE_MODEL env var)
        - **max_tokens**: 10
        - **temperature**: 0.1

Technical Details
-----------------

**Evaluation Prompt:**
    The LLM judge uses a structured prompt that includes:
    - Role definition as expert evaluator
    - Evaluation criteria (correctness, completeness, alignment)
    - Expected answer and given answer
    - Scoring scale (0.0 to 1.0)

**Scoring Scale:**
    - **1.0**: Perfect match or equivalent answer
    - **0.7-0.9**: Mostly correct with minor issues
    - **0.4-0.6**: Partially correct
    - **0.0-0.3**: Incorrect or irrelevant

**Score Processing:**
    - Extracts numerical score from LLM response
    - Clamps values to [0.0, 1.0] range
    - Defaults to 0.0 if parsing fails

**Example Usage:**

.. code-block:: python

    from agents.agents.rewards import get_reward_from_name
    
    # Get LLM judge reward function
    reward_fn = get_reward_from_name("llm_as_judge_reward")
    
    # Evaluate a response
    result = reward_fn(
        prediction="Paris is the capital of France and a major cultural center",
        golden_answer="Paris is the capital of France"
    )
    print(result)
    # {"reward": 0.95, "raw_score": 0.95}

**Configuration:**

Set custom judge model via environment variable:

.. code-block:: bash

    export LLM_JUDGE_MODEL="meta-llama/Llama-2-7b-chat-hf"

**Use Cases:**
    - Open-ended question evaluation
    - Creative writing assessment
    - Complex reasoning task evaluation
    - Multi-dimensional answer quality assessment 