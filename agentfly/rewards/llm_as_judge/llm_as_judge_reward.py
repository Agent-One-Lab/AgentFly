from ..reward_base import reward
from typing import Dict, Any, Optional
import os

@reward(name="llm_as_judge_reward")
def llm_as_judge_reward(prediction: str, golden_answer: str, llm_backend) -> Dict[str, float]:
    """
    Reward function that uses an LLM as a judge to evaluate the quality of predictions.
    
    Args:
        prediction: The model's prediction/response
        golden_answer: The ground truth answer
        llm_backend: The LLM backend instance to use for evaluation
        
    Returns:
        Dictionary containing the reward score and additional metrics
    """
    prompt = f"""You are an expert judge evaluating the quality of an answer. 
Please evaluate the following answer based on its correctness, completeness, and alignment with the expected answer.

Expected Answer: {golden_answer}
Given Answer: {prediction}

Please provide a score from 0.0 to 1.0, where:
- 1.0: Perfect match or equivalent answer
- 0.7-0.9: Mostly correct with minor issues
- 0.4-0.6: Partially correct
- 0.0-0.3: Incorrect or irrelevant

Score:"""

    response = llm_backend.generate(prompt)
    
    # Extract score from response
    try:
        score = float(response.split("Score:")[-1].strip())
        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
    except:
        score = 0.0
        
    return {
        "reward": score,
        "raw_score": score
    }

if __name__ == "__main__":
    # Test the reward function
    # Import here to avoid circular imports during normal operation
    from ..reward_base import get_reward_from_name
    
    reward_fn = get_reward_from_name("llm_as_judge_reward")
    result = reward_fn("I got answer is 2/3", "I got answer is 2/3")
    print(result)