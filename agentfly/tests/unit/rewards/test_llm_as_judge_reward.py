from ....rewards.llm_as_judge.llm_as_judge_client import llm_as_judge_client_math_reward
import pytest

# @pytest.mark.asyncio    
# async def test_llm_as_judge_client_reward():
#     prediction = "The answer is 10."
#     answer = "The answer is 10."
#     reward = await llm_as_judge_client_math_reward(prediction=prediction, answer=answer)
#     assert reward["reward"] == 1.0, f"Expected 1.0, got {reward}"

#     prediction = "The answer is 10."
#     answer = "The answer is 11."
#     reward = await llm_as_judge_client_math_reward(prediction=prediction, answer=answer)
#     assert reward["reward"] == 0.0, f"Expected 0.0, got {reward}"