# development note

now i have implemented for vlm_as_judge_pass_reward @/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/agentfly/rewards/vlm_as_judge/vlm_as_judge_reward.py, and i want to implement the llm_as_judge_pass_reward.


## workflow

the reward need to judge the code part like vlm_as_judge_pass_reward. but we have two more questions before the vlm_question, like 

"""
            all_questions = '1. Is the code can be executed? 2. Is the code can generate video?'
            all_weight= []
            try:
                for vlmindex in range(len(after_verify['vlm_questions'])):
                    subquestion = after_verify['vlm_questions'][vlmindex]['question']
                    all_questions += f'{vlmindex+3}. {subquestion}'

"""

and then like vlm_as_jdudge_pass_reward, will pass the code, questions to server.

compute the reward after server response.