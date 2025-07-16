from ...tool_base import tool

@tool(name="answer", description="Give the final answer. The answer should be put inside the \\boxed{} tag.", status="finish")
def answer(answer: str):
    """
    args:
        answer (str): The final answer to the question.
    """
    return str(answer)