from ...tool_base import tool

@tool(name="answer", description="Give the final answer. The answer should be put inside the \\boxed{} tag.", status="finish")
def answer_math(answer: str):
    """
    A helper tool to give the final answer. The answer should be put inside the \\boxed{} tag.
    Args:
        answer (str): The final answer to the question.
    Returns:
        str: The final answer to the question.
    """
    return str(answer)


@tool(name="answer", description="Give the final answer. The answer should be a simple, short, and direct.", status="finish")
def answer_qa(answer: str):
    """
    A helper tool to give the final answer. The answer should be a simple, short, and direct.
    Args:
        answer (str): The final answer to the question.
    Returns:
        str: The final answer to the question.
    """
    return str(answer)