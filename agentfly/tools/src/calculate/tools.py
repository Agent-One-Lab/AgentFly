from tool_base import tool


@tool(name="calculator", description="Calculate the result of a mathematical expression.")
def calculate(expression: str) -> float:
    """
    Calculate the result of a mathematical expression.
    Args:
        expression (str): A mathematical expression to calculate.
    """
    return eval(expression)