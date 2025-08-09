# Tutorial 7: Complete Pipeline - End-to-End Agent Training

This comprehensive tutorial walks through the complete process of creating and training a customized AgentFly agent from scratch. You'll build a math problem-solving agent that can use tools to solve mathematical problems step by step.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Creating Custom Tools](#creating-custom-tools)
4. [Defining Reward Functions](#defining-reward-functions)
5. [Building a Custom Agent](#building-a-custom-agent)
6. [Preparing Training Data](#preparing-training-data)
7. [Configuring Templates](#configuring-templates)
8. [Setting Up Training](#setting-up-training)
9. [Running Training](#running-training)
10. [Evaluation and Testing](#evaluation-and-testing)
11. [Deployment](#deployment)

## Project Overview

We'll create a **MathAgent** that can:
- Solve various types of mathematical problems
- Use a calculator tool for computations
- Show step-by-step reasoning
- Validate its answers
- Learn from feedback through reinforcement learning

### Project Structure

```bash
math_agent_project/
├── src/
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   └── validator.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── math_rewards.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── math_agent.py
│   └── templates/
│       ├── __init__.py
│       └── math_template.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── configs/
│   ├── agent_config.yaml
│   └── training_config.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── models/
│   ├── logs/
│   └── results/
└── requirements.txt
```

## Environment Setup

### 1. Create Project Environment

```bash
# Create project directory
mkdir math_agent_project
cd math_agent_project

# Create virtual environment
python -m venv math_agent_env
source math_agent_env/bin/activate  # On Windows: math_agent_env\Scripts\activate

# Install AgentFly
git clone https://github.com/Agent-One-Lab/AgentFly
cd AgentFly
pip install -e .
pip install -e '.[verl]' --no-build-isolation
cd ..

# Install additional requirements
pip install wandb jupyter matplotlib seaborn pandas
```

### 2. Initialize Project Structure

```python
# scripts/init_project.py
import os
from pathlib import Path

def create_project_structure():
    """Create the project directory structure."""
    
    directories = [
        "src/tools",
        "src/rewards", 
        "src/agents",
        "src/templates",
        "data/raw",
        "data/processed",
        "configs",
        "scripts",
        "notebooks",
        "outputs/models",
        "outputs/logs",
        "outputs/results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            init_file.touch()
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
```

Run the initialization:

```bash
python scripts/init_project.py
```

## Creating Custom Tools

### 1. Calculator Tool

Create `src/tools/calculator.py`:

```python
"""
Advanced calculator tool for mathematical operations.
"""

import math
import re
import ast
import operator
from typing import Dict, Any
from agents.tools.tool_base import tool

# Safe operators for mathematical expressions
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe functions
SAFE_FUNCTIONS = {
    'abs': abs,
    'min': min,
    'max': max,
    'round': round,
    'sum': sum,
    'pow': pow,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'factorial': math.factorial,
    'gcd': math.gcd,
    'pi': math.pi,
    'e': math.e,
}

class SafeMathEvaluator(ast.NodeVisitor):
    """Safe evaluator for mathematical expressions."""
    
    def visit_Expression(self, node):
        return self.visit(node.body)
    
    def visit_Constant(self, node):
        return node.value
    
    def visit_Num(self, node):  # For Python < 3.8 compatibility
        return node.n
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator_func = SAFE_OPERATORS.get(type(node.op))
        
        if operator_func is None:
            raise ValueError(f"Unsafe operator: {type(node.op).__name__}")
        
        try:
            return operator_func(left, right)
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Math error: {str(e)}")
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        operator_func = SAFE_OPERATORS.get(type(node.op))
        
        if operator_func is None:
            raise ValueError(f"Unsafe unary operator: {type(node.op).__name__}")
        
        return operator_func(operand)
    
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Unsafe function: {func_name}")
        
        args = [self.visit(arg) for arg in node.args]
        
        try:
            return SAFE_FUNCTIONS[func_name](*args)
        except Exception as e:
            raise ValueError(f"Function error: {str(e)}")
    
    def visit_Name(self, node):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        else:
            raise ValueError(f"Undefined variable: {node.id}")
    
    def generic_visit(self, node):
        raise ValueError(f"Unsafe operation: {type(node).__name__}")

def safe_eval(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    try:
        tree = ast.parse(expression, mode='eval')
        evaluator = SafeMathEvaluator()
        result = evaluator.visit(tree)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

@tool(
    name="advanced_calculator",
    description="Advanced calculator that can perform arithmetic, trigonometric, logarithmic operations and more"
)
def advanced_calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate mathematical expressions safely.
    
    Supports:
    - Basic arithmetic: +, -, *, /, **, %
    - Functions: sin, cos, tan, log, sqrt, abs, etc.
    - Constants: pi, e
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        Dict containing the result and metadata
        
    Examples:
        "2 + 3 * 4" -> 14
        "sqrt(16)" -> 4.0
        "sin(pi/2)" -> 1.0
    """
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Replace common text with symbols
        replacements = {
            ' plus ': ' + ',
            ' minus ': ' - ',
            ' times ': ' * ',
            ' divided by ': ' / ',
            ' to the power of ': ' ** ',
            ' squared': ' ** 2',
            ' cubed': ' ** 3',
        }
        
        for text, symbol in replacements.items():
            expression = expression.replace(text, symbol)
        
        # Evaluate the expression
        result = safe_eval(expression)
        
        # Format result appropriately
        if result.is_integer():
            formatted_result = str(int(result))
        else:
            formatted_result = f"{result:.10g}"  # Remove trailing zeros
        
        return {
            "observation": f"Result: {formatted_result}",
            "result": result,
            "formatted_result": formatted_result,
            "expression": expression,
            "success": True
        }
        
    except Exception as e:
        return {
            "observation": f"Error: {str(e)}",
            "error": str(e),
            "expression": expression,
            "success": False
        }

@tool(
    name="step_by_step_solver",
    description="Solve mathematical problems step by step with explanations"
)
def step_by_step_solver(problem: str) -> Dict[str, Any]:
    """
    Solve mathematical problems with step-by-step explanations.
    
    Args:
        problem (str): Mathematical problem description
        
    Returns:
        Dict containing steps and solution
    """
    try:
        # Pattern matching for common problem types
        steps = []
        
        # Quadratic equation solver
        quadratic_pattern = r'(\w+)\s*=\s*([+-]?\d*\.?\d*)\s*\*?\s*(\w+)\^?2\s*([+-])\s*(\d*\.?\d*)\s*\*?\s*(\w+)\s*([+-])\s*(\d+)'
        if re.search(quadratic_pattern, problem):
            steps.append("Identified as quadratic equation")
            steps.append("Using quadratic formula: x = (-b ± √(b² - 4ac)) / 2a")
            
        # Linear equation solver
        linear_pattern = r'(\d*\.?\d*)\s*\*?\s*(\w+)\s*([+-])\s*(\d+)\s*=\s*(\d+)'
        linear_match = re.search(linear_pattern, problem)
        if linear_match:
            a, var, op, b, result = linear_match.groups()
            a = float(a) if a else 1.0
            b = float(b)
            result = float(result)
            
            steps.append(f"Linear equation: {a}*{var} {op} {b} = {result}")
            
            if op == '+':
                steps.append(f"Subtract {b} from both sides: {a}*{var} = {result - b}")
                solution = (result - b) / a
            else:  # op == '-'
                steps.append(f"Add {b} to both sides: {a}*{var} = {result + b}")
                solution = (result + b) / a
            
            steps.append(f"Divide by {a}: {var} = {solution}")
            
            return {
                "observation": f"Solution: {var} = {solution}\n\nSteps:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)]),
                "steps": steps,
                "solution": solution,
                "success": True
            }
        
        # Percentage problems
        if 'percent' in problem.lower() or '%' in problem:
            percentage_pattern = r'(\d+\.?\d*)\s*%?\s*of\s*(\d+\.?\d*)'
            match = re.search(percentage_pattern, problem)
            if match:
                percent, number = match.groups()
                percent = float(percent)
                number = float(number)
                
                steps.append(f"Calculate {percent}% of {number}")
                steps.append(f"Convert percentage to decimal: {percent}% = {percent/100}")
                steps.append(f"Multiply: {number} × {percent/100} = {number * percent/100}")
                
                result = number * percent / 100
                
                return {
                    "observation": f"Result: {result}\n\nSteps:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)]),
                    "steps": steps,
                    "result": result,
                    "success": True
                }
        
        # Default: try to extract and solve expression
        expressions = re.findall(r'[\d+\-*/().\s]+', problem)
        if expressions:
            expr = max(expressions, key=len).strip()
            if len(expr) > 3:  # Basic validation
                calc_result = advanced_calculator(expr)
                if calc_result["success"]:
                    steps.append(f"Extract mathematical expression: {expr}")
                    steps.append(f"Evaluate: {calc_result['formatted_result']}")
                    
                    return {
                        "observation": f"Result: {calc_result['formatted_result']}\n\nSteps:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)]),
                        "steps": steps,
                        "result": calc_result['result'],
                        "success": True
                    }
        
        return {
            "observation": "Could not solve this problem automatically. Please provide a clearer mathematical expression.",
            "error": "Problem type not recognized",
            "success": False
        }
        
    except Exception as e:
        return {
            "observation": f"Error solving problem: {str(e)}",
            "error": str(e),
            "success": False
        }
```

### 2. Math Validator Tool

Create `src/tools/validator.py`:

```python
"""
Mathematical answer validation tool.
"""

import re
import math
from typing import Dict, Any, Union
from agents.tools.tool_base import tool

@tool(
    name="math_validator",
    description="Validate mathematical answers and check if they are equivalent"
)
def math_validator(answer1: str, answer2: str, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate if two mathematical answers are equivalent.
    
    Args:
        answer1 (str): First answer to compare
        answer2 (str): Second answer (reference)
        tolerance (float): Numerical tolerance for comparison
        
    Returns:
        Dict containing validation results
    """
    try:
        # Extract numerical values from answers
        def extract_number(text: str) -> Union[float, str]:
            """Extract numerical value from text."""
            # Remove common formatting
            text = text.strip().lower()
            
            # Handle fractions
            fraction_pattern = r'(\d+)/(\d+)'
            fraction_match = re.search(fraction_pattern, text)
            if fraction_match:
                num, den = fraction_match.groups()
                return float(num) / float(den)
            
            # Handle percentages
            if '%' in text:
                percent_pattern = r'(\d+\.?\d*)%'
                percent_match = re.search(percent_pattern, text)
                if percent_match:
                    return float(percent_match.group(1)) / 100
            
            # Handle boxed format (LaTeX)
            boxed_pattern = r'\\boxed\{([^}]+)\}'
            boxed_match = re.search(boxed_pattern, text)
            if boxed_match:
                return extract_number(boxed_match.group(1))
            
            # Handle decimal numbers
            number_pattern = r'-?\d+\.?\d*'
            number_match = re.search(number_pattern, text)
            if number_match:
                return float(number_match.group(0))
            
            # Return original text if no number found
            return text
        
        num1 = extract_number(answer1)
        num2 = extract_number(answer2)
        
        # If both are numbers, compare numerically
        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            difference = abs(num1 - num2)
            is_equivalent = difference <= tolerance
            
            return {
                "observation": f"Numerical comparison: {num1} vs {num2} -> {'✓ Equivalent' if is_equivalent else '✗ Different'}",
                "is_equivalent": is_equivalent,
                "value1": num1,
                "value2": num2,
                "difference": difference,
                "tolerance": tolerance,
                "comparison_type": "numerical",
                "success": True
            }
        
        # Text comparison (case-insensitive)
        text1 = str(num1).strip().lower()
        text2 = str(num2).strip().lower()
        is_equivalent = text1 == text2
        
        return {
            "observation": f"Text comparison: '{text1}' vs '{text2}' -> {'✓ Equivalent' if is_equivalent else '✗ Different'}",
            "is_equivalent": is_equivalent,
            "value1": text1,
            "value2": text2,
            "comparison_type": "text",
            "success": True
        }
        
    except Exception as e:
        return {
            "observation": f"Validation error: {str(e)}",
            "error": str(e),
            "success": False
        }

@tool(
    name="answer_formatter",
    description="Format mathematical answers in different styles"
)
def answer_formatter(answer: str, format_type: str = "clean") -> Dict[str, Any]:
    """
    Format mathematical answers in different styles.
    
    Args:
        answer (str): Answer to format
        format_type (str): Format type ("clean", "boxed", "fraction", "percentage")
        
    Returns:
        Dict containing formatted answer
    """
    try:
        # Extract numerical value
        number_pattern = r'-?\d+\.?\d*'
        number_match = re.search(number_pattern, answer.strip())
        
        if not number_match:
            return {
                "observation": f"Could not extract number from: {answer}",
                "formatted_answer": answer,
                "success": False
            }
        
        value = float(number_match.group(0))
        
        if format_type == "clean":
            # Remove unnecessary decimal places
            if value.is_integer():
                formatted = str(int(value))
            else:
                formatted = f"{value:.10g}"
        
        elif format_type == "boxed":
            # LaTeX boxed format
            if value.is_integer():
                formatted = f"\\boxed{{{int(value)}}}"
            else:
                formatted = f"\\boxed{{{value:.10g}}}"
        
        elif format_type == "fraction":
            # Convert to fraction if possible
            from fractions import Fraction
            frac = Fraction(value).limit_denominator(1000)
            if frac.denominator == 1:
                formatted = str(frac.numerator)
            else:
                formatted = f"{frac.numerator}/{frac.denominator}"
        
        elif format_type == "percentage":
            # Convert to percentage
            formatted = f"{value * 100}%" if value < 1 else f"{value}%"
        
        else:
            formatted = str(value)
        
        return {
            "observation": f"Formatted answer: {formatted}",
            "formatted_answer": formatted,
            "original_value": value,
            "format_type": format_type,
            "success": True
        }
        
    except Exception as e:
        return {
            "observation": f"Formatting error: {str(e)}",
            "error": str(e),
            "success": False
        }
```

## Defining Reward Functions

Create `src/rewards/math_rewards.py`:

```python
"""
Reward functions for mathematical problem solving.
"""

import re
import math
from typing import Dict, List, Any
from agents.rewards.reward_base import reward

def extract_final_answer(text: str) -> str:
    """Extract the final answer from agent response."""
    # Look for boxed answers first
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for "Final answer:" pattern
    final_pattern = r'final\s*answer\s*:?\s*([^\n]+)'
    final_match = re.search(final_pattern, text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    
    # Look for "Answer:" pattern
    answer_pattern = r'answer\s*:?\s*([^\n]+)'
    answer_match = re.search(answer_pattern, text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Look for numerical values at the end
    lines = text.strip().split('\n')
    for line in reversed(lines):
        number_pattern = r'-?\d+\.?\d*'
        if re.search(number_pattern, line):
            return line.strip()
    
    return text.strip()

def normalize_answer(answer: str) -> float:
    """Normalize answer to numerical value for comparison."""
    try:
        # Remove common formatting
        answer = answer.strip().lower()
        
        # Handle fractions
        fraction_pattern = r'(\d+)/(\d+)'
        fraction_match = re.search(fraction_pattern, answer)
        if fraction_match:
            num, den = fraction_match.groups()
            return float(num) / float(den)
        
        # Handle percentages
        if '%' in answer:
            percent_pattern = r'(\d+\.?\d*)%'
            percent_match = re.search(percent_pattern, answer)
            if percent_match:
                return float(percent_match.group(1))
        
        # Extract number
        number_pattern = r'-?\d+\.?\d*'
        number_match = re.search(number_pattern, answer)
        if number_match:
            return float(number_match.group(0))
        
        return 0.0
    except:
        return 0.0

@reward(name="math_accuracy_reward")
def math_accuracy_reward(prediction: str, answer: str, **kwargs) -> Dict[str, Any]:
    """
    Reward based on mathematical accuracy.
    
    Args:
        prediction (str): Agent's prediction
        answer (str): Correct answer
        **kwargs: Additional parameters
        
    Returns:
        Dict containing reward and metadata
    """
    # Extract final answers
    pred_answer = extract_final_answer(prediction)
    true_answer = answer.strip()
    
    # Normalize answers
    pred_value = normalize_answer(pred_answer)
    true_value = normalize_answer(true_answer)
    
    # Calculate accuracy
    if true_value == 0:
        # Exact match for zero
        exact_match = abs(pred_value - true_value) < 1e-10
    else:
        # Relative tolerance for non-zero
        relative_error = abs(pred_value - true_value) / abs(true_value)
        exact_match = relative_error < 1e-6
    
    # Partial credit for being close
    if not exact_match and true_value != 0:
        relative_error = abs(pred_value - true_value) / abs(true_value)
        if relative_error < 0.01:  # Within 1%
            partial_score = 0.8
        elif relative_error < 0.05:  # Within 5%
            partial_score = 0.6
        elif relative_error < 0.1:  # Within 10%
            partial_score = 0.4
        else:
            partial_score = 0.0
    else:
        partial_score = 0.0
    
    final_score = 1.0 if exact_match else partial_score
    
    return {
        "reward": final_score,
        "exact_match": exact_match,
        "predicted_answer": pred_answer,
        "true_answer": true_answer,
        "predicted_value": pred_value,
        "true_value": true_value,
        "partial_score": partial_score
    }

@reward(name="math_process_reward")
def math_process_reward(prediction: str, answer: str, trajectory: List[Dict], **kwargs) -> Dict[str, Any]:
    """
    Reward based on mathematical reasoning process.
    
    Args:
        prediction (str): Agent's prediction
        answer (str): Correct answer
        trajectory (List[Dict]): Agent's trajectory
        **kwargs: Additional parameters
        
    Returns:
        Dict containing process-based reward
    """
    # Base accuracy reward
    accuracy_result = math_accuracy_reward(prediction, answer, **kwargs)
    accuracy_score = accuracy_result["reward"]
    
    # Analyze process quality
    process_score = 0.0
    
    # Tool usage analysis
    tool_calls = [step for step in trajectory if step.get("role") == "tool"]
    calculator_calls = [call for call in tool_calls if "calculator" in call.get("name", "")]
    
    tool_usage_score = 0.0
    if calculator_calls:
        tool_usage_score += 0.3  # Used calculator
        
        # Check if calculations are correct
        correct_calculations = 0
        for call in calculator_calls:
            if call.get("content", {}).get("success", False):
                correct_calculations += 1
        
        if correct_calculations == len(calculator_calls):
            tool_usage_score += 0.2  # All calculations correct
    
    # Reasoning analysis
    reasoning_score = 0.0
    trajectory_text = " ".join([step.get("content", "") for step in trajectory])
    
    # Look for step-by-step reasoning
    reasoning_indicators = [
        "first", "second", "third", "next", "then", "therefore",
        "step 1", "step 2", "step 3", "let's", "we need to"
    ]
    
    reasoning_count = sum(1 for indicator in reasoning_indicators 
                         if indicator in trajectory_text.lower())
    
    if reasoning_count >= 3:
        reasoning_score = 0.3
    elif reasoning_count >= 2:
        reasoning_score = 0.2
    elif reasoning_count >= 1:
        reasoning_score = 0.1
    
    # Explanation quality
    explanation_score = 0.0
    if len(trajectory_text) > 100:  # Substantial explanation
        explanation_score += 0.1
        
        # Check for mathematical notation
        if any(symbol in trajectory_text for symbol in ["=", "+", "-", "*", "/", "^"]):
            explanation_score += 0.1
    
    # Combine scores
    process_score = tool_usage_score + reasoning_score + explanation_score
    
    # Final reward combines accuracy and process
    final_reward = 0.6 * accuracy_score + 0.4 * min(process_score, 1.0)
    
    return {
        "reward": final_reward,
        "accuracy_score": accuracy_score,
        "process_score": process_score,
        "tool_usage_score": tool_usage_score,
        "reasoning_score": reasoning_score,
        "explanation_score": explanation_score,
        "tool_calls_count": len(tool_calls),
        "trajectory_length": len(trajectory)
    }

@reward(name="math_comprehensive_reward")
def math_comprehensive_reward(prediction: str, answer: str, trajectory: List[Dict], 
                            difficulty: str = "medium", **kwargs) -> Dict[str, Any]:
    """
    Comprehensive reward combining accuracy, process, and difficulty.
    
    Args:
        prediction (str): Agent's prediction
        answer (str): Correct answer
        trajectory (List[Dict]): Agent's trajectory
        difficulty (str): Problem difficulty level
        **kwargs: Additional parameters
        
    Returns:
        Dict containing comprehensive reward
    """
    # Get process reward
    process_result = math_process_reward(prediction, answer, trajectory, **kwargs)
    base_reward = process_result["reward"]
    
    # Difficulty multiplier
    difficulty_multipliers = {
        "easy": 0.8,
        "medium": 1.0,
        "hard": 1.3,
        "expert": 1.6
    }
    
    multiplier = difficulty_multipliers.get(difficulty, 1.0)
    
    # Efficiency bonus/penalty
    trajectory_length = len(trajectory)
    if trajectory_length <= 5:
        efficiency_bonus = 0.1  # Efficient solution
    elif trajectory_length <= 10:
        efficiency_bonus = 0.0  # Normal length
    else:
        efficiency_bonus = -0.1  # Too verbose
    
    # Final reward calculation
    final_reward = (base_reward + efficiency_bonus) * multiplier
    final_reward = max(0.0, min(final_reward, 2.0))  # Clamp to [0, 2]
    
    return {
        "reward": final_reward,
        "base_reward": base_reward,
        "difficulty_multiplier": multiplier,
        "efficiency_bonus": efficiency_bonus,
        "difficulty": difficulty,
        **process_result  # Include all process metrics
    }
```

## Building a Custom Agent

Create `src/agents/math_agent.py`:

```python
"""
Specialized mathematical problem-solving agent.
"""

import re
import json
from typing import List, Dict, Any
from agents.agents.agent_base import BaseAgent

class MathAgent(BaseAgent):
    """
    Specialized agent for mathematical problem solving.
    
    Features:
    - Step-by-step reasoning
    - Tool usage for calculations
    - Answer validation
    - Mathematical formatting
    """
    
    def __init__(self, 
                 show_work: bool = True,
                 validate_answers: bool = True,
                 max_calculation_steps: int = 10,
                 **kwargs):
        """
        Initialize MathAgent.
        
        Args:
            show_work (bool): Whether to show step-by-step work
            validate_answers (bool): Whether to validate answers
            max_calculation_steps (int): Maximum calculation steps
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(**kwargs)
        self.show_work = show_work
        self.validate_answers = validate_answers
        self.max_calculation_steps = max_calculation_steps
        
        # Mathematical problem patterns
        self.problem_patterns = {
            'arithmetic': r'[\d\+\-\*/\(\)\s]+',
            'algebra': r'.*solve.*for.*[a-zA-Z]',
            'geometry': r'.*(area|perimeter|volume|circumference).*',
            'percentage': r'.*\d+.*percent.*|.*\d+%.*',
            'word_problem': r'.*(how many|how much|what is|calculate).*'
        }
    
    async def generate_async(self, messages_list: List[List[Dict]], **args):
        """
        Generate responses with mathematical reasoning enhancement.
        
        Args:
            messages_list: List of message conversations
            **args: Additional generation arguments
            
        Returns:
            List of generated responses
        """
        # Enhance messages with mathematical context
        enhanced_messages = self._enhance_math_context(messages_list)
        
        # Generate responses
        return await self.llm_engine.generate_async(enhanced_messages, **args)
    
    def _enhance_math_context(self, messages_list: List[List[Dict]]) -> List[List[Dict]]:
        """Add mathematical reasoning context to messages."""
        enhanced_messages = []
        
        for messages in messages_list:
            enhanced_conv = []
            
            for message in messages:
                if message.get("role") == "user":
                    # Analyze problem type
                    content = message.get("content", "")
                    problem_type = self._identify_problem_type(content)
                    
                    # Add mathematical guidance
                    if problem_type != "unknown":
                        math_guidance = self._get_math_guidance(problem_type)
                        
                        # Add guidance as system message
                        guidance_msg = {
                            "role": "system",
                            "content": math_guidance
                        }
                        enhanced_conv.append(guidance_msg)
                
                enhanced_conv.append(message)
            
            enhanced_messages.append(enhanced_conv)
        
        return enhanced_messages
    
    def _identify_problem_type(self, content: str) -> str:
        """Identify the type of mathematical problem."""
        content_lower = content.lower()
        
        for problem_type, pattern in self.problem_patterns.items():
            if re.search(pattern, content_lower):
                return problem_type
        
        return "unknown"
    
    def _get_math_guidance(self, problem_type: str) -> str:
        """Get problem-specific guidance."""
        guidance_templates = {
            'arithmetic': """
For arithmetic problems:
1. Identify the numbers and operations
2. Use the calculator tool for calculations
3. Show each step clearly
4. Format your final answer clearly
""",
            'algebra': """
For algebra problems:
1. Identify the variable to solve for
2. Isolate the variable step by step
3. Use the calculator for each arithmetic step
4. Check your answer by substitution
""",
            'geometry': """
For geometry problems:
1. Identify the shape and given measurements
2. Recall the appropriate formula
3. Substitute values into the formula
4. Calculate step by step using tools
""",
            'percentage': """
For percentage problems:
1. Identify what percentage of what number
2. Convert percentage to decimal (divide by 100)
3. Multiply using the calculator
4. Express answer appropriately
""",
            'word_problem': """
For word problems:
1. Read carefully and identify what's being asked
2. Extract the relevant numbers and operations
3. Set up the mathematical expression
4. Solve step by step using tools
"""
        }
        
        return guidance_templates.get(problem_type, "Solve this mathematical problem step by step.")
    
    def parse(self, responses: List[str], tools) -> List[Dict]:
        """
        Parse responses with mathematical reasoning analysis.
        
        Args:
            responses: List of LLM responses
            tools: Available tools
            
        Returns:
            List of parsed actions
        """
        parsed_actions = []
        
        for response in responses:
            # Standard parsing
            action = self._parse_single_response(response, tools)
            
            # Add mathematical analysis
            math_analysis = self._analyze_math_reasoning(response)
            action.update(math_analysis)
            
            parsed_actions.append(action)
        
        return parsed_actions
    
    def _parse_single_response(self, response: str, tools) -> Dict:
        """Parse a single response for tool calls."""
        # Look for tool calls in various formats
        tool_calls = []
        
        # JSON format: {"name": "tool_name", "arguments": {...}}
        json_pattern = r'\{[^}]*"name"[^}]*"arguments"[^}]*\}'
        json_matches = re.findall(json_pattern, response)
        
        for match in json_matches:
            try:
                tool_call = json.loads(match)
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        # Function call format: tool_name(arguments)
        func_pattern = r'(\w+)\((.*?)\)'
        func_matches = re.findall(func_pattern, response)
        
        tool_names = [tool.name for tool in tools] if tools else []
        
        for func_name, func_args in func_matches:
            if func_name in tool_names:
                # Parse arguments (simplified)
                tool_calls.append({
                    "name": func_name,
                    "arguments": {"input": func_args.strip('"\'\'"""')}
                })
        
        return {
            "response": response,
            "tool_calls": tool_calls,
            "should_continue": len(tool_calls) > 0
        }
    
    def _analyze_math_reasoning(self, response: str) -> Dict:
        """Analyze mathematical reasoning in response."""
        analysis = {
            "has_calculations": False,
            "shows_steps": False,
            "uses_notation": False,
            "has_final_answer": False,
            "reasoning_quality": 0.0
        }
        
        response_lower = response.lower()
        
        # Check for calculations
        if any(op in response for op in ['+', '-', '*', '/', '=']):
            analysis["has_calculations"] = True
        
        # Check for step indicators
        step_indicators = ["step", "first", "next", "then", "therefore", "so"]
        if any(indicator in response_lower for indicator in step_indicators):
            analysis["shows_steps"] = True
        
        # Check for mathematical notation
        math_notation = ["√", "²", "³", "π", "°", "∠", "∆", "≈", "±"]
        if any(notation in response for notation in math_notation):
            analysis["uses_notation"] = True
        
        # Check for final answer
        final_patterns = ["final answer", "answer:", "result:", "\\boxed"]
        if any(pattern in response_lower for pattern in final_patterns):
            analysis["has_final_answer"] = True
        
        # Calculate reasoning quality score
        quality_score = 0.0
        if analysis["shows_steps"]:
            quality_score += 0.3
        if analysis["has_calculations"]:
            quality_score += 0.2
        if analysis["uses_notation"]:
            quality_score += 0.2
        if analysis["has_final_answer"]:
            quality_score += 0.3
        
        analysis["reasoning_quality"] = quality_score
        
        return analysis
```

## Preparing Training Data

Create `scripts/prepare_data.py`:

```python
#!/usr/bin/env python3
"""
Prepare training data for MathAgent.
"""

import json
import random
import math
from typing import List, Dict, Any
from pathlib import Path

class MathDataGenerator:
    """Generate synthetic mathematical problems for training."""
    
    def __init__(self):
        """Initialize the data generator."""
        self.problem_types = [
            "arithmetic", "algebra", "geometry", "percentage", 
            "fractions", "word_problems", "trigonometry"
        ]
    
    def generate_arithmetic_problems(self, count: int) -> List[Dict]:
        """Generate arithmetic problems."""
        problems = []
        
        for i in range(count):
            # Random numbers and operations
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            operation = random.choice(['+', '-', '*', '/'])
            
            if operation == '+':
                question = f"Calculate {a} + {b}"
                answer = str(a + b)
                difficulty = "easy"
            elif operation == '-':
                # Ensure positive result
                if a < b:
                    a, b = b, a
                question = f"Calculate {a} - {b}"
                answer = str(a - b)
                difficulty = "easy"
            elif operation == '*':
                question = f"Calculate {a} × {b}"
                answer = str(a * b)
                difficulty = "medium"
            else:  # division
                # Ensure clean division
                a = a * b
                question = f"Calculate {a} ÷ {b}"
                answer = str(a // b)
                difficulty = "medium"
            
            problems.append({
                "question": question,
                "answer": answer,
                "id": f"arith_{i:04d}",
                "type": "arithmetic",
                "difficulty": difficulty,
                "expected_tools": ["advanced_calculator"]
            })
        
        return problems
    
    def generate_algebra_problems(self, count: int) -> List[Dict]:
        """Generate algebra problems."""
        problems = []
        
        for i in range(count):
            # Linear equations: ax + b = c
            a = random.randint(2, 10)
            x = random.randint(1, 20)
            b = random.randint(1, 50)
            c = a * x + b
            
            question = f"Solve for x: {a}x + {b} = {c}"
            answer = str(x)
            
            problems.append({
                "question": question,
                "answer": answer,
                "id": f"algebra_{i:04d}",
                "type": "algebra",
                "difficulty": "medium",
                "solution_steps": [
                    f"Subtract {b} from both sides: {a}x = {c - b}",
                    f"Divide by {a}: x = {x}"
                ],
                "expected_tools": ["advanced_calculator", "step_by_step_solver"]
            })
        
        return problems
    
    def generate_geometry_problems(self, count: int) -> List[Dict]:
        """Generate geometry problems."""
        problems = []
        
        shapes = [
            ("rectangle", "area"),
            ("rectangle", "perimeter"),
            ("circle", "area"),
            ("circle", "circumference"),
            ("triangle", "area")
        ]
        
        for i in range(count):
            shape, measurement = random.choice(shapes)
            
            if shape == "rectangle":
                length = random.randint(5, 20)
                width = random.randint(3, 15)
                
                if measurement == "area":
                    question = f"Find the area of a rectangle with length {length} and width {width}"
                    answer = str(length * width)
                else:  # perimeter
                    question = f"Find the perimeter of a rectangle with length {length} and width {width}"
                    answer = str(2 * (length + width))
            
            elif shape == "circle":
                radius = random.randint(3, 15)
                
                if measurement == "area":
                    question = f"Find the area of a circle with radius {radius} (use π ≈ 3.14159)"
                    answer = f"{math.pi * radius * radius:.2f}"
                else:  # circumference
                    question = f"Find the circumference of a circle with radius {radius} (use π ≈ 3.14159)"
                    answer = f"{2 * math.pi * radius:.2f}"
            
            else:  # triangle
                base = random.randint(4, 20)
                height = random.randint(3, 15)
                question = f"Find the area of a triangle with base {base} and height {height}"
                answer = f"{0.5 * base * height:.1f}"
            
            problems.append({
                "question": question,
                "answer": answer,
                "id": f"geo_{i:04d}",
                "type": "geometry",
                "difficulty": "medium",
                "shape": shape,
                "measurement": measurement,
                "expected_tools": ["advanced_calculator"]
            })
        
        return problems
    
    def generate_percentage_problems(self, count: int) -> List[Dict]:
        """Generate percentage problems."""
        problems = []
        
        for i in range(count):
            percentage = random.choice([10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
            number = random.randint(20, 500)
            
            question = f"What is {percentage}% of {number}?"
            answer = str(int(number * percentage / 100))
            
            problems.append({
                "question": question,
                "answer": answer,
                "id": f"percent_{i:04d}",
                "type": "percentage",
                "difficulty": "easy",
                "percentage": percentage,
                "base_number": number,
                "expected_tools": ["advanced_calculator"]
            })
        
        return problems
    
    def generate_word_problems(self, count: int) -> List[Dict]:
        """Generate word problems."""
        problems = []
        
        templates = [
            {
                "template": "Sarah has {a} apples. She buys {b} more apples. How many apples does she have now?",
                "operation": "addition",
                "answer_func": lambda a, b: a + b
            },
            {
                "template": "Tom has {a} marbles. He gives away {b} marbles. How many marbles does he have left?",
                "operation": "subtraction", 
                "answer_func": lambda a, b: a - b
            },
            {
                "template": "A box contains {a} rows of {b} items each. How many items are in the box?",
                "operation": "multiplication",
                "answer_func": lambda a, b: a * b
            },
            {
                "template": "Lisa has {a} stickers. She wants to divide them equally among {b} friends. How many stickers does each friend get?",
                "operation": "division",
                "answer_func": lambda a, b: a // b
            }
        ]
        
        for i in range(count):
            template_info = random.choice(templates)
            
            if template_info["operation"] == "division":
                b = random.randint(2, 10)
                a = b * random.randint(2, 20)  # Ensure clean division
            else:
                a = random.randint(5, 50)
                b = random.randint(2, 20)
                
                if template_info["operation"] == "subtraction" and b > a:
                    a, b = b, a  # Ensure positive result
            
            question = template_info["template"].format(a=a, b=b)
            answer = str(template_info["answer_func"](a, b))
            
            problems.append({
                "question": question,
                "answer": answer,
                "id": f"word_{i:04d}",
                "type": "word_problem",
                "difficulty": "medium",
                "operation": template_info["operation"],
                "numbers": [a, b],
                "expected_tools": ["advanced_calculator"]
            })
        
        return problems
    
    def generate_dataset(self, total_samples: int = 1000) -> List[Dict]:
        """Generate complete dataset."""
        # Distribution across problem types
        distribution = {
            "arithmetic": 0.3,
            "algebra": 0.2,
            "geometry": 0.15,
            "percentage": 0.15,
            "word_problems": 0.2
        }
        
        dataset = []
        
        for problem_type, ratio in distribution.items():
            count = int(total_samples * ratio)
            
            if problem_type == "arithmetic":
                problems = self.generate_arithmetic_problems(count)
            elif problem_type == "algebra":
                problems = self.generate_algebra_problems(count)
            elif problem_type == "geometry":
                problems = self.generate_geometry_problems(count)
            elif problem_type == "percentage":
                problems = self.generate_percentage_problems(count)
            elif problem_type == "word_problems":
                problems = self.generate_word_problems(count)
            
            dataset.extend(problems)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return dataset

def main():
    """Generate and save training data."""
    generator = MathDataGenerator()
    
    # Generate datasets
    print("Generating training data...")
    train_data = generator.generate_dataset(2000)
    
    print("Generating validation data...")
    val_data = generator.generate_dataset(400)
    
    print("Generating test data...")
    test_data = generator.generate_dataset(200)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save datasets
    with open(data_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(data_dir / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    with open(data_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Generated {len(train_data)} training samples")
    print(f"Generated {len(val_data)} validation samples")
    print(f"Generated {len(test_data)} test samples")
    
    # Print sample
    print("\nSample problems:")
    for i, problem in enumerate(train_data[:3]):
        print(f"{i+1}. {problem['question']}")
        print(f"   Answer: {problem['answer']}")
        print(f"   Type: {problem['type']}, Difficulty: {problem['difficulty']}")
        print()

if __name__ == "__main__":
    main()
```

Run data generation:

```bash
cd math_agent_project
python scripts/prepare_data.py
```

## Configuring Templates

Create `src/templates/math_template.py`:

```python
"""
Custom template for mathematical problem solving.
"""

from agents.agents.templates.templates import Template, register_template

# Math-specific template
math_template = Template(
    name="math_problem_template",
    
    system_template="""<|im_start|>system
{system_message}

When solving mathematical problems:
1. Read the problem carefully
2. Identify what is being asked
3. Show your work step by step
4. Use tools for calculations
5. Format your final answer clearly

Available tools:
{tools}

To use a tool, format as:
{{"name": "tool_name", "arguments": {{"parameter": "value"}}}}
<|im_end|>
""",
    
    system_message="""You are an expert mathematics tutor. You solve problems step by step, explain your reasoning clearly, and use appropriate tools for calculations. Always show your work and format your final answer clearly.""",
    
    user_template="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
    tool_template="<|im_start|>tool\nTool Result: {observation}<|im_end|>\n",
    
    stop_words=["<|im_end|>"],
    
    system_template_with_tools="""<|im_start|>system
{system_message}

# Mathematical Problem Solving Process

1. **Understand**: Read and understand what is being asked
2. **Plan**: Identify the approach and tools needed
3. **Execute**: Solve step by step using tools
4. **Verify**: Check your answer makes sense
5. **Present**: Format your final answer clearly

# Available Tools

{tools}

# Tool Usage Format

To use a tool, write:
{{"name": "tool_name", "arguments": {{"input": "your_input"}}}}

# Example

Question: "What is 15% of 240?"

Step 1: Understand - Find 15% of 240
Step 2: Plan - Convert percentage to decimal and multiply
Step 3: Execute calculation
{{"name": "advanced_calculator", "arguments": {{"expression": "240 * 0.15"}}}}

Always show your reasoning and use tools for calculations!
<|im_end|>
"""
)

# Register the template
register_template(math_template)
```

## Setting Up Training

Create `configs/training_config.yaml`:

```yaml
# Complete training configuration for MathAgent

# Agent Configuration
agent:
  agent_type: "react"
  model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  template: "math_problem_template"
  backend: "async_verl"
  
  # Tools
  tools:
    - "advanced_calculator"
    - "step_by_step_solver"
    - "math_validator"
    - "answer"
  
  # Agent behavior
  max_steps: 10
  num_chains: 8
  use_agent: true
  
  # Reward function
  reward_name: "math_comprehensive_reward"
  
  # System prompt
  system_prompt: "You are an expert mathematics tutor who solves problems step by step."

# Data Configuration
data:
  train_files: "./data/train.json"
  val_files: "./data/val.json"
  train_batch_size: 32
  val_batch_size: 16

# Model Configuration
model:
  path: "Qwen/Qwen2.5-7B-Instruct"
  use_remove_padding: false
  enable_gradient_checkpointing: false

# Training Configuration
training:
  algorithm: "grpo"
  learning_rate: 5e-7
  total_training_steps: 500
  kl_coef: 0.001
  entropy_coeff: 0.001
  response_length: 1024
  save_freq: 50
  test_freq: 20

# Hardware Configuration
hardware:
  num_gpus: 8
  num_nodes: 1
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.6
  use_ray: true
  ray_port: 6379

# Monitoring Configuration
monitoring:
  loggers: ["console", "wandb"]
  project_name: "MathAgent-Training"
  experiment_name: "math_agent_v1"
  log_level: "INFO"

# Optimization Configuration
optimization:
  ppo_mini_batch_size: 32
  ppo_micro_batch_size_per_gpu: 2
  use_kl_loss: true
  kl_loss_type: "mse"
  fsdp_param_offload: true
  fsdp_optimizer_offload: true
  critic_warmup: 0
```

## Running Training

Create `scripts/train.py`:

```python
#!/usr/bin/env python3
"""
Training script for MathAgent.
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_environment():
    """Setup training environment."""
    # Set environment variables
    os.environ['VLLM_USE_V1'] = '1'
    os.environ['HYDRA_FULL_ERROR'] = '1'
    
    # Initialize Ray
    print("Setting up Ray cluster...")
    subprocess.run(['ray', 'stop'], check=False)
    subprocess.run(['rm', '-rf', '/tmp/ray/ray_current_cluster'], check=False)
    
    # Start Ray head node
    head_node_ip = subprocess.check_output(['hostname', '--ip-address']).decode().strip()
    ray_cmd = [
        'ray', 'start', '--head',
        f'--node-ip-address={head_node_ip}',
        '--port=6379',
        '--num-cpus=192',
        '--num-gpus=8'
    ]
    
    result = subprocess.run(ray_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to start Ray: {result.stderr}")
        return False
    
    print(f"Ray cluster started on {head_node_ip}")
    return True

def load_config(config_file: str) -> dict:
    """Load training configuration."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def build_training_command(config: dict) -> list:
    """Build the training command."""
    cmd = ['python3', '-m', 'verl.trainer.main_ppo']
    
    # Add configuration parameters
    agent_config = config['agent']
    data_config = config['data']
    training_config = config['training']
    hardware_config = config['hardware']
    monitoring_config = config['monitoring']
    
    # Convert tools list to string format
    tools_str = str(agent_config['tools']).replace("'", '"')
    
    # Build command arguments
    cmd.extend([
        f"algorithm.adv_estimator={training_config['algorithm']}",
        f"algorithm.kl_ctrl.kl_coef={training_config['kl_coef']}",
        
        f"data.train_files={data_config['train_files']}",
        f"data.val_files={data_config['val_files']}",
        f"data.train_batch_size={data_config['train_batch_size']}",
        
        f"agent.agent_type={agent_config['agent_type']}",
        f"agent.model_name_or_path={agent_config['model_name_or_path']}",
        f"agent.template={agent_config['template']}",
        f"agent.tools={tools_str}",
        f"agent.reward_name={agent_config['reward_name']}",
        f"agent.max_steps={agent_config['max_steps']}",
        f"agent.num_chains={agent_config['num_chains']}",
        f"agent.backend={agent_config['backend']}",
        "agent.use_agent=True",
        
        f"actor_rollout_ref.actor.optim.lr={training_config['learning_rate']}",
        f"actor_rollout_ref.model.path={agent_config['model_name_or_path']}",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={data_config['train_batch_size']}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={training_config['kl_coef']}",
        "actor_rollout_ref.actor.kl_loss_type=mse",
        f"actor_rollout_ref.actor.entropy_coeff={training_config['entropy_coeff']}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={hardware_config['tensor_parallel_size']}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.response_length={training_config['response_length']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={hardware_config['gpu_memory_utilization']}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        
        f"critic.model.path={agent_config['model_name_or_path']}",
        f"critic.ppo_mini_batch_size={data_config['train_batch_size']}",
        "critic.ppo_micro_batch_size_per_gpu=2",
        
        f"trainer.total_training_steps={training_config['total_training_steps']}",
        f"trainer.save_freq={training_config['save_freq']}",
        f"trainer.test_freq={training_config['test_freq']}",
        "trainer.val_before_train=False",
        "trainer.critic_warmup=0",
        f"trainer.logger={monitoring_config['loggers']}",
        f"trainer.project_name={monitoring_config['project_name']}",
        f"trainer.experiment_name={monitoring_config['experiment_name']}",
        f"trainer.n_gpus_per_node={hardware_config['num_gpus']}",
        f"trainer.nnodes={hardware_config['num_nodes']}"
    ])
    
    return cmd

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MathAgent")
    parser.add_argument("--config", default="configs/training_config.yaml",
                        help="Training configuration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without executing")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip environment setup")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    if not args.skip_setup:
        if not setup_environment():
            print("Failed to setup environment")
            sys.exit(1)
    
    # Build training command
    cmd = build_training_command(config)
    
    if args.dry_run:
        print("Would execute:")
        print(" ".join(cmd))
        return
    
    # Create output directory
    output_dir = Path("outputs") / config['monitoring']['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_backup = output_dir / "training_config.yaml"
    with open(config_backup, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    print("Starting training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("Training completed successfully!")
        else:
            print(f"Training failed with return code {result.returncode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Running Training

Now you can start training your MathAgent:

```bash
# Make sure you're in the project directory
cd math_agent_project

# Register your custom components
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Start training
python scripts/train.py --config configs/training_config.yaml
```

## Evaluation and Testing

Create `scripts/evaluate.py`:

```python
#!/usr/bin/env python3
"""
Evaluation script for trained MathAgent.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.agents.react.react_agent import ReactAgent
from tools.calculator import advanced_calculator
from tools.validator import math_validator
from rewards.math_rewards import math_accuracy_reward

class MathAgentEvaluator:
    """Evaluate trained MathAgent performance."""
    
    def __init__(self, model_path: str, template: str = "math_problem_template"):
        """Initialize evaluator."""
        self.model_path = model_path
        self.template = template
        self.agent = None
        
    async def setup_agent(self):
        """Setup the agent for evaluation."""
        from tools.calculator import advanced_calculator
        from src.tools.validator import math_validator
        
        self.agent = ReactAgent(
            model_name_or_path=self.model_path,
            template=self.template,
            tools=[advanced_calculator, math_validator],
            system_prompt="You are an expert mathematics tutor. Solve problems step by step.",
            backend="transformers"  # Use simpler backend for evaluation
        )
        
    async def evaluate_dataset(self, dataset_file: str) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        # Load dataset
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        print(f"Evaluating on {len(dataset)} problems...")
        
        results = {
            "total_problems": len(dataset),
            "correct_answers": 0,
            "accuracy": 0.0,
            "by_type": {},
            "by_difficulty": {},
            "detailed_results": []
        }
        
        for i, problem in enumerate(dataset):
            print(f"Problem {i+1}/{len(dataset)}: {problem['type']}")
            
            # Prepare input
            messages = [{
                "messages": [
                    {"role": "user", "content": problem["question"]}
                ],
                "question": problem["question"],
                **problem  # Include all problem metadata
            }]
            
            # Run agent
            try:
                await self.agent.run_async(
                    max_steps=8,
                    start_messages=messages,
                    num_chains=1
                )
                
                # Get agent response
                if self.agent.trajectories:
                    trajectory = self.agent.trajectories[0]
                    prediction = trajectory[-1].get("content", "") if trajectory else ""
                else:
                    prediction = ""
                
                # Calculate reward
                reward_result = math_accuracy_reward(
                    prediction, 
                    problem["answer"],
                    **problem
                )
                
                is_correct = reward_result["exact_match"]
                
                # Update results
                if is_correct:
                    results["correct_answers"] += 1
                
                # Track by type
                prob_type = problem.get("type", "unknown")
                if prob_type not in results["by_type"]:
                    results["by_type"][prob_type] = {"total": 0, "correct": 0}
                results["by_type"][prob_type]["total"] += 1
                if is_correct:
                    results["by_type"][prob_type]["correct"] += 1
                
                # Track by difficulty
                difficulty = problem.get("difficulty", "unknown")
                if difficulty not in results["by_difficulty"]:
                    results["by_difficulty"][difficulty] = {"total": 0, "correct": 0}
                results["by_difficulty"][difficulty]["total"] += 1
                if is_correct:
                    results["by_difficulty"][difficulty]["correct"] += 1
                
                # Store detailed result
                results["detailed_results"].append({
                    "problem_id": problem.get("id", f"prob_{i}"),
                    "question": problem["question"],
                    "true_answer": problem["answer"],
                    "predicted_answer": reward_result.get("predicted_answer", ""),
                    "is_correct": is_correct,
                    "reward": reward_result["reward"],
                    "type": prob_type,
                    "difficulty": difficulty,
                    "agent_response": prediction
                })
                
            except Exception as e:
                print(f"Error on problem {i+1}: {str(e)}")
                results["detailed_results"].append({
                    "problem_id": problem.get("id", f"prob_{i}"),
                    "question": problem["question"],
                    "error": str(e),
                    "is_correct": False
                })
        
        # Calculate final accuracy
        results["accuracy"] = results["correct_answers"] / results["total_problems"]
        
        # Calculate accuracy by type
        for type_name, type_results in results["by_type"].items():
            type_results["accuracy"] = type_results["correct"] / type_results["total"]
        
        # Calculate accuracy by difficulty
        for diff_name, diff_results in results["by_difficulty"].items():
            diff_results["accuracy"] = diff_results["correct"] / diff_results["total"]
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"Overall Accuracy: {results['accuracy']:.3f} ({results['correct_answers']}/{results['total_problems']})")
        
        print("\nAccuracy by Problem Type:")
        for type_name, type_results in results["by_type"].items():
            print(f"  {type_name}: {type_results['accuracy']:.3f} ({type_results['correct']}/{type_results['total']})")
        
        print("\nAccuracy by Difficulty:")
        for diff_name, diff_results in results["by_difficulty"].items():
            print(f"  {diff_name}: {diff_results['accuracy']:.3f} ({diff_results['correct']}/{diff_results['total']})")
        
        print("\nSample Incorrect Answers:")
        incorrect_samples = [r for r in results["detailed_results"] if not r["is_correct"]][:5]
        for i, sample in enumerate(incorrect_samples):
            print(f"\n{i+1}. Question: {sample['question']}")
            print(f"   True Answer: {sample['true_answer']}")
            print(f"   Predicted: {sample.get('predicted_answer', 'N/A')}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save detailed results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {output_file}")

async def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MathAgent")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", default="data/test.json", help="Test dataset")
    parser.add_argument("--output", default="outputs/evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MathAgentEvaluator(args.model)
    await evaluator.setup_agent()
    
    # Run evaluation
    results = await evaluator.evaluate_dataset(args.data)
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results, args.output)

if __name__ == "__main__":
    asyncio.run(main())
```

## Deployment

Once training is complete, you can deploy your trained MathAgent:

Create `scripts/inference_server.py`:

```python
#!/usr/bin/env python3
"""
Simple inference server for trained MathAgent.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = FastAPI(title="MathAgent API", version="1.0.0")

# Global agent instance
agent = None

class ProblemRequest(BaseModel):
    question: str
    max_steps: int = 8
    show_work: bool = True

class ProblemResponse(BaseModel):
    question: str
    answer: str
    reasoning: List[str]
    confidence: float
    tool_calls: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    
    from agents.react.react_agent import ReactAgent
    from tools.calculator import advanced_calculator
    from tools.validator import math_validator
    
    # Load trained model
    model_path = "outputs/math_agent_v1/checkpoints/final_model"
    
    agent = ReactAgent(
        model_name_or_path=model_path,
        template="math_problem_template",
        tools=[advanced_calculator, math_validator],
        system_prompt="You are an expert mathematics tutor. Solve problems step by step.",
        backend="transformers"
    )
    
    print("MathAgent loaded and ready!")

@app.post("/solve", response_model=ProblemResponse)
async def solve_problem(request: ProblemRequest):
    """Solve a mathematical problem."""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Prepare input
        messages = [{
            "messages": [
                {"role": "user", "content": request.question}
            ],
            "question": request.question
        }]
        
        # Run agent
        await agent.run_async(
            max_steps=request.max_steps,
            start_messages=messages,
            num_chains=1
        )
        
        # Extract response
        if agent.trajectories:
            trajectory = agent.trajectories[0]
            
            # Get final answer
            final_response = trajectory[-1].get("content", "") if trajectory else ""
            
            # Extract reasoning steps
            reasoning_steps = []
            tool_calls = []
            
            for step in trajectory:
                if step.get("role") == "assistant":
                    content = step.get("content", "")
                    if content and "tool_call" not in content.lower():
                        reasoning_steps.append(content)
                
                elif step.get("role") == "tool":
                    tool_calls.append({
                        "tool": step.get("name", "unknown"),
                        "input": step.get("arguments", {}),
                        "output": step.get("content", "")
                    })
            
            # Extract final answer
            from rewards.math_rewards import extract_final_answer
            answer = extract_final_answer(final_response)
            
            return ProblemResponse(
                question=request.question,
                answer=answer,
                reasoning=reasoning_steps if request.show_work else [],
                confidence=0.95,  # Placeholder
                tool_calls=tool_calls if request.show_work else []
            )
        
        else:
            raise HTTPException(status_code=500, detail="No response generated")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent_loaded": agent is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Conclusion

Congratulations! You've built a complete pipeline for training a customized mathematical problem-solving agent with AgentFly. Here's what you've accomplished:

### 🎯 What You Built

1. **Custom Tools**: Advanced calculator, step-by-step solver, and answer validator
2. **Reward Functions**: Comprehensive rewards that evaluate both accuracy and reasoning process
3. **Custom Agent**: MathAgent with specialized mathematical reasoning capabilities
4. **Training Data**: 2,600 synthetic mathematical problems across different types and difficulties
5. **Training Pipeline**: Complete RL training setup with monitoring and evaluation
6. **Deployment**: Inference server for serving the trained model

### 🚀 Key Features

- **Multi-type Problem Solving**: Arithmetic, algebra, geometry, percentages, and word problems
- **Step-by-step Reasoning**: Agent shows work and explains its thinking
- **Tool Integration**: Uses calculators and validators for accurate computations
- **Process Rewards**: Learns not just correct answers but good reasoning processes
- **Comprehensive Evaluation**: Detailed metrics by problem type and difficulty

### 📈 Next Steps

1. **Expand Problem Types**: Add calculus, statistics, or advanced mathematics
2. **Improve Reasoning**: Add more sophisticated reasoning patterns
3. **Multi-modal Support**: Handle mathematical diagrams and graphs
4. **Fine-tune Rewards**: Optimize reward functions based on training results
5. **Scale Up**: Train on larger datasets and more powerful models

### 🛠️ Customization Tips

- **Modify Tools**: Add domain-specific calculators or validators
- **Adjust Rewards**: Weight accuracy vs. process differently
- **Change Templates**: Customize conversation format for your use case
- **Extend Data**: Generate more diverse and challenging problems
- **Experiment with Models**: Try different base models and architectures

You now have a complete understanding of how to build, train, and deploy customized agents with AgentFly. This pipeline can be adapted for many other domains beyond mathematics - just replace the tools, rewards, and data with domain-specific components!

Happy training! 🎉