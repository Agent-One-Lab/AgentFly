# Tutorial 4: Data Preparation

This tutorial covers how to prepare and format training data for AgentFly agents. Proper data preparation is crucial for successful reinforcement learning training.

## Table of Contents

1. [Understanding Data Format](#understanding-data-format)
2. [Basic Data Structure](#basic-data-structure)
3. [Creating Training Data](#creating-training-data)
4. [Data Processing Scripts](#data-processing-scripts)
5. [Validation and Quality Control](#validation-and-quality-control)
6. [Advanced Data Patterns](#advanced-data-patterns)
7. [Best Practices](#best-practices)

## Understanding Data Format

AgentFly uses JSON format for training data. Each dataset is a list of dictionaries where each dictionary represents one training example. The framework supports both simple question-answer pairs and complex multi-turn conversations.

### Basic Requirements

- **question**: Required field used to format input messages
- **answer**: Typically the target answer (used in reward functions)
- **Additional fields**: Can be used in reward functions and agents

## Basic Data Structure

### Simple Question-Answer Format

```json
[
    {
        "question": "What is 15% of 240?",
        "answer": "36",
        "id": "math_001",
        "difficulty": "easy",
        "category": "percentage"
    },
    {
        "question": "Solve the equation: 2x + 5 = 13",
        "answer": "x = 4",
        "id": "math_002", 
        "difficulty": "medium",
        "category": "algebra"
    }
]
```

### Multi-Field Format for Complex Tasks

```json
[
    {
        "question": "Write a Python function to calculate the factorial of a number",
        "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "test_cases": [
            "assert factorial(5) == 120",
            "assert factorial(0) == 1",
            "assert factorial(3) == 6"
        ],
        "difficulty": "medium",
        "topic": "recursion",
        "expected_tools": ["code_interpreter"],
        "id": "code_001"
    }
]
```

### Conversation Format

```json
[
    {
        "question": "Help me debug this Python code that's supposed to sort a list",
        "code_snippet": "def sort_list(lst):\n    for i in range(len(lst)):\n        lst[i] = lst[i+1]\n    return lst",
        "expected_fix": "Fix index out of bounds error and implement proper sorting",
        "answer": "def sort_list(lst):\n    return sorted(lst)",
        "error_type": "index_error",
        "id": "debug_001"
    }
]
```

## Creating Training Data

### Math Problems Dataset

```python
import json
import random
from typing import List, Dict

def create_math_dataset(num_samples: int = 1000) -> List[Dict]:
    """Create a synthetic math problems dataset."""
    
    dataset = []
    problem_types = [
        "arithmetic", "algebra", "geometry", "percentage", 
        "word_problems", "fractions", "decimals"
    ]
    
    for i in range(num_samples):
        problem_type = random.choice(problem_types)
        
        if problem_type == "arithmetic":
            problem = create_arithmetic_problem(i)
        elif problem_type == "algebra":
            problem = create_algebra_problem(i)
        elif problem_type == "percentage":
            problem = create_percentage_problem(i)
        elif problem_type == "word_problems":
            problem = create_word_problem(i)
        else:
            problem = create_basic_problem(i, problem_type)
        
        dataset.append(problem)
    
    return dataset

def create_arithmetic_problem(problem_id: int) -> Dict:
    """Create an arithmetic problem."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        question = f"What is {a} + {b}?"
        answer = str(a + b)
    elif operation == '-':
        question = f"What is {a} - {b}?"
        answer = str(a - b)
    else:  # multiplication
        question = f"What is {a} × {b}?"
        answer = str(a * b)
    
    return {
        "question": question,
        "answer": answer,
        "id": f"arith_{problem_id}",
        "type": "arithmetic",
        "difficulty": "easy" if max(a, b) <= 20 else "medium",
        "operation": operation,
        "operands": [a, b]
    }

def create_algebra_problem(problem_id: int) -> Dict:
    """Create an algebra problem."""
    x_value = random.randint(1, 20)
    coefficient = random.randint(2, 10)
    constant = random.randint(1, 50)
    
    # Create equation: coefficient * x + constant = result
    result = coefficient * x_value + constant
    
    question = f"Solve for x: {coefficient}x + {constant} = {result}"
    answer = f"x = {x_value}"
    
    return {
        "question": question,
        "answer": answer,
        "id": f"algebra_{problem_id}",
        "type": "algebra",
        "difficulty": "medium",
        "equation_type": "linear",
        "solution": x_value
    }

def create_percentage_problem(problem_id: int) -> Dict:
    """Create a percentage problem."""
    base_number = random.randint(50, 500)
    percentage = random.choice([10, 15, 20, 25, 30, 50, 75])
    
    question = f"What is {percentage}% of {base_number}?"
    answer = str(int(base_number * percentage / 100))
    
    return {
        "question": question,
        "answer": answer,
        "id": f"percent_{problem_id}",
        "type": "percentage",
        "difficulty": "easy",
        "percentage": percentage,
        "base_number": base_number
    }

def create_word_problem(problem_id: int) -> Dict:
    """Create a word problem."""
    templates = [
        {
            "template": "Sarah has {a} apples. She gives away {b} apples. How many apples does she have left?",
            "operation": "subtraction",
            "answer_formula": lambda a, b: a - b
        },
        {
            "template": "A store has {a} items. They receive {b} more items. How many items do they have in total?",
            "operation": "addition", 
            "answer_formula": lambda a, b: a + b
        },
        {
            "template": "Tom buys {a} packs of cards. Each pack has {b} cards. How many cards does he have in total?",
            "operation": "multiplication",
            "answer_formula": lambda a, b: a * b
        }
    ]
    
    template_info = random.choice(templates)
    a = random.randint(5, 50)
    b = random.randint(2, 20)
    
    question = template_info["template"].format(a=a, b=b)
    answer = str(template_info["answer_formula"](a, b))
    
    return {
        "question": question,
        "answer": answer,
        "id": f"word_{problem_id}",
        "type": "word_problem",
        "difficulty": "medium",
        "operation": template_info["operation"],
        "numbers": [a, b]
    }

# Generate and save dataset
math_dataset = create_math_dataset(1000)

with open("math_training_data.json", "w") as f:
    json.dump(math_dataset, f, indent=2)

print(f"Created dataset with {len(math_dataset)} problems")
```

### Code Generation Dataset

```python
import json
from typing import List, Dict

def create_code_dataset(num_samples: int = 500) -> List[Dict]:
    """Create a code generation dataset."""
    
    dataset = []
    programming_tasks = [
        "basic_functions", "list_operations", "string_manipulation",
        "conditionals", "loops", "data_structures", "algorithms"
    ]
    
    for i in range(num_samples):
        task_type = random.choice(programming_tasks)
        
        if task_type == "basic_functions":
            task = create_function_task(i)
        elif task_type == "list_operations":
            task = create_list_task(i)
        elif task_type == "string_manipulation":
            task = create_string_task(i)
        elif task_type == "algorithms":
            task = create_algorithm_task(i)
        else:
            task = create_general_task(i, task_type)
        
        dataset.append(task)
    
    return dataset

def create_function_task(task_id: int) -> Dict:
    """Create a function writing task."""
    tasks = [
        {
            "description": "Write a function that takes two numbers and returns their sum",
            "function_name": "add_numbers",
            "parameters": ["a", "b"],
            "example_code": "def add_numbers(a, b):\n    return a + b",
            "test_cases": [
                "assert add_numbers(2, 3) == 5",
                "assert add_numbers(-1, 1) == 0",
                "assert add_numbers(0, 0) == 0"
            ]
        },
        {
            "description": "Write a function that checks if a number is even",
            "function_name": "is_even",
            "parameters": ["num"],
            "example_code": "def is_even(num):\n    return num % 2 == 0",
            "test_cases": [
                "assert is_even(4) == True",
                "assert is_even(7) == False",
                "assert is_even(0) == True"
            ]
        }
    ]
    
    task = random.choice(tasks)
    
    return {
        "question": f"{task['description']}. The function should be named '{task['function_name']}' and take parameters: {', '.join(task['parameters'])}.",
        "answer": task["example_code"],
        "id": f"func_{task_id}",
        "type": "function_writing",
        "difficulty": "easy",
        "function_name": task["function_name"],
        "test_cases": task["test_cases"],
        "expected_tools": ["code_interpreter"]
    }

def create_algorithm_task(task_id: int) -> Dict:
    """Create an algorithm implementation task."""
    algorithms = [
        {
            "name": "Binary Search",
            "description": "Implement binary search to find an element in a sorted list",
            "template": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
            "test_cases": [
                "assert binary_search([1, 2, 3, 4, 5], 3) == 2",
                "assert binary_search([1, 2, 3, 4, 5], 6) == -1",
                "assert binary_search([], 1) == -1"
            ]
        },
        {
            "name": "Bubble Sort",
            "description": "Implement bubble sort algorithm to sort a list of numbers",
            "template": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
            "test_cases": [
                "assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]",
                "assert bubble_sort([1]) == [1]",
                "assert bubble_sort([]) == []"
            ]
        }
    ]
    
    algorithm = random.choice(algorithms)
    
    return {
        "question": f"{algorithm['description']}. Make sure to handle edge cases properly.",
        "answer": algorithm["template"],
        "id": f"algo_{task_id}",
        "type": "algorithm",
        "difficulty": "hard",
        "algorithm_name": algorithm["name"],
        "test_cases": algorithm["test_cases"],
        "expected_tools": ["code_interpreter"]
    }

# Generate code dataset
code_dataset = create_code_dataset(500)

with open("code_training_data.json", "w") as f:
    json.dump(code_dataset, f, indent=2)
```

### Question Answering Dataset

```python
def create_qa_dataset(num_samples: int = 800) -> List[Dict]:
    """Create a question answering dataset."""
    
    categories = [
        "science", "history", "geography", "literature", 
        "technology", "sports", "general_knowledge"
    ]
    
    qa_templates = {
        "science": [
            ("What is the chemical symbol for {element}?", "chemistry"),
            ("What is the speed of light in vacuum?", "physics"),
            ("What is photosynthesis?", "biology")
        ],
        "history": [
            ("When did World War II end?", "modern_history"),
            ("Who was the first president of the United States?", "american_history"),
            ("What year did the Berlin Wall fall?", "modern_history")
        ],
        "geography": [
            ("What is the capital of {country}?", "capitals"),
            ("Which is the longest river in the world?", "physical_geography"),
            ("What is the largest ocean on Earth?", "physical_geography")
        ]
    }
    
    # Predefined facts for template filling
    facts = {
        "elements": ["Gold", "Silver", "Iron", "Carbon", "Oxygen"],
        "symbols": ["Au", "Ag", "Fe", "C", "O"],
        "countries": ["France", "Germany", "Japan", "Brazil", "Australia"],
        "capitals": ["Paris", "Berlin", "Tokyo", "Brasilia", "Canberra"]
    }
    
    dataset = []
    
    for i in range(num_samples):
        category = random.choice(categories)
        
        if category in qa_templates:
            template, subcategory = random.choice(qa_templates[category])
            
            # Fill templates with facts
            if "{element}" in template:
                element_idx = random.randint(0, len(facts["elements"]) - 1)
                question = template.format(element=facts["elements"][element_idx])
                answer = facts["symbols"][element_idx]
            elif "{country}" in template:
                country_idx = random.randint(0, len(facts["countries"]) - 1)
                question = template.format(country=facts["countries"][country_idx])
                answer = facts["capitals"][country_idx]
            else:
                question = template
                answer = get_predefined_answer(template)
            
            dataset.append({
                "question": question,
                "answer": answer,
                "id": f"qa_{i}",
                "category": category,
                "subcategory": subcategory,
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "type": "factual_qa"
            })
    
    return dataset

def get_predefined_answer(question: str) -> str:
    """Get predefined answers for template questions."""
    answers = {
        "What is the speed of light in vacuum?": "299,792,458 meters per second",
        "What is photosynthesis?": "The process by which plants convert sunlight into energy",
        "When did World War II end?": "1945",
        "Who was the first president of the United States?": "George Washington",
        "What year did the Berlin Wall fall?": "1989",
        "Which is the longest river in the world?": "The Nile River",
        "What is the largest ocean on Earth?": "Pacific Ocean"
    }
    return answers.get(question, "Unknown")

# Generate QA dataset  
qa_dataset = create_qa_dataset(800)

with open("qa_training_data.json", "w") as f:
    json.dump(qa_dataset, f, indent=2)
```

## Data Processing Scripts

### Data Conversion Script

```python
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class DataProcessor:
    """Utility class for processing training data."""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def convert_csv_to_json(self, csv_file: str, question_col: str, answer_col: str, **kwargs) -> str:
        """Convert CSV file to AgentFly JSON format."""
        df = pd.read_csv(csv_file)
        
        dataset = []
        for idx, row in df.iterrows():
            item = {
                "question": str(row[question_col]),
                "answer": str(row[answer_col]),
                "id": f"item_{idx}"
            }
            
            # Add additional columns
            for col in df.columns:
                if col not in [question_col, answer_col]:
                    item[col.lower()] = row[col]
            
            # Add any additional kwargs
            item.update(kwargs)
            
            dataset.append(item)
        
        output_file = self.output_dir / f"{Path(csv_file).stem}.json"
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Converted {len(dataset)} items from {csv_file} to {output_file}")
        return str(output_file)
    
    def merge_datasets(self, dataset_files: List[str], output_name: str = "merged_dataset.json") -> str:
        """Merge multiple JSON datasets."""
        merged_data = []
        
        for file_path in dataset_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                merged_data.extend(data)
        
        # Re-assign IDs to avoid conflicts
        for idx, item in enumerate(merged_data):
            item["id"] = f"merged_{idx}"
        
        output_file = self.output_dir / output_name
        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"Merged {len(merged_data)} items into {output_file}")
        return str(output_file)
    
    def split_dataset(self, dataset_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split dataset into train, validation, and test sets."""
        with open(dataset_file, "r") as f:
            data = json.load(f)
        
        random.shuffle(data)
        
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        base_name = Path(dataset_file).stem
        
        # Save splits
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        
        output_files = {}
        for split_name, split_data in splits.items():
            output_file = self.output_dir / f"{base_name}_{split_name}.json"
            with open(output_file, "w") as f:
                json.dump(split_data, f, indent=2)
            output_files[split_name] = str(output_file)
            print(f"Saved {len(split_data)} {split_name} samples to {output_file}")
        
        return output_files
    
    def filter_dataset(self, dataset_file: str, filter_func: callable, output_name: str = None) -> str:
        """Filter dataset based on a function."""
        with open(dataset_file, "r") as f:
            data = json.load(f)
        
        filtered_data = [item for item in data if filter_func(item)]
        
        if output_name is None:
            output_name = f"{Path(dataset_file).stem}_filtered.json"
        
        output_file = self.output_dir / output_name
        with open(output_file, "w") as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Filtered {len(data)} items to {len(filtered_data)} items, saved to {output_file}")
        return str(output_file)
    
    def add_metadata(self, dataset_file: str, metadata_func: callable, output_name: str = None) -> str:
        """Add metadata to each item in the dataset."""
        with open(dataset_file, "r") as f:
            data = json.load(f)
        
        for item in data:
            metadata = metadata_func(item)
            item.update(metadata)
        
        if output_name is None:
            output_name = f"{Path(dataset_file).stem}_with_metadata.json"
        
        output_file = self.output_dir / output_name
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Added metadata to {len(data)} items, saved to {output_file}")
        return str(output_file)

# Usage example
processor = DataProcessor()

# Convert CSV to JSON
# processor.convert_csv_to_json("questions.csv", "question", "answer", difficulty="medium")

# Filter for specific difficulty
def filter_easy_problems(item):
    return item.get("difficulty") == "easy"

# filtered_file = processor.filter_dataset("math_training_data.json", filter_easy_problems)

# Add complexity metadata
def add_complexity_metadata(item):
    question_length = len(item["question"])
    answer_length = len(item["answer"])
    
    complexity = "simple"
    if question_length > 100 or answer_length > 50:
        complexity = "complex"
    elif question_length > 50 or answer_length > 20:
        complexity = "medium"
    
    return {
        "complexity": complexity,
        "question_length": question_length,
        "answer_length": answer_length
    }

# enhanced_file = processor.add_metadata("math_training_data.json", add_complexity_metadata)
```

### Data Validation Script

```python
import json
from typing import List, Dict, Set
from pathlib import Path

class DataValidator:
    """Validate training data for AgentFly."""
    
    def __init__(self):
        self.required_fields = {"question"}
        self.recommended_fields = {"answer", "id"}
        self.errors = []
        self.warnings = []
    
    def validate_dataset(self, dataset_file: str) -> Dict[str, Any]:
        """Validate a complete dataset."""
        self.errors = []
        self.warnings = []
        
        try:
            with open(dataset_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON format: {str(e)}")
            return {"valid": False, "errors": self.errors}
        except FileNotFoundError:
            self.errors.append(f"File not found: {dataset_file}")
            return {"valid": False, "errors": self.errors}
        
        if not isinstance(data, list):
            self.errors.append("Dataset must be a list of dictionaries")
            return {"valid": False, "errors": self.errors}
        
        # Validate each item
        for idx, item in enumerate(data):
            self._validate_item(item, idx)
        
        # Dataset-level validations
        self._validate_ids(data)
        self._validate_consistency(data)
        
        stats = self._calculate_stats(data)
        
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": stats
        }
    
    def _validate_item(self, item: Dict, idx: int):
        """Validate a single data item."""
        if not isinstance(item, dict):
            self.errors.append(f"Item {idx}: Must be a dictionary")
            return
        
        # Check required fields
        for field in self.required_fields:
            if field not in item:
                self.errors.append(f"Item {idx}: Missing required field '{field}'")
            elif not item[field] or (isinstance(item[field], str) and not item[field].strip()):
                self.errors.append(f"Item {idx}: Field '{field}' is empty")
        
        # Check recommended fields
        for field in self.recommended_fields:
            if field not in item:
                self.warnings.append(f"Item {idx}: Missing recommended field '{field}'")
        
        # Validate question field
        if "question" in item:
            question = item["question"]
            if isinstance(question, str):
                if len(question) > 10000:
                    self.warnings.append(f"Item {idx}: Question is very long ({len(question)} chars)")
                if len(question) < 10:
                    self.warnings.append(f"Item {idx}: Question is very short ({len(question)} chars)")
            else:
                self.errors.append(f"Item {idx}: Question must be a string")
        
        # Validate answer field
        if "answer" in item:
            answer = item["answer"]
            if not isinstance(answer, (str, list, dict)):
                self.warnings.append(f"Item {idx}: Answer should be string, list, or dict")
        
        # Validate test_cases if present
        if "test_cases" in item:
            test_cases = item["test_cases"]
            if not isinstance(test_cases, list):
                self.errors.append(f"Item {idx}: test_cases must be a list")
            elif len(test_cases) == 0:
                self.warnings.append(f"Item {idx}: test_cases is empty")
    
    def _validate_ids(self, data: List[Dict]):
        """Validate ID uniqueness."""
        ids = []
        for idx, item in enumerate(data):
            if "id" in item:
                if item["id"] in ids:
                    self.errors.append(f"Duplicate ID found: {item['id']}")
                ids.append(item["id"])
            else:
                self.warnings.append(f"Item {idx}: No ID field")
    
    def _validate_consistency(self, data: List[Dict]):
        """Validate data consistency."""
        # Check field consistency across items
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        field_coverage = {}
        for field in all_fields:
            coverage = sum(1 for item in data if field in item) / len(data)
            field_coverage[field] = coverage
            
            if coverage < 0.8 and field not in ["id", "metadata"]:
                self.warnings.append(f"Field '{field}' only present in {coverage:.1%} of items")
        
        # Check for potential data quality issues
        question_lengths = [len(item.get("question", "")) for item in data]
        avg_length = sum(question_lengths) / len(question_lengths)
        
        if avg_length < 20:
            self.warnings.append(f"Average question length is very short ({avg_length:.1f} chars)")
        elif avg_length > 500:
            self.warnings.append(f"Average question length is very long ({avg_length:.1f} chars)")
    
    def _calculate_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        stats = {
            "total_items": len(data),
            "fields": {},
            "question_stats": {},
            "answer_stats": {}
        }
        
        # Field statistics
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        for field in all_fields:
            field_count = sum(1 for item in data if field in item)
            stats["fields"][field] = {
                "count": field_count,
                "coverage": field_count / len(data)
            }
        
        # Question statistics
        questions = [item.get("question", "") for item in data]
        if questions:
            question_lengths = [len(q) for q in questions]
            stats["question_stats"] = {
                "avg_length": sum(question_lengths) / len(question_lengths),
                "min_length": min(question_lengths),
                "max_length": max(question_lengths)
            }
        
        # Answer statistics  
        answers = [item.get("answer", "") for item in data if "answer" in item]
        if answers:
            answer_lengths = [len(str(a)) for a in answers]
            stats["answer_stats"] = {
                "avg_length": sum(answer_lengths) / len(answer_lengths),
                "min_length": min(answer_lengths),
                "max_length": max(answer_lengths)
            }
        
        return stats

# Usage
validator = DataValidator()
result = validator.validate_dataset("math_training_data.json")

if result["valid"]:
    print("✅ Dataset is valid!")
else:
    print("❌ Dataset has errors:")
    for error in result["errors"]:
        print(f"  - {error}")

if result["warnings"]:
    print("⚠️  Warnings:")
    for warning in result["warnings"]:
        print(f"  - {warning}")

print(f"\n📊 Dataset stats: {result['stats']['total_items']} items")
```

## Advanced Data Patterns

### Multi-Turn Conversation Data

```json
[
    {
        "conversation_id": "conv_001",
        "turns": [
            {
                "role": "user",
                "content": "I need help with a Python error"
            },
            {
                "role": "assistant", 
                "content": "I'd be happy to help! What error are you getting?"
            },
            {
                "role": "user",
                "content": "IndexError: list index out of range"
            },
            {
                "role": "assistant",
                "content": "This error occurs when you try to access an index that doesn't exist. Can you show me your code?"
            }
        ],
        "final_question": "Help debug my Python IndexError",
        "context": "debugging",
        "expected_resolution": "Identify and fix index bounds issue"
    }
]
```

### Tool-Specific Data

```json
[
    {
        "question": "Calculate the area of a circle with radius 5",
        "answer": "78.54",
        "expected_tools": ["code_interpreter"],
        "tool_sequence": [
            {
                "tool": "code_interpreter",
                "input": "import math\nradius = 5\narea = math.pi * radius ** 2\nprint(f'Area: {area:.2f}')"
            }
        ],
        "verification_code": "import math; assert abs(math.pi * 25 - 78.54) < 0.01"
    }
]
```

### Hierarchical Task Data

```json
[
    {
        "main_task": "Build a simple calculator",
        "subtasks": [
            {
                "subtask_id": "calc_1",
                "description": "Create addition function",
                "question": "Write a function that adds two numbers",
                "answer": "def add(a, b): return a + b"
            },
            {
                "subtask_id": "calc_2", 
                "description": "Create subtraction function",
                "question": "Write a function that subtracts two numbers",
                "answer": "def subtract(a, b): return a - b"
            }
        ],
        "integration_task": "Combine all functions into a calculator class",
        "difficulty": "medium",
        "estimated_time": "30 minutes"
    }
]
```

## Best Practices

### 1. Data Quality Guidelines

```python
def ensure_data_quality():
    """Guidelines for high-quality training data."""
    
    guidelines = {
        "questions": [
            "Be specific and clear",
            "Avoid ambiguous wording", 
            "Include necessary context",
            "Vary complexity levels"
        ],
        "answers": [
            "Provide complete solutions",
            "Include step-by-step reasoning when appropriate",
            "Use consistent formatting",
            "Verify correctness"
        ],
        "metadata": [
            "Add difficulty levels",
            "Include topic/category tags",
            "Specify expected tools",
            "Add time estimates"
        ]
    }
    
    return guidelines
```

### 2. Balanced Dataset Creation

```python
def create_balanced_dataset(specifications: Dict) -> List[Dict]:
    """Create a balanced dataset across different criteria."""
    
    dataset = []
    
    # Balance by difficulty
    difficulty_distribution = {
        "easy": 0.4,
        "medium": 0.4, 
        "hard": 0.2
    }
    
    # Balance by topic
    topic_distribution = {
        "math": 0.3,
        "coding": 0.3,
        "reasoning": 0.2,
        "factual": 0.2
    }
    
    total_samples = specifications["total_samples"]
    
    for difficulty, diff_ratio in difficulty_distribution.items():
        for topic, topic_ratio in topic_distribution.items():
            num_samples = int(total_samples * diff_ratio * topic_ratio)
            
            samples = generate_samples(
                topic=topic,
                difficulty=difficulty,
                count=num_samples
            )
            
            dataset.extend(samples)
    
    # Shuffle to avoid ordering bias
    random.shuffle(dataset)
    
    return dataset
```

### 3. Data Versioning

```python
import hashlib
from datetime import datetime

def version_dataset(dataset: List[Dict], description: str = "") -> Dict:
    """Create a versioned dataset with metadata."""
    
    # Calculate content hash
    content_str = json.dumps(dataset, sort_keys=True)
    content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    version_info = {
        "version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "content_hash": content_hash,
        "created_at": datetime.now().isoformat(),
        "description": description,
        "total_items": len(dataset),
        "schema_version": "1.0"
    }
    
    versioned_dataset = {
        "metadata": version_info,
        "data": dataset
    }
    
    return versioned_dataset
```

### 4. Quality Assurance Checks

```python
def quality_assurance_pipeline(dataset_file: str) -> bool:
    """Run comprehensive QA checks on dataset."""
    
    checks = [
        ("JSON validity", validate_json_format),
        ("Required fields", validate_required_fields),
        ("Data consistency", validate_data_consistency),
        ("Content quality", validate_content_quality),
        ("Balance check", validate_data_balance)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed = check_func(dataset_file)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check_name}: {status}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"{check_name}: ❌ ERROR - {str(e)}")
            all_passed = False
    
    return all_passed
```

## Next Steps

Now that you understand data preparation, proceed to:
- [Tutorial 5: Template Configuration](05_template_configuration.md) to learn about conversation templates
- [Tutorial 6: Training Setup](06_training_setup.md) to understand the training process

For a complete example, see [Tutorial 7: Complete Pipeline](07_complete_pipeline.md).