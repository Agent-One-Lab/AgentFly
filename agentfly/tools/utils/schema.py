import re
import inspect
import warnings
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

def extract_signatures(func):
    sig = inspect.signature(func)
    signature = {}
    for name, param in sig.parameters.items():
        if not name in signature:
            signature[name] = {}
        if param.annotation is not inspect.Parameter.empty:
            signature[name]['type'] = param.annotation.__name__
        if param.default is not inspect.Parameter.empty:
            signature[name]['default'] = param.default

    return signature


def parse_docstring(docstring):
    """
    Parse a function's docstring to extract both the overall description and
    its parameters. This parser supports Google-style docstrings that include an
    "Args:" (or "Arguments:" / "Parameters:") section.

    It returns a dictionary with two keys:
        "description": A string of the function's main description (everything
                       before the parameters section).
        "params": A dictionary where each key is a parameter name and its value
                  is another dictionary with:
                  - "type": A string representation of the parameter's type (or None if not provided).
                  - "description": The parameter's description text.

    Examples of supported parameter lines:
        a (int): The first number.
        b: The second number.

    Args:
        docstring (str): The function's docstring.

    Returns:
        dict: A dictionary containing the overall description and the parameters info.
    """
    if not docstring:
        return {"description": "", "params": {}}

    # Split the docstring into individual lines and clean them up.
    lines = docstring.strip().splitlines()

    # Look for the start of the parameter section.
    param_section_index = None
    for i, line in enumerate(lines):
        lower_line = line.strip().lower()
        if lower_line in ("args:", "arguments:", "parameters:"):
            param_section_index = i
            break

    # Everything before the parameter section becomes the overall description.
    if param_section_index is None:
        description = "\n".join(line.strip() for line in lines)
        return {"description": description, "params": {}}
    else:
        description = "\n".join(line.strip() for line in lines[:param_section_index]).strip()

    params = {}
    current_param = None

    # The parameter lines start immediately after the section header.
    param_lines = lines[param_section_index+1:]

    # Establish the base indent level for the parameter block (from the first non-empty line).
    base_indent = None
    for line in param_lines:
        if line.strip():
            base_indent = len(line) - len(line.lstrip())
            break
    if base_indent is None:
        return {"description": description, "params": {}}

    # A regex pattern to capture a parameter line.
    # It accepts both cases: with a type (inside parentheses) and without.
    # Examples:
    #   a (int): Description of a.
    #   b: Description of b.
    pattern = re.compile(r'^\s*(\w+)(?:\s*\(([^)]+)\))?:\s*(.*)')

    # Process each line in the parameter block.
    for line in param_lines:
        # If a line is less indented than base_indent, it likely indicates a new section.
        if line and (len(line) - len(line.lstrip())) < base_indent:
            break

        if not line.strip():
            continue

        match = pattern.match(line)
        if match:
            param_name = match.group(1)
            param_type = match.group(2) if match.group(2) else None
            param_desc = match.group(3).strip()
            params[param_name] = {"type": param_type, "description": param_desc}
            current_param = param_name
        else:
            # If the line does not match a new parameter, consider it a continuation of the previous description.
            if current_param is not None:
                params[current_param]["description"] += " " + line.strip()

    return {"description": description, "params": params}


# Example function with a Google-style docstring.
def add(a, b):
    """
    Adds two numbers together.

    This is a simple function that takes two numeric inputs and returns their sum.
    It demonstrates how to document a function using Google-style docstrings.

    Args:
        a (int): The first number.
        b: The second number, which can be an int or float.

    Returns:
        The sum of a and b.
    """
    return a + b

def validate_schema(name, description, signature, docs):
    docs_description = docs['description']
    if description and docs_description and docs_description != description:
        # raise ValueError(f"Description mismatch: {description} != {docs_description}")
        # warnings.warn(f"Description mismatch: {description} != {docs_description}, use the specified description by default.")
        # TODO: currently we don't do anything here and prioritize the specified description by default.
        pass


    docs_params = docs['params']
    for param, param_info in docs_params.items():
        if param not in signature:
            raise ValueError(f"Parameter {param} in docstring not found in function signature.")
        if ("type" in param_info and "type" in signature[param]) and (param_info['type'] != signature[param]['type']):
            raise ValueError(f"Parameter {param} type mismatch: \"{param_info['type']}\" != \"{signature[param]['type']}\"")
    
    required_params = [param for param in signature if "default" not in signature[param]]
    properties = {}
    for param in signature:
        properties[param] = {}
        
        if "type" in signature[param]:
            properties[param]['type'] = signature[param]['type']
        elif param in docs_params and "type" in docs_params[param]:
            properties[param]['type'] = docs_params[param]['type']
        else:
            # May be should raise an error
            properties[param]['type'] = "unknown"
            logger.warning(f"Parameter {param} has no type in signature or docstring.")
        
        if "default" in signature[param]:
            properties[param]['default'] = signature[param]['default']

        if param in docs_params and "description" in docs_params[param]:
            properties[param]['description'] = docs_params[param]['description']

    # Postprocess for schema type compatibility
    for param_name, param_info in properties.items():
        for k, v in param_info.items():
            if k == "type":
                if v == "str":
                    param_info[k] = "string"

    # we need to remove env from the schema
    if "env" in properties:
        del properties["env"]
        required_params.remove("env")

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    **properties
                },
                "required": required_params
            }
        }
    }
    arguments = deepcopy(signature)
    if "env" in arguments:
        del arguments["env"]

    return {
        "schema": schema,
        "args": arguments
    }

if __name__ == '__main__':
    # Retrieve and parse the docstring using inspect.
    doc = inspect.getdoc(add)
    result = parse_docstring(doc)
    print("Function description:")
    print(result["description"])
    print("\nParameters:")
    for param, details in result["params"].items():
        print(f"  {param}:")
        print(f"    Type: {details['type']}")
        print(f"    Description: {details['description']}")

