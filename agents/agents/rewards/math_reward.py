
import re
from typing import List
import sys
import mpmath
sys.modules['sympy.mpmath'] = mpmath
from typing import Dict
from .reward_base import reward
from sympy.parsing.latex import parse_latex
from itertools import islice, zip_longest
try:
    from math_verify import parse, verify
except ImportError:

    print("math_verify is not installed in this environment")
    parse = None
    verify = None
from tenacity import retry, stop_after_attempt, stop_after_delay, retry_if_exception_type
    

def extract_final_answers_batch(responses: List[str]) -> List[str]:
    # pattern = re.compile(r"(\\boxed{.*})")
    pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
    results = []
    for response in responses:
        matches = re.findall(pattern, response)
        results.append(matches[-1] if matches else "")
    return results

def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def _is_latex_equal(str1, str2):
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:  # noqa
            return norm1 == norm2
    return False


@retry(
    stop=(stop_after_delay(1)),
    retry_error_callback=lambda _: False  # Return False on failure
)
def is_latex_equal(str1, str2, math_mode="legacy"):
    if math_mode == "legacy":
        if (len(str1) > 128 and repeatness(str1)) or (len(str2) > 128 and repeatness(str2)):
            return False
        return _is_latex_equal(str1, str2)
    elif math_mode == "math_verify":
        return verify(parse(str1), parse(str2))
    else:
        raise NotImplementedError(f"Math mode {math_mode} is not implemented")
    

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:  # noqa
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:  # noqa
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace(",", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False) -> bool:
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        try:
            return float(ss1) == (float(ss2))
        except Exception:  # noqa
            return ss1 == ss2
    except Exception:  # noqa
        return str1 == str2
    

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None
    

def get_answer_str(s: str) -> str:
    res = remove_boxed(last_boxed_only_string(s))
    if res is not None:
        return res
    return s

def solution2answer(solution: str, math_mode="eval_peeking") -> str:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")
    return answer

def is_equal(str1, str2, math_mode="legacy"):
    first_equal = is_equiv(str1, str2)
    if first_equal:
        return True
    return is_latex_equal(str1, str2, math_mode)

def symbolic_math_equal(gold_answer, final_answer):
    return is_equal(solution2answer(gold_answer), solution2answer(final_answer))

def evaluate_math(gold_answer, final_answer):
    reward = symbolic_math_equal(gold_answer, final_answer)
    return {"reward": reward}


def extract_last_number(s: str) -> str:
    """
    Extracts the last number found in the input string.

    Parameters:
        s (str): The string to search.

    Returns:
        str: The last number found in the string as a string, or an empty string if no number is found.
    """
    # This regular expression finds a group of digits that isn't followed by any other digit in the string.
    match = re.search(r'\d+(?!.*\d)', s)
    if match:
        return match.group()
    return ""

def math_equal_naive(gold_answer, final_answer):
    # Extract the last number in the string
    final_number = extract_last_number(final_answer)
    return gold_answer == final_number


@reward(name="math_reward")
def math_reward(prediction: str, golden_answer: str, **kwargs) -> float:
    if symbolic_math_equal(prediction, golden_answer):
        return 1.0
    else:
        return 0.0

@reward(name="math_reward_tool")
def math_reward_tool(prediction: str, answer: str, trajectory: List[Dict]) -> float:
    has_called_tool = False
    for msg in trajectory:
        if msg["role"] == "tool":
            has_called_tool = True
            break

    
    reward = 0.0
    answer_correct = symbolic_math_equal(prediction, answer)
    if not has_called_tool:
        reward = 0.0
    elif has_called_tool and not answer_correct:
        reward = 0.1
    elif has_called_tool and answer_correct:
        reward = 1.0
    else:
        raise ValueError(f"Invalid prediction or trajectory for math reward with format: Trajectory: {trajectory}")
    return {
        "reward": reward,
        "acc": 1.0 if answer_correct else 0.0,
    }

@reward(name="math_reward_thought")
def math_reward_thought(prediction: str, answer: str, trajectory: List[Dict]) -> float:
    has_called_tool = False
    for msg in trajectory:
        if msg["role"] == "tool":
            has_called_tool = True
            break

    all_have_thought = True
    for msg in trajectory:
        if msg["role"] == "assistant":
            if isinstance(msg["content"], str):
                content = msg["content"]
            elif isinstance(msg["content"], list):
                content = msg["content"][-1]["text"]
            else:
                raise ValueError(f"Invalid content type: {type(msg['content'])}")
            if not content.strip().lower().startswith("thought"):
                all_have_thought = False
                break
    
    reward = 0.0
    answer_correct = symbolic_math_equal(prediction, answer)
    if not has_called_tool:
        reward = 0.0
    elif has_called_tool and not all_have_thought and not answer_correct:
        reward = 0.0
    elif has_called_tool and all_have_thought and not answer_correct:
        reward = 0.1
    elif has_called_tool and not all_have_thought and answer_correct:
        reward = 0.1
    elif has_called_tool and all_have_thought and answer_correct:
        reward = 1.0
    else:
        raise ValueError(f"Invalid prediction or trajectory for math reward with format: Trajectory: {trajectory}")
    return {
        "reward": reward,
        "acc": 1.0 if answer_correct else 0.0,
    }


@reward(name="math_reward_thought_with_tool")
def math_reward_thought_with_tool(prediction: str, answer: str, trajectory: List[Dict]) -> float:
    has_called_tool = False
    for msg in trajectory:
        if msg["role"] == "tool":
            has_called_tool = True
            break

    all_have_thought = True
    for msg in trajectory:
        if msg["role"] == "assistant":
            if isinstance(msg["content"], str):
                content = msg["content"]
            elif isinstance(msg["content"], list):
                content = msg["content"][-1]["text"]
            else:
                raise ValueError(f"Invalid content type: {type(msg['content'])}")
            if not content.strip().lower().startswith("thought"):
                all_have_thought = False
                break
    
    reward = 0.0
    answer_correct = symbolic_math_equal(prediction, answer)
    if not has_called_tool:
        reward = 0.0
    elif has_called_tool and not all_have_thought and not answer_correct:
        reward = 0.0
    elif has_called_tool and all_have_thought and not answer_correct:
        reward = 0.1
    elif has_called_tool and not all_have_thought and answer_correct:
        reward = 0.0
    elif has_called_tool and all_have_thought and answer_correct:
        reward = 1.0
    else:
        raise ValueError(f"Invalid prediction or trajectory for math reward with format: Trajectory: {trajectory}")
    return {
        "reward": reward,
        "acc": 1.0 if answer_correct else 0.0,
    }


def parse_thinking_response(response: str):
    try:
        # First try to match complete <think>...</think> pattern
        thinking_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = re.search(thinking_pattern, response)
        if match:
            thinking = match.group(1)
        else:
            # If no complete pattern found, try to match thinking that starts directly and ends with </think>
            thinking_pattern_direct = re.compile(r"^(.*?)</think>", re.DOTALL)
            match = re.search(thinking_pattern_direct, response)
            if match:
                thinking = match.group(1)
            else:
                thinking = None
    except Exception as e:
        thinking = None
    
    try:
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        match = re.search(answer_pattern, response)
        if match:
            answer = match.group(1)
        else:
            
            answer = None
    except Exception as e:
        answer = None
    
    return thinking, answer

@reward(name="math_reward_think")
def math_reward_think(prediction: str, answer: str, trajectory: List[Dict]) -> float:
    """
    This reward function is used to reward the agent for using thinking and tool calling. The reward contains two parts:
    format: if there is <think> in the response, and the agent has called the tool, then the reward is 1.0 if the answer is correct, otherwise 0.1. if there is no <think> or the agent has not called the tool, then the reward is 0.0.
    """
    has_called_tool = False
    for msg in trajectory:
        if msg["role"] == "tool":
            has_called_tool = True
            break
    
    all_have_thinking = True
    all_have_answer = True
    for msg in trajectory:
        if msg["role"] == "assistant":
            if isinstance(msg["content"], str):
                content = msg["content"]
            elif isinstance(msg["content"], list):
                content = msg["content"][-1]["text"]
            else:
                raise ValueError(f"Invalid content type: {type(msg['content'])}")
            extracted_thinking, extracted_answer = parse_thinking_response(content)
            if extracted_thinking is None:
                all_have_thinking = False
            if extracted_answer is None:
                all_have_answer = False
    
    if not all_have_thinking or not has_called_tool:
        return {
            "reward": 0.0,
            "acc": 0.0,
        }
    
    answer_correct = symbolic_math_equal(prediction, answer)
    if answer_correct:
        return {
            "reward": 1.0,
            "acc": 1.0,
        }
    else:
        return {
            "reward": 0.1,
            "acc": 0.0,
        }

if __name__ == "__main__":
    result = symbolic_math_equal("I got answer is \\boxed{2/3}", "May be it's \\boxed{\\frac{2}{3}}")
    print(result)