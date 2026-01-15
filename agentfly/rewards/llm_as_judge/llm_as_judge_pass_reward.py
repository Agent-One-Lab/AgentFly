"""LLM as Judge Pass Reward

This reward evaluates whether a generated Python program is executable and
whether it would generate a video, then asks an LLM to judge if the program's
output video would satisfy a set of domain questions. It returns a binary
pass/fail reward: 1.0 if all questions are satisfied (with the uncertainty
policy below), otherwise 0.0.

Flow:
- Extract Python code from the agent prediction (```python ... ```)
- Build an evaluation question list with two prepended code checks:
  1) Is the code executable?
  2) Will the code generate a video?
- Append the domain VLM-style questions (renumbered to start at index 3)
- Send the code + questions to the LLM server and get structured judgments
- Compute a pass/fail reward with the same policy as VLM pass reward

Notes:
- We do not execute the code here; instead we instruct the LLM to analyze the
  code and provide True/False/Not sure with a confidence score.
- For “Not sure”, we mirror the VLM pass/fail semantics: low-confidence
  uncertainty (confidence < 4) is treated as pass; explicit False or
  high-confidence uncertainty causes failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..reward_base import reward
from .llm_as_judge_client import LLMClient

# Reuse helper and scoring policy from VLM flow to keep behavior consistent
from ..vlm_as_judge.vlm_as_judge_reward import pass_fail_reward

logger = logging.getLogger(__name__)


def _extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from the agent response.

    - Removes optional <think>...</think> blocks
    - Extracts the first fenced code block labeled as python
    """
    if not response:
        return None

    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, cleaned, re.DOTALL)
    if matches:
        return matches[0]
    return None


def _extract_vlm_questions(data: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Extract summarize and the VLM-style questions list from input data.

    Supports structures like:
        {"vlm_questions": {"summarize": str, "vlm_questions": [ {index, question, weight?}, ... ]}}
    or directly:
        {"vlm_questions": [ {index, question, weight?}, ... ]}

    Returns:
        (all_questions_str, summarize, questions_list)
        - all_questions_str: formatted lines "i. question" for the original indices
        - summarize: string summary (may be empty)
        - questions_list: list of {index, question, weight?}
    """
    all_questions = ""
    summarize = ""
    questions_list: List[Dict[str, Any]] = []

    if "vlm_questions" in data:
        vlm = data["vlm_questions"]
        if isinstance(vlm, dict):
            summarize = vlm.get("summarize", "")
            inner = vlm.get("vlm_questions", [])
            if isinstance(inner, list):
                questions_list = [q for q in inner if isinstance(q, dict)]
        elif isinstance(vlm, list):
            questions_list = [q for q in vlm if isinstance(q, dict)]

    # Fallback: some pipelines might pass via "after_verify"
    if not questions_list and isinstance(data.get("after_verify"), dict):
        av = data["after_verify"]
        if isinstance(av.get("vlm_questions"), list):
            questions_list = [q for q in av["vlm_questions"] if isinstance(q, dict)]

    # Build original-index text (not used for final prompt numbering here)
    for q in questions_list:
        idx = str(q.get("index", "")).strip()
        question = str(q.get("question", "")).strip()
        if idx and question:
            all_questions += f"{idx}. {question}\n"

    return all_questions.strip(), summarize, questions_list


def _build_combined_questions(questions_list: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Create the combined question list and display text for the LLM.

    We prepend two code-centric checks, then append the provided questions
    re-indexed to start at 3.

    Returns:
        (all_questions_text, combined_questions_list)
    """
    base_questions = [
        {"index": "1", "question": "Is the code executable?"},
        {"index": "2", "question": "Will the code generate a video file (e.g., .mp4)?"},
    ]

    combined_list: List[Dict[str, Any]] = []
    combined_list.extend(base_questions)

    # Re-index domain questions to start at 3
    display_lines = ["1. Is the code executable?", "2. Will the code generate a video file (e.g., .mp4)?"]
    for i, q in enumerate(questions_list):
        new_idx = str(i + 3)
        question = str(q.get("question", "")).strip()
        if not question:
            continue
        combined_list.append({
            "index": new_idx,
            "question": question,
            # carry through weight if present (not used in pass/fail but kept for parity)
            **({"weight": q.get("weight")} if "weight" in q else {}),
        })
        display_lines.append(f"{new_idx}. {question}")

    all_questions_text = "\n".join(display_lines)
    return all_questions_text, combined_list


DEFAULT_LLM_CODE_JUDGE_PROMPT = """You are a code analysis judge. Read the Python code and determine:
1) Whether it would execute without errors in a typical Python environment, and
2) Whether it would generate a video file (e.g., .mp4) by writing frames.
Then, based on the code's intended behavior, judge the domain questions.

Instructions:
- Do not execute the code; reason from static analysis of the code.
- Assume a standard Python 3 environment with common libraries available
  (e.g., numpy, opencv-python) unless the code clearly requires missing resources.
- For generating video: look for proper creation of a video writer, writing frames,
  and releasing/closing to finalize the file.
- For each question, return one of "True", "False", or "Not sure" and a confidence
  score from 1 (very uncertain) to 5 (very certain).

Context:
- Summary of expected visual content (if provided): {summarize}
- Questions to evaluate:\n{all_questions}

Provide your output strictly as a JSON list with entries like:
[
  {{
    "index": "1",
    "question": "Is the code executable?",
    "analysis": "...your reasoning...",
    "result": "True|False|Not sure",
    "confidence_score": "1|2|3|4|5"
  }},
  ...
]

Here is the Python code to evaluate:
```python
{code}
```
"""


@reward(name="llm_as_judge_pass_reward")
async def llm_as_judge_pass_reward(
    prediction: str,
    trajectory: Dict[str, Any] | None = None,
    vlm_questions: Dict[str, Any] | None = None,
    **data_fields: Any,
) -> Dict[str, float]:
    """LLM-as-Judge pass/fail reward for code + domain questions.

    Args:
        prediction: Model output that should contain a Python code block.
        trajectory: Optional trajectory info (unused).
        vlm_questions: Optional VLM question payload (dict or list).
        **data_fields: Additional fields that may include vlm_questions or after_verify.

    Returns:
        {"reward": 1.0} if all questions pass under the pass/fail rule, else {"reward": 0.0}.
    """
    try:
        logger.info("=" * 60)
        logger.info("llm_as_judge_pass_reward called")
        logger.info(f"Prediction length: {len(prediction) if prediction else 0}")

        # Preview prediction for debugging
        if prediction:
            logger.info("Prediction preview (first 500 chars):")
            logger.info(prediction[:500])
            if len(prediction) > 500:
                logger.info("... (truncated)")

        # Gather question data
        all_data: Dict[str, Any] = dict(data_fields)
        if vlm_questions is not None:
            all_data["vlm_questions"] = vlm_questions

        _orig_all_q, summarize, orig_questions = _extract_vlm_questions(all_data)

        # Extract code
        code = _extract_code_from_response(prediction)
        if not code:
            logger.warning("No Python code found in prediction")
            return {"reward": 0.0}

        # Build combined questions (prepend two code checks)
        all_questions_text, combined_questions = _build_combined_questions(orig_questions)
        if not combined_questions:
            logger.warning("No questions provided; cannot compute pass/fail")
            return {"reward": 0.0}

        # Build prompt for LLM
        prompt = DEFAULT_LLM_CODE_JUDGE_PROMPT.format(
            summarize=summarize or "",
            all_questions=all_questions_text,
            code=code,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        # Send to LLM server
        client = LLMClient(
            model="Qwen/Qwen2.5-72B-Instruct",
            timeout_seconds=120,
        )

        # Wait briefly for availability
        for _ in range(10):
            if client.is_available():
                break
            await asyncio.sleep(1)
        else:
            logger.error("LLM client not available")
            return {"reward": 0.0}

        responses = await client.process_all_inputs([messages], model="Qwen/Qwen2.5-72B-Instruct", temperature=0.1)
        if not responses or not responses[0] or not responses[0][0]:
            logger.error("No response from LLM server")
            return {"reward": 0.0}

        output_text = responses[0][0]

        # Parse returned JSON list
        # We inline a small tolerant parser to avoid cross-module dependency here.
        results: List[Dict[str, Any]] = []
        try:
            import json
            try:
                parsed = json.loads(output_text)
                if isinstance(parsed, list):
                    results = parsed
                else:
                    raise ValueError("Top-level JSON is not a list")
            except Exception:
                # Fallback: extract substring between first '[' and last ']'
                start = output_text.find('[')
                end = output_text.rfind(']')
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(output_text[start:end+1])
                    if isinstance(parsed, list):
                        results = parsed
                if not results:
                    raise ValueError("Failed to extract JSON list from LLM output")
        except Exception as e:
            logger.error(f"Error parsing LLM JSON output: {e}")
            return {"reward": 0.0}

        # Compute pass/fail using the same policy as VLM pass reward
        reward_score = pass_fail_reward(results, combined_questions)
        return {"reward": float(reward_score)}

    except Exception as e:
        logger.error(f"Error in llm_as_judge_pass_reward: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": 0.0}

