"""VLM as Judge Reward Function for AgentFly RL Training"""

import os
import sys
import re
import json
import uuid
import asyncio
import logging
import time
import concurrent.futures
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .video_generation import VideoGenerator, can_open_video, extract_code_from_response
from .prompt_utils import create_vlm_prompt, create_vlm_prompt_custom, create_vlm_prompt_from_template

from ...core import Context
from ..reward_base import reward
from ...resources import APIModelResourceSpec
from ..llm_as_judge.llm_as_judge_client import LLMClient


def _extract_json_list(output_str: str) -> List[Dict[str, Any]]:
    """Extract the VLM JSON list from the model output.

    Tries strict JSON first; then looks for fenced JSON blocks and
    substring-extracts list/object payloads.
    """
    def _has_unquoted_colon(value: str) -> bool:
        in_string = False
        escaped = False
        for ch in value:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if ch == ":" and not in_string:
                return True
        return False

    def _fix_bare_string_fields(value: str) -> str:
        lines = value.splitlines()
        fixed_lines = []
        changed = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"') and (stripped.endswith('"') or stripped.endswith('",')):
                if not _has_unquoted_colon(stripped):
                    indent = line[:len(line) - len(line.lstrip())]
                    has_comma = stripped.endswith(",")
                    raw_value = stripped[:-1].strip() if has_comma else stripped
                    fixed_lines.append(
                        f'{indent}"question": {raw_value}{"," if has_comma else ""}'
                    )
                    changed = True
                    continue
            fixed_lines.append(line)
        return "\n".join(fixed_lines) if changed else value

    def _fix_missing_object_braces(value: str) -> str:
        lines = value.splitlines()
        fixed_lines = []
        in_object = False
        changed = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("{"):
                in_object = True
                fixed_lines.append(line)
                continue
            if stripped.startswith("}"):
                in_object = False
                fixed_lines.append(line)
                continue
            if stripped.startswith('"index"') and not in_object:
                indent = line[:len(line) - len(line.lstrip())]
                fixed_lines.append(f"{indent}{{")
                fixed_lines.append(line)
                in_object = True
                changed = True
                continue
            fixed_lines.append(line)
        return "\n".join(fixed_lines) if changed else value

    def _coerce_list(value: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return None

    def _try_parse(candidate: str) -> Optional[List[Dict[str, Any]]]:
        try:
            parsed = json.loads(candidate)
            coerced = _coerce_list(parsed)
            if coerced is not None:
                return coerced
        except Exception:
            pass

        for start_char, end_char in (('[', ']'), ('{', '}')):
            start = candidate.find(start_char)
            end = candidate.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(candidate[start:end + 1])
                    coerced = _coerce_list(parsed)
                    if coerced is not None:
                        return coerced
                except Exception:
                    continue
        return None

    candidates: List[str] = [output_str.strip()]

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", output_str, flags=re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    unterminated = re.search(r"```(?:json)?\s*(.*)$", output_str, flags=re.DOTALL | re.IGNORECASE)
    if unterminated:
        tail = unterminated.group(1).strip()
        if tail:
            candidates.append(tail)

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
        fixed_candidate = _fix_bare_string_fields(candidate)
        if fixed_candidate != candidate:
            parsed = _try_parse(fixed_candidate)
            if parsed is not None:
                return parsed
        brace_fixed = _fix_missing_object_braces(fixed_candidate)
        if brace_fixed != fixed_candidate:
            parsed = _try_parse(brace_fixed)
            if parsed is not None:
                return parsed
    raise ValueError("Failed to parse VLM JSON list from output")


logger = logging.getLogger(__name__)

# Multi-model VLM endpoint config: model at index i uses server at index i.
# Configure via the VLM_MODELS / VLM_SERVER_IPS environment variables, each a
# comma-separated list. The two must have the same length; see
# examples/train_scripts/simuscene/test_vlm_as_judge_train.sh for an example.
def _parse_csv_env(name: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]

VLM_MODELS: List[str] = _parse_csv_env("VLM_MODELS")
VLM_SERVER_IPS: List[str] = _parse_csv_env("VLM_SERVER_IPS")
VLM_ENSEMBLE_PROMPT_TEMPLATE = """You are given a video and several visual verification questions. Your task is to judge each question as true or false based only on what can be seen or reasonably inferred from the video. If the visual evidence is insufficient to confirm the statement, or if the statement directly contradicts the video, answer 'false'.
 
⸻
 
Visual Reasoning Rules
    1.  Perspective Changes:Objects can look different from different angles. such as a circular path may look curved, straight, or wavy depending on the viewpoint, or a cylinder may look like a circle (top view) or a rectangle (side view)
    2.  Simplified / Symbolic Drawings: The video may use simple shapes or colors to represent real objects. If the motion or layout still matches the description, treat it as True.
    3.  Container Boundaries: If no container is drawn, assume the video frame is the boundary. If a container is shown, treat it as see-through if you can see inside it. If something is not visible, don’t assume it is inside the container.
    4.  Focus on Shape and Motion, ignore assumptions about material, weight, texture, or color. If the described motion does not match (e.g., “slides down” vs. actually moving up), answer False.
    5.  Occlusion: If something is partly hidden, use nearby cues to infer how it is moving or positioned.
 
Input
Questions:{all_questions}
 
Output Format:
Return a JSON list of objects. **Do not include any conversational filler or Markdown formatting tags.** Fields for each object:
- `"index"`: The question index.
- `"question"`: The full question text.
- `"analysis"`: A concise step-by-step breakdown of what was observed vs. what the question claims.
- `"result"`: "true" if visually confirmed, "false" if contradicted or evidence is insufficient.
 
Example Format:
[
  {{
    "index": "1",
    "question": "The ball rolls along the circular path.",
    "analysis": "The red sphere moves in a consistent arc that completes a 360-degree loop.",
    "result": "true"
  }}
]
"""
VLM_ENSEMBLE_MIN_RESPONSES = 2
LADDER_CODE_EXTRACTION_REWARD = 0.04
LADDER_CODE_RENDER_REWARD = 0.06
LADDER_VIDEO_OPEN_REWARD = 0.1
LADDER_VLM_REWARD = 1.0 - LADDER_CODE_EXTRACTION_REWARD - LADDER_CODE_RENDER_REWARD - LADDER_VIDEO_OPEN_REWARD
MULTI_MODEL_LADDER_CODE_REWARD = 0.02
MULTI_MODEL_LADDER_VIDEO_SAVED_REWARD = 0.04
MULTI_MODEL_LADDER_VIDEO_OPEN_REWARD = 0.06

LLM_AS_JUDGE_ABLATION_PROMPT = """You are a code analysis judge. Read the Python code and determine:
1) Does the code execute without errors?
2) Does the code generate and save a video?
Then, based on the code's intended behavior, judge the domain questions.

Instructions:
- Do not execute the code; reason from static analysis of the code.
- Assume a standard Python 3 environment with common libraries available
  (e.g., numpy, opencv-python) unless the code clearly requires missing resources.
- For generating video: look for proper creation of a video writer, writing frames,
  and releasing/closing to finalize the file.
- For each question, return one of "True" or "False".

Context:
- Summary of expected visual content (if provided): {summarize}
- Questions to evaluate:\n{all_questions}

Provide your output strictly as a JSON list with entries like:
[
  {{
    "index": "1",
    "question": "Does the code execute without errors?",
    "analysis": "...your reasoning...",
    "result": "True|False"
  }}
]

Here is the Python code to evaluate:
```python
{code}
```
"""


def _format_video_uri(path: str) -> str:
    if not path:
        return ""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", path):
        return path
    return f"file://{os.path.abspath(path)}"


def _build_vlm_messages(video_path: str, prompt_text: str) -> List[Dict[str, Any]]:
    video_uri = _format_video_uri(video_path)

    return [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_uri}},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def _build_api_model_spec(
    model: str,
    server_ip: Optional[str] = None,
    port: int = 8000,
    api_key: str = "token-abc123",
) -> APIModelResourceSpec:
    return APIModelResourceSpec(
        model_name_or_path=model,
        host=server_ip,
        port=port,
        api_key=api_key,
    )


def _result_is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"

def extract_vlm_questions_from_data(data: Dict[str, Any]) -> Tuple[str, str, List[Dict]]:
    """Extract VLM questions and summary from data
    
    Args:
        data: Dictionary containing vlm_questions data
        
    Returns:
        Tuple of (all_questions_str, summarize, questions_list)
    """
    vlm_data = data.get("vlm_questions")
    if vlm_data is None:
        logger.warning("No vlm_questions field in data. Available fields: %s", list(data.keys()))
        return "", "Evaluate the visual content based on the questions provided.", []
    if not isinstance(vlm_data, dict):
        logger.warning("vlm_questions is not a dict: %s", type(vlm_data))
        return "", "Evaluate the visual content based on the questions provided.", []

    summarize = vlm_data.get("summarize", "") or "Evaluate the visual content based on the questions provided."
    questions_raw = vlm_data.get("vlm_questions", [])
    if not isinstance(questions_raw, list):
        logger.warning("vlm_questions inner field is not a list: %s", type(questions_raw))
        return "", summarize, []

    questions_list = [q for q in questions_raw if isinstance(q, dict)]
    all_questions = "\n".join(
        f"{q.get('index', '')}. {q.get('question', '')}" for q in questions_list
    ).strip()
    logger.info("Extracted %d questions from VLM data", len(questions_list))
    return all_questions, summarize, questions_list


def calculate_weighted_reward(results: List[Dict], questions_list: List[Dict]) -> float:
    """Calculate weighted reward based on VLM results and question weights
    
    Args:
        results: List of VLM evaluation results
        questions_list: Original questions with weights
        
    Returns:
        Weighted reward score between 0.0 and 1.0
    """
    if not results or not questions_list:
        return 0.0
    
    # Create weight mapping
    weight_map = {}
    for q in questions_list:
        idx = str(q.get("index", ""))
        weight = float(q.get("weight", 1.0))
        weight_map[idx] = weight
    
    scores = []
    weights = []
    
    for result in results:
        idx = str(result.get("index", ""))
        result_value = str(result.get("result", "")).strip().lower()
        
        # Get weight for this question
        weight = weight_map.get(idx, 1.0)
        
        # Calculate score based on result
        if result_value == "true":
            score = 1.0
        else:
            score = 0.0
        
        scores.append(score)
        weights.append(weight)
    
    # Calculate weighted average
    # if weights:
    #     weighted_sum = sum(s * w for s, w in zip(scores, weights))
    #     total_weight = sum(weights)
    #     reward = weighted_sum / total_weight if total_weight > 0 else 0.0
    # else:
    reward = sum(scores) / len(scores) if scores else 0.0
    
    return reward

def pass_fail_reward(results: List[Dict], questions_list: List[Dict]) -> float:
    """Calculate a binary pass/fail score from VLM results.

    Returns 1.0 only when every question is judged as "true", otherwise 0.0.
    """
    if not results or not questions_list:
        return 0.0

    result_map = {
        str(r.get("index", "")).strip(): r
        for r in results
        if str(r.get("index", "")).strip()
    }

    for question in questions_list:
        idx = str(question.get("index", "")).strip()
        if not idx:
            logger.warning("Question without index encountered in pass/fail reward")
            return 0.0

        result = result_map.get(idx)
        if result is None:
            logger.warning("Missing VLM result for question index %s", idx)
            return 0.0

        result_value = str(result.get("result", "")).strip().lower()
        if result_value != "true":
            return 0.0

    return 1.0


def _all_true_pass(results: List[Dict], questions_list: List[Dict]) -> bool:
    """Strict pass/fail: only True counts; any non-True or missing result fails."""
    if not results or not questions_list:
        return False

    result_map = {
        str(r.get("index", "")).strip(): r
        for r in results
        if str(r.get("index", "")).strip()
    }

    for question in questions_list:
        idx = str(question.get("index", "")).strip()
        if not idx:
            logger.warning("Question without index encountered in strict pass/fail")
            return False

        result = result_map.get(idx)
        if result is None:
            logger.warning("Missing VLM result for question index %s in strict pass/fail", idx)
            return False

        result_value = str(result.get("result", "")).strip().lower()
        if result_value != "true":
            return False

    return True


def _true_rate(results: List[Dict], questions_list: List[Dict]) -> float:
    """Compute true-rate over questions; missing/False counts as 0."""
    if not questions_list:
        return 0.0

    result_map = {
        str(r.get("index", "")).strip(): r
        for r in results
        if str(r.get("index", "")).strip()
    }

    true_count = 0
    for question in questions_list:
        idx = str(question.get("index", "")).strip()
        if not idx:
            logger.warning("Question without index encountered in true-rate reward")
            return 0.0

        result = result_map.get(idx)
        result_value = str(result.get("result", "")).strip().lower() if result else ""
        if result_value == "true":
            true_count += 1

    return true_count / len(questions_list)


def _build_question_result_map(results: List[Dict[str, Any]]) -> Dict[str, Optional[bool]]:
    result_map: Dict[str, Optional[bool]] = {}
    if not results:
        return result_map
    for result in results:
        idx = str(result.get("index", "")).strip()
        if not idx:
            continue
        result_map[idx] = _result_is_true(result.get("result"))
    return result_map


def _calculate_vlm_agreement_stats(
    result_maps: List[Dict[str, Optional[bool]]],
    questions_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_questions = 0
    complete_questions = 0
    agreement_count = 0
    disagreement_count = 0
    agreement_true_count = 0
    agreement_false_count = 0
    disagreement_2_true_1_false = 0
    disagreement_2_false_1_true = 0
    missing_count = 0

    for question in questions_list:
        idx = str(question.get("index", "")).strip()
        if not idx:
            continue
        total_questions += 1

        votes: List[bool] = []
        missing = False
        for result_map in result_maps:
            if idx not in result_map:
                missing = True
                break
            vote = result_map.get(idx)
            if vote is None:
                missing = True
                break
            votes.append(vote)

        if missing or len(votes) != len(result_maps):
            missing_count += 1
            continue

        complete_questions += 1
        true_votes = sum(1 for vote in votes if vote)
        false_votes = len(votes) - true_votes

        if true_votes == len(votes) or false_votes == len(votes):
            agreement_count += 1
            if true_votes == len(votes):
                agreement_true_count += 1
            else:
                agreement_false_count += 1
        else:
            disagreement_count += 1
            if len(votes) == 3:
                if true_votes == 2:
                    disagreement_2_true_1_false += 1
                elif true_votes == 1:
                    disagreement_2_false_1_true += 1

    coverage_rate = complete_questions / total_questions if total_questions else 0.0
    agreement_rate = agreement_count / complete_questions if complete_questions else 0.0
    disagreement_rate = disagreement_count / complete_questions if complete_questions else 0.0

    return {
        "questions_total": total_questions,
        "questions_complete": complete_questions,
        "missing_count": missing_count,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_rate": agreement_rate,
        "disagreement_rate": disagreement_rate,
        "agreement_true_count": agreement_true_count,
        "agreement_false_count": agreement_false_count,
        "disagreement_2_true_1_false": disagreement_2_true_1_false,
        "disagreement_2_false_1_true": disagreement_2_false_1_true,
    }


def _format_model_label(model_result: Dict[str, Any]) -> str:
    model_name = model_result.get("model") or "unknown_model"
    server_ip = model_result.get("server_ip")
    if server_ip:
        return f"{model_name}@{server_ip}"
    return model_name



def _calculate_vlm_per_model_stats(
    model_results: List[Dict[str, Any]],
    questions_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    per_model: Dict[str, Dict[str, Any]] = {}
    model_entries: List[Dict[str, Any]] = []

    for result in model_results:
        label = _format_model_label(result)
        entry = {
            "label": label,
            "model": result.get("model"),
            "server_ip": result.get("server_ip"),
            "result_map": result.get("result_map", {}),
        }
        model_entries.append(entry)
        per_model[label] = {
            "questions_total": 0,
            "answered_count": 0,
            "missing_count": 0,
            "agree_with_majority_count": 0,
            "disagree_with_majority_count": 0,
            "no_majority_count": 0,
            "agree_with_majority_rate": 0.0,
            "disagree_with_majority_rate": 0.0,
        }

    for question in questions_list:
        idx = str(question.get("index", "")).strip()
        if not idx:
            continue

        votes: Dict[str, bool] = {}
        for entry in model_entries:
            result_map = entry["result_map"]
            if idx not in result_map:
                continue
            vote = result_map.get(idx)
            if vote is None:
                continue
            votes[entry["label"]] = vote

        true_votes = sum(1 for vote in votes.values() if vote)
        false_votes = len(votes) - true_votes
        if len(votes) >= 2 and true_votes != false_votes:
            majority = true_votes > false_votes
        else:
            majority = None

        for entry in model_entries:
            label = entry["label"]
            stats = per_model[label]
            stats["questions_total"] += 1
            result_map = entry["result_map"]
            if idx not in result_map or result_map.get(idx) is None:
                stats["missing_count"] += 1
                continue
            stats["answered_count"] += 1
            if majority is None:
                stats["no_majority_count"] += 1
                continue
            if result_map[idx] == majority:
                stats["agree_with_majority_count"] += 1
            else:
                stats["disagree_with_majority_count"] += 1

    for label, stats in per_model.items():
        compared = stats["agree_with_majority_count"] + stats["disagree_with_majority_count"]
        if compared:
            stats["agree_with_majority_rate"] = stats["agree_with_majority_count"] / compared
            stats["disagree_with_majority_rate"] = stats["disagree_with_majority_count"] / compared

    return per_model


def _build_vlm_data(vlm_questions: Optional[Dict[str, Any]], data_fields: Dict[str, Any]) -> Dict[str, Any]:
    all_data = dict(data_fields)
    if vlm_questions is not None:
        all_data["vlm_questions"] = vlm_questions
    return all_data


def _video_generation_metrics(
    code_extracted: bool = False,
    video_saved: bool = False,
    video_opened: bool = False,
) -> Dict[str, float]:
    return {
        "code_extracted": 1.0 if code_extracted else 0.0,
        "video_saved": 1.0 if video_saved else 0.0,
        "video_opened": 1.0 if video_opened else 0.0,
    }


async def _prepare_video_and_questions(
    final_response: str,
    *,
    vlm_questions: Optional[Dict[str, Any]],
    data_fields: Dict[str, Any],
) -> Dict[str, Any]:
    video_gen = VideoGenerator()
    all_data = _build_vlm_data(vlm_questions, data_fields)
    all_questions, summarize, questions_list = extract_vlm_questions_from_data(all_data)
    if not questions_list:
        logger.warning("No VLM questions found in data.")
        return {"ok": False, "prepared": None, "video_metrics": _video_generation_metrics()}

    code = video_gen.extract_code_from_response(final_response)
    if not code:
        logger.warning("No Python code found in final_response")
        return {
            "ok": False,
            "prepared": None,
            "video_metrics": _video_generation_metrics(code_extracted=False),
        }
    video_metrics = _video_generation_metrics(code_extracted=True)

    video_filename = f"video_{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(video_gen.output_dir, video_filename)
    success = await video_gen.generate_video_from_code(code, video_path)
    if not success:
        logger.error("Failed to generate video from code")
        return {
            "ok": False,
            "prepared": None,
            "video_metrics": _video_generation_metrics(code_extracted=True, video_saved=False),
        }
    video_metrics["video_saved"] = 1.0
    if can_open_video(video_path):
        video_metrics["video_opened"] = 1.0

    prepared = {
        "video_path": video_path,
        "all_questions": all_questions,
        "summarize": summarize,
        "questions_list": questions_list,
    }
    return {"ok": True, "prepared": prepared, "video_metrics": video_metrics}


def _vlm_specs_from_globals() -> List[APIModelResourceSpec]:
    """Build one APIModelResourceSpec per (model, host) global pair."""
    models_raw = VLM_MODELS
    ips_raw = VLM_SERVER_IPS

    if not isinstance(models_raw, list) or not isinstance(ips_raw, list):
        logger.error("VLM_MODELS and VLM_SERVER_IPS must both be lists of strings.")
        return []

    if not models_raw or not ips_raw:
        logger.error("VLM_MODELS and VLM_SERVER_IPS must be non-empty.")
        return []

    if len(models_raw) != len(ips_raw):
        logger.error(
            "VLM_MODELS length (%d) must match VLM_SERVER_IPS length (%d).",
            len(models_raw),
            len(ips_raw),
        )
        return []

    specs: List[APIModelResourceSpec] = []
    for model_name, server_ip in zip(models_raw, ips_raw):
        if not isinstance(model_name, str) or not model_name.strip():
            logger.error("Each entry in VLM_MODELS must be a non-empty string.")
            return []
        if not isinstance(server_ip, str) or not server_ip.strip():
            logger.error("Each entry in VLM_SERVER_IPS must be a non-empty string.")
            return []
        specs.append(
            _build_api_model_spec(model=model_name.strip(), server_ip=server_ip.strip())
        )
    return specs


def _extract_response_text(responses: List[List[str]]) -> Optional[str]:
    if not responses:
        return None
    first_batch = responses[0]
    if not first_batch:
        return None
    text = first_batch[0]
    if not text:
        return None
    return text


def _parse_vlm_results(output_text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    try:
        results = _extract_json_list(output_text)
    except Exception:
        return None, "parse_error"
    if not results:
        return None, "no_results"
    return results, None


async def _run_one_vlm_model(
    context: Context,
    *,
    messages: List[Dict[str, Any]],
    spec: APIModelResourceSpec,
    questions_list: List[Dict[str, Any]],
    timeout_seconds: int = 600,
) -> Dict[str, Any]:
    model_name = str(spec.model_name_or_path)
    server_ip = spec.host
    try:
        host = server_ip or "default"
        rid = f"vlm_api::{model_name}::{host}:{spec.port or 8000}"
        resource = await context.acquire_resource(
            id=rid,
            spec=spec,
            scope="rollout",
            backend=context.resource_backend,
            timeout=timeout_seconds,
        )
        outputs = await resource.generate_async([messages], temperature=0)
        responses = [outputs]
        output_text = _extract_response_text(responses)
        if output_text is None:
            return {
                "model": model_name,
                "server_ip": server_ip,
                "error": "empty_response",
                "response_success": False,
                "format_success": False,
            }
        response_success = True
        results, parse_error = _parse_vlm_results(output_text)
        if parse_error is not None:
            return {
                "model": model_name,
                "server_ip": server_ip,
                "error": parse_error,
                "response_success": response_success,
                "format_success": False,
            }

        passed = _all_true_pass(results, questions_list)
        return {
            "model": model_name,
            "server_ip": server_ip,
            "error": None,
            "response_success": response_success,
            "format_success": True,
            "results": results,
            "passed": passed,
            "true_rate": _true_rate(results, questions_list),
            "result_map": _build_question_result_map(results),
        }
    except Exception as exc:
        return {
            "model": model_name,
            "server_ip": server_ip,
            "error": str(exc),
            "response_success": False,
            "format_success": False,
        }


async def _run_vlm_backbone(
    context: Context,
    *,
    final_response: str,
    vlm_questions: Optional[Dict[str, Any]],
    data_fields: Dict[str, Any],
    model_specs: List[APIModelResourceSpec],
    prompt_text: str,
) -> Dict[str, Any]:
    prep = await _prepare_video_and_questions(
        final_response,
        vlm_questions=vlm_questions,
        data_fields=data_fields,
    )
    if not prep["ok"]:
        return {
            "ok": False,
            "reward": 0.0,
            "prepared": None,
            "model_results": [],
            "available_results": [],
            "video_metrics": prep["video_metrics"],
        }

    prepared = prep["prepared"]
    video_path = prepared["video_path"]
    messages = _build_vlm_messages(video_path, prompt_text)
    model_results = await asyncio.gather(
        *[
            _run_one_vlm_model(
                context,
                messages=messages,
                spec=spec,
                questions_list=prepared["questions_list"],
            )
            for spec in model_specs
        ]
    )
    available_results = [r for r in model_results if r.get("error") is None]
    return {
        "ok": True,
        "prepared": prepared,
        "video_path": video_path,
        "model_results": model_results,
        "available_results": available_results,
        "video_metrics": prep["video_metrics"],
    }


def _cleanup_video(video_path: Optional[str]) -> None:
    if video_path:
        try:
            os.remove(video_path)
        except Exception:
            pass


def _zero_multi_model_metrics(model_specs: List[APIModelResourceSpec]) -> Dict[str, float]:
    metrics: Dict[str, float] = {"reward": 0.0, **_video_generation_metrics()}
    model_count = len(model_specs)
    for spec in model_specs:
        model_name = str(spec.model_name_or_path)
        metrics[f"model_pass@{model_name}"] = 0.0
        metrics[f"model_response_success@{model_name}"] = 0.0
        metrics[f"model_format_success@{model_name}"] = 0.0
    for i in range(1, max(0, model_count) + 1):
        metrics[f"pass#{i}"] = 0.0
    return metrics


def _build_multi_model_metrics(
    model_results: List[Dict[str, Any]],
    model_specs: List[APIModelResourceSpec],
    video_metrics: Dict[str, float],
) -> Dict[str, float]:
    metrics = _zero_multi_model_metrics(model_specs)
    metrics.update(video_metrics)
    model_count = len(model_specs)
    passed_flags = [bool(r.get("passed")) for r in model_results[:model_count]]
    pass_count = sum(1 for flag in passed_flags if flag)

    for idx, spec in enumerate(model_specs):
        model_name = str(spec.model_name_or_path)
        result = model_results[idx] if idx < len(model_results) else {}
        metrics[f"model_pass@{model_name}"] = 1.0 if bool(result.get("passed")) else 0.0
        metrics[f"model_response_success@{model_name}"] = (
            1.0 if bool(result.get("response_success")) else 0.0
        )
        metrics[f"model_format_success@{model_name}"] = (
            1.0 if bool(result.get("format_success")) else 0.0
        )
    for threshold in range(1, model_count + 1):
        metrics[f"pass#{threshold}"] = 1.0 if pass_count >= threshold else 0.0

    ladder_reward = 0.0
    if video_metrics.get("code_extracted", 0.0) >= 1.0:
        ladder_reward = MULTI_MODEL_LADDER_CODE_REWARD
    if video_metrics.get("video_saved", 0.0) >= 1.0:
        ladder_reward = MULTI_MODEL_LADDER_VIDEO_SAVED_REWARD
    if video_metrics.get("video_opened", 0.0) >= 1.0:
        ladder_reward = MULTI_MODEL_LADDER_VIDEO_OPEN_REWARD

    majority_threshold = (model_count // 2) + 1
    if pass_count >= majority_threshold:
        metrics["reward"] = 1.0
    else:
        metrics["reward"] = ladder_reward
    return metrics


@reward(name="vlm_as_judge_pass_reward")
async def vlm_as_judge_pass_reward(
    final_response: str, 
    vlm_questions: Dict[str, Any] = None,
    context: Optional[Context] = None,
    **data_fields
) -> Dict[str, float]:
    video_path = None
    try:
        if context is None:
            logger.error("Context is required for vlm_as_judge_pass_reward.")
            return {"reward": 0.0, **_video_generation_metrics()}
        prep = await _prepare_video_and_questions(
            final_response,
            vlm_questions=vlm_questions,
            data_fields=data_fields,
        )
        if not prep["ok"]:
            return {"reward": 0.0, **prep["video_metrics"]}
        prepared = prep["prepared"]

        video_path = prepared["video_path"]
        prompt_text = create_vlm_prompt(prepared["summarize"], prepared["all_questions"])
        messages = _build_vlm_messages(video_path, prompt_text)
        specs = _vlm_specs_from_globals()
        if not specs:
            return {"reward": 0.0, **prep["video_metrics"]}
        if len(specs) == 1:
            model_result = await _run_one_vlm_model(
                context,
                messages=messages,
                spec=specs[0],
                questions_list=prepared["questions_list"],
            )
            if model_result.get("error") is not None:
                logger.error("VLM evaluation failed: %s", model_result.get("error"))
                return {"reward": 0.0, **prep["video_metrics"]}
            return {
                "reward": pass_fail_reward(model_result["results"], prepared["questions_list"]),
                **prep["video_metrics"],
            }
        model_results = await asyncio.gather(
            *[
                _run_one_vlm_model(
                    context,
                    messages=messages,
                    spec=spec,
                    questions_list=prepared["questions_list"],
                )
                for spec in specs
            ]
        )
        if any(r.get("error") is not None for r in model_results):
            for r in model_results:
                if r.get("error") is not None:
                    logger.error("VLM evaluation failed: %s", r.get("error"))
            return {"reward": 0.0, **prep["video_metrics"]}
        if not all(r.get("passed") for r in model_results):
            return {"reward": 0.0, **prep["video_metrics"]}
        return {"reward": 1.0, **prep["video_metrics"]}
    except Exception as e:
        logger.error("Error in vlm_as_judge_pass_reward: %s", e)
        return {"reward": 0.0, **_video_generation_metrics()}
    finally:
        _cleanup_video(video_path)



@reward(name="vlm_as_judge_pass_reward_multi_model")
async def vlm_as_judge_pass_reward_multi_model(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    context: Optional[Context] = None,
    **data_fields,
) -> Dict[str, float]:
    video_path = None
    model_specs: List[APIModelResourceSpec] = []
    try:
        if context is None:
            logger.error("Context is required for vlm_as_judge_pass_reward_multi_model.")
            return {"reward": 0.0, **_video_generation_metrics()}
        model_specs = _vlm_specs_from_globals()
        if not model_specs:
            return {"reward": 0.0, **_video_generation_metrics()}
        all_data = _build_vlm_data(vlm_questions, data_fields)
        all_questions, _, _ = extract_vlm_questions_from_data(all_data)
        prompt_text = create_vlm_prompt_from_template(
            VLM_ENSEMBLE_PROMPT_TEMPLATE,
            variables={"all_questions": all_questions},
        )
        backbone = await _run_vlm_backbone(
            context,
            final_response=final_response,
            vlm_questions=vlm_questions,
            data_fields=data_fields,
            model_specs=model_specs,
            prompt_text=prompt_text,
        )
        if not backbone["ok"]:
            metrics = _zero_multi_model_metrics(model_specs)
            metrics.update(backbone.get("video_metrics", _video_generation_metrics()))
            return metrics
        video_path = backbone["video_path"]
        model_results = backbone["model_results"]
        available_results = backbone["available_results"]
        if len(available_results) < len(model_specs):
            metrics = _zero_multi_model_metrics(model_specs)
            metrics.update(backbone.get("video_metrics", _video_generation_metrics()))
            return metrics
        return _build_multi_model_metrics(
            model_results,
            model_specs,
            backbone.get("video_metrics", _video_generation_metrics()),
        )
    except Exception as e:
        logger.error("Error in vlm_as_judge_pass_reward_multi_model: %s", e)
        metrics = _zero_multi_model_metrics(model_specs)
        metrics.update(_video_generation_metrics())
        return metrics
    finally:
        _cleanup_video(video_path)