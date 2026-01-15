"""VLM as Judge Reward Function for AgentFly RL Training"""

import os
import re
import json
import uuid
import tempfile
import subprocess
import shutil
import asyncio
import logging
import time
import concurrent.futures
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Support running both as a package module and as a standalone script
try:
    from ..reward_base import reward
    from .vlm_as_judge_client import (
        VLMClient,
        create_vlm_prompt,
        create_vlm_prompt_from_template,
        _extract_json_list,
    )
except ImportError:  # Running as a script without package context
    import sys
    # Add repo root to sys.path so absolute imports work when invoked directly
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from agentfly.rewards.reward_base import reward
    from agentfly.rewards.vlm_as_judge.vlm_as_judge_client import (
        VLMClient,
        create_vlm_prompt,
        _extract_json_list,
        create_vlm_prompt_from_template,
        create_vlm_prompt_custom,
        DEFAULT_VLM_PROMPT_TEMPLATE,
    )

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
VLM_MODEL_SERVER_IPS = {

}
_VLM_CLIENTS: Dict[Tuple[str, Tuple[str, ...], int, int, int, str], VLMClient] = {}
DEFAULT_VLM_ENSEMBLE_MODEL_SPECS = [

]
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


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no")


def _log_vlm_raw_output(model_name: str, output_text: str, reason: str, force: bool = False) -> None:
    if not (force or _env_flag_enabled("VLM_LOG_RAW_OUTPUT")):
        return
    if force:
        log_fn = logger.error
    else:
        log_fn = logger.warning
    log_fn("VLM raw output (%s) for model '%s':\n%s", reason, model_name, output_text)


def _normalize_server_ips(value: Optional[Any]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        ips = [str(ip).strip() for ip in value if str(ip).strip()]
        return ips or None
    if isinstance(value, str):
        ips = [ip.strip() for ip in value.split(",") if ip.strip()]
        return ips or None
    return None


def _resolve_vlm_model(explicit_model: Optional[str], data_fields: Dict[str, Any]) -> str:
    if explicit_model:
        return explicit_model
    for key in ("vlm_model", "vlm_model_name", "vlm_model_name_or_path"):
        candidate = data_fields.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return os.getenv("VLM_JUDGE_MODEL", DEFAULT_VLM_MODEL)


def _resolve_vlm_server_ips(
    model: str,
    explicit_ips: Optional[Any],
    data_fields: Dict[str, Any],
) -> Optional[List[str]]:
    ips = _normalize_server_ips(explicit_ips)
    if ips:
        return ips
    ips = _normalize_server_ips(data_fields.get("vlm_server_ips"))
    if ips:
        return ips
    env_json = os.getenv("VLM_SERVER_IPS_JSON")
    if env_json:
        try:
            mapping = json.loads(env_json)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid VLM_SERVER_IPS_JSON: %s", exc)
        else:
            if isinstance(mapping, dict):
                env_ips = _normalize_server_ips(mapping.get(model))
                if env_ips:
                    return env_ips
    ips = VLM_MODEL_SERVER_IPS.get(model)
    if ips:
        return list(ips)
    logger.warning(
        "No VLM server IPs configured for model '%s'; falling back to VLMClient default.",
        model,
    )
    return None


def _get_vlm_ensemble_log_path() -> str:
    env_path = os.getenv("VLM_ENSEMBLE_LOG_PATH")
    if env_path:
        return env_path
    data_dir = os.getenv("AGENT_DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "vlm_ensemble_endpoints.jsonl")
    default_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    return os.path.join(default_data_dir, "vlm_ensemble_endpoints.jsonl")


def _append_vlm_ensemble_log(record: Dict[str, Any]) -> None:
    log_path = _get_vlm_ensemble_log_path()
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.debug("Failed to write VLM ensemble log: %s", exc)


def _get_vlm_client(
    model: str,
    timeout_seconds: int = 60,
    max_window_size_per_instance: int = 10,
    port: int = 8000,
    api_key: str = "token-abc123",
    server_ips: Optional[List[str]] = None,
) -> VLMClient:
    ips_key = tuple(server_ips) if server_ips else ("__default__",)
    key = (model, ips_key, timeout_seconds, max_window_size_per_instance, port, api_key)
    client = _VLM_CLIENTS.get(key)
    if client is None:
        client = VLMClient(
            model=model,
            timeout_seconds=timeout_seconds,
            max_window_size_per_instance=max_window_size_per_instance,
            port=port,
            api_key=api_key,
            server_ips=server_ips,
        )
        _VLM_CLIENTS[key] = client
    return client


def _get_vlm_wait_settings() -> Tuple[int, float]:
    wait_seconds_env = os.getenv("VLM_CLIENT_WAIT_SECONDS")
    poll_interval_env = os.getenv("VLM_CLIENT_POLL_INTERVAL")

    try:
        wait_seconds = int(wait_seconds_env) if wait_seconds_env is not None else 30
    except (TypeError, ValueError):
        wait_seconds = 30

    try:
        poll_interval = float(poll_interval_env) if poll_interval_env is not None else 1.0
    except (TypeError, ValueError):
        poll_interval = 1.0

    if wait_seconds < 0:
        wait_seconds = 0
    if poll_interval <= 0:
        poll_interval = 1.0

    return wait_seconds, poll_interval


async def _wait_for_vlm_availability(client: VLMClient) -> bool:
    wait_seconds, poll_interval = _get_vlm_wait_settings()
    if wait_seconds == 0:
        return client.is_available()

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if client.is_available():
            return True
        await asyncio.sleep(poll_interval)

    return client.is_available()


def _normalize_model_specs(value: Optional[Any]) -> Optional[List[Dict[str, Any]]]:
    if not value or not isinstance(value, list):
        return None

    specs = []
    for item in value:
        if not isinstance(item, dict):
            continue
        model = item.get("model") or item.get("model_name")
        if not isinstance(model, str) or not model.strip():
            continue
        server_ips = _normalize_server_ips(item.get("server_ips") or item.get("ips"))
        specs.append({"model": model.strip(), "server_ips": server_ips})

    return specs or None


def _resolve_vlm_model_specs(explicit_specs: Optional[Any], data_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = _normalize_model_specs(explicit_specs)
    if not specs:
        specs = _normalize_model_specs(data_fields.get("vlm_model_specs"))

    if not specs:
        return [dict(item) for item in DEFAULT_VLM_ENSEMBLE_MODEL_SPECS]

    for spec in specs:
        if not spec.get("server_ips"):
            spec["server_ips"] = _resolve_vlm_server_ips(spec["model"], None, data_fields)
    return specs


def _parse_confidence_score(value: Any, default: int = 1) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(5, score))


def _result_is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _aggregate_vlm_results(
    results_by_model: List[Dict[str, Dict[str, Any]]],
    questions_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregated = []

    for question in questions_list:
        idx = str(question.get("index", "")).strip()
        if not idx:
            continue

        votes = []
        confidences = []
        for result_map in results_by_model:
            result = result_map.get(idx)
            if not result:
                continue
            votes.append(_result_is_true(result.get("result")))
            confidences.append(_parse_confidence_score(result.get("confidence_score")))

        if len(votes) < VLM_ENSEMBLE_MIN_RESPONSES:
            final_value = False
        else:
            true_votes = sum(1 for vote in votes if vote)
            false_votes = len(votes) - true_votes
            final_value = true_votes > false_votes

        avg_conf = int(round(sum(confidences) / len(confidences))) if confidences else 1
        aggregated.append({
            "index": idx,
            "question": question.get("question", ""),
            "analysis": "Ensemble vote across available models.",
            "result": "True" if final_value else "False",
            "confidence_score": str(max(1, min(5, avg_conf))),
        })

    return aggregated


class VideoGenerator:
    """Helper class to generate videos from code"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize video generator
        
        Args:
            output_dir: Directory to save generated videos
        """
        # Prefer a shared directory accessible by the VLM server if provided.
        if output_dir is None:
            output_dir = os.getenv("VLM_SHARED_VIDEO_DIR", "/tmp/vlm_videos")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from model response
        
        Args:
            response: Model response containing code
            
        Returns:
            Extracted Python code or None
        """
        # Remove <think> tags if present
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Extract code from ```python blocks
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, cleaned, re.DOTALL)
        
        if matches:
            return matches[0]
        return None
    
    async def generate_video_from_code(self, code: str, output_path: str) -> bool:
        """Execute Python code to generate video (async version)
        
        Args:
            code: Python code to execute
            output_path: Path to save the generated video
            
        Returns:
            True if video generation successful, False otherwise
        """
        temp_file = None
        try:
            # Create a temporary Python file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Modify code to use the specified output path
                modified_code = code
                
                # Handle sys.argv usage for output filename
                if 'sys.argv[1]' in code:
                    modified_code = code.replace('sys.argv[1]', f'"{output_path}"')
                elif 'sys.argv' in code and 'len(sys.argv)' in code:
                    # Add sys.argv mock at the beginning
                    modified_code = f"import sys\nsys.argv = ['script.py', '{output_path}']\n" + code
                else:
                    # If no sys.argv usage, try to modify output filename assignments
                    # Look for common patterns like output_file = ... or out = cv2.VideoWriter(...)
                    if 'output_file' in code:
                        # Replace output_file assignment
                        modified_code = re.sub(
                            r'output_file\s*=\s*["\'].*?["\']',
                            f'output_file = "{output_path}"',
                            code
                        )
                    elif 'VideoWriter(' in code:
                        # Try to replace the first string argument in VideoWriter
                        modified_code = re.sub(
                            r'VideoWriter\s*\(\s*["\'].*?["\']',
                            f'VideoWriter("{output_path}"',
                            code
                        )
                    else:
                        # Last resort: append output path assignment
                        modified_code = f"output_file = '{output_path}'\n" + code
                
                f.write(modified_code)
                temp_file = f.name
            
            # Execute the code asynchronously
            # Always pass the output path as an argument so scripts that expect
            # sys.argv[1] or check len(sys.argv) continue without exiting.
            process = await asyncio.create_subprocess_exec(
                'python', temp_file, output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.output_dir  # Run in output directory
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120.0  # Increased timeout for video generation
                )
            except asyncio.TimeoutError:
                logger.error("Video generation timed out")
                process.kill()
                await process.wait()
                if temp_file:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                return False
            
            # Clean up temp file
            if temp_file:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            # Check if video was created and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
                logger.info(f"Successfully generated video: {output_path} ({os.path.getsize(output_path)} bytes)")
                return True
            else:
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ''
                logger.error(f"Video generation failed or file too small. stderr: {stderr_text}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            if temp_file:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return False


def _can_open_video(video_path: str) -> bool:
    if not video_path or not os.path.exists(video_path):
        return False

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return False
            ok, _ = cap.read()
            return bool(ok)
        finally:
            cap.release()

    try:
        import imageio.v3 as iio  # type: ignore
        for _ in iio.imiter(video_path):
            return True
        return False
    except Exception:
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=nw=1:nk=1",
                    video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except Exception:
            pass

    return False


def extract_vlm_questions_from_data(data: Dict[str, Any]) -> Tuple[str, str, List[Dict]]:
    """Extract VLM questions and summary from data
    
    Args:
        data: Dictionary containing vlm_questions data
        
    Returns:
        Tuple of (all_questions_str, summarize, questions_list)
    """
    all_questions = ""
    summarize = ""
    questions_list = []
    
    if "vlm_questions" in data:
        vlm_data = data["vlm_questions"]
        if isinstance(vlm_data, dict):
            # Get summary
            summarize = vlm_data.get("summarize", "")
            
            # Extract questions from nested vlm_questions field
            if "vlm_questions" in vlm_data:
                questions_list = vlm_data["vlm_questions"]
                if isinstance(questions_list, list):
                    for q in questions_list:
                        if isinstance(q, dict):
                            idx = q.get("index", "")
                            question = q.get("question", "")
                            all_questions += f"{idx}. {question}\n"
                else:
                    logger.warning(f"vlm_questions inner field is not a list: {type(questions_list)}")
        else:
            logger.warning(f"vlm_questions is not a dict: {type(vlm_data)}")
    else:
        logger.warning(f"No vlm_questions field in data. Available fields: {list(data.keys())}")
    
    all_questions = all_questions.strip()
    
    if not summarize:
        summarize = "Evaluate the visual content based on the questions provided."
    
    logger.info(f"Extracted {len(questions_list)} questions from VLM data")
    
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


@reward(name="vlm_as_judge_pass_reward")
async def vlm_as_judge_pass_reward(
    final_response: str, 
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model: Optional[str] = None,
    vlm_server_ips: Optional[List[str]] = None,
    **data_fields
) -> Dict[str, float]:
        """VLM as Judge reward function for evaluating agent trajectories
    
    This reward function:
    1. Extracts Python code from the prediction 
    2. Generates a video using the code
    3. Uses VLM server to evaluate the video against provided questions
    4. Returns a binary pass/fail score based on VLM judgments
    
    Args:
        prediction: Agent's generated response (should contain Python code)
        trajectory: Agent trajectory information
        **data_fields: Additional data fields from the RL data, including vlm_questions
        
    Returns:
        pass/fail reward score between 0.0 and 1.0
    """
        try:
            # Log incoming data for debugging
            logger.info(f"=" * 60)
            logger.info(f"vlm_as_judge_reward called")
            logger.info(f"Prediction length: {len(final_response) if final_response else 0}")
            
            # Print the actual prediction content
            logger.info(f"Prediction content (first 500 chars):")
            logger.info(f"{final_response[:500] if final_response else 'No prediction'}")
            if final_response and len(final_response) > 500:
                logger.info(f"... (truncated, total length: {len(final_response)} chars)")
            
            logger.info(f"vlm_questions parameter: {vlm_questions is not None}")
            logger.info(f"Additional data_fields keys: {list(data_fields.keys())}")
            
            # Initialize video generator
            video_gen = VideoGenerator()
            
            # Combine vlm_questions with data_fields for extraction
            all_data = dict(data_fields)
            if vlm_questions is not None:
                all_data['vlm_questions'] = vlm_questions
                logger.info(f"vlm_questions type: {type(vlm_questions)}")
                if isinstance(vlm_questions, dict):
                    logger.info(f"vlm_questions keys: {vlm_questions.keys()}")
                    if 'vlm_questions' in vlm_questions:
                        inner_vlm = vlm_questions['vlm_questions']
                        logger.info(f"Inner vlm_questions type: {type(inner_vlm)}")
                        if isinstance(inner_vlm, list):
                            logger.info(f"Number of questions in inner list: {len(inner_vlm)}")
            
            # Extract VLM questions from data
            all_questions, summarize, questions_list = extract_vlm_questions_from_data(all_data)
            
            if not questions_list:
                logger.warning(f"No VLM questions found in data. Available fields: {list(all_data.keys())}")
                return {"reward": 0.0}
            
            # Extract code from prediction
            code = video_gen.extract_code_from_response(final_response)
            if not code:
                logger.warning("No Python code found in prediction")
                logger.warning(f"Prediction was: {final_response[:1000] if final_response else 'None'}")
                return {"reward": 0.0}
            
            logger.info(f"Extracted Python code ({len(code)} chars)")
            logger.info(f"Code preview (first 300 chars):")
            logger.info(f"{code[:300]}...")
            if len(code) > 300:
                logger.info(f"... (truncated, total length: {len(code)} chars)")
            
            # Generate unique video filename
            video_filename = f"video_{uuid.uuid4().hex}.mp4"
            video_path = os.path.join(video_gen.output_dir, video_filename)
            
            # Generate video from code
            success = await video_gen.generate_video_from_code(code, video_path)
            if not success:
                logger.error("Failed to generate video from code")
                return {"reward": 0.0}
            
            model_name = _resolve_vlm_model(vlm_model, data_fields)
            server_ips = _resolve_vlm_server_ips(model_name, vlm_server_ips, data_fields)
            # Run VLM evaluation directly since we're already async
            client = _get_vlm_client(
                model=model_name,
                timeout_seconds=120,
                server_ips=server_ips,
            )
            
            # Wait for client availability
            if not await _wait_for_vlm_availability(client):
                logger.error("VLM client not available")
                return {"reward": 0.0}
            
            # Create VLM prompt
            prompt_text = create_vlm_prompt(summarize, all_questions)
            
            # Build message using <video> tag in text content to match server expectations
            user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
            messages = [{
                "role": "user",
                "content": user_text
            }]
            
            # Process the request
            responses = await client.process_all_inputs(
                [messages],
                model=model_name,
                temperature=0,
            )
            
            if not responses or not responses[0] or not responses[0][0]:
                logger.error("No response from VLM server")
                # Clean up video file
                try:
                    os.remove(video_path)
                except:
                    pass
                return {"reward": 0.0}
            
            output_text = responses[0][0]
            
            try:
                # Parse VLM results
                results = _extract_json_list(output_text)
                if not results:
                    logger.error("Failed to parse VLM results")
                    # Clean up video file
                    try:
                        os.remove(video_path)
                    except:
                        pass
                    return {"reward": 0.0}
                
                # Calculate weighted reward
                reward_score = pass_fail_reward(results, questions_list)
                
                logger.info(f"VLM evaluation completed. Reward: {reward_score:.3f}")
                
                # Clean up video file
                try:
                    os.remove(video_path)
                except:
                    pass
                
                return {"reward": reward_score}
                
            except Exception as e:
                logger.error(f"Error processing VLM results: {e}")
                # Clean up video file
                try:
                    os.remove(video_path)
                except:
                    pass
                return {"reward": 0.0}
            
        except Exception as e:
            logger.error(f"Error in vlm_as_judge_reward: {e}")
            import traceback
            traceback.print_exc()
            return {"reward": 0.0}
    

@reward(name="vlm_as_judge_pass_reward_multi_model")
async def vlm_as_judge_pass_reward_multi_model(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model_specs: Optional[List[Dict[str, Any]]] = None,
    **data_fields,
) -> Dict[str, float]:
    """VLM as Judge pass/fail reward using multiple models with an ensemble vote."""
    video_path = None
    try:
        logger.info(f"=" * 60)
        logger.info("vlm_as_judge_pass_reward_multi_model called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")

        video_gen = VideoGenerator()

        all_data = dict(data_fields)
        if vlm_questions is not None:
            all_data["vlm_questions"] = vlm_questions

        all_questions, _, questions_list = extract_vlm_questions_from_data(all_data)
        if not questions_list:
            logger.warning("No VLM questions found in data.")
            return {"reward": 0.0}

        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in final_response")
            return {"reward": 0.0}

        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)

        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": 0.0}

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        prompt_text = create_vlm_prompt_from_template(
            VLM_ENSEMBLE_PROMPT_TEMPLATE,
            variables={"all_questions": all_questions},
        )
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [
            {"role": "user", "content": user_text},
        ]

        model_specs = _resolve_vlm_model_specs(vlm_model_specs, data_fields)
        if not model_specs:
            logger.error("No VLM model specs configured for ensemble.")
            return {"reward": 0.0}

        async def _run_model(spec: Dict[str, Any], spec_index: int) -> Dict[str, Any]:
            model_name = spec["model"]
            server_ips = spec.get("server_ips")
            try:
                client = _get_vlm_client(
                    model=model_name,
                    timeout_seconds=120,
                    server_ips=server_ips,
                )

                if not await _wait_for_vlm_availability(client):
                    logger.error("VLM client not available for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "client_unavailable",
                    }

                responses = await client.process_all_inputs(
                    [messages],
                    model=model_name,
                    temperature=0,
                )

                if not responses or not responses[0] or not responses[0][0]:
                    if _env_flag_enabled("VLM_LOG_RAW_OUTPUT"):
                        logger.warning(
                            "VLM raw responses for model '%s' were empty: %s",
                            model_name,
                            responses,
                        )
                    logger.error("No response from VLM server for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "empty_response",
                    }

                output_text = responses[0][0]
                _log_vlm_raw_output(model_name, output_text, "response")
                try:
                    results = _extract_json_list(output_text)
                except Exception as exc:
                    logger.error("Failed to parse VLM results for model '%s': %s", model_name, exc)
                    # print(f"output_text: {output_text}")
                    _log_vlm_raw_output(model_name, output_text, "parse_error", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "parse_error",
                    }

                if not results:
                    logger.error("Empty VLM results for model '%s'", model_name)
                    _log_vlm_raw_output(model_name, output_text, "no_results", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "no_results",
                    }

                passed = _all_true_pass(results, questions_list)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": passed,
                    "endpoint_reward": 1.0 if passed else 0.0,
                    "error": None,
                }
            except Exception as exc:
                logger.error("Error running VLM model '%s': %s", model_name, exc)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": None,
                    "endpoint_reward": None,
                    "error": str(exc),
                }

        tasks = [_run_model(spec, idx) for idx, spec in enumerate(model_specs, start=1)]
        model_results = await asyncio.gather(*tasks)
        failed_results = [r for r in model_results if r["passed"] is None]
        if failed_results:
            failed_summary = ", ".join(
                f"{result['model']}:{result.get('error')}" for result in failed_results
            )
            logger.error(
                "VLM ensemble all-or-nothing: %d/%d models failed (%s); dropping partial results.",
                len(failed_results),
                len(model_specs),
                failed_summary,
            )
            available_results = []
        else:
            available_results = model_results

        required_responses = len(model_specs)
        if len(available_results) < required_responses:
            logger.error(
                "Only %d/%d VLM models returned results; need at least %d.",
                len(available_results),
                len(model_specs),
                required_responses,
            )
            _append_vlm_ensemble_log(
                {
                    "timestamp": time.time(),
                    "reward": 0.0,
                    "available": len(available_results),
                    "required": required_responses,
                    "endpoints": model_results,
                }
            )
            return {"reward": 0.0}

        pass_votes = sum(1 for result in available_results if result["passed"])
        fail_votes = len(available_results) - pass_votes
        reward_score = 1.0 if pass_votes > fail_votes else 0.0
        logger.info(
            "Ensemble VLM evaluation completed. Pass votes: %d/%d. Reward: %.3f",
            pass_votes,
            len(available_results),
            reward_score,
        )
        _append_vlm_ensemble_log(
            {
                "timestamp": time.time(),
                "reward": reward_score,
                "available": len(available_results),
                "required": required_responses,
                "endpoints": model_results,
            }
        )
        return {"reward": reward_score}

    except Exception as e:
        logger.error(f"Error in vlm_as_judge_pass_reward_multi_model: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": 0.0}
    finally:
        if video_path:
            try:
                os.remove(video_path)
            except Exception:
                pass


@reward(name="vlm_as_judge_pass_reward_multi_model_pass_at_3")
async def vlm_as_judge_pass_reward_multi_model_pass_at_3(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model_specs: Optional[List[Dict[str, Any]]] = None,
    **data_fields,
) -> Dict[str, float]:
    """VLM as Judge pass@N reward using multiple models (reward=1 if any model passes)."""
    video_path = None
    try:
        logger.info(f"=" * 60)
        logger.info("vlm_as_judge_pass_reward_multi_model_pass_at_3 called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")

        video_gen = VideoGenerator()

        all_data = dict(data_fields)
        if vlm_questions is not None:
            all_data["vlm_questions"] = vlm_questions

        all_questions, _, questions_list = extract_vlm_questions_from_data(all_data)
        if not questions_list:
            logger.warning("No VLM questions found in data.")
            return {"reward": 0.0}

        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in final_response")
            return {"reward": 0.0}

        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)

        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": 0.0}

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        prompt_text = create_vlm_prompt_from_template(
            VLM_ENSEMBLE_PROMPT_TEMPLATE,
            variables={"all_questions": all_questions},
        )
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [
            {"role": "user", "content": user_text},
        ]

        model_specs = _resolve_vlm_model_specs(vlm_model_specs, data_fields)
        if not model_specs:
            logger.error("No VLM model specs configured for ensemble.")
            return {"reward": 0.0}

        async def _run_model(spec: Dict[str, Any], spec_index: int) -> Dict[str, Any]:
            model_name = spec["model"]
            server_ips = spec.get("server_ips")
            try:
                client = _get_vlm_client(
                    model=model_name,
                    timeout_seconds=120,
                    server_ips=server_ips,
                )

                if not await _wait_for_vlm_availability(client):
                    logger.error("VLM client not available for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "client_unavailable",
                    }

                responses = await client.process_all_inputs(
                    [messages],
                    model=model_name,
                    temperature=0,
                )

                if not responses or not responses[0] or not responses[0][0]:
                    if _env_flag_enabled("VLM_LOG_RAW_OUTPUT"):
                        logger.warning(
                            "VLM raw responses for model '%s' were empty: %s",
                            model_name,
                            responses,
                        )
                    logger.error("No response from VLM server for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "empty_response",
                    }

                output_text = responses[0][0]
                _log_vlm_raw_output(model_name, output_text, "response")
                try:
                    results = _extract_json_list(output_text)
                except Exception as exc:
                    logger.error("Failed to parse VLM results for model '%s': %s", model_name, exc)
                    _log_vlm_raw_output(model_name, output_text, "parse_error", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "parse_error",
                    }

                if not results:
                    logger.error("Empty VLM results for model '%s'", model_name)
                    _log_vlm_raw_output(model_name, output_text, "no_results", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "no_results",
                    }

                passed = _all_true_pass(results, questions_list)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": passed,
                    "endpoint_reward": 1.0 if passed else 0.0,
                    "error": None,
                }
            except Exception as exc:
                logger.error("Error running VLM model '%s': %s", model_name, exc)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": None,
                    "endpoint_reward": None,
                    "error": str(exc),
                }

        tasks = [_run_model(spec, idx) for idx, spec in enumerate(model_specs, start=1)]
        model_results = await asyncio.gather(*tasks)
        failed_results = [r for r in model_results if r["passed"] is None]
        if failed_results:
            failed_summary = ", ".join(
                f"{result['model']}:{result.get('error')}" for result in failed_results
            )
            logger.error(
                "VLM ensemble all-or-nothing: %d/%d models failed (%s); dropping partial results.",
                len(failed_results),
                len(model_specs),
                failed_summary,
            )
            available_results = []
        else:
            available_results = model_results

        required_responses = len(model_specs)
        if len(available_results) < required_responses:
            logger.error(
                "Only %d/%d VLM models returned results; need at least %d.",
                len(available_results),
                len(model_specs),
                required_responses,
            )
            _append_vlm_ensemble_log(
                {
                    "timestamp": time.time(),
                    "reward": 0.0,
                    "available": len(available_results),
                    "required": required_responses,
                    "endpoints": model_results,
                }
            )
            return {"reward": 0.0}

        pass_votes = sum(1 for result in available_results if result["passed"])
        reward_score = 1.0 if pass_votes >= 1 else 0.0
        logger.info(
            "Pass@%d VLM evaluation completed. Pass votes: %d/%d. Reward: %.3f",
            len(available_results),
            pass_votes,
            len(available_results),
            reward_score,
        )
        _append_vlm_ensemble_log(
            {
                "timestamp": time.time(),
                "reward": reward_score,
                "available": len(available_results),
                "required": required_responses,
                "endpoints": model_results,
            }
        )
        return {"reward": reward_score}

    except Exception as e:
        logger.error(f"Error in vlm_as_judge_pass_reward_multi_model_pass_at_3: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": 0.0}
    finally:
        if video_path:
            try:
                os.remove(video_path)
            except Exception:
                pass


@reward(name="vlm_as_judge_pass_reward_multi_model_ladder")
async def vlm_as_judge_pass_reward_multi_model_ladder(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model_specs: Optional[List[Dict[str, Any]]] = None,
    **data_fields,
) -> Dict[str, float]:
    """VLM as Judge ladder reward using multiple models with an ensemble vote."""
    video_path = None
    reward_score = 0.0
    vlm_judge_score = 0.0
    try:
        logger.info(f"=" * 60)
        logger.info("vlm_as_judge_pass_reward_multi_model_ladder called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")

        video_gen = VideoGenerator()

        all_data = dict(data_fields)
        if vlm_questions is not None:
            all_data["vlm_questions"] = vlm_questions

        all_questions, _, questions_list = extract_vlm_questions_from_data(all_data)
        if not questions_list:
            logger.warning("No VLM questions found in data.")
            return {"reward": 0.0, "vlm_judge": vlm_judge_score}

        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in final_response")
            return {"reward": 0.0, "vlm_judge": vlm_judge_score}
        reward_score += LADDER_CODE_EXTRACTION_REWARD

        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)

        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}
        reward_score += LADDER_CODE_RENDER_REWARD

        if not _can_open_video(video_path):
            logger.error("Failed to open generated video")
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}
        reward_score += LADDER_VIDEO_OPEN_REWARD

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        prompt_text = create_vlm_prompt_from_template(
            VLM_ENSEMBLE_PROMPT_TEMPLATE,
            variables={"all_questions": all_questions},
        )
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [
            {"role": "user", "content": user_text},
        ]

        model_specs = _resolve_vlm_model_specs(vlm_model_specs, data_fields)
        if not model_specs:
            logger.error("No VLM model specs configured for ensemble.")
            return {
                "reward": reward_score,
                "vlm_judge": vlm_judge_score,
            }

        async def _run_model(spec: Dict[str, Any], spec_index: int) -> Dict[str, Any]:
            model_name = spec["model"]
            server_ips = spec.get("server_ips")
            try:
                client = _get_vlm_client(
                    model=model_name,
                    timeout_seconds=120,
                    server_ips=server_ips,
                )

                if not await _wait_for_vlm_availability(client):
                    logger.error("VLM client not available for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "client_unavailable",
                    }

                responses = await client.process_all_inputs(
                    [messages],
                    model=model_name,
                    temperature=0,
                )

                if not responses or not responses[0] or not responses[0][0]:
                    if _env_flag_enabled("VLM_LOG_RAW_OUTPUT"):
                        logger.warning(
                            "VLM raw responses for model '%s' were empty: %s",
                            model_name,
                            responses,
                        )
                    logger.error("No response from VLM server for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "empty_response",
                    }

                output_text = responses[0][0]
                _log_vlm_raw_output(model_name, output_text, "response")
                try:
                    results = _extract_json_list(output_text)
                except Exception as exc:
                    logger.error("Failed to parse VLM results for model '%s': %s", model_name, exc)
                    _log_vlm_raw_output(model_name, output_text, "parse_error", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "parse_error",
                    }

                if not results:
                    logger.error("Empty VLM results for model '%s'", model_name)
                    _log_vlm_raw_output(model_name, output_text, "no_results", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "no_results",
                    }

                passed = _all_true_pass(results, questions_list)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": passed,
                    "endpoint_reward": 1.0 if passed else 0.0,
                    "error": None,
                }
            except Exception as exc:
                logger.error("Error running VLM model '%s': %s", model_name, exc)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": None,
                    "endpoint_reward": None,
                    "error": str(exc),
                }

        tasks = [_run_model(spec, idx) for idx, spec in enumerate(model_specs, start=1)]
        model_results = await asyncio.gather(*tasks)
        failed_results = [r for r in model_results if r["passed"] is None]
        if failed_results:
            failed_summary = ", ".join(
                f"{result['model']}:{result.get('error')}" for result in failed_results
            )
            logger.error(
                "VLM ensemble all-or-nothing: %d/%d models failed (%s); dropping partial results.",
                len(failed_results),
                len(model_specs),
                failed_summary,
            )
            available_results = []
        else:
            available_results = model_results

        required_responses = len(model_specs)
        if len(available_results) < required_responses:
            logger.error(
                "Only %d/%d VLM models returned results; need at least %d.",
                len(available_results),
                len(model_specs),
                required_responses,
            )
            _append_vlm_ensemble_log(
                {
                    "timestamp": time.time(),
                    "reward": reward_score,
                    "available": len(available_results),
                    "required": required_responses,
                    "endpoints": model_results,
                }
            )
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}

        pass_votes = sum(1 for result in available_results if result["passed"])
        fail_votes = len(available_results) - pass_votes
        if pass_votes > fail_votes:
            reward_score += LADDER_VLM_REWARD
            vlm_judge_score = 1.0
        logger.info(
            "Ladder VLM evaluation completed. Pass votes: %d/%d. Reward: %.3f",
            pass_votes,
            len(available_results),
            reward_score,
        )
        _append_vlm_ensemble_log(
            {
                "timestamp": time.time(),
                "reward": reward_score,
                "available": len(available_results),
                "required": required_responses,
                "endpoints": model_results,
            }
        )
        return {"reward": reward_score, "vlm_judge": vlm_judge_score}

    except Exception as e:
        logger.error(f"Error in vlm_as_judge_pass_reward_multi_model_ladder: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": reward_score, "vlm_judge": vlm_judge_score}
    finally:
        if video_path:
            try:
                os.remove(video_path)
            except Exception:
                pass


@reward(name="vlm_as_judge_pass_reward_multi_model_pass_at_3_ladder")
async def vlm_as_judge_pass_reward_multi_model_pass_at_3_ladder(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model_specs: Optional[List[Dict[str, Any]]] = None,
    **data_fields,
) -> Dict[str, float]:
    """VLM as Judge ladder pass@N reward using multiple models."""
    video_path = None
    reward_score = 0.0
    vlm_judge_score = 0.0
    try:
        logger.info(f"=" * 60)
        logger.info("vlm_as_judge_pass_reward_multi_model_pass_at_3_ladder called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")

        video_gen = VideoGenerator()

        all_data = dict(data_fields)
        if vlm_questions is not None:
            all_data["vlm_questions"] = vlm_questions

        all_questions, _, questions_list = extract_vlm_questions_from_data(all_data)
        if not questions_list:
            logger.warning("No VLM questions found in data.")
            return {"reward": 0.0, "vlm_judge": vlm_judge_score}

        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in final_response")
            return {"reward": 0.0, "vlm_judge": vlm_judge_score}
        reward_score += LADDER_CODE_EXTRACTION_REWARD

        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)

        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}
        reward_score += LADDER_CODE_RENDER_REWARD

        if not _can_open_video(video_path):
            logger.error("Failed to open generated video")
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}
        reward_score += LADDER_VIDEO_OPEN_REWARD

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        expected_indices = [
            str(question.get("index", "")).strip()
            for question in questions_list
            if str(question.get("index", "")).strip()
        ]
        if not expected_indices:
            expected_indices = [str(i + 1) for i in range(len(questions_list))]
        expected_index_set = set(expected_indices)

        prompt_text = create_vlm_prompt_from_template(
            VLM_ENSEMBLE_PROMPT_TEMPLATE,
            variables={"all_questions": all_questions},
        )
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [
            {"role": "user", "content": user_text},
        ]

        model_specs = _resolve_vlm_model_specs(vlm_model_specs, data_fields)
        if not model_specs:
            logger.error("No VLM model specs configured for ensemble.")
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}

        async def _run_model(spec: Dict[str, Any], spec_index: int) -> Dict[str, Any]:
            model_name = spec["model"]
            server_ips = spec.get("server_ips")
            try:
                client = _get_vlm_client(
                    model=model_name,
                    timeout_seconds=120,
                    server_ips=server_ips,
                )

                if not await _wait_for_vlm_availability(client):
                    logger.error("VLM client not available for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "client_unavailable",
                    }

                responses = await client.process_all_inputs(
                    [messages],
                    model=model_name,
                    temperature=0,
                )

                if not responses or not responses[0] or not responses[0][0]:
                    if _env_flag_enabled("VLM_LOG_RAW_OUTPUT"):
                        logger.warning(
                            "VLM raw responses for model '%s' were empty: %s",
                            model_name,
                            responses,
                        )
                    logger.error("No response from VLM server for model '%s'", model_name)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "empty_response",
                    }

                output_text = responses[0][0]
                _log_vlm_raw_output(model_name, output_text, "response")
                try:
                    results = _extract_json_list(output_text)
                except Exception as exc:
                    logger.error("Failed to parse VLM results for model '%s': %s", model_name, exc)
                    _log_vlm_raw_output(model_name, output_text, "parse_error", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "parse_error",
                    }

                if not results:
                    logger.error("Empty VLM results for model '%s'", model_name)
                    _log_vlm_raw_output(model_name, output_text, "no_results", force=True)
                    return {
                        "index": spec_index,
                        "model": model_name,
                        "server_ips": server_ips,
                        "passed": None,
                        "endpoint_reward": None,
                        "error": "no_results",
                    }

                passed = _all_true_pass(results, questions_list)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": passed,
                    "endpoint_reward": 1.0 if passed else 0.0,
                    "error": None,
                }
            except Exception as exc:
                logger.error("Error running VLM model '%s': %s", model_name, exc)
                return {
                    "index": spec_index,
                    "model": model_name,
                    "server_ips": server_ips,
                    "passed": None,
                    "endpoint_reward": None,
                    "error": str(exc),
                }

        tasks = [_run_model(spec, idx) for idx, spec in enumerate(model_specs, start=1)]
        model_results = await asyncio.gather(*tasks)
        failed_results = [r for r in model_results if r["passed"] is None]
        if failed_results:
            failed_summary = ", ".join(
                f"{result['model']}:{result.get('error')}" for result in failed_results
            )
            logger.error(
                "VLM ensemble all-or-nothing: %d/%d models failed (%s); dropping partial results.",
                len(failed_results),
                len(model_specs),
                failed_summary,
            )
            available_results = []
        else:
            available_results = model_results

        required_responses = len(model_specs)
        if len(available_results) < required_responses:
            logger.error(
                "Only %d/%d VLM models returned results; need at least %d.",
                len(available_results),
                len(model_specs),
                required_responses,
            )
            _append_vlm_ensemble_log(
                {
                    "timestamp": time.time(),
                    "reward": reward_score,
                    "available": len(available_results),
                    "required": required_responses,
                    "endpoints": model_results,
                }
            )
            return {"reward": reward_score, "vlm_judge": vlm_judge_score}

        pass_votes = sum(1 for result in available_results if result["passed"])
        if pass_votes >= 1:
            reward_score += LADDER_VLM_REWARD
            vlm_judge_score = 1.0
        logger.info(
            "Ladder pass@%d VLM evaluation completed. Pass votes: %d/%d. Reward: %.3f",
            len(available_results),
            pass_votes,
            len(available_results),
            reward_score,
        )
        _append_vlm_ensemble_log(
            {
                "timestamp": time.time(),
                "reward": reward_score,
                "available": len(available_results),
                "required": required_responses,
                "endpoints": model_results,
            }
        )
        return {"reward": reward_score, "vlm_judge": vlm_judge_score}

    except Exception as e:
        logger.error(f"Error in vlm_as_judge_pass_reward_multi_model_pass_at_3_ladder: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": reward_score, "vlm_judge": vlm_judge_score}
    finally:
        if video_path:
            try:
                os.remove(video_path)
            except Exception:
                pass


@reward(name="vlm_as_judge_reward_multi_model")
async def vlm_as_judge_reward_multi_model(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model_specs: Optional[List[Dict[str, Any]]] = None,
    **data_fields,
) -> Dict[str, float]:
    """VLM as Judge reward using multiple models, returning true-rate (0-1)."""
    video_path = None
    try:
        logger.info(f"=" * 60)
        logger.info("vlm_as_judge_reward_multi_model called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")

        video_gen = VideoGenerator()

        all_data = dict(data_fields)
        if vlm_questions is not None:
            all_data["vlm_questions"] = vlm_questions

        all_questions, _, questions_list = extract_vlm_questions_from_data(all_data)
        if not questions_list:
            logger.warning("No VLM questions found in data.")
            return {"reward": 0.0}

        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in final_response")
            return {"reward": 0.0}

        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)

        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": 0.0}

        prompt_text = create_vlm_prompt_from_template(
            VLM_ENSEMBLE_PROMPT_TEMPLATE,
            variables={"all_questions": all_questions},
        )
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [
            {"role": "user", "content": user_text},
        ]

        model_specs = _resolve_vlm_model_specs(vlm_model_specs, data_fields)
        if not model_specs:
            logger.error("No VLM model specs configured for ensemble.")
            return {"reward": 0.0}

        async def _run_model(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            model_name = spec["model"]
            server_ips = spec.get("server_ips")
            try:
                client = _get_vlm_client(
                    model=model_name,
                    timeout_seconds=120,
                    server_ips=server_ips,
                )

                if not await _wait_for_vlm_availability(client):
                    logger.error("VLM client not available for model '%s'", model_name)
                    return None

                responses = await client.process_all_inputs(
                    [messages],
                    model=model_name,
                    temperature=0,
                )

                if not responses or not responses[0] or not responses[0][0]:
                    logger.error("No response from VLM server for model '%s'", model_name)
                    return None

                output_text = responses[0][0]
                try:
                    results = _extract_json_list(output_text)
                except Exception as exc:
                    logger.error("Failed to parse VLM results for model '%s': %s", model_name, exc)
                    return None

                if not results:
                    logger.error("Empty VLM results for model '%s'", model_name)
                    return None

                return {
                    "model": model_name,
                    "true_rate": _true_rate(results, questions_list),
                }
            except Exception as exc:
                logger.error("Error running VLM model '%s': %s", model_name, exc)
                return None

        tasks = [_run_model(spec) for spec in model_specs]
        model_results = [r for r in await asyncio.gather(*tasks) if r]

        if len(model_results) < VLM_ENSEMBLE_MIN_RESPONSES:
            logger.error(
                "Only %d/%d VLM models returned results; need at least %d.",
                len(model_results),
                len(model_specs),
                VLM_ENSEMBLE_MIN_RESPONSES,
            )
            return {"reward": 0.0}

        reward_score = sum(r["true_rate"] for r in model_results) / len(model_results)
        logger.info(
            "Ensemble VLM evaluation completed. True-rate avg: %.3f (%d models).",
            reward_score,
            len(model_results),
        )
        return {"reward": reward_score}

    except Exception as e:
        logger.error(f"Error in vlm_as_judge_reward_multi_model: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": 0.0}
    finally:
        if video_path:
            try:
                os.remove(video_path)
            except Exception:
                pass


@reward(name="vlm_as_judge_pass_reward_rebuttal")
async def vlm_as_judge_pass_reward_rebuttal(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    question: str = None,
    vlm_model: Optional[str] = None,
    vlm_server_ips: Optional[List[str]] = None,
    **data_fields
) -> Dict[str, float]:
    """VLM as Judge pass/fail reward for evaluating video generation based on question/prompt

    This is a simplified reward function that:
    1. Extracts Python code from the prediction
    2. Generates a video using the code
    3. Uses VLM to judge if the video correctly represents the question/prompt
    4. Returns a binary pass/fail score

    Args:
        prediction: Agent's generated response (should contain Python code)
        trajectory: Agent trajectory information
        question: The original question/prompt describing what the video should show
        **data_fields: Additional data fields from the RL data

    Returns:
        Binary pass/fail reward: 1.0 if passed, 0.0 if failed
    """
    try:
        # Log incoming data for debugging
        logger.info(f"=" * 60)
        logger.info(f"vlm_as_judge_pass_reward_rebuttal called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")
        logger.info(f"Question: {question[:200] if question else 'None'}...")

        # Initialize video generator
        video_gen = VideoGenerator()

        # Get question from parameters
        if question is None:
            question = data_fields.get("question", "")

        if not question:
            logger.warning("No question found in data")
            return {"reward": 0.0}

        # Extract code from prediction
        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in prediction")
            logger.warning(f"Prediction was: {final_response[:1000] if final_response else 'None'}")
            return {"reward": 0.0}

        logger.info(f"Extracted Python code ({len(code)} chars)")

        # Generate unique video filename
        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)

        # Generate video from code
        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": 0.0}

        model_name = _resolve_vlm_model(vlm_model, data_fields)
        server_ips = _resolve_vlm_server_ips(model_name, vlm_server_ips, data_fields)
        # Run VLM evaluation
        client = _get_vlm_client(
            model=model_name,
            timeout_seconds=120,
            server_ips=server_ips,
        )

        # Wait for client availability
        if not await _wait_for_vlm_availability(client):
            logger.error("VLM client not available")
            try:
                os.remove(video_path)
            except:
                pass
            return {"reward": 0.0}

        # Create VLM prompt for pass/fail judgment
        prompt_text = f"""You are evaluating a video that was supposed to demonstrate the following scenario:

{question}

Please watch the video carefully and determine whether it correctly demonstrates the described scenario.

Consider:
1. Does the video show the objects and elements described?
2. Does the motion/behavior match what was described?
3. Are the physical relationships and interactions correct?
4. Does the overall visualization accurately represent the scenario?

**Important**: Focus on the core physical behavior and motion. Minor visual simplifications (like using 2D views of 3D objects, or symbolic representations) are acceptable as long as the fundamental physics and motion are correct.

Provide your judgment as a JSON object with:
- "passed": true or false (whether the video correctly demonstrates the scenario)
- "reasoning": your detailed reasoning for the judgment
- "confidence": 1-5 (how confident you are in your judgment)

Return only the JSON object, no additional text."""

        # Build message using <video> tag
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [{
            "role": "user",
            "content": user_text
        }]

        # Process the request
        responses = await client.process_all_inputs(
            [messages],
            model=model_name,
            temperature=0,
        )

        if not responses or not responses[0] or not responses[0][0]:
            logger.error("No response from VLM server")
            try:
                os.remove(video_path)
            except:
                pass
            return {"reward": 0.0}

        output_text = responses[0][0]

        try:
            # Parse VLM result
            # Try to extract JSON from the response
            result = None
            try:
                result = json.loads(output_text)
            except:
                # Try to find JSON in the text
                start = output_text.find('{')
                end = output_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    result = json.loads(output_text[start:end+1])

            if not result:
                logger.error("Failed to parse VLM result")
                logger.error(f"VLM output: {output_text}")
                try:
                    os.remove(video_path)
                except:
                    pass
                return {"reward": 0.0}

            # Extract pass/fail decision
            passed = result.get("passed", False)
            reasoning = result.get("reasoning", "")
            confidence = result.get("confidence", 1)

            # Calculate reward: 1.0 if passed, 0.0 if failed
            reward_score = 1.0 if passed else 0.0

            logger.info(f"VLM judgment: {'PASSED' if passed else 'FAILED'}")
            logger.info(f"Reasoning: {reasoning}")
            logger.info(f"Confidence: {confidence}/5")
            logger.info(f"Reward: {reward_score}")

            # Clean up video file
            try:
                os.remove(video_path)
            except:
                pass

            return {"reward": reward_score}

        except Exception as e:
            logger.error(f"Error processing VLM result: {e}")
            import traceback
            traceback.print_exc()
            try:
                os.remove(video_path)
            except:
                pass
            return {"reward": 0.0}

    except Exception as e:
        logger.error(f"Error in vlm_as_judge_pass_reward_rebuttal: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": 0.0}


@reward(name="vlm_as_judge_reward")
async def vlm_as_judge_reward(
    final_response: str,
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
    vlm_model: Optional[str] = None,
    vlm_server_ips: Optional[List[str]] = None,
    **data_fields
) -> Dict[str, float]:
    """VLM as Judge reward function for evaluating agent trajectories
    
    This reward function:
    1. Extracts Python code from the prediction 
    2. Generates a video using the code
    3. Uses VLM server to evaluate the video against provided questions
    4. Returns a weighted score based on VLM judgments
    
    Args:
        prediction: Agent's generated response (should contain Python code)
        trajectory: Agent trajectory information
        **data_fields: Additional data fields from the RL data, including vlm_questions
        
    Returns:
        Weighted reward score between 0.0 and 1.0
    """
    try:
        # Log incoming data for debugging
        logger.info(f"=" * 60)
        logger.info(f"vlm_as_judge_reward called")
        logger.info(f"Prediction length: {len(final_response) if final_response else 0}")
        
        # Print the actual prediction content
        logger.info(f"Prediction content (first 500 chars):")
        logger.info(f"{final_response[:500] if final_response else 'No prediction'}")
        if final_response and len(final_response) > 500:
            logger.info(f"... (truncated, total length: {len(final_response)} chars)")
        
        logger.info(f"vlm_questions parameter: {vlm_questions is not None}")
        logger.info(f"Additional data_fields keys: {list(data_fields.keys())}")
        
        # Initialize video generator
        video_gen = VideoGenerator()
        
        # Combine vlm_questions with data_fields for extraction
        all_data = dict(data_fields)
        if vlm_questions is not None:
            all_data['vlm_questions'] = vlm_questions
            logger.info(f"vlm_questions type: {type(vlm_questions)}")
            if isinstance(vlm_questions, dict):
                logger.info(f"vlm_questions keys: {vlm_questions.keys()}")
                if 'vlm_questions' in vlm_questions:
                    inner_vlm = vlm_questions['vlm_questions']
                    logger.info(f"Inner vlm_questions type: {type(inner_vlm)}")
                    if isinstance(inner_vlm, list):
                        logger.info(f"Number of questions in inner list: {len(inner_vlm)}")
        
        # Extract VLM questions from data
        all_questions, summarize, questions_list = extract_vlm_questions_from_data(all_data)
        
        if not questions_list:
            logger.warning(f"No VLM questions found in data. Available fields: {list(all_data.keys())}")
            return {"reward": 0.0}
        
        # Extract code from prediction
        code = video_gen.extract_code_from_response(final_response)
        if not code:
            logger.warning("No Python code found in prediction")
            logger.warning(f"Prediction was: {final_response[:1000] if final_response else 'None'}")
            return {"reward": 0.0}
        
        logger.info(f"Extracted Python code ({len(code)} chars)")
        logger.info(f"Code preview (first 300 chars):")
        logger.info(f"{code[:300]}...")
        if len(code) > 300:
            logger.info(f"... (truncated, total length: {len(code)} chars)")
        
        # Generate unique video filename
        video_filename = f"video_{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(video_gen.output_dir, video_filename)
        
        # Generate video from code
        success = await video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": 0.0}
        
        model_name = _resolve_vlm_model(vlm_model, data_fields)
        server_ips = _resolve_vlm_server_ips(model_name, vlm_server_ips, data_fields)
        # Run VLM evaluation directly since we're already async
        client = _get_vlm_client(
            model=model_name,
            timeout_seconds=120,
            server_ips=server_ips,
        )
        
        # Wait for client availability
        if not await _wait_for_vlm_availability(client):
            logger.error("VLM client not available")
            return {"reward": 0.0}
        
        # Create VLM prompt
        prompt_text = create_vlm_prompt(summarize, all_questions)
        
        # Build message using <video> tag in text content to match server expectations
        user_text = f"<video>{video_path}</video>\n\n{prompt_text}"
        messages = [{
            "role": "user",
            "content": user_text
        }]
        
        # Process the request
        responses = await client.process_all_inputs(
            [messages],
            model=model_name,
            temperature=0,
        )
        
        if not responses or not responses[0] or not responses[0][0]:
            logger.error("No response from VLM server")
            # Clean up video file
            try:
                os.remove(video_path)
            except:
                pass
            return {"reward": 0.0}
        
        output_text = responses[0][0]
        
        try:
            # Parse VLM results
            results = _extract_json_list(output_text)
            if not results:
                logger.error("Failed to parse VLM results")
                # Clean up video file
                try:
                    os.remove(video_path)
                except:
                    pass
                return {"reward": 0.0}
            
            # Calculate weighted reward
            reward_score = calculate_weighted_reward(results, questions_list)
            
            logger.info(f"VLM evaluation completed. Reward: {reward_score:.3f}")
            
            # Clean up video file
            try:
                os.remove(video_path)
            except:
                pass
            
            return {"reward": reward_score}
            
        except Exception as e:
            logger.error(f"Error processing VLM results: {e}")
            # Clean up video file
            try:
                os.remove(video_path)
            except:
                pass
            return {"reward": 0.0}
        
    except Exception as e:
        logger.error(f"Error in vlm_as_judge_reward: {e}")
        import traceback
        traceback.print_exc()
        return {"reward": 0.0}
