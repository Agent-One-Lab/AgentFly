import os
import re
import json
import glob
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional

from pathlib import Path
from openai import AsyncOpenAI
import sys
from tqdm.asyncio import tqdm_asyncio

from ... import AGENT_HOME

logger = logging.getLogger(__name__)


def _get_retry_settings() -> tuple[int, float]:
    retry_seconds_env = os.getenv("VLM_CLIENT_RETRY_SECONDS")
    if retry_seconds_env is None:
        retry_seconds_env = os.getenv("VLM_CLIENT_WAIT_SECONDS")

    retry_interval_env = os.getenv("VLM_CLIENT_RETRY_INTERVAL")
    if retry_interval_env is None:
        retry_interval_env = os.getenv("VLM_CLIENT_POLL_INTERVAL")

    try:
        retry_seconds = int(retry_seconds_env) if retry_seconds_env is not None else -1
    except (TypeError, ValueError):
        retry_seconds = -1

    try:
        retry_interval = float(retry_interval_env) if retry_interval_env is not None else 1.0
    except (TypeError, ValueError):
        retry_interval = 1.0

    if retry_seconds < 0:
        retry_seconds = -1
    if retry_interval <= 0:
        retry_interval = 1.0

    return retry_seconds, retry_interval


def _resolve_server_status_dir() -> str:
    """Resolve the directory that contains vLLM server status JSON files.

    Priority:
    1) env `VLLM_SERVER_STATUS_DIR`
    2) env `DATA_PROCESS_HOME` + /vllm_server/server_status
    3) AGENT_HOME + /data-process/vllm_server/server_status
    4) Explicit fallback to /mnt/weka/... path as provided by user
    """
    # 1) Explicit override
    override = os.getenv("VLLM_SERVER_STATUS_DIR")
    if override and os.path.isdir(override):
        return override

    # 2) data-process home
    dp_home = os.getenv("DATA_PROCESS_HOME")
    if dp_home:
        candidate = os.path.join(dp_home, "vllm_server", "server_status")
        if os.path.isdir(candidate):
            return candidate

    # 3) default relative to AGENT_HOME
    candidate = os.path.join(AGENT_HOME, "data-process", "vllm_server", "server_status")
    if os.path.isdir(candidate):
        return candidate

    # Last resort: return the AGENT_HOME-based path even if missing (caller errors out)
    return candidate


def get_server_ips(model: str) -> List[str]:
    """Get list of server IPs from the most recent complete server instances file for a specific model."""
    server_status_dir = _resolve_server_status_dir()

    # Clean model name for filename matching (replace / and - with _)
    model_clean = model.replace('/', '_').replace('-', '_')
    
    # Try multiple patterns to match the server files
    patterns = [
        f"server_instances_complete_{model_clean}_*.json",
        f"server_instances_complete_vllm_{model_clean}_*.json",
        f"server_instances_complete_*{model_clean}*.json"
    ]
    
    json_files = []
    for pattern in patterns:
        search_pattern = os.path.join(server_status_dir, pattern)
        found_files = glob.glob(search_pattern)
        if found_files:
            json_files = found_files
            break

    if not json_files:
        # Fallback: try to find any server instances file and filter by model in the JSON content
        fallback_pattern = os.path.join(server_status_dir, "server_instances_complete_*.json")
        all_files = glob.glob(fallback_pattern)

        for file in all_files:
            try:
                with open(file, 'r') as f:
                    server_info = json.load(f)

                # Check if any server in this file matches our model
                matching_servers = [info for info in server_info if info.get('model') == model]
                if matching_servers:
                    json_files = [file]
                    logger.info(f"Found servers for model '{model}' in fallback file: {file}")
                    break
            except Exception as e:
                logger.warning(f"Error reading file {file}: {e}")
                continue

        if not json_files:
            raise RuntimeError(
                f"No server instances file found for model '{model}' in {search_pattern} or any fallback file under {server_status_dir}"
            )

    # Get the most recent file
    latest_file = max(json_files, key=os.path.getctime)

    with open(latest_file, 'r') as f:
        server_info = json.load(f)

    # Filter servers by model and extract IPs
    ips = []
    for info in server_info:
        if info.get('model') == model and 'ip' in info:
            ips.append(info['ip'])

    if not ips:
        raise RuntimeError(f"No IPs found for model '{model}' in server instances file {latest_file}")

    logger.info(f"Found {len(ips)} server instances for model '{model}': {ips}")
    return ips


class RateLimiter:
    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.semaphore = asyncio.Semaphore(max_window_size)

    async def acquire(self):
        await self.semaphore.acquire()

    async def release(self):
        self.semaphore.release()


class RoundRobinClient:
    def __init__(self, ips: List[str], port: int, api_key: str, timeout: int, rate_limiters: List[RateLimiter]):
        self.ips = ips
        self.current_index = 0
        self.port = port
        self.api_key = api_key
        self.clients = [
            AsyncOpenAI(
                base_url=f"http://{ip}:{port}/v1",
                api_key=api_key,
                timeout=timeout,
            ) for ip in ips
        ]
        self.rate_limiters = rate_limiters

    async def get_next_available_client(self) -> tuple[AsyncOpenAI, RateLimiter]:
        # Find the instance with the most available slots
        max_available = -1
        best_client = None
        best_limiter = None
        best_index = -1

        for i in range(len(self.clients)):
            available = self.rate_limiters[i].semaphore._value
            if available > max_available and available > 0:
                max_available = available
                best_client = self.clients[i]
                best_limiter = self.rate_limiters[i]
                best_index = i

        if best_client is not None:
            await best_limiter.acquire()
            return best_client, best_limiter

        # If no instance has available slots, wait on all and race
        wait_tasks = [(i, asyncio.create_task(limiter.semaphore.acquire()))
                      for i, limiter in enumerate(self.rate_limiters)]
        done, pending = await asyncio.wait(
            [task for _, task in wait_tasks],
            return_when=asyncio.FIRST_COMPLETED
        )

        for _, task in wait_tasks:
            if task not in done:
                task.cancel()

        for i, task in wait_tasks:
            if task in done:
                return self.clients[i], self.rate_limiters[i]

        raise RuntimeError("No instance became available despite wait completion")


class VLMClient:
    def __init__(self,
                 model: str,
                 timeout_seconds: int = 60,
                 max_window_size_per_instance: int = 10,
                 port: int = 8000,
                 api_key: str = "token-abc123",
                 server_ips: Optional[List[str]] = None):
        self.timeout_seconds = timeout_seconds
        # Prefer explicit server IPs when provided; otherwise keep existing default.
        # server_ips = get_server_ips(model)
        if server_ips is None:
            server_ips = ["10.24.3.228"]
        rate_limiters = [RateLimiter(max_window_size_per_instance) for _ in server_ips]
        self.client_manager = RoundRobinClient(server_ips, port, api_key, timeout_seconds, rate_limiters)

    async def single_call(self, inputs, model, **kwargs):
        retry_seconds, retry_interval = _get_retry_settings()
        start_time = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                client, rate_limiter = await self.client_manager.get_next_available_client()
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=inputs,
                        timeout=self.timeout_seconds,
                        **kwargs
                    )
                    return response.choices[0].message.content
                finally:
                    await rate_limiter.release()
            except asyncio.TimeoutError:
                logger.error(
                    "Request timed out after %ss (attempt %d)",
                    self.timeout_seconds,
                    attempt,
                )
            except Exception as e:
                logger.error("Error processing request (attempt %d): %s", attempt, e)

            if retry_seconds == 0:
                return None

            if retry_seconds > 0 and (time.monotonic() - start_time) >= retry_seconds:
                return None

            await asyncio.sleep(retry_interval)

    async def process_all_inputs(self, inputs_list, num_generations=1, model=None, **kwargs):
        all_tasks = []
        for inputs in inputs_list:
            for _ in range(num_generations):
                all_tasks.append(self.single_call(inputs, model=model, **kwargs))

        responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing VLM requests", file=sys.stdout)

        grouped_responses = []
        for i in range(0, len(responses), num_generations):
            grouped_responses.append(responses[i:i + num_generations])

        return grouped_responses

    def check_availability(self) -> Dict[str, Any]:
        # Mirror llm client basic stats
        total_capacity = 0
        total_available = 0
        total_used = 0
        instance_details = []

        for i in range(len(self.client_manager.clients)):
            available = self.client_manager.rate_limiters[i].semaphore._value
            capacity = self.client_manager.rate_limiters[i].max_window_size
            used = capacity - available

            total_capacity += capacity
            total_available += available
            total_used += used

            instance_details.append({
                'instance_id': i,
                'ip': self.client_manager.ips[i],
                'port': self.client_manager.port,
                'available_slots': available,
                'used_slots': used,
                'total_slots': capacity,
            })

        return {
            'total_instances': len(self.client_manager.clients),
            'total_capacity': total_capacity,
            'total_available': total_available,
            'total_used': total_used,
            'has_available_slots': total_available > 0,
            'instances': instance_details
        }

    def is_available(self) -> bool:
        availability = self.check_availability()
        return availability['has_available_slots']


DEFAULT_VLM_PROMPT_TEMPLATE = """You are given a set of visual verification questions and a description of objects and motion observed in an input medium (e.g., image or video).

Your task is to **evaluate each question** based on whether it is **correctly reflected in the visual content**, considering visual cues, shape changes from viewpoint, and possible symbolic representations.


---

 **Visual Reasoning Guidelines**:

1. **Perspective Awareness**:  
   Objects may appear different based on viewpoint. For example:
   - A **cylinder** may look like a **circle (top view)** or a **rectangle/square (side view)**.
   - A **circular path** may appear as a **wave-like curve or straight line** in 2D projection.

2. **Symbolic Representations**:  
   Common simplifications may be used. You should **reasonably infer** their meaning:
   - A series of **dots or circles** may represent **foam markers** or control points.
   - A **rectangle** may represent a **container** (e.g., cylindrical viewed from the side).
   - A **line** may represent a **rubber mat** or constraint boundary.
   - The object and track specifics might do not match directly, if the motion can be interpreted correctly, it is still true.
   - It might use color to represent different objects, such as a green line to represent the flat surface is covered with a felt-like material.
   - The rotation of the object might cannot be judged from the video, but the motion can be interpreted correctly, it is still true.

3. **Container Boundaries**:
   - If **no container is drawn**, you may assume the **video frame itself is the container boundary**.
   - If a **container is visible**, treat it as **transparent** if inner content is visible.
   - If the object is not visible, you should not assume it is in the container.

4. **Focus on Shape & Position**, **not material**:
   - Ignore assumptions about object **material**, **color**, or **texture**.
   - Base your decisions entirely on **observable geometry** (e.g., shape, layout, structure) and **motion** (e.g., direction, trajectory).
   - Use visible movement and positioning to judge truthfulness — even if the object type is unknown.
   - If the described motion is **sliding down a slope**, but the video shows an **upward movement**, the result should be `"False"` — regardless of material or appearance.
   - Make geometric and motion-based reasoning the core of your judgment, even when objects are **partially occluded**.

5. **Occlusion Handling**:
   - If an object is **partially blocked**, assess based on surrounding evidence whether its state or motion can still be inferred.

6. **Avoid excessive uncertainty**:
   - If there is enough visual context and logical structure, make a **confident judgment**.
   - Use "Not sure" only when the evidence is **truly insufficient or ambiguous**.

---

 **Input**:
- Questions: {all_questions}
- Object and motion description: {summarize}

---

 **For each question**, return:
- `"index"`: the question index
- `"question"`: the full question text
- `"analysis"`: your reasoning process and visual inference
- `"result"`: one of `"True"`, `"False"`, or `"Not sure"`
- `"confidence_score"`: an integer from 1 (very uncertain) to 5 (very certain)

---

**Output Format**:
Return a JSON list like this:
[
    {{
        "index": "1",
        "question": "The ball rolls along the circular path.",
        "analysis": "The object follows a closed curve consistent with a circular path from the top view.",
        "result": "True",
        "confidence_score": "5"
    }},
    ...
]
"""


def _format_keywords(text: str,
                     keywords: Optional[List[str]] = None,
                     style: str = "bold",
                     case_sensitive: bool = False) -> str:
    """Format given keywords in text.

    Args:
        text: The input text to process.
        keywords: List of keywords to highlight/format.
        style: One of "bold" (default), "bracket", or "caps".
        case_sensitive: Whether to match case-sensitively.

    Returns:
        Formatted text with keywords highlighted.
    """
    if not keywords:
        return text

    # Deduplicate and sort by length (longest first) to avoid partial overlapping matches
    unique_keywords = [k for k in sorted(set(keywords), key=lambda s: len(s or ""), reverse=True) if k]
    if not unique_keywords:
        return text

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile("|".join(re.escape(k) for k in unique_keywords), flags)

    def repl(match: re.Match) -> str:
        start, end = match.start(), match.end()
        # Avoid double-formatting if already bolded ("**word**")
        if style == "bold":
            prev2 = text[max(0, start - 2):start]
            next2 = text[end:end + 2]
            if prev2 == "**" and next2 == "**":
                return match.group(0)
            return f"**{match.group(0)}**"
        elif style == "bracket":
            return f"[{match.group(0)}]"
        elif style == "caps":
            return match.group(0).upper()
        else:
            return f"**{match.group(0)}**"

    return pattern.sub(repl, text)


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def create_vlm_prompt_from_template(
    prompt_template: str,
    variables: Optional[Dict[str, Any]] = None,
    keywords: Optional[List[str]] = None,
    style: str = "bold",
    case_sensitive: bool = False,
) -> str:
    """Create a VLM prompt from a template with optional keyword formatting.

    Args:
        prompt_template: Template string containing placeholders like {summarize}, {all_questions}.
        variables: Mapping used to format the template.
        keywords: Keywords to highlight after formatting.
        style: Keyword highlight style: "bold" (default), "bracket", or "caps".
        case_sensitive: Whether keyword matching is case-sensitive.

    Returns:
        Final prompt string.
    """
    text = prompt_template
    if variables:
        try:
            text = prompt_template.format_map(_SafeDict(variables))
        except Exception:
            # Fall back to the original template if formatting fails
            text = prompt_template
    return _format_keywords(text, keywords=keywords, style=style, case_sensitive=case_sensitive)


def create_vlm_prompt_custom(
    prompt: str,
    keywords: Optional[List[str]] = None,
    style: str = "bold",
    case_sensitive: bool = False,
) -> str:
    """Create a VLM prompt from a raw prompt string and optional keywords.

    This function does not inject any default template; it only formats
    the provided prompt and highlights keywords as requested.

    Args:
        prompt: The prompt content to send to the VLM.
        keywords: List of keywords to format/highlight.
        style: Highlight style: "bold" (default), "bracket", or "caps".
        case_sensitive: Whether keyword matching is case-sensitive.

    Returns:
        Final prompt string.
    """
    return _format_keywords(prompt, keywords=keywords, style=style, case_sensitive=case_sensitive)


def create_vlm_prompt(summarize: str, all_questions: str) -> str:
    """Create the default VLM prompt (backward-compatible).

    Existing call sites expect (summarize, all_questions). This delegates to
    a template-based builder to enable future customization.
    """
    return create_vlm_prompt_from_template(
        DEFAULT_VLM_PROMPT_TEMPLATE,
        variables={"summarize": summarize, "all_questions": all_questions},
    )


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
