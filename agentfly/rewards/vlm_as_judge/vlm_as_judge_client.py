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
from tqdm.asyncio import tqdm_asyncio

from ... import AGENT_HOME

logger = logging.getLogger(__name__)


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

    # 4) user-provided absolute path - CORRECTED PATH
    hardcoded = "/mnt/weka/home/renxi.wang/yxwang/data-process/src/server_status"
    if os.path.isdir(hardcoded):
        return hardcoded

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
                 api_key: str = "token-abc123"):
        self.timeout_seconds = timeout_seconds
        # server_ips = get_server_ips(model)
        server_ips = ["10.24.3.24"]
        rate_limiters = [RateLimiter(max_window_size_per_instance) for _ in server_ips]
        self.client_manager = RoundRobinClient(server_ips, port, api_key, timeout_seconds, rate_limiters)

    async def single_call(self, inputs, model, **kwargs):
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
            logger.error(f"Request timed out after {self.timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return None

    async def process_all_inputs(self, inputs_list, num_generations=1, model=None, **kwargs):
        all_tasks = []
        for inputs in inputs_list:
            for _ in range(num_generations):
                all_tasks.append(self.single_call(inputs, model=model, **kwargs))

        responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing VLM requests")

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


def create_vlm_prompt(summarize: str, all_questions: str) -> str:
    return f"""You are given a set of visual verification questions and a description of objects and motion observed in an input medium (e.g., image or video).

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


def _extract_json_list(output_str: str) -> List[Dict[str, Any]]:
    """Extract the VLM JSON list from the model output.

    Tries strict JSON first; falls back to extracting the substring between
    the first '[' and the last ']' and parsing that.
    """
    try:
        parsed = json.loads(output_str)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: try to extract JSON list portion
    start = output_str.find('[')
    end = output_str.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(output_str[start:end+1])
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise ValueError("Failed to parse VLM JSON list from output")


async def run_vlm_as_judge(
    *,
    question: str,
    prediction: str,
    video_path: str,
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    fps: int = 5,
    summarize: Optional[str] = None,
    resized_width: Optional[int] = None,
    resized_height: Optional[int] = None,
    temperature: float = 0.0,
    timeout_seconds: int = 120,
) -> float:
    """Run VLM-as-judge once and return a binary score 0.0/1.0.

    Args:
        question: The question to evaluate against the video.
        prediction: The proposed answer to be judged.
        video_path: Absolute path accessible by the vLLM server(s).
        model: VLM model name registered by the server.
        fps: Frame sampling rate for the server-side video loader.
        resized_width/resized_height: Optional hints; if None, omitted.
        temperature: Generation temperature (kept low for judging determinism).
        timeout_seconds: Request timeout.
    Returns:
        1.0 if correct; 0.0 otherwise.
    """
    client = VLMClient(model=model, timeout_seconds=timeout_seconds)

    # Best-effort small wait for capacity
    for _ in range(10):
        if client.is_available():
            break
        time.sleep(1)

    # Align with get_vlm_result.py prompt: single-question variant
    if not summarize:
        summarize = "No additional description provided. Base your judgment solely on the video."
    all_questions = f"1. {question} (Proposed answer: {prediction})"
    text = create_vlm_prompt(summarize, all_questions)

    # Build message using <video> tag embedded in text content
    # This aligns with the server's parsing logic observed in tests.
    video_tag = f"<video>{video_path}</video>"
    # Note: resized_width/height and fps are not standard in text tag; server-side
    # may ignore or infer these. If needed, extend the tag format later.
    user_text = f"{video_tag}\n\n{text}"

    messages = [{
        "role": "user",
        "content": user_text,
    }]

    responses = await client.process_all_inputs([messages], model=model, temperature=temperature)

    if responses and responses[0] and responses[0][0]:
        output_text = responses[0][0]
        try:
            parsed_list = _extract_json_list(output_text)
            if not parsed_list:
                return 0.0
            # Expect first item corresponds to our only question
            first = parsed_list[0]
            result = str(first.get("result", "")).strip().lower()
            if result == "true":
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error parsing VLM output: {e}")
            return 0.0
    return 0.0


if __name__ == "__main__":
    """Test VLM client functionality"""
    import asyncio
    
    async def test_client():
        print("="*70)
        print("Testing VLM Client")
        print("="*70)
        
        # Test data
        test_questions = {
            "vlm_questions": {
                "summarize": "A ball rolls down a ramp",
                "vlm_questions": [
                    {"index": "1", "question": "A ball is visible", "weight": 1.0},
                    {"index": "2", "question": "The ball moves downward", "weight": 1.0}
                ]
            }
        }
        
        try:
            # Test client initialization
            client = VLMClient(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                timeout_seconds=60
            )
            print(f"✓ Client initialized")
            
            # Check availability
            is_available = client.is_available()
            print(f"✓ Client available: {is_available}")
            
            # Test prompt creation
            all_q = "1. A ball is visible\n2. The ball moves downward"
            prompt = create_vlm_prompt("A ball rolls down a ramp", all_q)
            print(f"✓ Prompt created ({len(prompt)} chars)")
            
            # Test JSON extraction
            test_response = '''[{"index": "1", "result": "True", "confidence_score": "5"}]'''
            results = _extract_json_list(test_response)
            print(f"✓ JSON extraction works: {len(results)} results")
            
            print("\nAll tests passed!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_client())
