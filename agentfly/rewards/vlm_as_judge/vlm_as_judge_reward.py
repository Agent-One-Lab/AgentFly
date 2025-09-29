"""VLM as Judge Reward Function for AgentFly RL Training"""

import os
import re
import json
import uuid
import tempfile
import subprocess
import asyncio
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Support running both as a package module and as a standalone script
try:
    from ..reward_base import reward
    from .vlm_as_judge_client import VLMClient, create_vlm_prompt, _extract_json_list
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
    
    def generate_video_from_code(self, code: str, output_path: str) -> bool:
        """Execute Python code to generate video
        
        Args:
            code: Python code to execute
            output_path: Path to save the generated video
            
        Returns:
            True if video generation successful, False otherwise
        """
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
            
            # Execute the code
            # Always pass the output path as an argument so scripts that expect
            # sys.argv[1] or check len(sys.argv) continue without exiting.
            result = subprocess.run(
                ['python', temp_file, output_path],
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for video generation
                cwd=self.output_dir  # Run in output directory
            )
            
            # Clean up temp file
            os.unlink(temp_file)
            
            # Check if video was created and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
                logger.info(f"Successfully generated video: {output_path} ({os.path.getsize(output_path)} bytes)")
                return True
            else:
                logger.error(f"Video generation failed or file too small. stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Video generation timed out")
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return False
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
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
        result_value = result.get("result", "Not sure")
        confidence = int(result.get("confidence_score", "1"))
        
        # Get weight for this question
        weight = weight_map.get(idx, 1.0)
        
        # Calculate score based on result
        if result_value == "True":
            score = 1.0
        elif result_value == "False":
            score = 0.0
        else:  # "Not sure"
            if confidence >= 4:
                score = 0.0  # High confidence "Not sure" -> False
            else:
                score = 1.0  # Low confidence "Not sure" -> True
        
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

    Returns 1.0 only when every question is judged as satisfied (or low-confidence
    "Not sure"), otherwise returns 0.0.
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

        result_value = str(result.get("result", "Not sure")).strip().lower()
        confidence_raw = result.get("confidence_score", "1")
        try:
            confidence = int(confidence_raw)
        except (TypeError, ValueError):
            confidence = 1

        if result_value == "true":
            continue
        if result_value == "not sure" and confidence < 4:
            continue

        # Any explicit false or high-confidence uncertainty causes failure
        return 0.0

    return 1.0

@reward(name="vlm_as_judge_pass_reward")
async def vlm_as_judge_pass_reward(
    prediction: str, 
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
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
            logger.info(f"Prediction length: {len(prediction) if prediction else 0}")
            
            # Print the actual prediction content
            logger.info(f"Prediction content (first 500 chars):")
            logger.info(f"{prediction[:500] if prediction else 'No prediction'}")
            if prediction and len(prediction) > 500:
                logger.info(f"... (truncated, total length: {len(prediction)} chars)")
            
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
            code = video_gen.extract_code_from_response(prediction)
            if not code:
                logger.warning("No Python code found in prediction")
                logger.warning(f"Prediction was: {prediction[:1000] if prediction else 'None'}")
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
            success = video_gen.generate_video_from_code(code, video_path)
            if not success:
                logger.error("Failed to generate video from code")
                return {"reward": 0.0}
            
            # Run VLM evaluation directly since we're already async
            client = VLMClient(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                timeout_seconds=120
            )
            
            # Wait for client availability
            for _ in range(10):
                if client.is_available():
                    break
                await asyncio.sleep(1)
            else:
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
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                temperature=0.1
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
    

@reward(name="vlm_as_judge_reward")
async def vlm_as_judge_reward(
    prediction: str, 
    trajectory: Dict[str, Any] = None,
    vlm_questions: Dict[str, Any] = None,
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
        logger.info(f"Prediction length: {len(prediction) if prediction else 0}")
        
        # Print the actual prediction content
        logger.info(f"Prediction content (first 500 chars):")
        logger.info(f"{prediction[:500] if prediction else 'No prediction'}")
        if prediction and len(prediction) > 500:
            logger.info(f"... (truncated, total length: {len(prediction)} chars)")
        
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
        code = video_gen.extract_code_from_response(prediction)
        if not code:
            logger.warning("No Python code found in prediction")
            logger.warning(f"Prediction was: {prediction[:1000] if prediction else 'None'}")
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
        success = video_gen.generate_video_from_code(code, video_path)
        if not success:
            logger.error("Failed to generate video from code")
            return {"reward": 0.0}
        
        # Run VLM evaluation directly since we're already async
        client = VLMClient(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            timeout_seconds=120
        )
        
        # Wait for client availability
        for _ in range(10):
            if client.is_available():
                break
            await asyncio.sleep(1)
        else:
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
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            temperature=0.1
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
