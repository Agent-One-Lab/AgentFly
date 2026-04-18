"""Video generation helpers for VLM-as-judge rewards."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from model response."""
    if not response:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, cleaned, re.DOTALL)
    if matches:
        return matches[0]
    return None


class VideoGenerator:
    """Helper class to generate videos from code."""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            output_dir = os.getenv(
                "VLM_SHARED_VIDEO_DIR",
                "./tmp/vlm_judge",
            )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_code_from_response(self, response: str) -> Optional[str]:
        return extract_code_from_response(response)

    async def generate_video_from_code(self, code: str, output_path: str) -> bool:
        """Execute Python code to generate video (async version)."""
        temp_file = None
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path_abs = os.path.abspath(output_path)
            output_parent = os.path.dirname(output_path_abs)
            if output_parent:
                os.makedirs(output_parent, exist_ok=True)
            logger.debug(
                "Video generation start: output_dir=%s output_path=%s code_len=%d",
                self.output_dir,
                output_path_abs,
                len(code or ""),
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                modified_code = code
                rewrite_mode = "none"
                if "sys.argv[1]" in code:
                    modified_code = code.replace("sys.argv[1]", f'"{output_path_abs}"')
                    rewrite_mode = "replace_sys_argv_1"
                elif "sys.argv" in code and "len(sys.argv)" in code:
                    modified_code = (
                        f"import sys\nsys.argv = ['script.py', '{output_path_abs}']\n" + code
                    )
                    rewrite_mode = "inject_sys_argv"
                else:
                    if "output_file" in code:
                        modified_code = re.sub(
                            r'output_file\s*=\s*["\'].*?["\']',
                            f'output_file = "{output_path_abs}"',
                            code,
                        )
                        rewrite_mode = "replace_output_file"
                    elif "VideoWriter(" in code:
                        modified_code = re.sub(
                            r'VideoWriter\s*\(\s*["\'].*?["\']',
                            f'VideoWriter("{output_path_abs}"',
                            code,
                        )
                        rewrite_mode = "replace_videowriter_path"
                    else:
                        modified_code = f"output_file = '{output_path_abs}'\n" + code
                        rewrite_mode = "prepend_output_file"

                f.write(modified_code)
                temp_file = f.name
            logger.debug(
                "Video generation script prepared: temp_file=%s rewrite_mode=%s",
                temp_file,
                rewrite_mode,
            )
            logger.debug("Video generation script preview:\n%s", modified_code[:600])

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_file,
                output_path_abs,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.output_dir,
            )
            logger.debug(
                "Spawned video generation process: python=%s temp_file=%s arg_output=%s cwd=%s pid=%s",
                sys.executable,
                temp_file,
                output_path_abs,
                self.output_dir,
                process.pid,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                logger.error("Video generation timed out")
                process.kill()
                await process.wait()
                if temp_file:
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass
                return False

            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            logger.debug(
                "Video generation process finished: returncode=%s stdout_len=%d stderr_len=%d",
                process.returncode,
                len(stdout_text),
                len(stderr_text),
            )
            if stdout_text.strip():
                logger.debug("Video generation stdout:\n%s", stdout_text)
            if stderr_text.strip():
                logger.debug("Video generation stderr:\n%s", stderr_text)

            if temp_file:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

            exists = os.path.exists(output_path_abs)
            size = os.path.getsize(output_path_abs) if exists else -1
            logger.debug(
                "Video generation output check: exists=%s size=%s threshold=1000 path=%s",
                exists,
                size,
                output_path_abs,
            )
            if exists and size > 1000:
                logger.debug(
                    "Successfully generated video: %s (%s bytes)",
                    output_path_abs,
                    size,
                )
                return True

            logger.error(
                "Video generation failed or file too small. returncode=%s exists=%s size=%s stderr: %s",
                process.returncode,
                exists,
                size,
                stderr_text,
            )
            return False

        except Exception as e:
            if isinstance(e, FileNotFoundError):
                logger.error(
                    "Error generating video: %s (missing: %s)",
                    e,
                    getattr(e, "filename", None),
                )
            else:
                logger.error("Error generating video: %s", e)
            if temp_file:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            return False


def can_open_video(video_path: str) -> bool:
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
