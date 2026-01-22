import logging
import random
import re

from ..utils.json import jsonish
from ..agent_base import BaseAgent
from typing import Dict, List
import json
from PIL import Image
from ...tools import tool
from ...utils.vision import image_to_data_uri, image_to_pil, open_image_from_any
import torch

logger = logging.getLogger(__name__)


IMAGE_AGENT_SYSTEM_PROMPT = """You are an ImageEditingAgent, a powerful AI assistant specialized in image editing and manipulation tasks.

Always provide clear, step-by-step instructions and call the appropriate tools for each task. If you have finished the task, describe what you have seen in the final image."""


class QwenImageEditTool:
    """
    Qwen-Image-Edit tool for instruction-based image editing.
    Based on https://huggingface.co/Qwen/Qwen-Image-Edit
    """

    def __init__(self, model_id="Qwen/Qwen-Image-Edit", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = None

    def _lazy_init(self):
        from diffusers import QwenImageEditPipeline

        """Lazy initialization of the pipeline to save memory."""
        if self._pipeline is None:
            print(f"INFO: Loading Qwen-Image-Edit pipeline on {self.device}...")
            self._pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            )
            self._pipeline.to(self.device)
            self._pipeline.set_progress_bar_config(disable=None)
            print("INFO: Qwen-Image-Edit pipeline loaded successfully!")

    def apply(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 50,
        seed: int = 0,
    ) -> Image.Image:
        """
        Applies the image editing based on the given prompt.

        Args:
            image: Input PIL Image
            prompt: Text prompt for editing
            negative_prompt: Negative prompt (default: " ")
            true_cfg_scale: CFG scale (default: 4.0)
            num_inference_steps: Number of inference steps (default: 50)
            seed: Random seed (default: 0)

        Returns:
            Edited PIL Image
        """
        self._lazy_init()

        inputs = {
            "image": image.convert("RGB"),
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }

        with torch.inference_mode():
            output = self._pipeline(**inputs)
            edited_image = output.images[0]

        return edited_image


def extract_tool_calls(action_input: str) -> List[Dict]:
    if action_input is None:
        return []

    tool_call_str = ""
    # Extract the tool call from the action input
    # 1. Extract with qwen style
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    m = pattern.search(action_input)
    # If we find a tool call, extract it
    if m:
        tool_call_str = m.group(1).strip()
        try:
            tool_call = jsonish(tool_call_str)
            return [tool_call]
        except Exception:
            pass

    # 2. Extract directly
    try:
        tool_call = jsonish(action_input)
        return [tool_call]
    except Exception:
        pass

    return []


def geneate_image_id(image: Image.Image) -> str:
    image_id = random.randint(0, 999999)
    return str(image_id)


class ImageEditingAgent(BaseAgent):
    def __init__(
        self,
        model_name_or_path: str,
        system_prompt: str = IMAGE_AGENT_SYSTEM_PROMPT,
        **kwargs,
    ):
        self._image_database = {}
        self._qwen_image_edit_tool = QwenImageEditTool()
        tools = [self.qwen_image_edit_tool]

        super().__init__(
            model_name_or_path=model_name_or_path,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )

    def _store_image(self, image: Image.Image) -> str:
        """Store an image in the instance database and return its ID"""
        image_id = geneate_image_id(image)
        self._image_database[image_id] = image
        return image_id

    def _get_image(self, image_id: str) -> Image.Image:
        """Retrieve an image from the instance database by ID"""
        if image_id not in self._image_database:
            raise ValueError(f"Image with ID {image_id} not found in database")
        return self._image_database[image_id]

    def save_image(self, image_id: str, path: str):
        """Save an image from the instance database to a path"""
        image = self._get_image(image_id)
        image = image_to_pil(image)
        image.save(path)

    def parse(self, responses: List[str | Dict], **kwargs) -> List[str]:
        logger.debug(f"[ImageEditingAgent.parse] responses: {responses}")
        new_messages_list = []
        for response in responses:
            formatted_tool_calls = []
            if isinstance(response, dict):
                response_text = response["response_text"]
                if response["tool_calls"] and len(response["tool_calls"]) > 0:
                    tool_calls = [
                        response["tool_calls"][0]
                    ]  # We only support one tool call for now
                else:
                    tool_calls = []
                new_messages_list.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response_text}],
                        "tool_calls": tool_calls,
                    }
                )
            else:
                tool_calls = extract_tool_calls(response)
                if len(tool_calls) == 1:
                    tool_call = tool_calls[0]
                    try:
                        tool_call = json.loads(tool_call)
                        # {"name": "...", "arguments": "..."}
                        if "name" in tool_call and "arguments" in tool_call:
                            name = tool_call["name"]
                            arguments = tool_call["arguments"]

                            formatted_tool_calls.append(
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            )
                    except Exception:
                        pass
                else:
                    pass
                new_messages_list.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                        "tool_calls": formatted_tool_calls,
                    }
                )
        return new_messages_list

    def _detect_and_insert_image_id(self, messages: List[dict]):
        for message in messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "image":
                        image = open_image_from_any(content["image"])
                        image_id = self._store_image(image)
                message["content"].insert(
                    0, {"type": "text", "text": f"Image id: {image_id}"}
                )
                break

    async def run(self, messages: List[dict], **kwargs):
        for message in messages:
            self._detect_and_insert_image_id(message["messages"])
        return await super().run(messages=messages, **kwargs)

    @tool(
        name="qwen_image_edit",
        description="Edit an image using Qwen-Image-Edit model with natural language instructions, return the image id and the edited image. Useful for tasks like changing colors, adding/removing elements, or style transfer.",
    )
    async def qwen_image_edit_tool(
        self,
        image_id: str,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 50,
        seed: int = 0,
    ) -> str:
        """
        Edit an image using Qwen-Image-Edit.

        Args:
            image_id: ID of the image to edit
            prompt: Natural language instruction for editing
            negative_prompt: Negative prompt (default: " ")
            true_cfg_scale: CFG scale (default: 4.0)
            num_inference_steps: Number of steps (default: 50)
            seed: Random seed (default: 0)

        Returns:
            JSON string with observation and edited image data
        """

        image = self._get_image(image_id)

        edited_image = self._qwen_image_edit_tool.apply(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        image_base64 = image_to_data_uri(edited_image)
        new_image_id = self._store_image(edited_image)

        result = {"observation": f"Image Id: {new_image_id}", "image": image_base64}
        return result
