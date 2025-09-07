import logging
import os
import random
import re

from ...utils.json import jsonish
from ...agent_base import BaseAgent
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import json
from PIL import Image
import io
import base64
import numpy as np
from ....tools import tool
from .utils import (
    Task, 
    GroundingDINOTool, 
    SAMRefiner,
    FluxKontextTool,
    SDInpaintTool,
    SDXLInpaintTool,
    InstructPix2PixTool,
    IPAdapterSDXLTool,
    QwenImageEditTool,
    dilate_mask,
    feather_mask,
    visualize_boxes,
    fetch_image
)
from ....utils.vision import image_to_data_uri, image_to_pil

logger = logging.getLogger(__name__)

# Global variables for tools
_qwen_image_edit_tool = None
_detector = None
_sam_refiner = None
_flux_tool = None
_sd_inpaint_tool = None
_sdxl_inpaint_tool = None
_instruct_pix2pix_tool = None
_ip_adapter_tool = None

IMAGE_AGENT_SYSTEM_PROMPT = """You are an ImageEditingAgent, a powerful AI assistant specialized in image editing and manipulation tasks.

Always provide clear, step-by-step instructions and call the appropriate tools for each task. If you have finished the task, describe what you have seen in the final image."""

def _get_tools():
    """Initialize tools if not already done"""
    global _detector, _sam_refiner, _flux_tool, _sd_inpaint_tool, _sdxl_inpaint_tool, _instruct_pix2pix_tool, _ip_adapter_tool, _qwen_image_edit_tool
    
    import torch
    
    if _detector is None:
        _detector = GroundingDINOTool(model_id="IDEA-Research/grounding-dino-base", device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _sam_refiner is None:
        _sam_refiner = SAMRefiner(model_id="facebook/sam-vit-base", device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _flux_tool is None:
        _flux_tool = FluxKontextTool(device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _sd_inpaint_tool is None:
        _sd_inpaint_tool = SDInpaintTool(device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _sdxl_inpaint_tool is None:
        _sdxl_inpaint_tool = SDXLInpaintTool(device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _instruct_pix2pix_tool is None:
        _instruct_pix2pix_tool = InstructPix2PixTool(device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _ip_adapter_tool is None:
        _ip_adapter_tool = IPAdapterSDXLTool(device="cuda:1" if torch.cuda.is_available() else "cpu")
    
    if _qwen_image_edit_tool is None:
        # 分配到不同的 GPU 以避免 OOM
        _qwen_image_edit_tool = QwenImageEditTool(device="cuda:1" if torch.cuda.is_available() else "cpu")

_get_tools()

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
        except:
            pass
    
    # 2. Extract directly
    try:
        tool_call = jsonish(action_input)
        return [tool_call]
    except:
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
        **kwargs
    ):
        self._image_database = {}
        tools = [self.qwen_edit_image_tool]
        super().__init__(
            model_name_or_path=model_name_or_path,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs
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
                if response['tool_calls'] and len(response["tool_calls"]) > 0:
                    tool_calls = [response["tool_calls"][0]] # We only support one tool call for now
                else:
                    tool_calls = []
                new_messages_list.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}],
                    "tool_calls": tool_calls,
                })
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

                            formatted_tool_calls.append({
                                "id": None,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments
                                }
                            })
                    except Exception as e:
                        pass
                else:
                    pass
                new_messages_list.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                    "tool_calls": formatted_tool_calls,
                })
        return new_messages_list

    def _detect_and_insert_image_id(self, messages: List[dict]):
        for message in messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "image":
                        image = fetch_image(content["image"])
                        image_id = self._store_image(image)
                message["content"].insert(0, {"type": "text", "text": f"Image id: {image_id}"})
                break

    async def run(self, messages: List[dict], **kwargs):
        for message in messages:
            self._detect_and_insert_image_id(message["messages"])
        return await super().run(messages=messages, **kwargs)

    @tool(
        name="detect_objects",
        description="Detect objects in an image using GroundingDINO. Returns bounding boxes, mask, and visualization."
    )
    async def detect_objects_tool(
        self,
        image_id: str, 
        text_prompt: str, 
        box_threshold: float = 0.3, 
        text_threshold: float = 0.25, 
        auto_mask_dilate: int = 1, 
        auto_mask_feather: int = 1
    ) -> str:
        """
        Detect objects in an image using GroundingDINO.
        
        Args:
            image_id: ID of the image stored in the database
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes (default: 0.3)
            text_threshold: Confidence threshold for text matching (default: 0.25)
            auto_mask_dilate: Number of pixels to dilate the mask (default: 1)
            auto_mask_feather: Number of pixels to feather the mask (default: 1)
            
        Returns:
            String containing detection results with image IDs
        """
        _get_tools()
        image = self._get_image(image_id)
        
        # Detect objects
        boxes = _detector.detect_boxes(
            image, 
            text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold
        )
        
        # Generate mask
        mask = _detector.detect_mask(
            image, 
            text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold
        )
        
        # Apply post-processing
        if auto_mask_dilate > 0:
            mask = dilate_mask(mask, radius=auto_mask_dilate)
        if auto_mask_feather > 0:
            mask = feather_mask(mask, radius=auto_mask_feather)
        
        # Store processed images
        mask_id = self._store_image(mask)
        
        # Create visualization
        vis_image = visualize_boxes(image, boxes)
        image_id = self._store_image(vis_image)
        
        result = {
            "mask_id": mask_id,
            "image_id": image_id,
        }
        return json.dumps(result)
            

    @tool(
        name="inpaint_image",
        description="Fill in masked areas of an image using AI generation."
    )
    async def inpaint_image_tool(
        self,
        image_id: str, 
        mask_id: str, 
        prompt: str, 
        guidance_scale: float = 5.5, 
        num_inference_steps: int = 30, 
        strength: float = 0.95, 
        seed: int = None, 
        negative_prompt: str = None
    ) -> str:
        """
        Fill in masked areas of an image using AI generation.
        
        Args:
            image_id: ID of the image to inpaint
            mask_id: ID of the mask image
            prompt: Text description of what to inpaint
            guidance_scale: Guidance scale (default: 5.5)
            num_inference_steps: Number of steps (default: 30)
            strength: Inpainting strength (default: 0.95)
            seed: Random seed
            negative_prompt: Negative prompt
            
        Returns:
            String containing the inpainted image ID
        """
        _get_tools()
        image = self._get_image(image_id)
        mask = self._get_image(mask_id)
        
        # Use SDXL inpainting for better quality
        image = _sdxl_inpaint_tool.apply(
            image=image,
            mask=mask,
            prompt=prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt
        )
        image_base64 = image_to_data_uri(image)
        image_id = self._store_image(image)
        # Store result
        result = {
            "observation": f"Image Id: {image_id}",
            "image": image_base64,
            "image_id": image_id
        }
        return result


    @tool(
        name="auto_inpaint_image",
        description="Automatically detect objects and inpaint them in one operation."
    )
    async def auto_inpaint_image_tool(
        self,
        image_id: str, 
        detect_prompt: str, 
        prompt: str, 
        box_threshold: float = 0.3, 
        text_threshold: float = 0.25, 
        guidance_scale: float = 5.5, 
        num_inference_steps: int = 30, 
        strength: float = 0.95, 
        seed: int = None, 
        negative_prompt: str = None, 
        auto_mask_dilate: int = 1, 
        auto_mask_feather: int = 1
    ) -> str:
        """
        Automatically detect objects and inpaint them.
        
        Args:
            image_id: ID of the image to process
            detect_prompt: Text description of objects to detect
            prompt: Text description of what to inpaint
            box_threshold: Confidence threshold for detection (default: 0.3)
            text_threshold: Confidence threshold for text matching (default: 0.25)
            guidance_scale: Guidance scale for generation (default: 5.5)
            num_inference_steps: Number of inference steps (default: 30)
            strength: Inpainting strength (default: 0.95)
            seed: Random seed for reproducibility
            negative_prompt: Negative prompt for generation
            auto_mask_dilate: Number of pixels to dilate the mask (default: 1)
            auto_mask_feather: Number of pixels to feather the mask (default: 1)
            
        Returns:
            String containing the inpainted image ID
        """
        # First detect objects
        detection_result = await self.detect_objects_tool(
            image_id=image_id, 
            text_prompt=detect_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold, 
            auto_mask_dilate=auto_mask_dilate, 
            auto_mask_feather=auto_mask_feather
        )
        detection_result = detection_result['observation']
        # Extract mask ID from detection result
        detection_result = json.loads(detection_result)
        mask_id = detection_result["mask_id"]

        
        # Perform inpainting
        inpaint_result = await self.inpaint_image_tool(
            image_id=image_id, 
            mask_id=mask_id, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps, 
            strength=strength, 
            seed=seed, 
            negative_prompt=negative_prompt
        )
        observation = inpaint_result['observation']
        image = inpaint_result['info']['image']
        image_id = inpaint_result['info']['image_id']
        result = {
            "observation": observation,
            "image": image,
            "image_id": image_id
        }
        # If want to save the image, uncomment the following line
        # self.save_image(image_id, f"inpainted_image_{image_id}.png")
        return result
    
    @tool(
        name="qwen_edit_image",
        description="Edit an image using Qwen-Image-Edit model with natural language instructions, return the image id and the edited image. Useful for tasks like changing colors, adding/removing elements, or style transfer."
    )
    async def qwen_edit_image_tool(
        self,
        image_id: str,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 50,
        seed: int = 0
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
        _get_tools()
        
        image = self._get_image(image_id)
        
        edited_image = _qwen_image_edit_tool.apply(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )
        
        image_base64 = image_to_data_uri(edited_image)
        new_image_id = self._store_image(edited_image)
        
        result = {
            "observation": f"Image Id: {new_image_id}",
            "image": image_base64
        }
        return result
