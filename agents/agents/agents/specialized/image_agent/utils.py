import io
import time
import json
import math
import enum
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import requests

from diffusers import (
    FluxKontextPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image


from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamProcessor, SamModel


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("edit_agent")

# -----------------------------
# Helpers
# -----------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def center_crop_to_square(image: Image.Image) -> Image.Image:
    """Center crop an image to a square shape."""
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2
    return image.crop((left, top, right, bottom))


def fetch_image(path_or_url: str) -> Image.Image:
    """Load an image from path or url into RGB."""
    try:
        img = load_image(path_or_url)
    except Exception:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            resp = requests.get(path_or_url, timeout=30)
            resp.raise_for_status()
            img =  Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")
    return center_crop_to_square(img)

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# -----------------------------
# Task taxonomy
# -----------------------------
class Task(enum.Enum):
    DETECTION = "detection"  # returns boxes + mask
    INPAINT = "inpaint"  # requires mask or detect_prompt for auto mask
    AUTO_INPAINT = "auto_inpaint"
    OBJECT_SWAP = "object_swap"
    OBJECT_REMOVAL = "object_removal"
    OBJECT_ADDITION = "object_addition"
    ATTRIBUTE_MOD = "attribute_modification"
    BACKGROUND_SWAP = "background_swap"
    ENV_CHANGE = "environment_change"
    STYLE_TRANSFER = "style_transfer"
    STYLE_REFERENCE_SYNTH = "style_reference_synthesis"
    EDIT_GENERAL = "edit_general"


# Canonical prompt templates
PROMPT_TEMPLATES: Dict[Task, Callable[[str], str]] = {
    Task.OBJECT_SWAP: lambda p: p,
    Task.OBJECT_REMOVAL: lambda p: f"Remove {p}",
    Task.OBJECT_ADDITION: lambda p: f"Add {p}",
    Task.ATTRIBUTE_MOD: lambda p: f"Change to {p}",
    Task.BACKGROUND_SWAP: lambda p: f"Replace background with {p}",
    Task.ENV_CHANGE: lambda p: f"Change scene to {p}",
    Task.STYLE_TRANSFER: lambda p: f"Apply {p} style to image",
    Task.INPAINT: lambda p: p,
    Task.AUTO_INPAINT: lambda p: p,
    Task.STYLE_REFERENCE_SYNTH: lambda p: p,
    Task.EDIT_GENERAL: lambda p: p,
}


# -----------------------------
# Detection → Mask (GroundingDINO)
# -----------------------------
class DetectionTool:
    def detect_mask(self, image: Image.Image, text_prompt: str, **kwargs) -> Image.Image:
        raise NotImplementedError

    def detect_boxes(self, image: Image.Image, text_prompt: str, **kwargs) -> np.ndarray:
        raise NotImplementedError


class GroundingDINOTool(DetectionTool):
    """Text-prompted detector using HF transformers weights.
    Produces rectangular masks from detected boxes (simple, no extra deps).
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[AutoModelForZeroShotObjectDetection] = None

    # Lazy init keeps startup light
    def _ensure_loaded(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            
        if self._model is None:
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def detect_boxes(self, image: Image.Image, text_prompt: str, **kwargs) -> np.ndarray:
        self._ensure_loaded()
        box_thr = float(kwargs.get("box_threshold", self.box_threshold))
        txt_thr = float(kwargs.get("text_threshold", self.text_threshold))
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = [image.size[::-1]]  # (H, W)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # box_threshold=box_thr,
            text_threshold=txt_thr,
            target_sizes=target_sizes,
        )[0]
        boxes = results.get("boxes")
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return boxes.detach().cpu().numpy().astype(np.float32)

    def _boxes_to_mask(self, image: Image.Image, boxes_xyxy: np.ndarray) -> Image.Image:
        w, h = image.size
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        for x0, y0, x1, y1 in boxes_xyxy:
            draw.rectangle([float(x0), float(y0), float(x1), float(y1)], fill=255)
        return mask_img

    def detect_mask(self, image: Image.Image, text_prompt: str, **kwargs) -> Image.Image:
        boxes_xyxy = self.detect_boxes(image, text_prompt, **kwargs)
        if boxes_xyxy.shape[0] == 0:
            return Image.new("L", image.size, 0)
        return self._boxes_to_mask(image, boxes_xyxy)


class SAMRefiner:
    """
    Use SAM to refine DINO boxes into pixel-accurate masks.
    Works with Hugging Face transformers SamModel/SamProcessor.
    """
    def __init__(
        self,
        model_id: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or get_device()
        self.model_id = model_id
        self.dtype = dtype
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._processor is None or self._model is None:
            self._processor = SamProcessor.from_pretrained(self.model_id)
            self._model = SamModel.from_pretrained(self.model_id).to(self.device)

    def refine_from_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: np.ndarray,
        mask_threshold: float = 0.0,     # logits > 0 → mask；你也可设 0.5
        topk_per_box: int = 1,           # SAM 每个提示通常输出3张；取前 k 张融合
    ) -> Image.Image:
        """
        Args:
            image: PIL Image (RGB)
            boxes_xyxy: np.ndarray of shape [N, 4] in XYXY pixel coords
            mask_threshold: threshold on logits after postprocess
            topk_per_box: take top-k by iou_scores for each box, then OR-merge
        Returns:
            PIL "L" image mask in {0,255}
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return Image.new("L", image.size, 0)

        self._ensure_loaded()

        # SAM 需要 list[List[boxes]]；坐标是原图像素 XYXY（processor 会自适配）
        inputs = self._processor(
            image,
            input_boxes=[boxes_xyxy.astype(np.float32).tolist()],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        masks_list = self._processor.post_process_masks(
            outputs.pred_masks,               # [B, num_masks(=3*N), 256, 256]
            inputs["original_sizes"],         # [(H, W)]
            inputs["reshaped_input_sizes"],   # [(H_in, W_in)]
        )[0]                                  # Tensor [3*N, H, W] in logits space

        # iou_scores: [3*N]
        iou_scores = outputs.iou_scores[0].detach().reshape(-1)
    


        H, W = masks_list.shape[-2], masks_list.shape[-1]
        masks_list = masks_list.view(-1, H, W)  # [3*N, H, W] logits
        
        union = torch.zeros((H, W), dtype=torch.bool, device=masks_list.device)

        num_boxes = boxes_xyxy.shape[0]
        for bi in range(num_boxes):
            start = bi * 3
            end = start + 3
            scores = iou_scores[start:end]
            idx = torch.topk(scores, k=min(topk_per_box, scores.numel())).indices
            chosen = masks_list[start:end][idx]  # [k, H, W] logits
            
            union |= (chosen > mask_threshold).any(dim=0)

        mask_img = (union.detach().cpu().numpy().astype(np.uint8) * 255)
        return Image.fromarray(mask_img, mode="L")

# -----------------------------
# Mask utilities
# -----------------------------

def dilate_mask(mask: Image.Image, radius: int = 8) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))


def feather_mask(mask: Image.Image, radius: int = 8) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def union_masks(masks: Sequence[Image.Image]) -> Image.Image:
    if not masks:
        raise ValueError("union_masks requires at least one mask")
    base = masks[0].copy().convert("L")
    for m in masks[1:]:
        base = Image.fromarray(np.maximum(np.array(base), np.array(m).astype(np.uint8)))
    return base


def visualize_boxes(image: Image.Image, boxes: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([float(x0), float(y0), float(x1), float(y1)], outline=color, width=3)
    return vis


# -----------------------------
# Editing Tools (generation/editing)
# -----------------------------
class EditingTool:
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def supported_tasks(self) -> List[Task]:
        return []

    def apply(self, image: Any, **kwargs) -> Image.Image:
        raise NotImplementedError


class _LazyPipe:
    """Lazy initializer for diffusion pipelines."""

    def __init__(self, fn_create: Callable[[], Any]):
        self._creator = fn_create
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._creator()
        return self._pipe


class FluxKontextTool(EditingTool):
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-Kontext-dev", dtype=torch.bfloat16, device: Optional[str] = None):
        device = device or get_device()
        def _create():
            pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        self._lazy = _LazyPipe(_create)

    def supported_tasks(self) -> List[Task]:
        return [
            Task.OBJECT_SWAP,
            Task.OBJECT_REMOVAL,
            Task.OBJECT_ADDITION,
            Task.ATTRIBUTE_MOD,
            Task.BACKGROUND_SWAP,
            Task.ENV_CHANGE,
            Task.STYLE_TRANSFER,
            Task.EDIT_GENERAL,
        ]

    def apply(self, image: Any, prompt: str, guidance_scale: float = 2.5, generator: Optional[torch.Generator] = None, **kwargs) -> Image.Image:
        out = self._lazy.pipe(image=image, prompt=prompt, guidance_scale=guidance_scale, generator=generator)
        return out.images[0]


class SDInpaintTool(EditingTool):
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", dtype=torch.float16, device: Optional[str] = None):
        device = device or get_device()
        def _create():
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        self._lazy = _LazyPipe(_create)

    def supported_tasks(self) -> List[Task]:
        return [Task.INPAINT]

    def apply(self, image: Image.Image, mask: Image.Image, prompt: str, strength: float = 0.95, guidance_scale: float = 7.5, num_inference_steps: int = 30, generator: Optional[torch.Generator] = None, **kwargs) -> Image.Image:
        out = self._lazy.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return out.images[0]


class SDXLInpaintTool(EditingTool):
    def __init__(self, model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", dtype=torch.float16, device: Optional[str] = None):
        device = device or get_device()
        def _create():
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        self._lazy = _LazyPipe(_create)

    def supported_tasks(self) -> List[Task]:
        return [Task.INPAINT]

    def apply(self, image: Image.Image, mask: Image.Image, prompt: str, strength: float = 0.95, guidance_scale: float = 5.5, num_inference_steps: int = 30, generator: Optional[torch.Generator] = None, **kwargs) -> Image.Image:
        out = self._lazy.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return out.images[0]


class InstructPix2PixTool(EditingTool):
    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix", dtype=torch.float16, device: Optional[str] = None):
        device = device or get_device()
        def _create():
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=dtype)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        self._lazy = _LazyPipe(_create)

    def supported_tasks(self) -> List[Task]:
        return [
            Task.OBJECT_SWAP,
            Task.OBJECT_REMOVAL,
            Task.OBJECT_ADDITION,
            Task.ATTRIBUTE_MOD,
            Task.STYLE_TRANSFER,
            Task.EDIT_GENERAL,
        ]

    def apply(self, image: Image.Image, prompt: str, guidance_scale: float = 7.5, image_guidance_scale: float = 1.5, num_inference_steps: int = 40, generator: Optional[torch.Generator] = None, **kwargs) -> Image.Image:
        out = self._lazy.pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return out.images[0]


class IPAdapterSDXLTool(EditingTool):
    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        ip_adapter_repo: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "sdxl_models",
        dtype=torch.float16,
        device: Optional[str] = None,
    ):
        device = device or get_device()
        def _create():
            pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=dtype)
            # try:
            pipe.load_ip_adapter(ip_adapter_repo, subfolder=ip_adapter_subfolder, weight_name="ip-adapter_sdxl.safetensors")
            # except Exception:
                # pipe.load_ip_adapter(ip_adapter_repo)
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            return pipe
        self._lazy = _LazyPipe(_create)

    def supported_tasks(self) -> List[Task]:
        return [Task.STYLE_REFERENCE_SYNTH, Task.STYLE_TRANSFER]

    def apply(self, image: Optional[Image.Image] = None, prompt: str = "", reference_image: Optional[Image.Image] = None, ip_adapter_scale: float = 0.8, guidance_scale: float = 5.0, num_inference_steps: int = 30, negative_prompt: Optional[str] = None, generator: Optional[torch.Generator] = None, **kwargs) -> Image.Image:
        if reference_image is None:
            raise ValueError("IP-Adapter requires a reference_image")
        pipe = self._lazy.pipe
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(ip_adapter_scale)
        kwargs_pipe = dict(
            prompt=prompt,
            # image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            ip_adapter_image=reference_image,
            generator=generator,
        )
        kwargs_pipe = {k: v for k, v in kwargs_pipe.items() if v is not None}
        out = pipe(**kwargs_pipe)
        return out.images[0]


# -----------------------------
# Registry & Results
# -----------------------------
@dataclass
class ToolSpec:
    tool: EditingTool
    priority: int = 0  # smaller is earlier


def default_registry() -> Dict[Task, List[ToolSpec]]:
    kflux = FluxKontextTool()
    sdinp = SDInpaintTool()
    sdxlinp = SDXLInpaintTool()
    ipx = InstructPix2PixTool()
    ip_adapter = IPAdapterSDXLTool()
    return {
        Task.OBJECT_SWAP: [ToolSpec(kflux, 0), ToolSpec(ipx, 1)],
        Task.OBJECT_REMOVAL: [ToolSpec(kflux, 0), ToolSpec(ipx, 1)],
        Task.OBJECT_ADDITION: [ToolSpec(kflux, 0), ToolSpec(ipx, 1)],
        Task.ATTRIBUTE_MOD: [ToolSpec(kflux, 0), ToolSpec(ipx, 1)],
        Task.BACKGROUND_SWAP: [ToolSpec(kflux, 0)],
        Task.ENV_CHANGE: [ToolSpec(kflux, 0)],
        Task.STYLE_TRANSFER: [ToolSpec(ip_adapter, 0), ToolSpec(kflux, 1), ToolSpec(ipx, 2)],
        Task.INPAINT: [ToolSpec(sdxlinp, 0), ToolSpec(sdinp, 1)],
        Task.AUTO_INPAINT: [ToolSpec(sdxlinp, 0), ToolSpec(sdinp, 1)],
        Task.STYLE_REFERENCE_SYNTH: [ToolSpec(ip_adapter, 0)],
        Task.EDIT_GENERAL: [ToolSpec(ipx, 0), ToolSpec(kflux, 1)],
        Task.DETECTION: [],
    }


@dataclass
class ToolResult:
    tool_name: str
    image: Optional[Image.Image]
    ok: bool
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Editing Agent
# -----------------------------
class EditingAgent:
    def __init__(self, gdino_model_id: str = "IDEA-Research/grounding-dino-base", sam_model_id: str = "facebook/sam-vit-base", registry: Optional[Dict[Task, List[ToolSpec]]] = None, device: Optional[str] = None):
        self.device = device or get_device()
        self.detector = GroundingDINOTool(model_id=gdino_model_id, device=self.device)
        self.registry = registry or default_registry()
        self._sam_refiner = None
        self._sam_model_id = sam_model_id

    # --- Introspection ---
    def available_tools(self, task: Task) -> List[EditingTool]:
        specs = self.registry.get(task)
        if specs is None:
            raise ValueError(f"Unknown task: {task}")
        return [s.tool for s in sorted(specs, key=lambda s: s.priority)]

    def _get_sam(self) -> SAMRefiner:
        if self._sam_refiner is None:
            self._sam_refiner = SAMRefiner(model_id=self._sam_model_id, device=self.device)
        return self._sam_refiner

    def plan(self, task: Task) -> List[str]:
        return [t.name for t in self.available_tools(task)]

    # --- Core execution ---
    def execute_task(
        self,
        task: Task | str,
        image_url: Optional[str] = None,
        prompt: str = "",
        mask_url: Optional[str] = None,
        reference_image_url: Optional[str] = None,
        return_all: bool = False,
        detect_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        # same options
        enable_sam: bool = False,
        sam_mask_threshold: float = 0.0,
        sam_topk_per_box: int = 1,
        # masking utils
        auto_mask_dilate: int = 1,
        auto_mask_feather: int = 1,
        # common diffusion params
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Any:
        if isinstance(task, str):
            task = Task(task)

        # I/O
        base_image = fetch_image(image_url) if image_url else None
        mask = fetch_image(mask_url).convert("L") if mask_url else None
        ref_image = fetch_image(reference_image_url) if reference_image_url else None

        # Generator for reproducibility
        generator = None
        if seed is not None:
            if self.device == "cuda":
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)

        # Pure detection
        if task == Task.DETECTION:
            if base_image is None or not detect_prompt:
                raise ValueError("detection requires image_url and detect_prompt")
            t0 = time.time()
            boxes = self.detector.detect_boxes(base_image, detect_prompt, **kwargs)
            
            if enable_sam and boxes.shape[0] > 0:
                mask_det = self._get_sam().refine_from_boxes(
                    base_image, boxes,
                    mask_threshold=sam_mask_threshold,
                    topk_per_box=sam_topk_per_box,
                )
            else:
                mask_det = self.detector.detect_mask(base_image, detect_prompt, **kwargs)
            # mask_det = self.detector.detect_mask(base_image, detect_prompt, **kwargs)
            if auto_mask_dilate:
                mask_det = dilate_mask(mask_det, radius=auto_mask_dilate)
            if auto_mask_feather:
                mask_det = feather_mask(mask_det, radius=auto_mask_feather)
            dt = time.time() - t0
            return {"boxes_xyxy": boxes, "mask": mask_det, "time_sec": dt}

        # Auto-generate mask for inpaint variants
        if (task in (Task.INPAINT, Task.AUTO_INPAINT)) and (mask is None):
            if base_image is None:
                raise ValueError("inpaint/auto_inpaint requires image_url")
            if not detect_prompt:
                raise ValueError("inpaint/auto_inpaint without mask requires detect_prompt")
            mask = self.detector.detect_mask(
                image=base_image,
                text_prompt=detect_prompt,
                box_threshold=kwargs.pop("box_threshold", 0.3),
                text_threshold=kwargs.pop("text_threshold", 0.25),
            )
            if auto_mask_dilate:
                mask = dilate_mask(mask, radius=auto_mask_dilate)
            if auto_mask_feather:
                mask = feather_mask(mask, radius=auto_mask_feather)

        # Prompt canonicalization
        final_prompt = PROMPT_TEMPLATES.get(task, lambda p: p)(prompt)

        # Execute through tools
        results: List[ToolResult] = []
        tools = self.available_tools(task)
        if task in (Task.INPAINT, Task.AUTO_INPAINT):
            if base_image is None or mask is None:
                raise ValueError("Inpainting requires image_url and a mask (or set detect_prompt)")
            for tool in tools:
                t0 = time.time()
                try:
                    img = tool.apply(
                        image=base_image,
                        mask=mask,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale or 5.5,
                        num_inference_steps=num_inference_steps or 30,
                        negative_prompt=negative_prompt,
                        generator=generator,
                        **kwargs,
                    )
                    results.append(ToolResult(tool.name, img, True, meta={"time_sec": time.time()-t0}))
                except Exception as e:
                    logger.warning(f"Tool {tool.name} failed: {e}")
                    results.append(ToolResult(f"{tool.name}", None, False, error=str(e), meta={"time_sec": time.time()-t0}))
        elif task == Task.STYLE_REFERENCE_SYNTH:
            if ref_image is None:
                raise ValueError("style_reference_synthesis requires reference_image_url")
            for tool in tools:
                t0 = time.time()
                try:
                    img = tool.apply(
                        image=base_image,
                        prompt=final_prompt,
                        reference_image=ref_image,
                        guidance_scale=guidance_scale or 5.0,
                        num_inference_steps=num_inference_steps or 30,
                        negative_prompt=negative_prompt,
                        generator=generator,
                        **kwargs,
                    )
                    results.append(ToolResult(tool.name, img, True, meta={"time_sec": time.time()-t0}))
                except Exception as e:
                    logger.warning(f"Tool {tool.name} failed: {e}")
                    results.append(ToolResult(f"{tool.name}", None, False, error=str(e), meta={"time_sec": time.time()-t0}))
        else:
            if base_image is None:
                raise ValueError("This task requires image_url")
            for tool in tools:
                t0 = time.time()
                try:
                    img = tool.apply(
                        image=base_image,
                        prompt=final_prompt,
                        guidance_scale=guidance_scale or 2.5,
                        num_inference_steps=num_inference_steps or 30,
                        negative_prompt=negative_prompt,
                        generator=generator,
                        **kwargs,
                    )
                    results.append(ToolResult(tool.name, img, True, meta={"time_sec": time.time()-t0}))
                except Exception as e:
                    logger.warning(f"Tool {tool.name} failed: {e}")
                    results.append(ToolResult(f"{tool.name}", None, False, error=str(e), meta={"time_sec": time.time()-t0}))

        if return_all:
            return results
        # pick first successful
        for r in results:
            if r.ok and r.image is not None:
                return r.image
        raise RuntimeError("All tools failed for the requested task.")


# -----------------------------
# Demos
# -----------------------------
if __name__ == "__main__":
    agent = EditingAgent(gdino_model_id="IDEA-Research/grounding-dino-base")

    # Demo 1: detection with visualization
    try:
        url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    # url_cat = "./000000039769.jpg"
        det = agent.execute_task(
            task=Task.DETECTION,
            image_url=url_cat,
            detect_prompt=[["a cat", "an eye"]],
            box_threshold=0.3,
            text_threshold=0.3,
            auto_mask_dilate=3,
            auto_mask_feather=3,
        )
        det["mask"].save("demo1_detection_mask.jpg")
        boxes = det["boxes_xyxy"]
        vis = visualize_boxes(fetch_image(url_cat), boxes)
        vis.save("demo1_detection_boxes.jpg")
        print("Demo 1 done.")
    except Exception as e:
        print("Demo 1 failed:", e)

    # Demo 2: auto inpaint remove object
    try:
        url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        out = agent.execute_task(
            task=Task.AUTO_INPAINT,
            image_url=url_cat,
            prompt="Dog in the photo",
            detect_prompt=[["a cat", "an eye"]],
            box_threshold=0.3,
            text_threshold=0.3,
            guidance_scale=5.5,
            num_inference_steps=25,
            seed=42,
        )
        out.save("demo2_auto_inpaint.jpg")
        print("Demo 2 done.")
    except Exception as e:
        print("Demo 2 failed:", e)


    url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    det = agent.execute_task(
        task=Task.DETECTION,
        enable_sam=True, 
        image_url=url_cat, 
        detect_prompt=[["a cat", "an eye"]],
        box_threshold=0.3,
        text_threshold=0.3,
        auto_mask_dilate=3,
        auto_mask_feather=3,)
    mask = det["mask"]
    mask.save("demo3_mask.jpg")
    out = agent.execute_task(
        task=Task.INPAINT,
        image_url=url_cat,
        mask_url="demo3_mask.jpg",
        prompt="Dog in the photo",
        guidance_scale=5.5,
        num_inference_steps=25,
    )
    out.save("demo3_manual_inpaint.jpg")
    print("Demo 3 done.")
    
    
    # Demo 3: manual inpaint with provided mask (here reuse auto mask for simplicity)
    try:
        url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        det = agent.execute_task(
            task=Task.DETECTION,
            enable_sam=True, 
            image_url=url_cat, 
            detect_prompt=[["a cat", "an eye"]],
            box_threshold=0.3,
            text_threshold=0.3,
            auto_mask_dilate=3,
            auto_mask_feather=3,)
        mask = det["mask"]
        mask.save("demo3_mask.jpg")
        out = agent.execute_task(
            task=Task.INPAINT,
            image_url=url_cat,
            mask_url="demo3_mask.jpg",
            prompt="Dog in the photo",
            guidance_scale=5.5,
            num_inference_steps=25,
        )
        out.save("demo3_manual_inpaint.jpg")
        print("Demo 3 done.")
    except Exception as e:
        print("Demo 3 failed:", e)

    # Demo 4: attribute modification (global edit)
    try:
        url_dog = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        out = agent.execute_task(
            task=Task.ATTRIBUTE_MOD,
            image_url=url_dog,
            prompt="make the photo look like a vintage film shot",
            guidance_scale=2.5,
            num_inference_steps=30,
            seed=123,
        )
        out.save("demo4_attribute_mod.jpg")
        print("Demo 4 done.")
    except Exception as e:
        print("Demo 4 failed:", e)

    # Demo 5: background swap
    try:
        url_dog = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        out = agent.execute_task(
            task=Task.BACKGROUND_SWAP,
            image_url=url_dog,
            prompt="a night city skyline with soft bokeh lights",
            guidance_scale=2.5,
            num_inference_steps=30,
        )
        out.save("demo5_background_swap.jpg")
        print("Demo 5 done.")
    except Exception as e:
        print("Demo 5 failed:", e)

    # Demo 6: style transfer using IP-Adapter reference
    try:
        url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        url_ref = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        out = agent.execute_task(
            task=Task.STYLE_REFERENCE_SYNTH,
            image_url=url_cat,
            reference_image_url=url_ref,
            prompt="keep composition but match the reference color palette",
            guidance_scale=5.0,
            num_inference_steps=30,
        )
        out.save("demo6_style_reference.jpg")
        print("Demo 6 done.")
    except Exception as e:
        print("Demo 6 failed:", e)

    # Demo 7: general edit via instruct-pix2pix
    try:
        url_city = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        out = agent.execute_task(
            task=Task.EDIT_GENERAL,
            image_url=url_city,
            prompt="make it look like a watercolor painting",
            guidance_scale=7.5,
            num_inference_steps=30,
        )
        out.save("demo7_edit_general.jpg")
        print("Demo 7 done.")
    except Exception as e:
        print("Demo 7 failed:", e)


