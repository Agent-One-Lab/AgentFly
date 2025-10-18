#!/usr/bin/env python3
"""
OpenAI-compatible API server for image-edit models with multi-GPU support.
Provides fast inference with automatic load balancing across multiple GPUs.
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image
import subprocess
import shutil
from diffusers import QwenImageEditPipeline
from urllib.parse import urlparse
from .....utils.vision import open_image_from_any, image_to_data_uri
from .utils import ImageEditRequest, ImageEditResponse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-GPU Image Edit API",
    description="OpenAI-compatible API for image-edit models with multi-GPU support",
    version="2.0.0"
)

# Abstract base class for image edit models
class ImageEditModel(ABC):
    """Abstract base class for image edit models."""
    
    def __init__(self, model_path: str, device: str, **kwargs):
        self.model_path = model_path
        self.device = device
        self.is_loaded = False
    
    @abstractmethod
    async def load_model(self):
        """Load the model on the specified device."""
        pass
    
    @abstractmethod
    def edit_image(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Edit image and return edited image."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

class QwenImageEditModel(ImageEditModel):
    """Qwen Image Edit model implementation using diffusers."""
    
    def __init__(self, model_path: str, device: str, use_fast: bool = True, use_ultra_fast: bool = False, **kwargs):
        super().__init__(model_path, device)
        self.use_fast = use_fast
        self.use_ultra_fast = use_ultra_fast
        self.temp_dir = kwargs.get('temp_dir', '/tmp/qwen_image_edit')
        self.temp_dir = Path(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Model components
        self.pipeline = None
        self.device_torch = torch.device(device)
        
        # Default settings
        self.default_num_inference_steps = 20 if use_ultra_fast else (50 if use_fast else 100)
        self.default_guidance_scale = 7.5
        self.default_timeout = 60  # Reduced to 1 minute default timeout
    
    async def load_model(self):
        """Load Qwen model using QwenImageEditPipeline."""
        try:
            logger.info(f"Loading Qwen model on device: {self.device}")
            
            # Load the Qwen image edit pipeline
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            )
            self.pipeline.to(self.device)
            # self.pipeline = ORTDiffusionPipeline.from_pretrained(self.model_path, provider="CUDAExecutionProvider", device=self.device)
            
            # Disable progress bar
            self.pipeline.set_progress_bar_config(disable=None)
            
            # Optimize for performance
            if self.use_ultra_fast or self.use_fast:
                # Enable memory efficient attention if available
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                
                if hasattr(self.pipeline, 'enable_memory_efficient_attention'):
                    self.pipeline.enable_memory_efficient_attention()
            
            # Enable CPU offload for memory efficiency
            # if self.device_torch.type == 'cuda':
            #     self.pipeline.enable_model_cpu_offload()
            
            self.is_loaded = True
            logger.info(f"Qwen model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model on {self.device}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def edit_image(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Edit image using diffusers pipeline (synchronous version for thread pool)."""
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Get parameters from kwargs
            negative_prompt = kwargs.get('negative_prompt')
            true_cfg_scale = kwargs.get('true_cfg_scale', self.default_guidance_scale)
            num_inference_steps = kwargs.get('num_inference_steps', self.default_num_inference_steps)
            seed = kwargs.get('seed')
            timeout = kwargs.get('timeout', self.default_timeout)
            
            # Generate edited image
            logger.info(f"Generating image on {self.device} with prompt: {prompt}, timeout: {timeout}s")
            
            # Prepare inputs
            inputs = {
                "image": image,
                "prompt": prompt,
                "generator": torch.Generator(device=self.device_torch).manual_seed(seed) if seed is not None else None,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
            }
            
            # Run inference synchronously (no async needed for the actual inference)
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                edited_image = output.images[0]
            
            logger.info(f"Image editing completed on {self.device}")
            return edited_image
            
        except Exception as e:
            logger.error(f"Image editing failed on {self.device}: {str(e)}")
            raise RuntimeError(f"Image editing error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Qwen model information."""
        return {
            "model_type": "qwen-image-edit-diffusers",
            "model_path": self.model_path,
            "device": self.device,
            "fast_mode": self.use_fast,
            "ultra_fast_mode": self.use_ultra_fast,
            "default_num_inference_steps": self.default_num_inference_steps,
            "default_guidance_scale": self.default_guidance_scale
        }

# Example implementation for other image-edit models
class GenericImageEditModel(ImageEditModel):
    """Generic image edit model implementation for any command-line tool."""
    
    def __init__(self, model_path: str, device: str, command_template: str, **kwargs):
        super().__init__(model_path, device)
        self.command_template = command_template
        self.temp_dir = kwargs.get('temp_dir', '/tmp/generic_image_edit')
        self.temp_dir = Path(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.model_name = kwargs.get('model_name', 'generic-image-edit')
    
    async def load_model(self):
        """Load generic model (check if command exists)."""
        try:
            # Extract command from template (first word)
            command = self.command_template.split()[0]
            result = subprocess.run([command, '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError(f"{command} not found. Please install the required tool.")
            self.is_loaded = True
            logger.info(f"Generic model loaded on device: {self.device}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError(f"Command not found. Please install the required tool.")
    
    def edit_image(self, input_path: Path, prompt: str, **kwargs) -> Path:
        """Edit image using generic command template."""
        output_id = str(uuid.uuid4())
        output_path = self.temp_dir / f"output_{output_id}.png"
        
        # Build command from template
        cmd = self.command_template.format(
            input_path=str(input_path),
            prompt=prompt,
            output_path=str(output_path),
            device=self.device
        ).split()
        
        try:
            logger.info(f"Running command on {self.device}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Command failed on {self.device}: {result.stderr}")
                raise RuntimeError(f"Image editing failed: {result.stderr}")
            
            if not output_path.exists():
                raise RuntimeError("Output image not generated")
            
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Image editing timeout")
        except Exception as e:
            raise RuntimeError(f"Image editing error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get generic model information."""
        return {
            "model_type": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "command_template": self.command_template
        }

class GPUManager:
    """Manages multiple GPU instances and load balancing."""
    
    def __init__(self, model_class, model_configs: List[Dict[str, Any]]):
        self.model_class = model_class
        self.model_configs = model_configs
        self.gpu_instances: Dict[int, ImageEditModel] = {}
        self.gpu_queues: Dict[int, queue.Queue] = {}
        self.gpu_locks: Dict[int, threading.Lock] = {}
        self.executor = ThreadPoolExecutor(max_workers=len(model_configs))
        self._initialize_gpus()
    
    def _initialize_gpus(self):
        """Initialize GPU instances."""
        for i, config in enumerate(self.model_configs):
            gpu_id = config.get('gpu_id', i)
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id < torch.cuda.device_count() else "cpu"
            
            # Update device in config
            config['device'] = device
            
            # Create model instance
            model = self.model_class(**config)
            self.gpu_instances[gpu_id] = model
            self.gpu_queues[gpu_id] = queue.Queue()
            self.gpu_locks[gpu_id] = threading.Lock()
            
            logger.info(f"Initialized GPU {gpu_id} with device {device}")
    
    async def load_all_models(self):
        """Load all models on their respective GPUs."""
        tasks = []
        for gpu_id, model in self.gpu_instances.items():
            tasks.append(model.load_model())
        
        await asyncio.gather(*tasks)
        logger.info(f"Loaded {len(self.gpu_instances)} models across {len(self.gpu_instances)} GPUs")
    
    def get_available_gpu(self) -> Optional[int]:
        """Get the least busy GPU."""
        available_gpus = []
        for gpu_id, gpu_queue in self.gpu_queues.items():
            if gpu_queue.qsize() < 5:  # Threshold for queue size
                available_gpus.append((gpu_id, gpu_queue.qsize()))
        
        if not available_gpus:
            return None
        
        # Return GPU with smallest queue
        return min(available_gpus, key=lambda x: x[1])[0]
    
    # in GPUManager.edit_image(...)
    async def edit_image(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        gpu_id = self.get_available_gpu()
        if gpu_id is None:
            raise HTTPException(status_code=503, detail="All GPUs are busy, please try again later")

        model = self.gpu_instances[gpu_id]
        self.gpu_queues[gpu_id].put(time.time())

        try:
            # Get timeout from kwargs, default to 300 seconds
            timeout = kwargs.get('timeout', 300)
            
            def _run():
                # serialize access to this pipeline
                with self.gpu_locks[gpu_id]:
                    # run the model call synchronously in this worker thread
                    try:
                        logger.debug(f"Input parameters: {kwargs}")
                        return model.edit_image(image, prompt, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in model.edit_image: {e}")
                        raise

            # Apply timeout to the entire operation (waiting + processing)
            # Note: This includes both queue waiting time and GPU processing time
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, _run),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"GPU {gpu_id} operation timed out after {timeout}s")
            raise HTTPException(status_code=504, detail=f"Request timeout after {timeout} seconds (includes queue wait + processing)")
        except Exception as e:
            logger.error(f"GPU {gpu_id} operation failed: {e}")
            raise HTTPException(status_code=500, detail=f"GPU operation failed: {str(e)}")
        finally:
            try:
                self.gpu_queues[gpu_id].get_nowait()
            except queue.Empty:
                pass

    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all GPUs."""
        status = {}
        for gpu_id, model in self.gpu_instances.items():
            status[f"gpu_{gpu_id}"] = {
                "device": model.device,
                "queue_size": self.gpu_queues[gpu_id].qsize(),
                "model_info": model.get_model_info()
            }
        return status

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MultiGPUImageEditAPI:
    """Main API handler for multi-GPU image editing."""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        
        logger.info(f"Initialized MultiGPUImageEditAPI with {len(self.gpu_manager.gpu_instances)} GPUs")
    
    async def edit_image(self, request: ImageEditRequest) -> ImageEditResponse:
        """Main image editing method with multi-GPU support."""
        start_time = time.time()
        
        try:
            # Load image using open_image_from_any
            try:
                image = open_image_from_any(request.image)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
            
            # Edit image using GPU manager with simplified parameters
            edited_image = await self.gpu_manager.edit_image(
                image, 
                request.prompt,
                negative_prompt=request.negative_prompt,
                true_cfg_scale=request.true_cfg_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                timeout=request.timeout
            )
            
            # Encode result
            edited_image_b64 = image_to_data_uri(edited_image)
            
            processing_time = time.time() - start_time
            logger.info(f"Image editing completed in {processing_time:.2f} seconds")
            
            return ImageEditResponse(image=edited_image_b64)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Global variables for API handler
api_handler: Optional[MultiGPUImageEditAPI] = None
gpu_manager: Optional[GPUManager] = None

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Multi-GPU Image Edit API is running", "status": "healthy"}

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    # Get model info from first GPU (assuming all GPUs use same model)
    first_gpu = next(iter(gpu_manager.gpu_instances.values()))
    model_info = first_gpu.get_model_info()
    
    return {
        "object": "list",
        "data": [
            {
                "id": model_info.get("model_type", "image-edit"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "multi-gpu-api"
            }
        ]
    }

@app.post("/v1/images/edits", response_model=ImageEditResponse)
async def create_image_edit(request: ImageEditRequest):
    """Create image edit (OpenAI compatible endpoint)."""
    if not api_handler:
        raise HTTPException(status_code=503, detail="API handler not initialized")
    
    try:
        return await api_handler.edit_image(request)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": {"message": e.detail, "type": "invalid_request_error"}}
        )

@app.post("/v1/images/edits/upload")
async def create_image_edit_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    true_cfg_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    timeout: Optional[int] = Form(None)
):
    """Create image edit with file upload."""
    if not api_handler:
        raise HTTPException(status_code=503, detail="API handler not initialized")
    
    try:
        # Read image data and convert to base64
        image_data = await image.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create request object
        request = ImageEditRequest(
            image=image_b64,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            timeout=timeout
        )
        
        return await api_handler.edit_image(request)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}}
        )

@app.get("/health")
async def health_check():
    """Detailed health check."""
    if not gpu_manager:
        return {
            "status": "unhealthy",
            "timestamp": int(time.time()),
            "error": "GPU manager not initialized"
        }
    
    gpu_status = gpu_manager.get_status()
    first_gpu = next(iter(gpu_manager.gpu_instances.values()))
    model_info = first_gpu.get_model_info()
    
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "gpu_count": len(gpu_manager.gpu_instances),
        "gpu_status": gpu_status,
        "model_info": model_info
    }

@app.get("/gpu/status")
async def gpu_status():
    """Get detailed GPU status."""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    return gpu_manager.get_status()

def create_model_configs(model_type: str, model_path: str, gpu_ids: List[int], 
                        use_fast: bool = True, use_ultra_fast: bool = False, **kwargs) -> List[Dict[str, Any]]:
    """Create model configurations for multiple GPUs."""
    configs = []
    for gpu_id in gpu_ids:
        config = {
            "model_path": model_path,
            "gpu_id": gpu_id,
            "use_fast": use_fast,
            "use_ultra_fast": use_ultra_fast,
            **kwargs
        }
        configs.append(config)
    return configs

def get_available_gpu_ids() -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return [0]  # Use CPU if no CUDA
    
    gpu_count = torch.cuda.device_count()
    return list(range(gpu_count))

async def initialize_api(model_type: str, model_path: str, gpu_ids: List[int], 
                        use_fast: bool = True, use_ultra_fast: bool = False, 
                        temp_dir: str = "/tmp/image_edit", **kwargs):
    """Initialize the multi-GPU API."""
    global api_handler, gpu_manager
    
    # Select model class based on type
    if model_type.lower() == "qwen":
        model_class = QwenImageEditModel
    elif model_type.lower() == "generic":
        model_class = GenericImageEditModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: qwen, generic")
    
    # Create model configurations
    model_configs = create_model_configs(
        model_type=model_type,
        model_path=model_path,
        gpu_ids=gpu_ids,
        use_fast=use_fast,
        use_ultra_fast=use_ultra_fast,
        temp_dir=temp_dir,
        **kwargs
    )
    
    # Initialize GPU manager
    gpu_manager = GPUManager(model_class, model_configs)
    
    # Load all models
    await gpu_manager.load_all_models()
    
    # Initialize API handler
    api_handler = MultiGPUImageEditAPI(gpu_manager)
    
    logger.info(f"Multi-GPU API initialized with {len(gpu_ids)} GPUs")

if __name__ == "__main__":
    """python -m agentfly.agents.specialized.image_agent.utils.diffuser_apis --model-type qwen --use-fast --gpu-ids 0 1"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU Image Edit API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-type", default="qwen", choices=["qwen", "generic"], help="Type of model to use")
    parser.add_argument("--command-template", help="Command template for generic model (e.g., 'my-tool edit -i {input_path} -p {prompt} -o {output_path}')")
    parser.add_argument("--model-name", default="generic-image-edit", help="Name for generic model")
    parser.add_argument("--model-path", default="Qwen/Qwen-Image-Edit", help="Model path")
    parser.add_argument("--gpu-ids", nargs="+", type=int, help="GPU IDs to use (default: all available)")
    parser.add_argument("--use-fast", action="store_true", help="Use fast mode (6x speedup)")
    parser.add_argument("--use-ultra-fast", action="store_true", help="Use ultra fast mode (12x speedup)")
    parser.add_argument("--temp-dir", default="/tmp/image_edit", help="Temporary directory")
    
    args = parser.parse_args()
    
    # Determine GPU IDs
    if args.gpu_ids is None:
        gpu_ids = get_available_gpu_ids()
    else:
        gpu_ids = args.gpu_ids
    
    logger.info(f"Using GPUs: {gpu_ids}")
    
    # Initialize API
    async def startup():
        kwargs = {}
        if args.model_type == "generic":
            if not args.command_template:
                raise ValueError("--command-template is required for generic model type")
            kwargs["command_template"] = args.command_template
            kwargs["model_name"] = args.model_name
        
        await initialize_api(
            model_type=args.model_type,
            model_path=args.model_path,
            gpu_ids=gpu_ids,
            use_fast=args.use_fast,
            use_ultra_fast=args.use_ultra_fast,
            temp_dir=args.temp_dir,
            **kwargs
        )
    
    # Run startup and then start server
    asyncio.run(startup())
    
    logger.info(f"Starting Multi-GPU Image Edit API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
