#!/usr/bin/env python3
"""
Tests for ImageEditingAgent tools

This module tests the functionality of all image editing tools including:
- Tool registration and schema validation
- Tool functionality with real image processing
- Error handling and edge cases
"""

import os
import tempfile
import shutil
import unittest
import json
import asyncio
from pathlib import Path

# Import the agent class
from .....agents import ImageEditingAgent
from .....tools import TOOL_REGISTRY
from .....utils.vision import open_image_from_any


class TestImageTools(unittest.TestCase):
    """Test suite for image editing tools"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test images
        self.temp_dir = "./temp/image_agent_test"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        
        # Sample image URL for testing
        self.sample_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        

        self.test_image = open_image_from_any(self.sample_image_url)
        self.test_image.save(self.test_image_path)
        
        # Create a fresh agent instance for each test
        self.agent = ImageEditingAgent(
            model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
            template="qwen2.5-vl",
            backend="client"
        )
        self.init_image_id = self.agent._store_image(self.test_image)
    
    def test_auto_inpaint_image_tool(self):
        """Test auto inpainting tool"""
        print("test_auto_inpaint_image_tool")
        
        async def run_test():
            print(f"Tool type: {type(self.agent.detect_objects_tool)}")
            print(f"Tool is_method: {self.agent.detect_objects_tool.is_method}")
            print(f"Tool instance: {self.agent.detect_objects_tool.instance}")
            print(f"Tool type: {type(self.agent.inpaint_image_tool)}")
            print(f"Tool is_method: {self.agent.inpaint_image_tool.is_method}")
            print(f"Tool instance: {self.agent.inpaint_image_tool.instance}")
            print(f"Tool type: {type(self.agent.auto_inpaint_image_tool)}")
            print(f"Tool is_method: {self.agent.auto_inpaint_image_tool.is_method}")
            print(f"Tool instance: {self.agent.auto_inpaint_image_tool.instance}")
            tool_call_result = await self.agent.auto_inpaint_image_tool(
                image_id=self.init_image_id,
                detect_prompt="a dog",
                prompt="a cat",
            )
            self.agent.save_image(tool_call_result["info"]["image_id"], os.path.join(self.temp_dir, "auto_inpaint_image_tool.jpg"))
            return tool_call_result
        
        # Run the async test
        result = asyncio.run(run_test())
        self.assertIsNotNone(result)
    
    
if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
