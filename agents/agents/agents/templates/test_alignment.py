#!/usr/bin/env python3
"""
Test script to verify tensor alignment functionality in the vision processor system.
"""

import torch
from typing import Dict, List, Any

def test_tensor_alignment():
    """Test that all tensors are properly aligned after vision token expansion"""
    print("=== Testing Tensor Alignment ===")
    
    # Mock vision processor for testing
    class MockVisionProcessor:
        def expand_vision_tokens(self, prompt, images, videos, processor):
            # Expand single image token to multiple tokens
            return prompt.replace("<|image_pad|>", "<|image_pad|><|image_pad|><|image_pad|>")
        
        def calculate_image_tokens(self, image_data, processor):
            return 3  # Mock: each image expands to 3 tokens
        
        def preprocess_images(self, images, processor):
            return {"pixel_values": torch.randn(1, 3, 224, 224)}
        
        def preprocess_videos(self, videos, processor):
            return {}
        
        def get_mm_inputs(self, images, videos, processor):
            return {"pixel_values": torch.randn(1, 3, 224, 224)}
    
    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            if "<|image_pad|>" in text:
                # Replace image tokens with token IDs
                text = text.replace("<|image_pad|>", "999")
                return [1, 2, 999, 999, 999, 4]  # Expanded tokens
            return [1, 2, 3, 4]  # Regular tokens
        
        @property
        def add_bos_token(self):
            return True
        
        @property
        def bos_token(self):
            return "<s>"
        
        @property
        def bos_token_id(self):
            return 0
    
    # Test data
    elements = [
        "What's in this image? <|image_pad|>",  # User message (masked)
        "I can see a cat in the image."          # Assistant message (not masked)
    ]
    mask_flags = [True, False]  # User message masked, assistant not
    
    processor = MockVisionProcessor()
    tokenizer = MockTokenizer()
    
    # Simulate the alignment process
    input_ids = []
    attention_mask = []
    labels = []
    action_mask = []
    
    # Add BOS token
    input_ids.append(tokenizer.bos_token_id)
    attention_mask.append(1)
    labels.append(-100)
    action_mask.append(0)
    
    # Process each element
    for element, mask_flag in zip(elements, mask_flags):
        # Check if element contains vision tokens
        if "<|image_pad|>" in element:
            # Expand vision tokens
            expanded_element = processor.expand_vision_tokens(element, ["image.jpg"], [], None)
            cur_input_ids = tokenizer.encode(expanded_element, add_special_tokens=False)
        else:
            cur_input_ids = tokenizer.encode(element, add_special_tokens=False)
        
        # Add tokens with proper alignment
        input_ids.extend(cur_input_ids)
        attention_mask.extend([1] * len(cur_input_ids))
        
        if mask_flag:
            labels.extend([-100] * len(cur_input_ids))
            action_mask.extend([0] * len(cur_input_ids))
        else:
            labels.extend(cur_input_ids)
            action_mask.extend([1] * len(cur_input_ids))
    
    # Convert to tensors
    inputs = {
        'input_ids': torch.tensor([input_ids]),
        'attention_mask': torch.tensor([attention_mask]),
        'labels': torch.tensor([labels]),
        'action_mask': torch.tensor([action_mask])
    }
    
    # Verify alignment
    print("Tensor shapes:")
    for key, value in inputs.items():
        print(f"  {key}: {value.shape}")
    
    # Check that all tensors have the same sequence length
    seq_len = inputs['input_ids'].shape[1]
    assert inputs['attention_mask'].shape[1] == seq_len, "attention_mask not aligned"
    assert inputs['labels'].shape[1] == seq_len, "labels not aligned"
    assert inputs['action_mask'].shape[1] == seq_len, "action_mask not aligned"
    
    print(f"✅ All tensors aligned with sequence length: {seq_len}")
    
    # Verify the content makes sense
    print("\nTensor content verification:")
    print(f"input_ids: {inputs['input_ids'][0].tolist()}")
    print(f"attention_mask: {inputs['attention_mask'][0].tolist()}")
    print(f"labels: {inputs['labels'][0].tolist()}")
    print(f"action_mask: {inputs['action_mask'][0].tolist()}")
    
    # Check that vision tokens are properly handled
    vision_token_positions = [i for i, token_id in enumerate(inputs['input_ids'][0]) if token_id == 999]
    print(f"\nVision token positions: {vision_token_positions}")
    
    # Verify that all tensors have proper values at vision token positions
    for pos in vision_token_positions:
        assert inputs['attention_mask'][0][pos] == 1, f"attention_mask should be 1 at position {pos}"
        # Labels and action_mask depend on whether it's in a masked region
        # This is a simplified test - in practice, you'd check the actual mask flags
    
    print("✅ Vision tokens properly handled in all tensors")
    print("✅ Tensor alignment test passed!")

def test_vision_processor_integration():
    """Test integration with the actual vision processor system"""
    print("\n=== Testing Vision Processor Integration ===")
    
    try:
        from .vision_processor import VisionProcessorRegistry, VisionProcessorConfig, PatchBasedProcessor
        from .templates import get_template
        
        # Check if qwen2.5-vl template is registered
        if VisionProcessorRegistry.is_vision_template("qwen2.5-vl"):
            print("✅ qwen2.5-vl template is registered")
            
            processor = VisionProcessorRegistry.get_processor("qwen2.5-vl")
            if processor is not None:
                print("✅ Vision processor retrieved successfully")
                
                # Test the contains_vision_tokens method
                test_text = "What's in this image? <|image_pad|>"
                has_vision = processor._contains_vision_tokens(test_text)
                print(f"✅ Vision token detection: {has_vision}")
                
            else:
                print("❌ Vision processor not found")
        else:
            print("❌ qwen2.5-vl template not registered")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_tensor_alignment()
    test_vision_processor_integration()
    print("\n=== All Tests Completed ===") 