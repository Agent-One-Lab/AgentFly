#!/usr/bin/env python3
"""
Test script for Qwen3Template implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentfly.templates.templates import Qwen3Template

def test_qwen3_template():
    """Test the Qwen3Template with various scenarios"""
    
    # Create a Qwen3Template instance
    template = Qwen3Template(
        name="qwen3-test",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        stop_words=["<|im_end|>"],
        generation_prompt="<|im_start|>assistant\n",
    )
    
    # Test case 1: Basic conversation without thinking
    print("=== Test Case 1: Basic conversation without thinking ===")
    messages1 = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    
    prompt1, elements1, roles1 = template.render(messages1, add_generation_prompt=False, enable_thinking=False)
    print("Prompt:")
    print(prompt1)
    print()
    
    # Test case 2: Conversation with thinking content that should be cleaned
    print("=== Test Case 2: Conversation with thinking content (should be cleaned) ===")
    messages2 = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>I need to add 2 and 2 together. This is basic arithmetic.</think>The answer is 4."}
    ]
    
    prompt2, elements2, roles2 = template.render(messages2, add_generation_prompt=False, enable_thinking=False)
    print("Prompt (thinking content should be removed):")
    print(prompt2)
    print()
    
    # Test case 3: With add_generation_prompt=True and enable_thinking=False
    print("=== Test Case 3: With generation prompt and enable_thinking=False ===")
    messages3 = [
        {"role": "user", "content": "Tell me a joke"}
    ]
    
    prompt3, elements3, roles3 = template.render(messages3, add_generation_prompt=True, enable_thinking=False)
    print("Prompt (should include empty think tokens):")
    print(prompt3)
    print()
    
    # Test case 4: With add_generation_prompt=True and enable_thinking=True
    print("=== Test Case 4: With generation prompt and enable_thinking=True ===")
    prompt4, elements4, roles4 = template.render(messages3, add_generation_prompt=True, enable_thinking=True)
    print("Prompt (should NOT include empty think tokens):")
    print(prompt4)
    print()
    
    # Test case 5: Last message is assistant with enable_thinking=False
    print("=== Test Case 5: Last message is assistant with enable_thinking=False ===")
    messages5 = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have access to current weather data."}
    ]
    
    prompt5, elements5, roles5 = template.render(messages5, add_generation_prompt=False, enable_thinking=False)
    print("Prompt (last assistant message should have empty think tokens):")
    print(prompt5)
    print()
    
    print("All tests completed!")

if __name__ == "__main__":
    test_qwen3_template()

