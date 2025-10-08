import sys
import os
from ....rewards.vlm_as_judge.vlm_as_judge_client import VLMClient, create_vlm_prompt, _extract_json_list
from ....rewards.vlm_as_judge.vlm_as_judge_reward import VideoGenerator, extract_vlm_questions_from_data, calculate_weighted_reward, pass_fail_reward
from ....rewards.vlm_as_judge.vlm_as_judge_reward import vlm_as_judge_pass_reward
from pathlib import Path
import asyncio
import sys
from pathlib import Path
import traceback
import time
import json
import os
import re
import subprocess
import tempfile
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, Any


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


    """Test VLM-as-judge reward function"""
    import sys
    
    # Test data - real physics example with charged sphere
    test_data = {
        "question": "A charged 0.6 kg aluminum sphere is placed at the center of a 1.5-meter by 1-meter by 1-meter glass tank filled with air at 25°C and 1 atm. The tank is horizontally divided into two equal compartments by a non-conductive partition. When a 450 N/C vertical electric field is applied, the sphere rises, clears the partition, and settles in the upper compartment over 13 seconds, as the field balances the gravitational force.",
        "Level": 3,
        "vlm_questions": {
            "enableAnnotator": "Yes",
            "summarize": "A charged 0.6 kg aluminum sphere is placed at the center of a 1.5m x 1m x 1m glass tank filled with air at 25°C and 1 atm. The tank is divided by a non-conductive partition. A 450 N/C vertical electric field causes the sphere to rise, clear the partition, and settle in the upper compartment, balancing gravitational force over 13 seconds.",
            "vlm_questions": [
                {
                    "index": "1",
                    "question": "A non-conductive partition divides the tank horizontally into two equal compartments.",
                    "weight": 1.0
                },
                {
                    "index": "2",
                    "question": "The sphere is initially placed at the center of the tank.",
                    "weight": 1.0
                },
                {
                    "index": "3",
                    "question": "The sphere rises vertically when the electric field is applied.",
                    "weight": 1.0
                },
                {
                    "index": "4",
                    "question": "The sphere clears the partition and enters the upper compartment.",
                    "weight": 1.0
                },
                {
                    "index": "5",
                    "question": "The sphere settles in the upper compartment after moving.",
                    "weight": 1.0
                }
            ]
        }
    }
    
    # Sample physics simulation code
    sample_code = '''
import sys
import subprocess
import importlib

required_libraries = ['cv2', 'numpy']
for lib in required_libraries:
    try:
        importlib.import_module(lib)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy'])
        break

import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python script.py output_filename.mp4")
    sys.exit(1)

output_file = sys.argv[1]

# Physical parameters
tank_length = 1.5
tank_width = 1.0
tank_height = 1.0
partition_height = 0.5
initial_z = 0.25
final_z = 0.75
total_time = 13.0
mass = 0.6
gravity = 9.8
E_field = 450.0
charge = (mass * gravity) / E_field

# Video parameters
fps = 30
width, height = 1280, 720
num_frames = int(total_time * fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Scaling factors for visualization
margin = 50
x_scale = (width - 2 * margin) / tank_length
z_scale = (height - 2 * margin) / tank_height
sphere_radius_px = 15
force_scale = 30

def world_to_pixel(x, z):
    px = int(margin + x * x_scale)
    pz = int(height - margin - z * z_scale)
    return px, pz

for frame_idx in range(num_frames):
    t = frame_idx / fps
    progress = min(1.0, t / total_time)
    current_z = initial_z + (final_z - initial_z) * progress
    current_pos = [tank_length/2, tank_width/2, current_z]
    velocity = (final_z - initial_z) / total_time
    
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw tank
    tank_tl = world_to_pixel(0, tank_height)
    tank_br = world_to_pixel(tank_length, 0)
    cv2.rectangle(img, tank_tl, tank_br, (200, 200, 255), 2)
    
    # Draw partition
    part_start = world_to_pixel(0, partition_height)
    part_end = world_to_pixel(tank_length, partition_height)
    cv2.line(img, part_start, part_end, (100, 100, 100), 2)
    
    # Draw sphere
    sphere_pos = world_to_pixel(tank_length/2, current_z)
    cv2.circle(img, sphere_pos, sphere_radius_px, (0, 0, 255), -1)
    
    # Draw force vectors
    g_vector_end = (sphere_pos[0], sphere_pos[1] + force_scale)
    cv2.arrowedLine(img, sphere_pos, g_vector_end, (0, 150, 0), 2, tipLength=0.3)
    
    e_vector_end = (sphere_pos[0], sphere_pos[1] - force_scale)
    cv2.arrowedLine(img, sphere_pos, e_vector_end, (255, 0, 0), 2, tipLength=0.3)
    
    # Draw text overlays
    cv2.putText(img, f"Time: {t:.2f}s / {total_time}s", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, f"Mass: {mass} kg", (width-300, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Gravity: {gravity} m/s^2", (width-300, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"E-Field: {E_field} N/C", (width-300, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Charge: {charge:.5f} C", (width-300, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Velocity: {velocity:.4f} m/s", (width-300, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Position: (0.75, 0.50, {current_z:.2f}) m", (width-300, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Write frame
    out.write(img)

out.release()
'''
    
    print("="*70)
    print("Testing VLM-as-Judge Reward")
    print("="*70)
    
    # Test video generator
    print("\n1. Testing VideoGenerator...")
    gen = VideoGenerator()
    
    # Test code extraction
    test_response = f"```python\n{sample_code}\n```"
    code = gen.extract_code_from_response(test_response)
    print(f"   ✓ Extracted {len(code) if code else 0} chars of code")
    
    # Test question extraction
    print("\n2. Testing question extraction...")
    all_q, summary, q_list = extract_vlm_questions_from_data(test_data)
    print(f"   ✓ Extracted {len(q_list)} questions")
    print(f"   ✓ Summary: {summary[:50]}...")
    
    # Test reward calculation
    print("\n3. Testing reward calculation...")
    test_results = [
        {"index": "1", "result": "True", "confidence_score": "5"},
        {"index": "2", "result": "True", "confidence_score": "4"},
        {"index": "3", "result": "False", "confidence_score": "3"}
    ]
    # reward = calculate_weighted_reward(test_results, q_list)
    reward = pass_fail_reward(test_results, q_list)
    print(f"   ✓ Calculated reward: {reward:.3f}")
    
    print("\n4. Testing full reward function...")
    print("   Note: This requires VLM server to be running")
    
    async def test_reward():
        """Async wrapper for testing the reward function"""
        try:
            # Test with physics simulation prediction including think tags
            prediction_with_think = f"<think>\n{test_data.get('think', 'Analyzing the physics problem...')}\n</think>\n```python\n{sample_code}\n```"
            
            # Alternative: Test with just code
            prediction = f"```python\n{sample_code}\n```"
            
            reward_value = await vlm_as_judge_pass_reward(
                prediction=prediction,
                trajectory={},
                **test_data
            )
            print(f"   ✓ Reward function returned: {reward_value}")
            return reward_value
        except Exception as e:
            print(f"   ⚠ Reward function test failed (expected if VLM server not running)")
            print(f"     Error: {e}")
            return None
    
    # Run the async test
    import asyncio
    result = asyncio.run(test_reward())
    
    print("\nTest complete!")
    if result:
        print(f"Final reward score: {result.get('reward', 0.0):.3f}")