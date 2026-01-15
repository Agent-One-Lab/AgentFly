#!/usr/bin/env python
"""Test script for vlm_as_judge_pass_reward_rebuttal"""

import asyncio
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from agentfly.rewards.vlm_as_judge.vlm_as_judge_reward import vlm_as_judge_pass_reward_rebuttal


# Sample test data
SAMPLE_QUESTION = """A charged 0.6 kg aluminum sphere is placed at the center of a 1.5-meter by 1-meter by 1-meter glass tank filled with air at 25°C and 1 atm. The tank is horizontally divided into two equal compartments by a non-conductive partition. When a 450 N/C vertical electric field is applied, the sphere rises, clears the partition, and settles in the upper compartment over 13 seconds, as the field balances the gravitational force."""

SAMPLE_PREDICTION = """```python
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

    # Write frame
    out.write(img)

out.release()
```"""


async def test_rebuttal_reward():
    """Test the vlm_as_judge_pass_reward_rebuttal function"""

    print("=" * 80)
    print("Testing vlm_as_judge_pass_reward_rebuttal")
    print("=" * 80)

    print(f"\nQuestion: {SAMPLE_QUESTION[:100]}...")
    print(f"\nPrediction length: {len(SAMPLE_PREDICTION)} characters")

    print("\nCalling reward function...")
    result = await vlm_as_judge_pass_reward_rebuttal(
        prediction=SAMPLE_PREDICTION,
        question=SAMPLE_QUESTION,
    )

    print(f"\n{'=' * 80}")
    print(f"RESULT: {result}")
    print(f"{'=' * 80}")

    reward = result.get("reward", 0.0)
    if reward == 1.0:
        print("✓ VIDEO PASSED")
    elif reward == 0.0:
        print("✗ VIDEO FAILED")
    else:
        print(f"? UNEXPECTED REWARD: {reward}")

    return result


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_rebuttal_reward())
    sys.exit(0 if result.get("reward", 0.0) > 0 else 1)
