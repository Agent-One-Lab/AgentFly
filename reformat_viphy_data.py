#!/usr/bin/env python
"""Script to reformat ViPhy data with system and user prompts in messages format"""

import json
import os

def reformat_data(input_path, output_path):
    """Reformat JSON data to include messages with system and user prompts"""
    
    system_prompt = """You are an expert computational physicist specializing in scientific visualization and simulation. 
You also an excellent programmer.
Your expertise includes creating educational physics simulations that effectively communicate complex physical phenomena to diverse audiences."""
    
    user_prompt_template = """Your task is to write a Python script that generates an educational video simulating a physical process. The video will be used in academic settings to help students better understand and visualize physics concepts.
Given a textual description of a physical process:
"{content}"
Write a Python script that simulates this process and outputs a video saved as:
"name.mp4", here, "name" is a user-defined parameter passed when running the script.

Requirements for the video output:
1. Use clear and distinct colors to represent different objects, trajectories, or forces.
2. Overlay a real-time timestamp that updates continuously throughout the simulation.
3. Display all relevant parameter values (e.g., gravity, speed, angle) clearly on the screen.
4. Ensure the camera view is wide enough to fully capture the entire motion, adjusting dynamically if needed. Ensure the camera view is the best view for the simulation to let the viewer see the whole process.
5. Provide smooth and continuous animation at a consistent frame rate (30 FPS).
6. Maintain a clean, uncluttered visual style with minimal distractions and a neutral background.
7. Keep the video duration between 10 and 20 seconds, slow enough to allow viewers to observe and understand the key transitions.
8. Save the output as an MP4 video in a suitable resolution (at least 360p).
9. When the process is finish, the video should finish also.
10. Use OpenCV to generate the video, and ensure the code is correct, complete, and runnable without any errors.

Focus on clarity, interpretability, and visual appeal to make the video intuitive and easy to understand for both technical and non-technical audiences.

Physics Simulation:
- Implement precise physical equations
- Use appropriate time steps for smooth motion
- Include relevant force vectors and trajectories

The final code should:
1. Initialize all necessary libraries and variables
2. Set up the video writer with specified parameters
3. Implement the physics calculations
4. Create and save the animation
5. Include error handling and resource cleanup

Ensure the code follows PEP 8 style guidelines and includes comments explaining key components. The simulation should prioritize educational value while maintaining scientific accuracy.

Instructions:
1. Output only the complete Python code. Do not include explanations or comments. The format should be like this:
```python
import cv2
import numpy as np
```
2. The code should install the dependencies in the code.
3. The code should be runnable without any errors.
4. The code should be complete and self-contained.
5. The code should be correct and accurate.
5. The code should be efficient and optimized.
6. The code should be easy to understand and modify."""
    
    # Read input data
    print(f"Reading from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Reformat each entry
    reformatted_data = []
    for item in data:
        # Extract the question content
        question_content = item.get("question", "").strip()
        
        # Create user prompt with the question content
        user_prompt = user_prompt_template.replace("{content}", question_content)
        
        # Create new formatted item
        new_item = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Add all other fields from the original item except 'question'
        for key, value in item.items():
            if key != "question":
                new_item[key] = value
        
        reformatted_data.append(new_item)
    
    # Save reformatted data
    print(f"Writing to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(reformatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully reformatted {len(reformatted_data)} entries")
    return len(reformatted_data)

def main():
    # Define input and output paths
    base_path = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata1/RL_format"
    
    files_to_process = [
        ("merged.json", "merged_reformatted.json"),
        ("val.json", "val_reformatted.json")
    ]
    
    for input_file, output_file in files_to_process:
        input_path = os.path.join(base_path, input_file)
        output_path = os.path.join(base_path, output_file)
        
        if os.path.exists(input_path):
            print(f"\nProcessing {input_file}...")
            count = reformat_data(input_path, output_path)
            print(f"✓ Completed: {count} entries reformatted")
        else:
            print(f"⚠ Warning: {input_path} not found, skipping...")
    
    print("\n✅ All files processed successfully!")

if __name__ == "__main__":
    main()