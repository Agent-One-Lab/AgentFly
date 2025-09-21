#!/usr/bin/env python
"""Script to reformat SFT data into RL format with messages"""

import json
import os
import glob

def create_messages_from_sft(question_content):
    """Create messages format from SFT data"""
    
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
    
    # Create user prompt with the question content
    user_prompt = user_prompt_template.replace("{content}", question_content.strip())
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def process_sft_directory(sft_dir, output_file):
    """Process all SFT data and convert to RL format"""
    
    all_data = []
    
    # Find all JSON files in the SFT directory
    json_files = glob.glob(os.path.join(sft_dir, "*/*.json"))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract fields from SFT data
            question = data.get("question", "").strip()
            
            # Create messages format
            messages = create_messages_from_sft(question)
            
            # Create new RL format item
            rl_item = {
                "messages": messages,
                "question": question  # Keep original question for compatibility
            }
            
            # Add all other fields from SFT data
            for key, value in data.items():
                if key not in ["question", "think"]:
                    rl_item[key] = value
            
            all_data.append(rl_item)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Save all data to output file
    print(f"Writing {len(all_data)} entries to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    return len(all_data)

def main():
    # Define paths
    sft_dir = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata1/SFT"
    output_dir = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata1/RL_format"
    output_file = os.path.join(output_dir, "sft_reformatted.json")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing SFT data from: {sft_dir}")
    print(f"Output will be saved to: {output_file}")
    
    # Process the data
    count = process_sft_directory(sft_dir, output_file)
    
    print(f"\n✅ Successfully reformatted {count} SFT entries to RL format!")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()