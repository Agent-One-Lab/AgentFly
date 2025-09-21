#!/usr/bin/env python
"""Reformat RL and val data in finaldata2 to messages format.

Outputs:
- /mnt/sharefs/users/haonan.li/ViPhy/finaldata2/RL_format/merged_reformatted.json
- /mnt/sharefs/users/haonan.li/ViPhy/finaldata2/val_format/val_reformatted.json

Schema:
{
  "messages": [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": USER_PROMPT_WITH_QUESTION}
  ],
  ... (preserve other original fields like Level, vlm_questions)
}
"""

import json
import glob
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert computational physicist specializing in scientific visualization and simulation. \n"
    "You also an excellent programmer.\n"
    "Your expertise includes creating educational physics simulations that effectively communicate complex physical phenomena to diverse audiences."
)

USER_PROMPT_TMPL = (
    "Your task is to write a Python script that generates an educational video simulating a physical process. The video will be used in academic settings to help students better understand and visualize physics concepts.\n"
    "Given a textual description of a physical process:\n"
    "\"{content}\"\n"
    "Write a Python script that simulates this process and outputs a video saved as:\n"
    "\"name.mp4\", here, \"name\" is a user-defined parameter passed when running the script.\n\n"
    "Requirements for the video output:\n"
    "1. Use clear and distinct colors to represent different objects, trajectories, or forces.\n"
    "2. Overlay a real-time timestamp that updates continuously throughout the simulation.\n"
    "3. Display all relevant parameter values (e.g., gravity, speed, angle) clearly on the screen.\n"
    "4. Ensure the camera view is wide enough to fully capture the entire motion, adjusting dynamically if needed. Ensure the camera view is the best view for the simulation to let the viewer see the whole process.\n"
    "5. Provide smooth and continuous animation at a consistent frame rate (30 FPS).\n"
    "6. Maintain a clean, uncluttered visual style with minimal distractions and a neutral background.\n"
    "7. Keep the video duration between 10 and 20 seconds, slow enough to allow viewers to observe and understand the key transitions.\n"
    "8. Save the output as an MP4 video in a suitable resolution (at least 360p).\n"
    "9. When the process is finish, the video should finish also.\n"
    "10. Use OpenCV to generate the video, and ensure the code is correct, complete, and runnable without any errors.\n\n"
    "Focus on clarity, interpretability, and visual appeal to make the video intuitive and easy to understand for both technical and non-technical audiences.\n\n"
    "Physics Simulation:\n"
    "- Implement precise physical equations\n"
    "- Use appropriate time steps for smooth motion\n"
    "- Include relevant force vectors and trajectories\n\n"
    "The final code should:\n"
    "1. Initialize all necessary libraries and variables\n"
    "2. Set up the video writer with specified parameters\n"
    "3. Implement the physics calculations\n"
    "4. Create and save the animation\n"
    "5. Include error handling and resource cleanup\n\n"
    "Ensure the code follows PEP 8 style guidelines and includes comments explaining key components. The simulation should prioritize educational value while maintaining scientific accuracy.\n\n"
    "Instructions:\n"
    "1. Output only the complete Python code. Do not include explanations or comments. The format should be like this:\n"
    "```python\n"
    "import cv2\n"
    "import numpy as np\n"
    "```\n"
    "2. The code should install the dependencies in the code.\n"
    "3. The code should be runnable without any errors.\n"
    "4. The code should be complete and self-contained.\n"
    "5. The code should be correct and accurate.\n"
    "5. The code should be efficient and optimized.\n"
    "6. The code should be easy to understand and modify.\n"
)


def to_messages(question: str):
    """Build messages array with substituted question content."""
    user_prompt = USER_PROMPT_TMPL.replace("{content}", (question or "").strip())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def reformat_dir(src_dir: str, out_path: str) -> int:
    """Aggregate *.json under src_dir (or src_dir/all) and write reformatted list to out_path."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not files and os.path.isdir(os.path.join(src_dir, "all")):
        files = sorted(glob.glob(os.path.join(src_dir, "all", "*.json")))

    items = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                d = json.load(f)
        except Exception as e:
            print(f"! Skip {fp}: {e}")
            continue

        q = d.get("question", "")
        new_d = {"messages": to_messages(q)}

        # Preserve other fields
        for k, v in d.items():
            if k != "question":
                new_d[k] = v

        items.append(new_d)

    # Ensure output directory exists and write
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    return len(items)


def main():
    rl_src = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata2/RL"
    rl_out = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata2/RL_format/merged_reformatted.json"
    val_src = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata2/val"
    val_out = "/mnt/sharefs/users/haonan.li/ViPhy/finaldata2/val_format/val_reformatted.json"

    print("Reformatting RL ...")
    rl_n = reformat_dir(rl_src, rl_out)
    print(f"RL items: {rl_n} -> {rl_out}")

    print("Reformatting val ...")
    val_n = reformat_dir(val_src, val_out)
    print(f"Val items: {val_n} -> {val_out}")

    print("Done.")


if __name__ == "__main__":
    main()

