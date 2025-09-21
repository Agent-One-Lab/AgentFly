# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentFly is an extensible framework for building and training LLM agents with reinforcement learning. It features:

- Multi-turn training with token-level masking
- Decorator-based tool and reward integration
- Asynchronous execution for high-throughput training
- Support for multi-modal agents (vision-language)
- Built-in environments: WebShop, ScienceWorld, ALFWorld, code interpreter

## Architecture

### Core Components

- **agentfly/agents/**: Agent implementations (BaseAgent, ReactAgent, CodeAgent, GuiAgent)
  - `agent_base.py`: Abstract base class all agents inherit from
  - `llm_backends/`: Supports vLLM, Transformers, OpenAI backends
  - `chain/`: Multi-chain rollout management

- **agentfly/tools/**: Tool system with @tool decorator
  - Tools are async by default, support stateful environments
  - Built-in: calculator, code_interpreter, search, webshop, UI automation

- **agentfly/rewards/**: Reward functions for RL training
  - Math, code, QA, GUI, WebShop, ScienceWorld rewards
  - LLM-as-judge reward support

- **agentfly/templates/**: Chat template system for multi-model support
  - Handles tool formatting, vision inputs, system prompts
  - Templates: qwen2.5, llama3, gemma2, etc.

- **agentfly/envs/**: Environment management
  - Containerized envs via enroot/docker
  - Resource pooling and warm start support

- **verl/**: RL training framework (submodule)
  - PPO, GRPO, RLOO, DAPO algorithms
  - FSDP/Megatron for distributed training
  - vLLM/SGLang for generation

## Common Commands

### Installation
```bash
# One-line install (requires conda with Python 3.10)
bash install.sh

# Manual install
pip install -e .
pip install -e '.[verl]' --no-build-isolation
```

### Testing
```bash
# Run unit tests
python -m pytest tests/unit/

# Run GPU tests
bash agentfly/tests/scripts/test_gpu_runs.sh

# Test specific component
python -m pytest tests/unit/agents/test_code_agent.py
```

### Training
```bash
# Start Ray cluster (required for distributed training)
ray start --head --port=6379 --num-cpus=192 --num-gpus=8

# Run code agent training with math reward
cd verl
bash run_agents/run_code_agent.sh

# Other agent training scripts
bash run_agents/run_webshop_agent.sh
bash run_agents/run_gui_agent.sh
bash run_agents/run_vlm_qa.sh  # Vision-language agent
```

### Key Training Parameters
- `model`: Model path (e.g., Qwen/Qwen2.5-3B-Instruct)
- `template`: Chat template (e.g., qwen2.5-no-system-tool)
- `agent_type`: Agent type (code, react, gui)
- `tools`: List of tools (e.g., [code_interpreter])
- `reward_name`: Reward function (e.g., math_reward_tool)
- `max_turns`: Max conversation turns (default: 8)
- `num_chains`: Parallel rollout chains
- `batch_size`: Training batch size

## Development Guidelines

### Adding New Tools
```python
from agentfly.tools import tool

@tool(name="my_tool")
async def my_tool(arg1, arg2):
    # Tool logic
    return {"observation": result}
```

### Adding New Rewards
```python
from agentfly.rewards import reward

@reward(name="my_reward")
def my_reward(prediction, trajectory, **data_fields):
    # Calculate reward
    return score
```

### Custom Agent Implementation
```python
from agentfly.agents import BaseAgent

class CustomAgent(BaseAgent):
    async def generate_async(self, messages_list, **args):
        return await self.llm_engine.generate_async(messages_list, **args)
    
    def parse(self, responses, tools):
        # Parse tool calls from responses
        pass
```

## Data Format

Training data should be JSON with this structure:
```json
[
    {
        "question": "...",
        "optional_field1": "...",
        "optional_field2": "..."
    }
]
```

The `question` field formats input messages. Other fields are passed to reward functions.

## Environment Setup

- Enroot required for containerized tools (code_interpreter, webshop, etc.)
- Redis optional for search result caching
- Ray required for distributed training
- VLLM_USE_V1=1 recommended for vLLM backend

## Key Dependencies

- vLLM 0.10.0 for generation
- Transformers for models
- Ray for distributed compute
- FSDP/Megatron for training
- Enroot for containerization

## Active Development

Now i am building the vlm_as_judge part, and @/mnt/weka/home/renxi.wang/yxwang/AgentFly/agentfly/rewards/vlm_as_judge/test.py works well. And now i want to implemente the vlm_as_judge_client.py and vlm_as_judge_reward.py, you can refer to @/mnt/weka/home/renxi.wang/yxwang/AgentFly/agentfly/rewards/llm_as_judge.

the VLM_as_judge workflow is:

First, prompting the policy model to generate the python code. the data example is below:

{
    "question": "        \" A charged 0.6 kg aluminum sphere is placed at the center of a 1.5-meter by 1-meter by 1-meter glass tank filled with air at 25°C and 1 atm. The tank is horizontally divided into two equal compartments by a non-conductive partition. When a 450 N/C vertical electric field is applied, the sphere rises, clears the partition, and settles in the upper compartment over 13 seconds, as the field balances the gravitational force. \"",
    "think": "<think>\nWe are going to simulate the described physical process.\n Steps:\n 1. We have a glass tank of dimensions 1.5m (length) x 1m (width) x 1m (height). Divided horizontally by a non-conductive partition at 0.5m height.\n 2. A charged aluminum sphere (mass = 0.6 kg) is initially at the center of the tank (so at (0.75, 0.5, 0.5) in a coordinate system where the tank goes from (0,0,0) to (1.5,1,1)).\n 3. The tank is filled with air at 25°C and 1 atm, but since the sphere is charged and we are applying an electric field, we might need to consider air resistance? However, the problem states that the electric field balances the gravitational force. So the sphere will rise at a constant velocity? But note: initially it must accelerate until the forces balance? Actually, the problem says \"the field balances the gravitational force\", meaning that the electric force exactly cancels gravity. Therefore, the net force is zero and the sphere should move at constant velocity? But note: when it clears the partition, there might be a change? However, the partition is non-conductive and we are not told it affects the electric field. So we assume the electric field is uniform.\n\n However, note: the problem says \"the sphere rises, clears the partition, and settles in the upper compartment over 13 seconds\". So the motion is over 13 seconds. Since the partition is at the middle (0.5m), the sphere must move from the bottom compartment (initially at center, which is at z=0.5? But note: the tank is divided horizontally, so the partition is at z=0.5. The center of the tank is at (0.75, 0.5, 0.5). The sphere is initially in the bottom compartment? Actually, the center of the tank is at z=0.5, which is exactly the partition. But the sphere has a radius? We are not given the radius. So we have to assume a small radius so that initially the sphere is below the partition? Or the problem says \"placed at the center\", meaning the center of the sphere is at the center of the tank? Then if the partition is at z=0.5, the sphere would be touching the partition? But to clear it, it must move upward.\n\n Let's assume the sphere is small enough that we can ignore its radius for the purpose of the simulation (or we can set a small radius, say 0.05m). Then initially, the center of the sphere is at (0.75, 0.5, 0.5). But the partition is at z=0.5? Then the sphere is exactly on the partition? That doesn't make sense. Alternatively, the tank is divided into two equal compartments, so the partition must be at z=0.5. The bottom compartment is from z=0 to z=0.5, and the top from z=0.5 to z=1. The sphere is placed at the center of the tank, which is at (0.75,0.5,0.5). But if the sphere has a radius, then part of it is in the top compartment? To avoid confusion, let's assume the partition is a thin plane at z=0.5. The sphere must move from below the partition to above. Therefore, we must assume that the sphere is initially entirely in the bottom compartment? How? The center of the sphere must be below 0.5 - radius. So we have to set an initial position that is below the partition.\n\n However, the problem states: \"placed at the center\". The center of the entire tank is at (0.75,0.5,0.5). But if the tank is divided at z=0.5, then the center of the bottom compartment is at (0.75,0.5,0.25). So let's assume the sphere is placed at the center of the bottom compartment? The problem does not specify. But it says \"at the center of a ... glass tank\". So the entire tank's center is (0.75,0.5,0.5). To avoid the sphere being stuck at the partition, we will assume the sphere is placed at (0.75, 0.5, 0.25) initially (the center of the bottom compartment). Then it rises to the center of the top compartment at (0.75,0.5,0.75) over 13 seconds.\n\n How does it move?\n  - The electric field is vertical and 450 N/C. The force on the sphere: F_e = q * E, where q is the charge.\n  - The gravitational force: F_g = m * g = 0.6 * 9.8 = 5.88 N downward.\n  - Since the electric force balances gravity, we have: q * E = m * g  => q = (m * g) / E = 5.88 / 450 = 0.013066... C.\n\n Therefore, the net force is zero? So the sphere should remain at rest? But the problem says it rises. This implies that the electric force must be upward and greater than gravity? Or perhaps the electric field is applied and then the sphere accelerates until it reaches a terminal velocity? But the problem says the field balances the gravitational force. So why does it move?\n\n Let me re-read: \"the field balances the gravitational force\" -> meaning that the electric force exactly cancels gravity? Then the net force is zero and the sphere should move at constant velocity? But initially it is at rest. How does it start moving? There must be an initial imbalance? Actually, the problem states that the sphere is charged and then the electric field is applied. So at the moment the field is applied, the electric force is upward and exactly equal to gravity? Then the sphere should not move? But then why does it rise?\n\n Alternative interpretation: the electric field is applied, and it provides a force that exactly cancels gravity, so the sphere becomes effectively weightless. Then any small disturbance (like air currents) would cause it to drift? But the problem says it rises and clears the partition. This is confusing.\n\n Another possibility: the electric field is uniform and vertical, but the sphere is charged and initially in contact with the bottom? Then when the field is applied, the sphere might experience a force that lifts it off the bottom? But the problem does not mention the bottom. It says \"placed at the center\", so we assume it is suspended? That doesn't make sense without the field.\n\n Actually, the problem states: \"When a 450 N/C vertical electric field is applied, the sphere rises\". So we must assume that without the field, the sphere would fall to the bottom. But then when the field is applied, it overcomes gravity and lifts the sphere. So initially the sphere is at the bottom of the tank? But the problem says \"placed at the center\". We have to reconcile this.\n\n Let me assume:\n  - Without the electric field, the sphere would fall to the bottom of the tank (z=0). Then when the field is applied, it rises. But the problem says it is placed at the center. So we must simulate the initial condition as the sphere at the bottom? But that contradicts \"placed at the center\".\n\n After careful thought, I think the intended initial position is the center of the bottom compartment: (0.75, 0.5, 0.25). Then the electric field is applied and the sphere rises to the center of the top compartment (0.75, 0.5, 0.75). The distance to travel is 0.5 meters (from z=0.25 to z=0.75). The time taken is 13 seconds. Therefore, the constant velocity would be: v = 0.5 m / 13 s ≈ 0.03846 m/s.\n\n But why does it move at constant velocity? Because the electric force exactly cancels gravity? Then there is no net force, so it should remain at rest? Unless there is an initial push? Or unless the electric field is applied and then the sphere is released? The problem doesn't specify.\n\n However, note that the sphere is in air. There is air resistance. But for such a slow velocity, air resistance might be negligible? Or we can ignore it because the net force is zero? Then the sphere would not accelerate.\n\n How do we explain the motion? We must assume that the sphere was initially held at the starting position and then released. Then when released, with zero net force, it would remain at rest? That doesn't help.\n\n Alternatively, the electric force is slightly greater than gravity? Then it would accelerate until air resistance balances the net force? But the problem says the field balances gravity.\n\n Given the ambiguity, we will simulate the motion as a constant velocity movement from the initial position (0.75, 0.5, 0.25) to the final position (0.75, 0.5, 0.75) over 13 seconds. This is consistent with the problem's description of the time (13 seconds) and the fact that the sphere settles in the upper compartment.\n\n Therefore, the motion in the simulation:\n   - The sphere moves in a straight line vertically from (0.75,0.5,0.25) to (0.75,0.5,0.75) at a constant velocity of 0.5/13 m/s.\n\n We will set:\n   - initial_position = [0.75, 0.5, 0.25]   # (x, y, z) in meters\n   - final_position   = [0.75, 0.5, 0.75]\n   - total_time = 13 seconds.\n\n We will simulate at 30 FPS, so total frames = 13 * 30 = 390.\n\n Steps in the code:\n\n 1. Import necessary libraries: cv2, numpy, and maybe math.\n 2. Define parameters: \n        tank_length = 1.5\n        tank_width = 1.0\n        tank_height = 1.0\n        partition_height = 0.5   # at z=0.5\n        initial_pos = np.array([0.75, 0.5, 0.25])\n        final_pos = np.array([0.75, 0.5, 0.75])\n        total_time = 13.0\n        fps = 30\n        num_frames = int(total_time * fps)\n\n 3. Set up video writer: \n        resolution: let's choose 720p (1280x720) for clarity, but we can adjust the view.\n        We'll create a 3D visualization? But OpenCV is 2D. So we need to project the 3D tank onto a 2D image.\n\n 4. How to visualize?\n    We can use an isometric projection or a fixed perspective. For educational purposes, we can show a side view (x-z plane) and ignore the y-dimension? But the tank has width (y). Alternatively, we can show a 3D-like view by using perspective projection.\n\n    However, for simplicity and clarity, we will use a side view (along the y-axis) so that we see the x and z dimensions. The tank will be represented as a rectangle of 1.5m (length) by 1m (height). The partition will be a horizontal line at z=0.5. The sphere will be a circle moving upward.\n\n    We need to map the physical coordinates to pixel coordinates.\n\n    Let the image be 1280x720. We will set:\n        margin = 50\n        x_pixels_per_meter = (1280 - 2*margin) / tank_length\n        z_pixels_per_meter = (720 - 2*margin) / tank_height\n\n    Then:\n        x_pixel = margin + x * x_pixels_per_meter\n        z_pixel = 720 - (margin + z * z_pixels_per_meter)   # because in images, y increases downward.\n\n    The sphere's position: we only care about x and z for the side view.\n\n 5. Colors:\n        tank: blue (for glass)\n        partition: gray\n        sphere: red\n\n 6. We'll also draw the force vectors? Since the net force is zero, we can draw two vectors: gravity (downward) and electric force (upward) of equal length? But they cancel. We can draw them at the sphere's position.\n\n    Gravity vector: from the sphere downward, length proportional to m*g (we choose a scale for vectors).\n    Electric force vector: from the sphere upward, same length.\n\n 7. Overlay:\n        timestamp: in seconds, with 2 decimal places.\n        parameters: gravity (9.8 m/s²), mass (0.6 kg), electric field (450 N/C), charge (q = (m*g)/E ≈ 0.01307 C), and the current position and velocity.\n\n 8. The simulation loop:\n        for frame in range(num_frames):\n            t = frame / fps   # time in seconds\n            fraction = t / total_time\n            # Linear interpolation from initial_pos to final_pos\n            current_pos = initial_pos + fraction * (final_pos - initial_pos)\n\n            # Create a blank white image\n            image = np.ones((720, 1280, 3), dtype=np.uint8) * 255   # white background\n\n            # Draw the tank (rectangle) and the partition (horizontal line at z=0.5)\n            # Convert tank corners to pixels: (0,0) to (1.5,1) in x,z\n            x0 = margin\n            z0 = margin\n            x1 = 1280 - margin\n            z1 = 720 - margin\n            # Draw the tank as a rectangle\n            cv2.rectangle(image, (x0, z1), (x1, z0), (0, 0, 255), 2)   # blue? but we use BGR: (255,0,0) for blue? -> let's use (0,0,255) for red? but we want blue for glass -> (255,0,0) for blue? Actually, let's use (200,200,200) for the tank? and the partition gray.\n            # Instead, let's use light blue for the tank: (255,200,100) in BGR? -> actually, we can choose.\n\n            # Partition: at z=0.5, so in the tank, from x=0 to x=1.5 at z=0.5.\n            z_partition = 720 - (margin + 0.5 * z_pixels_per_meter)\n            cv2.line(image, (x0, int(z_partition)), (x1, int(z_partition)), (100,100,100), 2)\n\n            # Draw the sphere: at (current_pos[0], current_pos[2])\n            sphere_x = margin + current_pos[0] * x_pixels_per_meter\n            sphere_z = 720 - (margin + current_pos[2] * z_pixels_per_meter)\n            radius_pixels = 10   # fixed radius for the sphere in pixels (since we don't have physical radius)\n            cv2.circle(image, (int(sphere_x), int(sphere_z)), radius_pixels, (0,0,255), -1)   # red sphere\n\n            # Draw force vectors: \n            # Gravity: downward from the sphere, length proportional to m*g (we choose a scale, say 50 pixels for 5.88 N? arbitrary)\n            # Electric force: upward, same length.\n            vector_scale = 50   # pixels per 5.88 N (so 5.88 N -> 50 pixels)\n            # Gravity vector: from center of sphere downward\n            g_start = (int(sphere_x), int(sphere_z))\n            g_end = (int(sphere_x), int(sphere_z + vector_scale))\n            cv2.arrowedLine(image, g_start, g_end, (0,255,0), 2)   # green for gravity\n\n            # Electric force vector: from center of sphere upward\n            e_start = (int(sphere_x), int(sphere_z))\n            e_end = (int(sphere_x), int(sphere_z - vector_scale))\n            cv2.arrowedLine(image, e_start, e_end, (255,0,0), 2)   # blue for electric\n\n            # Overlay text:\n            # Timestamp\n            cv2.putText(image, f\"Time: {t:.2f} s\", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)\n            # Parameters: we can show mass, gravity, E, and charge\n            cv2.putText(image, f\"Mass: 0.6 kg\", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)\n            cv2.putText(image, f\"Gravity: 9.8 m/s^2\", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)\n            cv2.putText(image, f\"Electric Field: 450 N/C\", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)\n            cv2.putText(image, f\"Charge: {0.6*9.8/450:.4f} C\", (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)\n            # Current position and velocity\n            velocity = (final_pos[2] - initial_pos[2]) / total_time   # constant velocity in z\n            cv2.putText(image, f\"Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) m\", (20,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)\n            cv2.putText(image, f\"Velocity: {velocity:.4f} m/s\", (20,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)\n\n            # Write frame to video\n\n 9. After the loop, release the video writer.\n\n However, note: the problem says the tank is 1.5m by 1m by 1m. Our side view shows the length and height, but the width is not shown. We are ignoring the y-dimension. This is acceptable for a side view.\n\n 10. We also have to consider that the sphere moves only in z, so in our side view we see the entire motion.\n\n 11. The code must accept the output file name as a command line argument.\n\n Implementation:\n\n    We'll use sys.argv to get the output file name.\n\n    Example: if the script is run as \"python script.py output.mp4\", then sys.argv[1] is \"output.mp4\".\n\n Steps:\n\n    import sys\n    if len(sys.argv) < 2:\n        print(\"Usage: python script.py output.mp4\")\n        sys.exit(1)\n    output_filename = sys.argv[1]\n\n 12. We'll set up the video writer with:\n        fourcc = cv2.VideoWriter_fourcc(*'mp4v')\n        out = cv2.VideoWriter(output_filename, fourcc, fps, (1280,720))\n\n 13. We'll do the loop for num_frames.\n\n 14. After the loop, release the video writer.\n\n 15. We assume that the sphere is moving at constant velocity. We are ignoring any initial acceleration because the problem states the field balances gravity. So we model constant velocity.\n\n 16. We are ignoring air resistance and any other forces.\n\n Let's code accordingly.\n\n Note: We are not simulating the dynamics with forces and acceleration because the net force is zero. We are just moving the sphere at constant velocity.\n\n But note: the problem says \"the sphere rises, clears the partition, and settles\". We are simulating that.\n\n We'll set the sphere to move from z=0.25 to z=0.75 in 13 seconds.\n\n However, what if we want to show the entire tank? The tank height is 1m, so we show from z=0 to z=1. Our initial position at z=0.25 and final at z=0.75 are within the tank.\n\n Let's code accordingly.\n\n We'll also draw the tank with the partition.\n\n Important: The partition is non-conductive, but we are not modeling any effect on the electric field. So the electric field is uniform throughout.\n\n We'll now write the code.\n\n Note: We must install opencv if not present. We can try to import and if fails, install using pip. But the problem says the code should install dependencies. We can do:\n\n    try:\n        import cv2\n        import numpy as np\n    except ImportError:\n        import subprocess\n        import sys\n        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"opencv-python\", \"numpy\"])\n        import cv2\n        import numpy as np\n\n But note: the problem says \"The code should install the dependencies in the code.\" However, it's better to do it at the beginning.\n\n However, if we are in an environment without pip, this might fail. But for the purpose of this simulation, we assume pip is available.\n\n Alternatively, we can just import and let it fail, but the requirement says to install in the code.\n\n We'll do:\n\n    import subprocess\n    import sys\n    import importlib\n\n    required = ['cv2', 'numpy']\n    for lib in required:\n        try:\n            importlib.import_module(lib)\n        except ImportError:\n            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy'])\n            break\n\n    import cv2\n    import numpy as np\n\n But note: we already broke the loop if one is missing, then we import. But we might have installed both even if only one is missing? We can do:\n\n    try:\n        import cv2\n        import numpy as np\n    except:\n        import subprocess\n        import sys\n        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy'])\n        import cv2\n        import numpy as np\n\n But note: this might be considered bad practice because it catches any exception. However, for the purpose of this simulation, we do it.\n\n Let's write the code accordingly.\n\n We'll also include sys for command line arguments.\n\n Code structure:\n\n    Install imports if needed\n    Import necessary modules\n    Parse command line arguments\n    Set up parameters\n    Set up video writer\n    Loop and create frames\n    Clean up\n\n Let's code.\n</think>\n```python\nimport sys\nimport subprocess\nimport importlib\n\nrequired_libraries = ['cv2', 'numpy']\nfor lib in required_libraries:\n    try:\n        importlib.import_module(lib)\n    except ImportError:\n        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', 'numpy'])\n        break\n\nimport cv2\nimport numpy as np\n\nif len(sys.argv) < 2:\n    print(\"Usage: python script.py output_filename.mp4\")\n    sys.exit(1)\n\noutput_file = sys.argv[1]\n\n# Physical parameters\ntank_length = 1.5\ntank_width = 1.0\ntank_height = 1.0\npartition_height = 0.5\ninitial_z = 0.25\nfinal_z = 0.75\ntotal_time = 13.0\nmass = 0.6\ngravity = 9.8\nE_field = 450.0\ncharge = (mass * gravity) / E_field\n\n# Video parameters\nfps = 30\nwidth, height = 1280, 720\nnum_frames = int(total_time * fps)\nfourcc = cv2.VideoWriter_fourcc(*'mp4v')\nout = cv2.VideoWriter(output_file, fourcc, fps, (width, height))\n\n# Scaling factors for visualization\nmargin = 50\nx_scale = (width - 2 * margin) / tank_length\nz_scale = (height - 2 * margin) / tank_height\nsphere_radius_px = 15\nforce_scale = 30\n\ndef world_to_pixel(x, z):\n    px = int(margin + x * x_scale)\n    pz = int(height - margin - z * z_scale)\n    return px, pz\n\nfor frame_idx in range(num_frames):\n    t = frame_idx / fps\n    progress = min(1.0, t / total_time)\n    current_z = initial_z + (final_z - initial_z) * progress\n    current_pos = [tank_length/2, tank_width/2, current_z]\n    velocity = (final_z - initial_z) / total_time\n    \n    # Create white background\n    img = np.ones((height, width, 3), dtype=np.uint8) * 255\n    \n    # Draw tank\n    tank_tl = world_to_pixel(0, tank_height)\n    tank_br = world_to_pixel(tank_length, 0)\n    cv2.rectangle(img, tank_tl, tank_br, (200, 200, 255), 2)\n    \n    # Draw partition\n    part_start = world_to_pixel(0, partition_height)\n    part_end = world_to_pixel(tank_length, partition_height)\n    cv2.line(img, part_start, part_end, (100, 100, 100), 2)\n    \n    # Draw sphere\n    sphere_pos = world_to_pixel(tank_length/2, current_z)\n    cv2.circle(img, sphere_pos, sphere_radius_px, (0, 0, 255), -1)\n    \n    # Draw force vectors\n    g_vector_end = (sphere_pos[0], sphere_pos[1] + force_scale)\n    cv2.arrowedLine(img, sphere_pos, g_vector_end, (0, 150, 0), 2, tipLength=0.3)\n    \n    e_vector_end = (sphere_pos[0], sphere_pos[1] - force_scale)\n    cv2.arrowedLine(img, sphere_pos, e_vector_end, (255, 0, 0), 2, tipLength=0.3)\n    \n    # Draw text overlays\n    cv2.putText(img, f\"Time: {t:.2f}s / {total_time}s\", (20, 40), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)\n    cv2.putText(img, f\"Mass: {mass} kg\", (width-300, 40), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n    cv2.putText(img, f\"Gravity: {gravity} m/s\", (width-300, 80), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n    cv2.putText(img, f\"E-Field: {E_field} N/C\", (width-300, 120), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n    cv2.putText(img, f\"Charge: {charge:.5f} C\", (width-300, 160), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n    cv2.putText(img, f\"Velocity: {velocity:.4f} m/s\", (width-300, 200), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n    cv2.putText(img, f\"Position: (0.75, 0.50, {current_z:.2f}) m\", (width-300, 240), \n                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)\n    \n    # Write frame\n    out.write(img)\n\nout.release()\n```"
}

Secondly, i get the output and then use the code to generate the video, here you can refer to @/mnt/weka/home/renxi.wang/yxwang/VLM-as-judge-proj, there are some files regarding code and video generation.

Thirdly, i have started a vlm_server as judge, you need to implemente the vlm_as_judge part, you can refer to @/mnt/weka/home/renxi.wang/yxwang/AgentFly/agentfly/rewards/vlm_as_judge/test.py. After send the request, you need to parse the overall reward as the vlm_as_judge reward.

the rl data format is below:

{
    "question": "        \" A 6 kg cube of polished marble, with a side length of 0.3 meters, is released from the top of a 40-degree inclined plane inside a 4-meter-long, 3-meter-wide, and 2-meter-high metal chamber. The plane is lined with a layer of felt, providing a friction coefficient of 0.3. The chamber's interior is heated to 30°C, causing a gentle convection current that introduces a slight, variable force acting against the marble's descent. Over 16 seconds, the marble cube slides down the 2.5-meter plane, the friction and convection currents adding complexity to its motion. \"",
    "Level": 3,
    "vlm_questions": {
        "enableAnnotator": "Yes",
        "summarize": "A 6 kg cube of polished marble is released from the top of a 40-degree inclined plane inside a metal chamber. The plane is lined with felt, providing friction, while convection currents from the heated chamber oppose the cube's descent. Over 16 seconds, the cube slides down the 2.5-meter plane, with friction and convection adding complexity to its motion.",
        "vlm_questions": [
            {
                "index": "1",
                "question": "A cube is released from the top of an inclined plane.",
                "weight": 1.0
            },
            {
                "index": "2",
                "question": "The inclined plane is inside a chamber.",
                "weight": 0.8
            },
            {
                "index": "3",
                "question": "The cube slides down the inclined plane over a period.",
                "weight": 0.7
            },
            {
                "index": "4",
                "question": "The cube's motion is influenced by forces acting against its descent.",
                "weight": 0.9
            },
            {
                "index": "5",
                "question": "The cube descends the entire length of the inclined plane.",
                "weight": 0.6
            }
        ]
    }
}