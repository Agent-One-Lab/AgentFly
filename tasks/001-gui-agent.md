# Task 001: Implement GUI Agent

## Overview
Implement a GUI agent within the AgentFly framework that can interact with graphical user interfaces through vision-language models and perform UI automation tasks. This implementation is based on an existing GUI agent but addresses training issues identified in the previous implementation.

## High-Level Specifications

### Agent Requirements
- Support for vision-language models (VLMs) for processing screenshots
- Ability to parse UI actions from model responses
- Integration with pyautogui for action execution
- Support for multi-modal inputs (text + images)
- Compatible with VERL training framework

### Key Components
1. **GUI Agent Class** (`agents/agents/specialized/gui_agent.py`)
   - Extends BaseAgent
   - Handles vision-language model interactions
   - Parses UI actions from responses

2. **UI Action Parser** (`agents/utils/ui_action_parser.py`)
   - Parses action strings to structured output
   - Converts coordinates between different formats
   - Generates pyautogui code

3. **GUI Tools** (`agents/tools/src/ui/tools.py`)
   - PyAutoGUI code generator tool
   - Screenshot capture and processing

4. **GUI Reward Function** (`agents/rewards/gui_reward.py`)
   - Format scoring for action parsing
   - Accuracy scoring for action matching
   - F1 score calculation

5. **Training Configuration**
   - VERL integration for RL training
   - Support for GUI-specific datasets
   - Vision-language model templates

## Relevant Files

### From Previous Implementation (Reference)
- `/mnt/weka/home/yongxin.wang/workspace/AgentFly/agents/agents/agents/specialized/ui_agent.py`
- `/mnt/weka/home/yongxin.wang/workspace/AgentFly/agents/agents/utils/ui_action_parser.py`
- `/mnt/weka/home/yongxin.wang/workspace/AgentFly/agents/agents/rewards/gui_reward.py`
- `/mnt/weka/home/yongxin.wang/workspace/AgentFly/verl/examples/run_agents/run_gui_test.sh`

### To Implement (New Repository)
- `/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/agents/agents/agents/specialized/gui_agent.py`
- `/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/agents/utils/ui_action_parser.py`
- `/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/agents/tools/src/ui/tools.py`
- `/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/agents/rewards/gui_reward.py`
- `/mnt/weka/home/yongxin.wang/workspace/Agent-One-Lab/AgentFly/verl/examples/run_agents/run_gui_agent.sh`

## Acceptance Criteria

- [ ] GUI agent can process screenshots and generate appropriate UI actions
- [ ] Action parser correctly handles all UI action types (click, type, scroll, drag, etc.)
- [ ] Reward function accurately scores predicted actions against ground truth
- [ ] Agent integrates with VERL training framework
- [ ] Training script runs without errors
- [ ] Agent supports vision-language models (Qwen2.5-VL, etc.)

## Implementation Steps

### Step 1: Create UI Action Parser
- [x] Copy and adapt ui_action_parser.py from reference implementation
- [x] Ensure compatibility with new AgentFly structure
- [x] Test parsing functionality

### Step 2: Implement GUI Tools
- [x] Create pyautogui_code_generator tool
- [x] Add screenshot processing utilities
- [x] Register tools with the framework

### Step 3: Create GUI Agent Class
- [x] Extend BaseAgent class
- [x] Implement vision-language model support
- [x] Add response parsing logic
- [x] Handle multi-turn interactions

### Step 4: Implement GUI Reward Function
- [x] Create reward function with format and accuracy scoring
- [x] Register with reward system using @reward decorator
- [x] Add to VERL reward score module

### Step 5: Set Up Training Configuration
- [x] Create training script for GUI agent
- [x] Configure VERL parameters
- [x] Add GUI-specific datasets

### Step 6: Test and Debug
- [x] Run unit tests for each component
- [x] Test end-to-end training pipeline
- [x] Fix identified issues

## Summary of Changes

### Files Created
- `/agents/agents/agents/specialized/gui_agent.py` - GUI agent implementation
- `/agents/utils/ui_action_parser.py` - UI action parsing utilities
- `/agents/tools/src/ui/tools.py` - PyAutoGUI code generation tools
- `/agents/rewards/gui_reward.py` - GUI reward function
- `/agents/rewards/reward_base.py` - Base reward function infrastructure
- `/verl/verl/utils/reward_score/gui.py` - VERL integration for GUI rewards
- `/verl/examples/run_agents/run_gui_agent.sh` - Training script for GUI agent
- `/agents/tests/unit/test_gui_agent.py` - Unit tests for GUI components
- `/test_gui_implementation.py` - Comprehensive test suite

### Files Modified
- `/agents/agents/agents/specialized/__init__.py` - Added GUI agent import
- `/agents/agents/agents/auto.py` - Registered GUI agent in factory
- `/tasks/001-gui-agent.md` - Task documentation
- `/ROADMAP.md` - Updated with completion status

### Issues Fixed
1. **Empty Response Handling**: Added robust default action handling and progressive fallback strategies
2. **Coordinate Format Inconsistencies**: Fixed coordinate normalization to handle both pixel and relative formats correctly
3. **Action Type Mapping**: Implemented comprehensive action type normalization between different naming conventions
4. **Vision Model Integration**: Added proper model type handling for different VLM architectures
5. **Reward Function Compatibility**: Ensured proper integration with VERL training framework

### Key Features
- Support for vision-language models (Qwen2.5-VL, UI-TARS, etc.)
- Comprehensive action space (click, type, scroll, drag, hotkey, etc.)
- Robust error handling and default behaviors
- Format and accuracy scoring for rewards
- VERL training integration
- Extensible tool system

## Training Issues to Address

Based on the previous implementation analysis:

1. **Empty Response Handling**: The agent sometimes generates empty responses, causing parsing failures
   - Solution: Add robust default action handling and response validation

2. **Coordinate Format Inconsistencies**: Different models output coordinates in different formats
   - Solution: Normalize coordinate handling across all formats

3. **Action Type Mapping**: Inconsistent action type naming between model outputs and reward functions
   - Solution: Implement comprehensive action type normalization

4. **Vision Model Integration**: Need proper multi-modal template support
   - Solution: Add vision processor and template configurations

5. **Reward Function Compatibility**: Ensure reward function works with VERL training
   - Solution: Proper integration with verl.utils.reward_score module

## Notes

- The implementation should follow AgentFly's design patterns (decorator-based tools and rewards)
- Ensure compatibility with async execution for high throughput
- Follow the existing code style and conventions
- Add comprehensive logging for debugging training issues