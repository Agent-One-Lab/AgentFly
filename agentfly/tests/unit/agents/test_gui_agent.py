#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
from pathlib import Path

# Add the agents module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ....agents.specialized.gui_agent import GUIAgent
from ....rewards.gui_reward import gui_reward
from ....utils.ui_action_parser import parse_action_to_structure_output, IMAGE_FACTOR
from ....tools import pyautogui_code_generator

class TestGUIAgent:
    """Test suite for GUI Agent implementation."""
    
    def test_gui_agent_initialization(self):
        """Test GUI agent can be initialized."""
        # Skip loading actual model for unit test
        agent = GUIAgent(
            model_name_or_path="ByteDance-Seed/UI-TARS-1.5-7B",
            template="qwen2.5-vl",
            tools=[pyautogui_code_generator],
            backend="async_vllm"
        )
        assert agent is not None
        assert agent.system_prompt is not None
        assert "GUI automation agent" in agent.system_prompt
    
    def test_gui_agent_parse_valid_response(self):
        """Test GUI agent can parse valid responses."""
        # Skip loading actual model for unit test
        agent = GUIAgent(
            model_name_or_path="ByteDance-Seed/UI-TARS-1.5-7B",
            template="qwen2.5-vl",
            tools=[],
            backend="async_vllm"
        )
        
        responses = [
            "Thought: I need to click on the button.\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"
        ]
        
        messages = agent.parse(responses, tools=[])
        
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["text"] == responses[0]
        assert len(messages[0]["tool_calls"]) == 1
        assert messages[0]["status"] == "continue"
    
    def test_gui_agent_parse_terminal_action(self):
        """Test GUI agent recognizes terminal actions."""
        # Skip loading actual model for unit test
        agent = GUIAgent(
            model_name_or_path="ByteDance-Seed/UI-TARS-1.5-7B",
            template="qwen2.5-vl",
            tools=[],
            backend="async_vllm"
        )
        
        responses = [
            "Thought: Task is complete.\nAction: finished(content='Task completed successfully')"
        ]
        
        messages = agent.parse(responses, tools=[])
        
        assert len(messages) == 1
        assert messages[0]["status"] == "terminal"
    
    def test_gui_agent_parse_empty_response(self):
        """Test GUI agent handles empty responses gracefully."""
        # Skip loading actual model for unit test
        agent = GUIAgent(
            model_name_or_path="ByteDance-Seed/UI-TARS-1.5-7B",
            template="qwen2.5-vl",
            tools=[],
            backend="async_vllm"
        )
        
        responses = [""]
        
        messages = agent.parse(responses, tools=[])
        
        assert len(messages) == 1
        assert "wait" in messages[0]["content"][0]["text"].lower()
        assert messages[0]["status"] == "continue"


class TestGUIReward:
    """Test suite for GUI reward function."""
    
    @pytest.mark.asyncio
    async def test_gui_reward_with_ground_truth(self):
        """Test GUI reward with ground truth data."""
        prediction = "Thought: I need to click on the button.\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"
        
        result = await gui_reward(
            prediction=prediction,
            gt_action="click",
            gt_bbox=[100, 200],
            gt_input_text=""
        )
        
        assert isinstance(result, dict)
        assert "reward" in result
        assert "format" in result
        assert "accuracy" in result
        assert result["format"] == 1.0  # Valid format
        assert result["accuracy"] == 1.0  # Exact match
        assert result["reward"] > 0.9  # High reward for perfect match
    
    @pytest.mark.asyncio
    async def test_gui_reward_without_ground_truth(self):
        """Test GUI reward without ground truth data."""
        prediction = "Thought: I need to click.\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"
        
        result = await gui_reward(prediction=prediction)
        
        assert isinstance(result, dict)
        assert result["reward"] == 0.0  # No ground truth
    
    @pytest.mark.asyncio
    async def test_gui_reward_empty_prediction(self):
        """Test GUI reward with empty prediction."""
        result = await gui_reward(
            prediction="",
            gt_action="click",
            gt_bbox=[100, 200],
            gt_input_text=""
        )
        
        assert isinstance(result, dict)
        assert result["format"] == 0.0  # Invalid format
    
    @pytest.mark.asyncio
    async def test_gui_reward_type_action(self):
        """Test GUI reward for typing action."""
        prediction = "Thought: I need to type text.\nAction: type(content='hello world')"
        
        result = await gui_reward(
            prediction=prediction,
            gt_action="type",
            gt_bbox=[],
            gt_input_text="hello world"
        )
        
        assert isinstance(result, dict)
        assert result["format"] == 1.0
        assert result["accuracy"] == 1.0  # Text matches


class TestUIActionParser:
    """Test suite for UI action parser."""
    
    def test_parse_click_action(self):
        """Test parsing click action."""
        text = "Thought: Click button.\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"
        
        result = parse_action_to_structure_output(
            text, IMAGE_FACTOR, 1080, 1920
        )
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["action_type"] == "click"
        assert "start_box" in result[0]["action_inputs"]
    
    def test_parse_type_action(self):
        """Test parsing type action."""
        text = "Thought: Type text.\nAction: type(content='hello world')"
        
        result = parse_action_to_structure_output(
            text, IMAGE_FACTOR, 1080, 1920
        )
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["action_type"] == "type"
        assert result[0]["action_inputs"]["content"] == "hello world"
    
    def test_parse_invalid_action(self):
        """Test parsing invalid action returns None."""
        text = "This is not a valid action format"
        
        result = parse_action_to_structure_output(
            text, IMAGE_FACTOR, 1080, 1920
        )
        
        assert result is None
    
    def test_parse_empty_text(self):
        """Test parsing empty text returns None."""
        result = parse_action_to_structure_output(
            "", IMAGE_FACTOR, 1080, 1920
        )
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])