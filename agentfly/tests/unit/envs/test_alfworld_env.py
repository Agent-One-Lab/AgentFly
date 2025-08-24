import asyncio
import pytest
from ....envs.alfworld_env import ALFWorldEnv

@pytest.mark.asyncio
async def test_alfworld_env_get_info():
    """Test the get_info method returns expected info structure with real environment data."""
    env = ALFWorldEnv()
    try:
        await env.start()
        
        # Test Case 1: After reset with real observation
        observation, reset_info = await env.reset()
        
        # Get info using the real environment state
        info = await env.get_info()
        
        # Verify the structure
        assert isinstance(info, dict)
        
        # Check for expected keys based on the get_info implementation
        assert "task" in info
        assert "goal" in info
        assert "won" in info
        assert "lost" in info
        assert "admissible_commands_count" in info
        
        # Verify data types
        assert isinstance(info["task"], str)
        assert isinstance(info["goal"], str)
        assert isinstance(info["won"], bool)
        assert isinstance(info["lost"], bool)
        assert isinstance(info["admissible_commands_count"], int)
        
        # Test Case 2: Verify goal extraction from actual observation
        if "Your task is to: " in observation:
            # Extract expected goal using the same logic as the method
            expected_goal_part = observation.split("Your task is to: ")[1]
            if '.' in expected_goal_part:
                expected_goal = expected_goal_part.split('.')[0].strip()
            else:
                expected_goal = expected_goal_part.split('\n')[0].strip()
            expected_goal = expected_goal.rstrip("'\".,)")
            
            assert info["goal"] == expected_goal, f"Expected goal '{expected_goal}', got '{info['goal']}'"
            assert info["goal"] != "", "Goal should not be empty when extracted from observation"
        
        # Test Case 3: Test after taking a step (should still maintain info)
        step_obs, reward, done, step_info = await env.step("look")
        
        # Get info after step
        info_after_step = await env.get_info()
        assert isinstance(info_after_step, dict)
        assert "goal" in info_after_step
        
        # Goal should be consistent
        if info["goal"]:  # If we had a goal before
            assert info_after_step["goal"] == info["goal"], "Goal should remain consistent after step"
        
        # Test Case 4: Clear cache and test fallback to HTTP
        original_info = env._current_info
        original_obs = getattr(env, '_current_obs', None)
        
        env._current_info = None
        env._current_obs = None
        
        fallback_info = await env.get_info()
        assert isinstance(fallback_info, dict)
        
        # Restore original state
        env._current_info = original_info
        env._current_obs = original_obs
        
        # Print for debugging
        print(f"Test results:")
        print(f"Original observation: {observation[:100]}...")
        print(f"Reset info: {reset_info}")
        print(f"Get info result: {info}")
        print(f"Info after step: {info_after_step}")
        print(f"Fallback info: {fallback_info}")
        
    finally:
        await env.aclose()

@pytest.mark.asyncio
async def test_alfworld_env_lifecycle():
    """Test basic environment lifecycle: start, reset, step, close."""
    env = ALFWorldEnv()
    
    try:
        # Start the environment
        await env.start()
        
        # Reset to get initial observation
        obs, info = await env.reset()
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert isinstance(info, dict)
        
        # Get admissible commands
        commands = await env.get_admissible_commands()
        assert isinstance(commands, list)
        
        # Take a simple action (look)
        obs, reward, done, info = await env.step("look")
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    finally:
        await env.aclose()


@pytest.mark.asyncio 
async def test_alfworld_env_multiple_steps():
    """Test taking multiple steps in the environment."""
    env = ALFWorldEnv()
    
    try:
        await env.start()
        obs, info = await env.reset()
        
        # Take several actions
        actions = ["look", "inventory", "go north"]
        for action in actions:
            obs, reward, done, info = await env.step(action)
            assert isinstance(obs, str)
            
            # If episode ends, break
            if done:
                break
                
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_alfworld_env_admissible_commands():
    """Test that admissible commands are properly returned."""
    env = ALFWorldEnv()
    
    try:
        await env.start()
        obs, info = await env.reset()
        
        # Check initial admissible commands
        commands = await env.get_admissible_commands()
        assert isinstance(commands, list)
        
        # Admissible commands should typically include basic actions
        # Note: exact commands depend on the specific task
        if commands:  # Some tasks might not provide admissible commands
            assert all(isinstance(cmd, str) for cmd in commands)
            
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_alfworld_env_inventory():
    """Test inventory functionality."""
    env = ALFWorldEnv()
    
    try:
        await env.start()
        await env.reset()
        
        # Get inventory
        inventory = await env.get_inventory()
        assert isinstance(inventory, str)
        
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_alfworld_env_reset_after_steps():
    """Test resetting environment after taking some steps."""
    env = ALFWorldEnv()
    
    try:
        await env.start()
        
        # First episode
        obs1, info1 = await env.reset()
        await env.step("look")
        await env.step("inventory")
        
        # Reset for second episode
        obs2, info2 = await env.reset()
        assert isinstance(obs2, str)
        assert isinstance(info2, dict)
        
        # Should be able to take actions in new episode
        obs, reward, done, info = await env.step("look")
        assert isinstance(obs, str)
        
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_alfworld_env_acquire():
    """Test the acquire static method."""
    env = await ALFWorldEnv.acquire()
    
    try:
        assert env is not None
        
        # Should be ready to use
        obs, info = await env.reset()
        assert isinstance(obs, str)
        
    finally:
        await env.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("train_eval", ["train"])
async def test_alfworld_env_train_eval_modes(train_eval):
    """Test environment in different modes."""
    env = ALFWorldEnv(train_eval=train_eval)
    
    try:
        await env.start()
        obs, info = await env.reset()
        
        assert isinstance(obs, str)
        assert len(obs) > 0
        
    finally:
        await env.aclose()


@pytest.mark.asyncio
async def test_alfworld_env_error_handling():
    """Test error handling for invalid actions."""
    env = ALFWorldEnv()
    
    try:
        await env.start()
        await env.reset()
        
        # Try an invalid/nonsense action
        obs, reward, done, info = await env.step("xyzabc123nonsense")
        
        # Should still return valid response structure
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    finally:
        await env.aclose()


# Additional tests based on test_env_run.py pattern but adapted for ALFWorld

N_REQUESTS = 3  # REDUCED from 10 for 16GB RAM safety

@pytest.mark.asyncio
async def test_alfworld_env_concurrent_requests():
    """Test concurrent requests to a single ALFWorld environment."""
    import time
    start_time = time.time()
    
    env = ALFWorldEnv(max_episodes=2)  # Only one reset needed
    await env.start()
    await env.reset()

    async def run_action(i: int):
        # Different actions based on index
        actions = ["look", "inventory", "help"]
        action = actions[i % len(actions)]
        obs, reward, done, info = await env.step(action)
        return i, obs, done

    tasks = [asyncio.create_task(run_action(i)) for i in range(N_REQUESTS)]
    results = await asyncio.gather(*tasks)

    # Verify all requests completed
    for i, obs, done in results:
        assert isinstance(obs, str)
        assert isinstance(done, bool)

    await env.aclose()
    
    end_time = time.time()
    print(f"Concurrent requests time: {end_time - start_time} seconds")


N_ENVS = 2         # REDUCED from 3 for 16GB RAM safety
MAX_PARALLEL = 2   # Keep at 2 for safety

# @pytest.mark.asyncio
# async def test_alfworld_env_many_instances():
#     """
#     Launch multiple ALFWorld environments sequentially to avoid memory pressure.
#     """
#     import time
    
#     errors = []
#     start_time = time.time()
    
#     # Run environments completely sequentially for memory safety
#     for i in range(N_ENVS):
#         env = ALFWorldEnv()
#         try:
#             await env.start()
#             obs, info = await env.reset()
            
#             # Take a simple action
#             obs, reward, done, info = await env.step("look")
#             assert isinstance(obs, str), f"id={i}: wrong output type {type(obs)}"
            
#         except Exception as exc:
#             errors.append(f"env_{i}: {exc}")
#         finally:
#             await env.aclose()

#     # Report any collected failures
#     if errors:
#         raise AssertionError(f"{len(errors)} failures: {errors[:3]}...")
    
#     end_time = time.time()
#     print(f"Sequential instances time: {end_time - start_time} seconds")

@pytest.mark.parametrize("observation,expected_goal", [
    (
        "Your task is to: heat some water in the microwave. Here's what you see...",
        "heat some water in the microwave"
    ),
    (
        "Your task is to: find a mug and place it on the counter. The kitchen is messy.",
        "find a mug and place it on the counter"
    ),
    (
        "Your task is to: clean the dishes... The sink is full.",  # Test with ellipsis
        "clean the dishes"
    ),
])
def test_extract_goal_from_observation(observation, expected_goal):
    """
    Test the _extract_goal_from_observation method with various observation formats.
    
    Args:
        observation (str): The input observation string
        expected_goal (str): The expected extracted goal
    """
    env = ALFWorldEnv()
    extracted_goal = env._extract_goal_from_observation(observation)
    
    if expected_goal is None:
        assert extracted_goal is None, f"Expected None but got '{extracted_goal}'"
    else:
        assert extracted_goal == expected_goal, f"Expected '{expected_goal}' but got '{extracted_goal}'"

# @pytest.mark.asyncio
# async def test_alfworld_env_stress_test_single_env():
#     """
#     Stress test a single ALFWorld environment with multiple episodes.
#     Resource-efficient version for 16GB RAM.
#     """
#     import time
    
#     start_time = time.time()
#     env = ALFWorldEnv(max_episodes=3)  # REDUCED from 5 for 16GB RAM safety
#     await env.start()
    
#     episodes_completed = 0
#     total_steps = 0
    
#     try:
#         for episode in range(2):  # REDUCED from 3 for 16GB RAM safety
#             obs, info = await env.reset()
#             episodes_completed += 1
            
#             # Take multiple steps per episode
#             for step in range(5):  # REDUCED from 10 for 16GB RAM safety
#                 actions = ["look", "inventory", "help"]
#                 action = actions[step % len(actions)]
                
#                 obs, reward, done, info = await env.step(action)
#                 total_steps += 1
                
#                 assert isinstance(obs, str)
#                 assert isinstance(reward, (int, float))
#                 assert isinstance(done, bool)
                
#                 if done:
#                     break
                    
#     finally:
#         await env.aclose()
    
#     end_time = time.time()
#     print(f"Stress test: {episodes_completed} episodes, {total_steps} steps in {end_time - start_time:.2f}s")
    
#     assert episodes_completed >= 2, "Should complete at least 2 episodes"  # REDUCED from 3
#     assert total_steps >= 2, "Should take at least 2 steps total"  # REDUCED from 3 