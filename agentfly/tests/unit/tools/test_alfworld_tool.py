import asyncio
import pytest
from ....tools import alfworld_step, alfworld_get_admissible_commands, alfworld_get_task_objective, alfworld_reset

@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_reset():
    """Test ALFWorld environment reset functionality"""
    result = await alfworld_reset(id='demo_reset')
    print(result)
    
    # Check the result is a dict with correct structure
    assert isinstance(result, dict)
    assert 'observation' in result
    assert 'status' in result
    assert result['status'] == 'success'
    assert isinstance(result['observation'], str)
    assert len(result['observation']) > 0
    
    await alfworld_reset.release(id='demo_reset')
    print('done')

@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_get_objective():
    """Test getting task objective"""
    # First, reset the environment to get a task
    await alfworld_reset(id='demo_objective')
    
    # Now, get the objective for that task
    result = await alfworld_get_task_objective(id='demo_objective')
    print(result)
    
    # Check the result is a dict with correct structure
    assert isinstance(result, dict)
    assert 'observation' in result
    assert 'status' in result
    assert result['status'] == 'success'
    
    # The observation should be a string containing the task description
    assert isinstance(result['observation'], str)
    assert len(result['observation']) > 0
    # A typical ALFWorld goal description
    assert 'Task:' in result['observation']
    assert len(result['observation'].split('Task:')[1].split('\n')[0].strip()) > 0
    
    # Clean up the environment
    await alfworld_get_task_objective.release(id='demo_objective')
    print('done')

@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_step():
    """Test ALFWorld step functionality"""
    await alfworld_reset(id='demo_step')
    result = await alfworld_step(action='look', id='demo_step')
    print(result)
    
    # Check the result is a dict with correct structure
    assert isinstance(result, dict)
    assert 'observation' in result
    assert 'status' in result
    assert result['status'] == 'success'
    assert isinstance(result['observation'], str)
    assert 'info' in result
    assert 'reward' in result['info']
    assert 'done' in result['info']
    assert isinstance(result['info']['reward'], (int, float))
    await alfworld_step.release(id='demo_step')
    print('done')


@pytest.mark.asyncio(loop_scope="session")
async def test_alfworld_commands():
    """Test getting admissible commands"""
    await alfworld_reset(id='demo_commands')
    result = await alfworld_get_admissible_commands(id='demo_commands')
    print(result)
    
    # Check the result is a dict with correct structure
    assert isinstance(result, dict)
    assert 'observation' in result
    assert 'status' in result
    assert result['status'] == 'success'
    # The observation should contain a list of commands
    assert isinstance(result['observation'], (list, str))  # Some tools return list, others string representation
    
    await alfworld_get_admissible_commands.release(id='demo_commands')
    print('done')


@pytest.mark.asyncio(loop_scope="session")
async def test_pool_async_calls():
    """Test pool with async calls - REDUCED CONCURRENCY FOR 16GB RAM"""
    async def one_chain(i):
        reset_result = await alfworld_reset(id=f"c{i}")
        assert reset_result['status'] == 'success'
        
        step_result = await alfworld_step(action="look", id=f"c{i}")
        assert step_result['status'] == 'success'
        
        await alfworld_step.release(id=f"c{i}")

    await asyncio.gather(*[
        one_chain(i) for i in range(3)   # Safe for 16GB RAM
    ])


@pytest.mark.asyncio(loop_scope="session")
async def test_double_release():
    """Test double release"""
    reset_result = await alfworld_reset(id="x")
    assert reset_result['status'] == 'success'
    
    step_result = await alfworld_step(action="look", id="x")
    assert step_result['status'] == 'success'
    
    # manual double call
    await alfworld_step.release(id="x")
    await alfworld_step.release(id="x")   # must return instantly


@pytest.mark.asyncio(loop_scope="session")
async def test_global_clean():
    """Test global cleanup - REDUCED CONCURRENCY FOR 16GB RAM"""
    async def one_chain(i):
        reset_result = await alfworld_reset(id=f"c{i}")
        assert reset_result['status'] == 'success'
        
        step_result = await alfworld_step(action="look", id=f"c{i}")
        assert step_result['status'] == 'success'
        # We don't release the env here, so it will be cleaned up automatically

    await asyncio.gather(*[
        one_chain(i) for i in range(2)   # Safe for 16GB RAM
    ])

 