import pytest
from ....envs.webshop_text_env import WebAgentTextEnv
from ast import literal_eval

STANDARD_BUTTONS = [
    'buy Now',
    'next >',
    '< prev',
    'back to search',
    'description',
    'features',
    'reviews',
    'attributes',
]

# @pytest.mark.asyncio
# async def test_env_initialization():
#     env = WebAgentTextEnv()
#     assert env.image == "rifoag/webshop-simulator-env:latest"
#     assert env.runtime == "runc"
#     assert env.cpu == 2
#     assert env.mem == "2g"
#     assert env.host_ip == "127.0.0.1"
#     assert env.observation_mode == 'text'
    
# @pytest.mark.asyncio
# async def test_env_start_and_close():
#     env = WebAgentTextEnv()
#     await env.start()
#     assert env._client is not None
#     await env.reset()
#     await env.close()
#     assert env._client is None

@pytest.mark.asyncio
async def test_env_full_shopping_flow():
    env = WebAgentTextEnv()
    await env.start()
    await env.reset(env_args={'question': 'Buy serta executive chair'})
    # Start on homepage and search for shoes
    actions = env.get_available_actions()
    assert actions['has_search_bar'] is True
    observation = await env.step('search[serta executive]')
    
    # Click first product
    actions = env.get_available_actions()
    assert len(actions['clickables']) > 0
    product_list = [button.lower() for button in actions['clickables'] if button.lower() not in STANDARD_BUTTONS]
    first_product = product_list[0]
    observation = await env.step(f'click[{first_product}]')
    current_page = env.state['url'].split('/')[1]
    assert current_page == 'item_page'
    
    # Click through product pages
    observation = await env.step('click[description]')
    current_page = env.state['url'].split('/')[1]
    current_sub_page = env.state['url'].split('/')[-2]
    assert current_page == 'item_sub_page'
    assert current_sub_page.lower() == 'description'
    observation = await env.step('click[features]') 
    current_page = env.state['url'].split('/')[1]
    current_sub_page = env.state['url'].split('/')[-2]
    assert current_page == 'item_sub_page'
    assert current_sub_page.lower() == 'features'
    observation = await env.step('click[reviews]')
    current_page = env.state['url'].split('/')[1]
    current_sub_page = env.state['url'].split('/')[-2]
    assert current_page == 'item_sub_page'
    assert current_sub_page.lower() == 'reviews'
    
    # Select two product attributes, skipped for now due to most of the product not having options
    # actions = env.get_available_actions()
    # print(observation)
    # observation = await env.step(f'click[black magic]')
    # options = literal_eval(env.state['url'].split('/')[-1])
    # assert len(options) == 1
    # actions = env.get_available_actions()
    # observation = await env.step(f'click[1.37 pound (pack of 1)]')
    # options = literal_eval(env.state['url'].split('/')[-1])
    # assert len(options) == 2
    
    # Complete purchase
    observation = await env.step('click[buy now]')
    current_page = env.state['url'].split('/')[1]
    assert current_page == 'done'
    assert 'observation' in observation
    assert 'reward' in observation
    
    await env.aclose()

# @pytest.mark.asyncio
# async def test_pagination_navigation():
#     env = WebAgentTextEnv()
#     await env.start()
#     await env.reset(env_args={'id': 0, 'question': 'Buy a pair of shoes'})
#     # Start on homepage and search for shoes
#     actions = env.get_available_actions()
#     assert actions['has_search_bar'] is True
#     observation = await env.step('search[shoes]')
    
#     # Navigate through pages
#     actions = env.get_available_actions()
#     current_page = env.state['url'].split('/')[-1]
#     assert current_page == '1'
    
#     observation = await env.step('click[next >]')
#     current_page = env.state['url'].split('/')[-1] 
#     assert current_page == '2'
    
#     observation = await env.step('click[next >]')
#     current_page = env.state['url'].split('/')[-1]
#     assert current_page == '3'
    
#     observation = await env.step('click[next >]')
#     current_page = env.state['url'].split('/')[-1]
#     assert current_page == '4'
    
#     observation = await env.step('click[< prev]')
#     current_page = env.state['url'].split('/')[-1]
#     assert current_page == '3'
    
#     await env.close()

# @pytest.mark.asyncio
# async def test_back_to_search_navigation():
#     env = WebAgentTextEnv()
#     await env.start()
#     await env.reset(env_args={'id': 0, 'question': 'Buy a pair of shoes'})
#     # Search for shirts
#     actions = env.get_available_actions()
#     assert actions['has_search_bar'] is True
#     observation = await env.step('search[shirt]')
    
#     # Click first product
#     actions = env.get_available_actions()
#     assert len(actions['clickables']) > 0
#     product_list = [button.lower() for button in actions['clickables'] if button.lower() not in STANDARD_BUTTONS]
#     first_product = product_list[0]
#     observation = await env.step(f'click[{first_product}]')
#     current_page = env.state['url'].split('/')[1]
#     assert current_page == 'item_page'
    
#     # Click back to search
#     actions = env.get_available_actions()
#     observation = await env.step('click[back to search]')
#     current_page = env.state['url'].split('/')[1]
#     assert current_page == 'index'
    
#     await env.close()
