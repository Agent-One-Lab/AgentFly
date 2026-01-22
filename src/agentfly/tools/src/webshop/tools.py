import traceback

from ...decorator import tool

from ....envs.webshop_text_env import WebAgentTextEnv

END_BUTTON = "Buy Now"
NEXT_PAGE = "Next >"
PREV_PAGE = "< Prev"
BACK_TO_SEARCH = "Back to Search"

ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


@tool(
    env_cls=WebAgentTextEnv,
    name="webshop_browser",
    description="Browse the webshop by searching or clicking. The action is either 'search' or 'click' and the value is the search query or the element to click. Clickables: 'Buy Now', 'Next >', '< Prev', 'Back to Search', 'Description', 'Features', 'Reviews', 'Attributes', product ASIN or ID like 'B079HGJ5MH' and their attributes or variants like 'Yellow', 'Blue', 'Small', 'Large', 'XL', '40x60', etc.",
    stateful=True,
    pool_size=8,
)
async def webshop_browser(action: str, value: str, env: WebAgentTextEnv):
    """
    Interact with the webshop environment by performing a search or clicking an element.

    Args:
        action (str): The action to perform, either 'search' or 'click'.
        value (str): The search query or the element to click (e.g., button, product ID, attribute).
        env (WebAgentTextEnv): The webshop text environment instance.

    Returns:
        str: The observation from the environment after performing the action, or an error message if the action is invalid or an exception occurs.
    """
    try:
        if action == "search":
            observation = await env.step(f"search[{value}]")
        elif action == "click":
            observation = await env.step(f"click[{value}]")
        else:
            return (
                f"Error: Invalid action '{action}'. Must be either 'search' or 'click'"
            )
        return observation
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    print(webshop_browser.schema)
