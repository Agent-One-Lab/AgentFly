import traceback

from ....core import Context
from ....envs.webshop_text_env import WebShopSpec
from ...decorator import tool


@tool(
    name="webshop_browser",
    description="Browse the webshop by searching or clicking. The action is either 'search' or 'click' and the value is the search query or the element to click. Clickables: 'Buy Now', 'Next >', '< Prev', 'Back to Search', 'Description', 'Features', 'Reviews', 'Attributes', product ASIN or ID like 'B079HGJ5MH' and their attributes or variants like 'Yellow', 'Blue', 'Small', 'Large', 'XL', '40x60', etc.",
    stateful=True,
)
async def webshop_browser(action: str, value: str, context: Context):
    """
    Interact with the webshop environment by performing a search or clicking an element.

    Args:
        action (str): The action to perform, either 'search' or 'click'.
        value (str): The search query or the element to click (e.g., button, product ID, attribute).
        context (Context): Injected rollout context; used to acquire the WebShop resource.

    Returns:
        str: The observation from the environment after performing the action, or an error message if the action is invalid or an exception occurs.
    """
    try:
        env = await context.acquire_resource(spec=WebShopSpec, scope="global", backend="local")
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
