import json
import logging
import os
import random
import string
from ast import literal_eval

from engine import (
    END_BUTTON,
    convert_web_app_string_to_var,
    get_product_per_page,
    get_top_n_product_from_keywords,
    init_search_engine,
    load_products,
    map_action_to_html,
)
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from goal import get_goals, get_reward
from utils import get_base_dir, get_file_path, init_basedir

logger = logging.getLogger(__name__)
app = FastAPI()

# Set up templates
templates_dir = os.path.join(get_base_dir(), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Mount static files
static_dir = os.path.join(get_base_dir(), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Add static file route for URL generation
@app.get("/static/{path:path}")
async def static_files(path: str):
    return FileResponse(os.path.join(static_dir, path))


init_basedir(dataset="full")
all_products, product_item_dict, product_prices, attribute_to_asins = load_products(
    filepath=get_file_path()
)
search_engine = init_search_engine()
goals = get_goals(all_products, product_prices)
# random.shuffle(goals)
weights = [goal["weight"] for goal in goals]

user_sessions = dict()
user_log_dir = None
SHOW_ATTRS_TAB = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def home():
    session_id = "".join(random.choices(string.ascii_lowercase, k=10))
    return RedirectResponse(url=f"/index/{session_id}")


@app.post("/")
async def home_post(request: Request):
    form = await request.form()
    session_id = form.get("session_id")
    if "search_query" in form:
        keywords = form["search_query"].lower().split(" ")
        return RedirectResponse(
            url=f"/search_results/{session_id}/{keywords}/1", status_code=303
        )
    return RedirectResponse(url=f"/index/{session_id}")


@app.get("/index/{session_id}")
@app.post("/index/{session_id}")
async def index(session_id: str, request: Request):
    global user_log_dir
    global \
        all_products, \
        product_item_dict, \
        product_prices, \
        attribute_to_asins, \
        search_engine, \
        goals, \
        weights, \
        user_sessions
    if session_id.isdigit() and session_id not in user_sessions:
        goal_idx = int(session_id)
        goal = goals[goal_idx]
        instruction_text = goal["instruction_text"]
        user_sessions[session_id] = {"goal": goal, "done": False}
        # if user_log_dir is not None:
        #     setup_logger(session_id, user_log_dir)
    elif session_id not in user_sessions:
        # Shuffle goals for each new session
        shuffled_goals = goals.copy()
        random.shuffle(shuffled_goals)
        goal = random.choices(shuffled_goals, weights)[0]
        instruction_text = goal["instruction_text"]
        user_sessions[session_id] = {"goal": goal, "done": False}
        # if user_log_dir is not None:
        #     setup_logger(session_id, user_log_dir)
    else:
        instruction_text = user_sessions[session_id]["goal"]["instruction_text"]

    # if request.method == "POST":
    #     form = await request.form()
    #     if "search_query" in form:
    #         keywords = form["search_query"].lower().split(" ")

    if user_log_dir is not None:
        logger = logging.getLogger(session_id)
        logger.info(
            json.dumps(
                dict(
                    page="index",
                    url=str(request.url),
                    goal=user_sessions[session_id]["goal"],
                )
            )
        )

    html_response = await map_action_to_html(
        "start",
        session_id=session_id,
        instruction_text=instruction_text,
        request=request,
    )
    return html_response


@app.get("/search_results/{session_id}/{keywords}/{page}")
@app.post("/search_results/{session_id}/{keywords}/{page}")
async def search_results(session_id: str, keywords: str, page: int, request: Request):
    instruction_text = user_sessions[session_id]["goal"]["instruction_text"]
    keywords = convert_web_app_string_to_var("keywords", keywords)
    top_n_products = get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins,
    )
    products = get_product_per_page(top_n_products, page)
    logger = logging.getLogger(session_id)
    logger.info(
        json.dumps(
            dict(
                page="search_results",
                url=str(request.url),
                goal=user_sessions[session_id]["goal"],
                content=dict(
                    keywords=keywords,
                    search_result_asins=[p["asin"] for p in products],
                    page=page,
                ),
            )
        )
    )
    html_response = await map_action_to_html(
        "search",
        session_id=session_id,
        products=products,
        keywords=keywords,
        page=page,
        total=len(top_n_products),
        instruction_text=instruction_text,
        request=request,
    )
    return html_response


@app.get("/item_page/{session_id}/{asin}/{keywords}/{page}/{options}")
@app.post("/item_page/{session_id}/{asin}/{keywords}/{page}/{options}")
async def item_page(
    session_id: str, asin: str, keywords: str, page: int, options: str, request: Request
):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    goal_instruction = user_sessions[session_id]["goal"]["instruction_text"]
    product_info["goal_instruction"] = goal_instruction

    logger = logging.getLogger(session_id)
    logger.info(
        json.dumps(
            dict(
                page="item_page",
                url=str(request.url),
                goal=user_sessions[session_id]["goal"],
                content=dict(
                    keywords=keywords,
                    page=page,
                    asin=asin,
                    options=options,
                ),
            )
        )
    )
    html_response = await map_action_to_html(
        "click",
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        show_attrs=SHOW_ATTRS_TAB,
        request=request,
    )
    return html_response


@app.get("/item_sub_page/{session_id}/{asin}/{keywords}/{page}/{sub_page}/{options}")
@app.post("/item_sub_page/{session_id}/{asin}/{keywords}/{page}/{sub_page}/{options}")
async def item_sub_page(
    session_id: str,
    asin: str,
    keywords: str,
    page: int,
    sub_page: str,
    options: str,
    request: Request,
):
    options = literal_eval(options)
    product_info = product_item_dict[asin]

    goal_instruction = user_sessions[session_id]["goal"]["instruction_text"]
    product_info["goal_instruction"] = goal_instruction

    logger = logging.getLogger(session_id)
    logger.info(
        json.dumps(
            dict(
                page="item_sub_page",
                url=str(request.url),
                goal=user_sessions[session_id]["goal"],
                content=dict(
                    keywords=keywords,
                    page=page,
                    asin=asin,
                    options=options,
                ),
            )
        )
    )
    html_response = await map_action_to_html(
        f"click[{sub_page}]",
        session_id=session_id,
        product_info=product_info,
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=goal_instruction,
        request=request,
    )
    return html_response


@app.get("/done/{session_id}/{asin}/{options}")
@app.post("/done/{session_id}/{asin}/{options}")
async def done(session_id: str, asin: str, options: str, request: Request):
    global goals
    options = literal_eval(options)

    # Extract task_id from request query params if present
    task_id = request.query_params.get("task_id")
    if task_id is not None:
        task_id = int(task_id)
        goal = goals[task_id]
    else:
        goal = user_sessions[session_id]["goal"]

    purchased_product = product_item_dict[asin]
    price = product_prices[asin]

    reward, reward_info = get_reward(
        purchased_product, goal, price=price, options=options, verbose=True
    )
    user_sessions[session_id]["done"] = True
    user_sessions[session_id]["reward"] = reward

    logger = logging.getLogger(session_id)
    logger.info(
        json.dumps(
            dict(
                page="done",
                url=str(request.url),
                goal=goal,
                content=dict(
                    asin=asin,
                    options=options,
                    price=price,
                ),
                reward=reward,
                reward_info=reward_info,
            )
        )
    )
    del logging.root.manager.loggerDict[session_id]

    html_response = await map_action_to_html(
        f"click[{END_BUTTON}]",
        session_id=session_id,
        reward=reward,
        asin=asin,
        options=options,
        reward_info=reward_info,
        query=purchased_product["query"],
        category=purchased_product["category"],
        product_category=purchased_product["product_category"],
        goal_attrs=user_sessions[session_id]["goal"]["attributes"],
        purchased_attrs=purchased_product["Attributes"],
        goal=goal,
        request=request,
    )

    # Get the rendered HTML content from the TemplateResponse
    rendered_html = html_response.body.decode()

    return JSONResponse(
        content={"observation": rendered_html, "reward": reward, "status_code": 200}
    )
