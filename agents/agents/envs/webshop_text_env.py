import time
import random
import string
import asyncio, httpx
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
from ast import literal_eval

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

from .env_base import BaseEnv, SupportsDocker
class WebAgentTextEnv(BaseEnv, SupportsDocker):
    """
    Text mode of WebShop environment.
    This class simulates a text-based shopping environment for agents to interact with a webshop.
    It manages the environment state, handles actions, and communicates with a backend server via HTTP.
    """
    def __init__(
            self,
            image: str = "rifoag/webshop-simulator-env:latest",
            runtime: str = "runc",
            cpu: int = 2,
            mem: str = "2g",
            start_timeout: float = 180.0,
            host_ip: str = "127.0.0.1",
            container_port: int = 3000,
            observation_mode: str = 'text',
        ):
        """
        Initialize the WebAgentTextEnv environment.
        Sets up Docker image, runtime, resource limits, and observation mode.
        """
        super().__init__()
        self.image = image
        self.runtime = runtime
        self.cpu = cpu
        self.mem = mem
        self.start_timeout = start_timeout
        self.host_ip = host_ip
        self.container_port = container_port
        self._client: httpx.AsyncClient | None = None

        self.session_id = ''.join(random.choices(string.ascii_lowercase, k=10))
        self.observation_mode = observation_mode
        
        self.home_html = None
        self.state = {'html': None, 'url': None}
        self.text_to_clickable = None

    async def _wait_ready(self) -> None:
        """
        Poll until the server answers or we time out.
        Raises RuntimeError if the server does not become ready in time.
        """
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            try:
                r = await self._client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.1)

        # Last-ditch diagnostics
        logs = self.get_container_logs()
        raise RuntimeError(
            f"WebShop server did not become ready within {self.start_timeout}s.\n{logs}"
        )

    async def start(self) -> None:
        """
        Start the environment and allocate any resources.
        Launches the Docker container, connects to the backend, and loads the home page.
        """
        await self._docker_start(
            image=self.image,
            runtime=self.runtime,
            cpu_count=self.cpu,
            mem_limit=self.mem,
            ports={f"{self.container_port}/tcp": None},
            read_only=True,
            cap_drop=["ALL"],
            pids_limit=256,
            tmpfs={
                '/tmp': 'rw,noexec,nosuid,size=100m',
                '/var/tmp': 'rw,noexec,nosuid,size=100m',
                '/usr/tmp': 'rw,noexec,nosuid,size=100m',
            },
        )
        
        await self._connect()
        await self._wait_ready()
        r = await self._client.get(f"/index/{self.session_id}")
        self.home_html = r.text
        self.state = {'html': self.home_html, 'url':  f"/index/{self.session_id}"}

    async def reset(self, env_args=None) -> str:        
        """
        Reset the environment to its initial state or to a specific task_id if provided.
        Args:
            env_args (dict, optional): Dictionary that may contain 'task_id'. Default is None. Used during training.
        Returns:
            str: None (kept for compatibility)
        """
        try:
            task_id = env_args.get('task_id', None)
            if task_id is not None:
                self.session_id = task_id
                r = await self._client.get(f"/index/{self.session_id}")
                self.state.update({
                    'html': r.text,
                    'url': f"/index/{self.session_id}",
                })
                self.text_to_clickable = None
            else:
                self.state = {'html': self.home_html, 'url':  f"/index/{self.session_id}"}
                self.text_to_clickable = None
        except Exception as e:
            self.state = {'html': self.home_html, 'url':  f"/index/{self.session_id}"}
            self.text_to_clickable = None
    
    async def step(self, action: str, task_id: int = None) -> str:
        """
        Take an action in the environment and return the observation.
        Args:
            action (str): An action in the form of 'click[value]' or 'search[keywords]'.
            task_id (int, optional): Task identifier for reward calculation.
        Returns:
            str or dict: The observation after the action, and reward if applicable.
        """
        available_actions = self.get_available_actions()
        has_search_bar = available_actions['has_search_bar']
        clickables = available_actions['clickables']
        # Determine action type (click, search) and argument
        action_name, action_arg = self.parse_action(action)
        if action_name == 'get_reward':
            current_url = self.state.get('url')
            if 'done' not in current_url:
                return {
                    'observation': self.observation,
                    'reward': 0
                }
            url_parts = current_url.split('/')
            asin = url_parts[3]
            options = literal_eval(url_parts[4])
            r = await self._client.get(f"/done/{self.session_id}/{asin}/{options}?task_id={task_id}")
            r = r.json()
            self.state.update({
                'html': r['observation'],
                'reward': r['reward'],
                'url': f"/done/{self.session_id}/{asin}/{options}",
            })
            try:
                return {
                    'observation': self.observation,
                    'reward': self.state['reward'],
                }
            except Exception as e:
                return {
                    "reward": 0.0,
                    "output": f"Error calculating reward: {e}",
                }
        
        if action_arg is not None:
            action_arg = action_arg.lower()
        
        if not action_name in ['click', 'search']:
            return "Invalid action, action name should be 'click' or 'search'." # invalid action, do nothing
        elif action_arg not in self.text_to_clickable.keys() and action_name == 'click':
            return "Invalid action, action argument should be one of the clickables: " + str(clickables)
        else:
            current_url = self.state.get('url')
            url_parts = current_url.split('/')
            current_page = url_parts[1]

            if current_page == 'index':
                if action_name == 'search':
                    keywords = action_arg.split()
                    r = await self._client.get(f"/search_results/{self.session_id}/{action_arg}/1")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/search_results/{self.session_id}/{keywords}/1",
                        })
            elif current_page == 'search_results':
                page_num = int(url_parts[-1])
                keywords = literal_eval(url_parts[3])
                if action_arg == NEXT_PAGE.lower():
                    next_page = page_num + 1
                    r = await self._client.get(f"/search_results/{self.session_id}/{keywords}/{next_page}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/search_results/{self.session_id}/{keywords}/{next_page}",
                        })
                elif action_arg == PREV_PAGE.lower() and PREV_PAGE.lower() in self.text_to_clickable.keys() and page_num > 1:
                    prev_page = page_num - 1
                    r = await self._client.get(f"/search_results/{self.session_id}/{keywords}/{prev_page}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/search_results/{self.session_id}/{keywords}/{prev_page}",
                        })
                elif action_arg == 'back to search':
                    r = await self._client.get(f"/index/{self.session_id}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/index/{self.session_id}",
                        })
                elif action_arg.lower() and len(action_arg) == 10 and action_arg.isalnum():
                    asin = action_arg.upper()
                    options = str({})
                    r = await self._client.get(f"/item_page/{self.session_id}/{asin}/{keywords}/{page_num}/{options}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/item_page/{self.session_id}/{asin}/{keywords}/{page_num}/{options}",
                        })
                else:
                    pass # invalid action, do nothing
            elif current_page == 'item_page':
                asin = url_parts[3]
                keywords = literal_eval(url_parts[4])
                page_num = url_parts[5]
                options = literal_eval(url_parts[6])
                if action_arg == BACK_TO_SEARCH.lower():
                    r = await self._client.get(f"/index/{self.session_id}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/index/{self.session_id}",
                        })
                elif action_arg in [k.lower() for k in ACTION_TO_TEMPLATE]: # click on sub page such as description, features, reviews, attributes
                    sub_page = action_arg.capitalize()
                    r = await self._client.get(f"/item_sub_page/{self.session_id}/{asin}/{keywords}/{page_num}/{sub_page}/{options}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/item_sub_page/{self.session_id}/{asin}/{keywords}/{page_num}/{sub_page}/{options}",
                        })
                elif action_arg == END_BUTTON.lower():
                    if task_id:
                        r = await self._client.get(f"/done/{self.session_id}/{asin}/{options}?task_id={task_id}")
                    else:
                        r = await self._client.get(f"/done/{self.session_id}/{asin}/{options}")
                    if r.status_code == 200:
                        r = r.json()
                        self.state.update({
                            'html': r['observation'],
                            'reward': r['reward'],
                            'url': f"/done/{self.session_id}/{asin}/{options}",
                        })
                elif action_arg in self.text_to_clickable.keys(): # click on product attributes
                    input_html = str(self.text_to_clickable[action_arg])
                    element = self._parse_html(input_html).find('input')
                    name_value = element.get('name')
                    options[name_value] = action_arg
                    r = await self._client.get(f"/item_page/{self.session_id}/{asin}/{keywords}/{page_num}/{options}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/item_page/{self.session_id}/{asin}/{keywords}/{page_num}/{options}",
                        })
                else:
                    pass # invalid action, do nothing
            elif current_page == 'item_sub_page':
                asin = url_parts[3]
                keywords = literal_eval(url_parts[4])
                page_num = url_parts[5]
                sub_page = url_parts[6].capitalize()
                options = literal_eval(url_parts[7])
                if action_arg == PREV_PAGE.lower():
                    r = await self._client.get(f"/item_page/{self.session_id}/{asin}/{keywords}/{page_num}/{options}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/item_page/{self.session_id}/{asin}/{keywords}/{page_num}/{options}",
                        })
                elif action_arg == BACK_TO_SEARCH.lower():
                    r = await self._client.get(f"/index/{self.session_id}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/index/{self.session_id}",
                        })
                elif action_arg in [k.lower() for k in ACTION_TO_TEMPLATE]: # click on sub page such as description, features, reviews, attributes
                    sub_page = action_arg.capitalize()
                    r = await self._client.get(f"/item_sub_page/{self.session_id}/{asin}/{keywords}/{page_num}/{sub_page}/{options}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/item_sub_page/{self.session_id}/{asin}/{keywords}/{page_num}/{sub_page}/{options}",
                        })
                elif action_arg == END_BUTTON.lower():
                    if task_id:
                        r = await self._client.get(f"/done/{self.session_id}/{asin}/{options}?task_id={task_id}")
                    else:
                        r = await self._client.get(f"/done/{self.session_id}/{asin}/{options}")
                    if r.status_code == 200:
                        r = r.json()
                        self.state.update({
                            'html': r['observation'],
                            'reward': r['reward'],
                            'url': f"/done/{self.session_id}/{asin}/{options}",
                        })
                elif action_arg in self.text_to_clickable.keys(): # click on product attributes
                    input_html = str(self.text_to_clickable[action_arg])
                    element = self._parse_html(input_html).find('input')
                    name_value = element.get('name')
                    options[name_value] = action_arg
                    r = await self._client.get(f"/item_sub_page/{self.session_id}/{asin}/{keywords}/{page_num}/{sub_page}/{options}")
                    if r.status_code == 200:
                        self.state.update({
                            'html': r.text,
                            'url': f"/item_sub_page/{self.session_id}/{asin}/{keywords}/{page_num}/{sub_page}/{options}",
                        })
                else:
                    if action_name == 'search' and not has_search_bar:
                        return "You are not in the search page, you cannot search."
                    else:
                        return "Invalid action: " + action
                            
        available_actions = self.get_available_actions()
        clickables = available_actions['clickables']
        ob = self.observation + '\n' + "Clickables: " + str(clickables)
        if 'reward' in self.state:
            return {
                'observation': ob,
                'reward': self.state['reward'],
            }
        else:
            return ob

    async def aclose(self) -> None:
        """
        Release everything allocated by the environment.
        Stops the Docker container and closes the HTTP client.
        """
        # empty the states
        self.state = {'html': None, 'url': None}
        self.text_to_clickable = None
        await self._docker_stop()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        """
        Release everything allocated by the environment (alias for aclose).
        """
        await self.aclose()

    async def _connect(self):
        """
        Discover which host port Docker chose and open an httpx client
        targeting http://127.0.0.1:<host_port>.
        Raises RuntimeError if port mapping is not found.
        """
        deadline = time.time() + self.start_timeout
        host_port = None

        while time.time() < deadline:
            ports = self._container.attrs["NetworkSettings"]["Ports"]
            binding = ports.get(f"{self.container_port}/tcp")
            if binding:
                host_port = binding[0]["HostPort"]
                break
            await asyncio.sleep(0.1)
            self._container.reload()

        if host_port is None:
            logs = self._container.logs().decode()
            raise RuntimeError(f"Port mapping not found. Logs:\n{logs}")
        
        base_url = f"http://{self.host_ip}:{host_port}"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=20.0)

    def get_available_actions(self):
        """
        Returns list of available actions at the current step.
        Scans the current HTML for clickable elements and search bar.
        Returns:
            dict: { 'has_search_bar': bool, 'clickables': list of str }
        """
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links  = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f'{b.get_text()}'.lower(): b
            for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )
    

    def get_instruction_text(self):
        """
        Get corresponding instruction text for current environment session.
        Returns:
            str: The instruction text from the HTML.
        """
        html_obj = self._parse_html(self.state['html'])
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object.
        Args:
            html (str, optional): HTML string to parse. If None, uses current state HTML.
        Returns:
            BeautifulSoup: Parsed HTML object.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj
    
    @property
    def observation(self):
        """
        Compiles state into either the `html`, `text`, `text_rich`, or `url` observation mode.
        Returns:
            str: The current observation in the selected mode.
        """
        html = self.state['html']
        current_url = self.state.get('url')
        url_parts = current_url.split('/')
        current_page = url_parts[1]

        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            if current_page == 'done':
                return "Thank you for shopping with us!"
            else:
                return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    
    def convert_html_to_text(self, html, simple=True):
        """
        Strip HTML of tags and add separators to convert observation into simple or rich text mode.
        Args:
            html (str): HTML string to convert.
            simple (bool): If True, use simple [SEP] separators. If False, use rich formatting.
        Returns:
            str: The converted text observation.
        """
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join([t.strip() for t in visible_texts if t != '\n'])
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            for t in visible_texts:
                if t == '\n': continue
                if t.parent.name == 'button':  # button
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':  # options
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ["product-link"]: # product asins
                    if f'{t}' in self.server.user_sessions[self.session_id]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else: # regular, unclickable text
                    processed_t =  str(t)
                observation += processed_t + '\n'
            return observation

    @staticmethod
    def tag_visible(element):
        """
        Determine if a tag is visible (not style/script/head/title/meta/document/comment).
        Args:
            element: BeautifulSoup element.
        Returns:
            bool: True if visible, False otherwise.
        """
        ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
        if element.parent.parent is not None:
            return (
                element.parent.name not in ignore and not isinstance(element, Comment)# and not (element.parent.parent.get('id') == 'instruction-text')
            )
        else:
            return element.parent.name not in ignore and not isinstance(element, Comment)
    
    @staticmethod
    async def acquire():
        """
        Acquire and start a new WebAgentTextEnv environment asynchronously.
        Returns:
            WebAgentTextEnv: The started environment instance.
        """
        env = WebAgentTextEnv()
        await env.start()
        await env.reset()
        return env
    
    @staticmethod
    def parse_action(action):
        """
        Parse action string to action name and its arguments.
        Args:
            action (str): Action string, e.g., 'click[value]' or 'search[keywords]'.
        Returns:
            tuple: (action_name, action_arg)
        """
        pattern = re.compile(r'(.+)\[(.+)\]')
        m = re.match(pattern, action)
        if m is None:
            action_name = action
            action_arg = None
        else:
            action_name, action_arg = m.groups()
        return action_name, action_arg
