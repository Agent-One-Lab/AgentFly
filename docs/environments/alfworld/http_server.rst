.. _alfworld_http_server:

ALFWorld HTTP Server
====================

The ALFWorld HTTP server provides a RESTful API for interacting with ALFWorld environments. It offers language-agnostic access to the environment functionality.

Server Module Reference
-----------------------

.. currentmodule:: agents.dockers.alfworld_env.alfworld_http_server
.. automodule:: agents.dockers.alfworld_env.alfworld_http_server
    :members:
    :undoc-members:
    :show-inheritance:

API Endpoints
-------------

Health Check
~~~~~~~~~~~~

**GET** ``/health``

Check if the server is running and ready to accept requests.

**Response:**
   - **200 OK**: Server is healthy
   - **Content**: ``{"status": "healthy"}``

Available Tasks
~~~~~~~~~~~~~~~

**GET** ``/available_tasks``

Get a list of available tasks for a given data split.

**Query Parameters:**
   - ``split`` (string): The data split (``train``, ``valid_seen``, ``valid_unseen``)

**Response:**
   - **200 OK**: Returns task list
   - **Content**: ``{"tasks": [{"task_id": "...", "split": "...", ...}, ...]}``

Reset Environment
~~~~~~~~~~~~~~~~~

**POST** ``/reset``

Reset the environment to start a new episode.

**Request Body:**
   - ``split`` (string): Data split to use
   - ``task_id`` (string, optional): Specific task ID to load

**Response:**
   - **200 OK**: Environment reset successful
   - **Content**: ``{"observation": "...", "info": {...}}``

Step Environment
~~~~~~~~~~~~~~~~

**POST** ``/step``

Execute an action in the environment.

**Request Body:**
   - ``action`` (string): The action command to execute

**Response:**
   - **200 OK**: Action executed successfully
   - **Content**: ``{"observation": "...", "reward": 0.0, "done": false, "info": {...}}``

Get Information
~~~~~~~~~~~~~~~

**GET** ``/info``

Get current environment state information.

**Response:**
   - **200 OK**: Current environment info
   - **Content**: ``{"info": {...}}``

Admissible Commands
~~~~~~~~~~~~~~~~~~~

**GET** ``/admissible_commands``

Get the list of valid commands for the current state.

**Response:**
   - **200 OK**: List of valid commands
   - **Content**: ``{"commands": ["go to kitchen", "take apple", ...]}``

Get Inventory
~~~~~~~~~~~~~

**GET** ``/inventory``

Get the current inventory contents.

**Response:**
   - **200 OK**: Current inventory
   - **Content**: ``{"inventory": "You are carrying: nothing."}``

Environment Configuration
-------------------------

The server can be configured using environment variables:

* **ALFWORLD_DATA**: Path to ALFWorld data directory (default: ``~/.cache/alfworld``)
* **ALFWORLD_CONFIG**: Path to ALFWorld config file (default: ``/srv/base_config.yaml``)
* **TRAIN_EVAL**: Default data split (default: ``train``)
* **BATCH_SIZE**: Batch size for ALFWorld environment (default: ``1``) 