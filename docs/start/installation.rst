Installation
==============

To install AgentFly, follow these steps:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/agentfly/agentfly.git
    cd agentfly

2. Initialize and update git submodules:

.. code-block:: bash

    git submodule init
    git submodule update

3. Install Python dependencies:

.. code-block:: bash

    pip install -r agents/requirements.txt
    pip install -r verl/requirements.txt

4. Install Redis server (required for caching search results):

.. code-block:: bash

    conda install conda-forge::redis-server==7.4.0

