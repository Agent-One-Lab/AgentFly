Tool
==============

Base Tool Class
---------------

The main tool wrapper class:

.. autoclass:: agentfly.tools.tool_base.Tool
    :members:
    :special-members: __call__

Tool Decorator
-------------

The main decorator for creating tools:

.. autofunction:: agentfly.tools.tool_base.tool

Predefined Tools
---------------

The following are predefined tool instances that can be used directly with agents.

Code Interpreter
^^^^^^^^^^^^^^^

.. autofunction:: agentfly.tools.src.code.tools.code_interpreter

Google Search
^^^^^^^^^^^^

.. autofunction:: agentfly.tools.src.search.google_search.google_search_serper

Calculator
^^^^^^^^^^

.. autofunction:: agentfly.tools.src.calculate.tools.calculator

Answer Tools
^^^^^^^^^^^

.. autofunction:: agentfly.tools.src.react.tools.answer_qa

.. autofunction:: agentfly.tools.src.react.tools.answer_math
