Chat Template System Documentation
=======================================


Structure
---------------------------------------
.. toctree::
    :maxdepth: 2

    core_components
    basic_usage
    custom_templates
    advanced_features
    vision_templates
    examples
    template
    chat

Quick Start
---------------------------------------

.. code-block:: python

    from agentfly.agents.templates import Chat, get_template

    # Get a pre-built template
    template = get_template("qwen2.5")

    # Create a chat instance
    chat = Chat(template="qwen2.5", messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ])

    # Generate a prompt
    prompt = chat.prompt()
    print(prompt)

Key Concepts
------------

Template Components
^^^^^^^^^^^^^^^^^^^

- **System Template**: Defines system message format
- **User Template**: How user messages are formatted
- **Assistant Template**: How assistant responses are formatted
- **Tool Template**: How tool responses are formatted

Policies
^^^^^^^^

- **System Policy**: Controls system message behavior
- **Tool Policy**: Manages tool integration strategy
- **Global Policy**: Template-wide behavior settings

Vision Support
^^^^^^^^^^^^^^

- **Image Processing**: Automatic image token expansion
- **Video Processing**: Video frame extraction and processing
- **Multi-Modal Alignment**: Proper tensor alignment for training

Getting Started
---------------

1. **Read the** :doc:`Architecture <architecture>` to understand the system design
2. **Follow** :doc:`Basic Usage <basic_usage>` for quick setup
3. **Explore** :doc:`Examples <examples>` to see practical implementations
4. **Create** :doc:`Custom Templates <custom_templates>` for your specific needs
5. **Leverage** :doc:`Advanced Features <advanced_features>` for complex use cases
6. **Add** :doc:`Vision Support <vision_templates>` for multi-modal capabilities

Design Philosophy
-----------------

The Chat Template System is inspired by **building block toys** - where complex structures are created by combining simple, standardized components. This philosophy manifests in:

- **Modularity**: Interchangeable, composable, extensible components
- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Strategy Pattern**: Different behaviors can be selected at runtime
- **Policy-Based Configuration**: Flexible behavior control without hardcoding

System Architecture
-------------------

::

    Messages + Tools → Template Processing → Vision Processing → LLM-Ready Inputs

The system follows a **three-step rendering process**:

1. **Tool Insertion**: Decide where and how to inject tool definitions
2. **Turn Encoding**: Convert each conversation turn to its textual representation
3. **Generation Prompt**: Optionally append generation prefixes

Key Features
------------

- **Modular Design**: Templates built from configurable components
- **Multi-Modal Support**: Built-in vision and video processing
- **Tool Integration**: Flexible tool placement and formatting strategies
- **Policy-Based Configuration**: Fine-grained control over behavior
- **Jinja Template Generation**: Automatic HuggingFace-compatible templates
- **Extensible Architecture**: Easy to add new template types and processors

Additional Resources
-----------------------

- **Source Code**: ``agentfly/templates/``
- **API Reference**: Check the source code for detailed method documentation
- **Issues & Discussions**: Use the project's issue tracker for questions

Contributing
------------

The template system is designed to be extensible. See :doc:`Custom Templates <custom_templates>` for guidance on adding new template types and processors.
