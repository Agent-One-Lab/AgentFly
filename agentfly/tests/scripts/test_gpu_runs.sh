#! /bin/bash

# Test GPU runs

python -m pytest -x tests/unit/agents/test_initialization.py || exit 1
python -m pytest -x tests/unit/agents/test_auto_agent.py || exit 1
python -m pytest -x tests/unit/agents/test_code_agent.py || exit 1
python -m pytest -x tests/unit/agents/test_react_agent.py || exit 1
python -m pytest -x tests/unit/agents/test_webshop_agent.py || exit 1
python -m pytest -x tests/unit/agents/test_vision_agent.py || exit 1