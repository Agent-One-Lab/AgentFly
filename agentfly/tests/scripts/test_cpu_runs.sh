#! /bin/bash

# Test CPU runs


pytest -x agentfly/tests/unit/tools/ || exit 1
pytest -x agentfly/tests/unit/envs/ || exit 1
pytest -x agentfly/tests/unit/rewards/ || exit 1
pytest -x agentfly/tests/unit/templates/ || exit 1