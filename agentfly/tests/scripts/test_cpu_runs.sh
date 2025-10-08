#! /bin/bash

# Test CPU runs


pytest -x agentfly/tests/unit/tools/
pytest -x agentfly/tests/unit/envs/
pytest -x agentfly/tests/unit/rewards/

pytest -x agentfly/tests/unit/templates/