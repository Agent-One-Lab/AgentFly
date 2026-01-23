#! /bin/bash

# Test CPU runs


pytest -x tests/unit/tools/ || exit 1
pytest -x tests/unit/envs/ || exit 1
pytest -x tests/unit/rewards/ || exit 1
pytest -x tests/unit/templates/ || exit 1
