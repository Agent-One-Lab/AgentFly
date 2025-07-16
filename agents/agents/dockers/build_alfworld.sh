#!/bin/bash
docker build --no-cache --network=host -t agentfly/alfworld-http-env:latest ./alfworld_env 