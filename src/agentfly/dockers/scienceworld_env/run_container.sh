#!/bin/bash

docker run \
    --name scienceworld-env \
    -p 2700:2700 \
    --rm \
    reasonwang/scienceworld-env:latest
