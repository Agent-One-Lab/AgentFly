#!/bin/bash

docker run \
    --name webshop-simulator \
    -p 3000:3000 \
    --rm \
    webshop-simulator-env:latest
