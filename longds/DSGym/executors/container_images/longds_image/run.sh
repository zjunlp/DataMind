#!/bin/bash

PORT=${1:-8432}
CODE_FOLDER=${2:-""}

# Build the Docker image
docker build -t ipython-executor .

# Run the container with configurable port and optional code folder
if [ -n "$CODE_FOLDER" ]; then
    docker run -p $PORT:$PORT -e PORT=$PORT -v "$CODE_FOLDER:/code:ro" ipython-executor
else
    docker run -p $PORT:$PORT -e PORT=$PORT ipython-executor
fi 