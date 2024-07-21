#!/bin/bash

# Variables for paths
MODEL_SOURCE_PATH="path/to/your/model" # Update this to the actual path of your model on the host
VOLUME_NAME="model-data"
CONTAINER_PATH="/app/model"

# Check if the model source path exists
if [ ! -d "$MODEL_SOURCE_PATH" ]; then
  echo "Model source path does not exist: $MODEL_SOURCE_PATH"
  exit 1
fi

# Copy model data to the Docker volume
docker run --rm -v ${VOLUME_NAME}:${CONTAINER_PATH} -v $(pwd)/${MODEL_SOURCE_PATH}:/model alpine sh -c "cp -r /model/* ${CONTAINER_PATH}/"

echo "Model data has been copied to the Docker volume: ${VOLUME_NAME}"
