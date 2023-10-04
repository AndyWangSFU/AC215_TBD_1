#!/bin/bash
data_dir="$(realpath "$(pwd)/../../data")"

# Run the Docker container with the data directory mounted
docker run --rm --gpus all -ti \
  --mount type=bind,source="$(pwd)",target="/app" \
  --mount type=bind,source="$data_dir",target="/app/data" training