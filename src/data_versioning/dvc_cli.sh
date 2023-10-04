#!/bin/bash

# mount the bucket
mkdir -p mounted_bucket
gcsfuse --key-file secrets/ac215-tbd-1.json fakenew_classifier_data_bucket mounted_bucket/
 
# Ask the user if they want to initialize DVC
read -p "Are you initializing DVC for the first time? (y/n): " initialize_dvc

# Convert the user input to lowercase
initialize_dvc=$(echo "$initialize_dvc" | tr '[:upper:]' '[:lower:]')

if [ "$initialize_dvc" = "y" ]; then
    # Run the DVC initialization command
    dvc init
    dvc remote add -d mounted_remote /mounted_bucket/dvc_store
else
    echo "DVC file tracking already initialized."
fi