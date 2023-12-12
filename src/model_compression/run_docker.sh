#!/bin/bash
export IMAGE_NAME="model_compression"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../secrets/
export GCP_PROJECT="ac215-project"
export GCS_BUCKET_NAME="fakenew_classifier_data_bucket"
export GCS_SERVICE_ACCOUNT="id-15-project@ac215-398714.iam.gserviceaccount.com"
export GCP_REGION="us-central1"


# Run the Docker container with the data directory mounted
docker run --rm -ti \
  --mount type=bind,source="$(pwd)",target="/app" \
  -v "$SECRETS_DIR":/secrets \
  -e GOOGLE_APPLICATION_CREDENTIALS=./secrets/ac215-tbd-1.json \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME compression