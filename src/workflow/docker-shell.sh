#!/bin/bash

set -e

export IMAGE_NAME="fakenews-detector-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
#export GCP_PROJECT="AC215"
export GCP_PROJECT="ac215-398714"
export GCS_BUCKET_NAME="fakenew_classifier_data_bucket"
export GCS_SERVICE_ACCOUNT="id-15-project@ac215-398714.iam.gserviceaccount.com"
export GCP_REGION="us-central1"
# export GCS_PACKAGE_URI="gs://fakenew_classifier-trainer-code"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
docker build -t fakenews-detector-workflow --platform=linux/amd64 -f Dockerfile .

# Run Container
docker run --rm --name fakenews-detector-workflow -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$BASE_DIR/../preprocessing":/preprocessing \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-tbd-1.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
-e GCP_REGION=$GCP_REGION \
$IMAGE_NAME

# -e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \