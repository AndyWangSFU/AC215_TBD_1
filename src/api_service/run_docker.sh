# Define some environment variables
export IMAGE_NAME="fake-news-classifier-api-service"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/secrets/
export GCS_BUCKET_NAME="fakenew_classifier_data_bucket"
export GCP_PROJECT="ac215-398714"


# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-p 9000:9000 \
-e DEV=1 \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-tbd-1.json \
-e WANDB_KEY=/secrets/wandb_key.txt \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
$IMAGE_NAME
