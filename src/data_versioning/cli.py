"""
Module that contains the command line app.
"""
import argparse
import os
import traceback
import time
from google.cloud import storage
import shutil
import glob
import json

GCP_PROJECT = "AC215"
BUCKET_NAME = "fakenew_classifier_data_bucket"


def download_data(filepath="cleaned_metadata"):
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=filepath)
    for blob in blobs:
        if not blob.name.endswith("/"):
            try:
            # Download the blob to a local file with the same name
                blob.download_to_filename(blob.name)
                print(f"Downloaded: {blob.name}")
            except Exception as e:
                print(f"Error downloading {blob.name}: {str(e)}")


def main(args=None):
    if args.download:
        download_data(args.me)
    


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Versioning CLI...")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download labeled data from a GCS Bucket",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        default="cleaned_metadata",
    )

    args = parser.parse_args()

    main(args)
