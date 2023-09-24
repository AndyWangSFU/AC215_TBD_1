# preprocess.py
"""
Module that contains the command line app.
"""
import argparse
import os
from google.cloud import storage

GCP_PROJECT = "AC215"
BUCKET_NAME = "fakenew_classifier_data_bucket"
# Initiate Storage client

def download():
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # create a local folder for the downloaded file if not exist
    os.makedirs(args.filepath, exist_ok=True)

    # Find all content in a bucket
    blobs = bucket.list_blobs(prefix=args.filepath)
    for blob in blobs:
        if not blob.name.endswith("/"):
            try:
            # Download the blob to a local file with the same name
                blob.download_to_filename(blob.name)
                print(f"Downloaded: {blob.name}")
            except Exception as e:
                print(f"Error downloading {blob.name}: {str(e)}")
    
def process():
    print("process")

def upload():
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)

    # Destination path in GCS 
    destination_blob_name = args.filepath
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename("test.txt")

    for root, dirs, files in os.walk(args.filepath):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob_name = os.path.relpath(local_file_path, start=args.filepath)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded: {blob_name}")

def main(args=None):

    print("Args:", args)

    if args.download:
        download()
    if args.process:
        process()
    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(
        description='Synthesis audio from text')

    parser.add_argument("-d", "--download", action='store_true',
                        help="Download paragraph of text from GCS bucket")
    
    parser.add_argument("-p", "--process", action='store_true',
                        help="Process the downloaded file")

    parser.add_argument("-u", "--upload", action='store_true',
                        help="Upload audio file to GCS bucket")
    
    parser.add_argument("-f", "--filepath", type=str,
                        help="Download/Upload the file with this filepath(prefix)")

    args = parser.parse_args()

    main(args)
    