# data_loader.py
"""
Module that download and upload data.
"""
import argparse
import os
from google.cloud import storage
import zipfile

GCP_PROJECT = "AC215"
BUCKET_NAME = "fakenew_classifier_data_bucket"
# Initiate Storage client

def download(filepath, max_num):
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # create a local folder for the downloaded file if not exist
    os.makedirs(filepath, exist_ok=True)

    # Find all content in a bucket
    blobs = bucket.list_blobs(prefix=filepath)
    for blob in blobs:
        if max_num == 0:
            break
        if not blob.name.endswith("/"):
            try:
            # Download the blob to a local file with the same name
                blob.download_to_filename(blob.name)
                print(f"Downloaded: {blob.name}")
            except Exception as e:
                print(f"Error downloading {blob.name}: {str(e)}")
            max_num -= 1


def upload(filepath, batch_size=10000):
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)

    # Destination path in GCS 
    destination_blob_name = filepath
    print(f"Uploading to {destination_blob_name}")
    
    # Create a list of image files
    image_files = [os.path.join(filepath, filename) for filename in os.listdir(filepath) if filename.endswith('.jpg')]
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    # Create zip files for each batch of images
    for i, batch in enumerate(batches):
        zip_filename = os.path.join(destination_blob_name, f'images_batch_{i + 1}.zip')
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for image_file in batch:
                zipf.write(image_file, os.path.basename(image_file))

        print(f"Created zip file: {zip_filename}")
        
        blob = bucket.blob(zip_filename)
        blob.upload_from_filename(zip_filename)  
        print(f"Uploaded: {zip_filename}")
        os.remove(zip_filename)

    # unused code to upload individual files
    # img_files = glob.glob(filepath + "/*.jpg")
    # for img_file in img_files:
    #     blob = bucket.blob(img_file)
    #     blob.upload_from_filename(img_file)
    #     print(f"Uploaded: {img_file}")


def main(args=None):

    # print("Args:", args)
    if not args.filepath:
        print("Using default filepath: data/")
        filepath = "raw_images/public_image_set"
    else:
        print("Using filepath:", args.filepath)
        filepath = args.filepath
    if not args.max:
        print("getting all files")
        max_num = -1
    else:
        print(f"getting first {args.max} files:")
        max_num = args.max

    if not args.batch_size:
        args.batch_size=10000
        
    if args.download:
        download(filepath, max_num)

    if args.upload:
        upload(filepath, args.batch_size)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(
        description='Upload/Download/Process images from GCS bucket')

    parser.add_argument("-d", "--download", action='store_true',
                        help="Download paragraph of text from GCS bucket")

    parser.add_argument("-u", "--upload", action='store_true',
                        help="Upload image to GCS bucket")
    
    parser.add_argument("-f", "--filepath", type=str, default="raw_images/public_image_set",
                        help="Download/Upload the file with this filepath(prefix)")
    
    parser.add_argument("-m", "--max", type=int,
                        help="Max number of files to download")
    
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Number of images to upload per batch")

    args = parser.parse_args()

    main(args)
    
