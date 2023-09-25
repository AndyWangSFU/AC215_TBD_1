# preprocess.py
"""
Module that contains the command line app.
"""
import argparse
import glob
import os
from google.cloud import storage
from PIL import Image
import numpy as np
import albumentations as A

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

def augment_img(img_path, size, augment_num=5, new_folder="data_augmented"):
    transform = A.Compose([
        A.RandomCrop(width=size, height=size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    for i in range(augment_num):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        transformed = transform(image=img)
        transformed_image = Image.fromarray(transformed["image"])
        transformed_image.save(
            os.path.join(
                new_folder, os.path.basename(img_path).split(".")[0] + f"_{i}.jpg"
            )
        )
    print(f"Augmented: {img_path}")

    
def process(filepath, size, file_suffix="_processed", augment=False, augment_num=5):
    # for each image, resize it to 128X128 (default) and save it to new folder
    img_paths = glob.glob(filepath + "/*.jpg")
    new_folder = filepath + file_suffix
    print(f"Saving processed images to {new_folder}")
    os.makedirs(new_folder, exist_ok=True)

    for img_path in img_paths:
        img = Image.open(img_path)
        if img.verify() == False:
            print(f"Error opening {img_path}. Skipping...")
            continue
        if augment:
            augment_img(img_path, size, augment_num, new_folder)
        else:
            img = img.resize((size, size))
            img.save(os.path.join(new_folder, os.path.basename(img_path)))
        print(f"Processed: {img_path}")
    

def upload(filepath):
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)

    # Destination path in GCS 
    destination_blob_name = filepath + "/"
    blob = bucket.blob(destination_blob_name)
    print(f"Uploading to {destination_blob_name}")
    img_files = glob.glob(filepath + "/*.jpg")
    for img_file in img_files:
        blob.upload_from_filename(img_file)
        print(f"Uploaded: {img_file}")

def main(args=None):

    # print("Args:", args)
    if not args.filepath:
        print("Using default filepath: data/")
        filepath = "data"
    else:
        print("Using filepath:", args.filepath)
        filepath = args.filepath
    if not args.max:
        print("getting all files")
        max_num = -1
    else:
        print(f"getting first {args.max} files:")
        max_num = args.max
    if not args.size:
        args.size=128
    if not args.suffix:
        args.suffix="_processed"
    if not args.augment_num:
        args.augment_num=5

    if args.download:
        download(filepath, max_num)
    if args.process:
        process(
            filepath, size=args.size, file_suffix=args.suffix, 
            augment=args.augment, augment_num=args.augment_num
        )
    if args.upload:
        upload(filepath)


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
    
    parser.add_argument("-m", "--max", type=int,
                        help="Max number of files to process")
    
    parser.add_argument("-s", "--size", type=int,
                        help="Dimension of the processed image")
    
    parser.add_argument("-S", "--suffix", type=str,
                        help="Suffix of the processed image")
    
    parser.add_argument("-a", "--augment", action='store_true',
                        help="Augment the processed image")
    
    parser.add_argument("-n", "--augment_num", type=int,
                        help="Number of augmented images per image")

    args = parser.parse_args()

    main(args)
    