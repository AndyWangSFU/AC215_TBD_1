import argparse
import glob
import os
import numpy as np
from PIL import Image
import albumentations as A
import pandas as pd


def update_metadata(inpath = 'raw_metadata/multimodal_train.tsv', image_directory = 'raw_images/public_image_set', outpath = 'cleaned_multimodal_train.tsv'):
    metadata = pd.read_csv(inpath, sep='\t')

    # drop unlabelled data
    metadata = metadata.dropna(subset=['2_way_label'])
    print(f"Originally {len(metadata)} rows")
    
    # List to store valid rows with image paths
    valid_rows = []
    corrupted_image = 0
    not_found_image = 0

    # Iterate through each row in the metadata
    for index, row in metadata.iterrows():
        # Extract the id from the current row
        image_id = row['id']
        
        # Construct the full path to the corresponding .jpg file
        image_path = os.path.join(image_directory, f"{image_id}.jpg")
        
        # Check if the .jpg file exists and can be opened
        if os.path.exists(image_path):
            try:
                # Attempt to open the image to ensure it is a valid image file
                with Image.open(image_path):
                    # If successfully opened, add the 'path' column and store the image path
                    row['path'] = os.path.join('data/public_image_set/', f"{image_id}.jpg")
                    valid_rows.append(row)
            except Exception as e:
                corrupted_image += 1
                #print(f"Error opening image {image_path}: {e}")
        else:
            not_found_image += 1
            #print(f"Image not found: {image_path}")

    print(f"Corrupted images: {corrupted_image}")
    print(f"Images not found: {not_found_image}")

    # Create a new DataFrame with valid rows
    clean_metadata = pd.DataFrame(valid_rows)

    # Save the cleaned metadata to a new directory called 'cleaned_metadata'
    os.makedirs('cleaned_metadata', exist_ok=True)

    print(f"Cleaned metadata saved to cleaned_metadata/{outpath}")
    clean_metadata.to_csv(os.path.join('cleaned_metadata', outpath), sep='\t', index=False)


def augment_img(img_path, size, augment_num=5, new_folder="data_augmented"):
    transform = A.Compose([
        A.RandomCrop(width=size, height=size, p = 1),
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

    
def process(filepath, size, process_suffix="_processed", augment=False, augment_suffix="_augmented", augment_num=5):
    # for each image, resize it to 128X128 (default) and save it to new folder
    img_paths = glob.glob(filepath + "/*.jpg")

    process_folder = "public_image_set" + process_suffix
    print(f"Saving processed images to {process_folder}")
    os.makedirs(process_folder, exist_ok=True)

    if augment:
        augment_folder = "public_image_set" + augment_suffix
        print(f"Saving augmented images to {augment_folder}")
        os.makedirs(augment_folder, exist_ok=True)
    
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        original_width, original_height = img.size
        crop_size = min(original_width, original_height, size)
        
        # check if the image is corrupted
        if img.verify() == False:
            print(f"Error opening {img_path}. Skipping...")
            continue
        img = img.resize((crop_size, crop_size))
        img.save(os.path.join(process_folder, os.path.basename(img_path)))
        print(f"Processed: {img_path}")

        if augment:
            augment_img(img_path, crop_size, augment_num, augment_folder)


def main(args=None):
    # print("Args:", args)
    if not args.filepath:
        print("Using default filepath: raw_images/public_image_set/")
        filepath = "raw_images/public_image_set"
    else:
        print("Using filepath:", args.filepath)
        filepath = args.filepath
    
    if not args.size:
        args.size=128

    if not args.pro_suf:
        args.pro_suf="_processed"
    if not args.aug_suf:
        args.aug_suf="_augmented"
    if not args.augment_num:
        args.augment_num=5

    if not args.inpath:
        args.inpath='raw_metadata/multimodal_train.tsv'
    if not args.inpath:
        args.outname='cleaned_multimodal_train.tsv'

    if args.process:
        process(
            filepath, size=args.size, process_suffix=args.pro_suf, 
            augment=args.augment, augment_suffix = args.aug_suf, augment_num=args.augment_num
        )

    if args.clean_meta:
        update_metadata(
            inpath = args.inpath, image_directory=filepath, outpath=args.outpath
        )

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    
    parser = argparse.ArgumentParser(
        description='Process images from GCS bucket')

    parser.add_argument("-p", "--process", action='store_true',
                        help="Process the downloaded file")
    
    parser.add_argument("-a", "--augment", action='store_true',
                        help="Augment the processed image")
    
    parser.add_argument("-c", "--clean_meta", action='store_true',
                        help="Clean up metadata")
    
    parser.add_argument("-f", "--filepath", type=str,
                        help="use the images in filepath(prefix)")
    
    parser.add_argument("-i", "--inpath", type=str,
                        help="input the files in this path")
    
    parser.add_argument("-o", "--outname", type=str,
                        help="store the resulting files with this name")

    parser.add_argument("-s", "--size", type=int,
                        help="Dimension of the processed image")
    
    parser.add_argument("--pro_suf", type=str,
                        help="Suffix of the processed image")
    
    parser.add_argument("--aug_suf", type=str,
                        help="Suffix of the processed image")
  
    parser.add_argument("-n", "--augment_num", type=int,
                        help="Number of augmented images per image")

    args = parser.parse_args()

    main(args)
    
