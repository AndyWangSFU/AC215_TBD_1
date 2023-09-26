AC215-Template (Milestone2)
==============================

AC215 - Milestone2

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── setup.py
      ├── .gitignore
      ├── .gitattributes
      └── src
            ├── preprocessing
            │   ├── Dockerfile
            │   ├── preprocess.py
            │   └── requirements.txt
            │   └── Pipfile
            │   └── Pipfile.lock
            └── data_versioning
                  ├── Dockerfile
                  ├── cli.py
                  └── docker-shell.sh
                  └── .dvcignore
                  └── Pipfile
                  └── Pipfile.lock
                  └── README.md
                  └── .dvc
                  
                  


--------
# AC215 - Milestone2 - “Multimodal Fake News Detector”

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1

**Project**
In this project we aim to develop an application that develops and deploys two models, one for detecting fake content when only text is given and another when text and images are inputs. The rationale is that when a user chooses to use our API and, for instance, check news they found online, they might not have an image but will most likely have text. However, if the user can provide an image associated with the text, the image may provide additional context that improves the discriminatory power of our model.

### Milestone2 ###

We are using the Fakeddit dataset from Nakamura, Levy, and Wang in 2020 for model training and validation. The dataset comprises more than one million text samples that have been categorized into six distinct groups: "True," "Satire," "Misleading Content," "Manipulated Content," "False Connection," and "Imposter Content." Of these samples, 682,996 are multimodal being accompanied by images. We have stored these images in a private Google Cloud Bucket. Following consultation with our TF (Jarrod Parks), for Milestone 2, we focus on creating containers for preprocessing training images and data versioning for these images. The text data is precleaned in our training data which is why we focus on preprocessing image data for training in Milestone 2.

**Preprocess container**

- This container downloads data from the Google Cloud Bucket, resizes and processes the data, stores it back to GCP.
- Our inputs for this container depend on whether an image is simply being resized or augmented. For image resizing, the main parameter is size of output image. If augmenting image, parameters is the extent of augmentation (e.g., 5X) and the size of output images.
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we do preprocessing on our dataset. This contains code to perform the above steps. The output is saved on GCS. 

(2) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - `Pillow`, `albumentations` 

(3) `src/preprocessing/Pipefile` and `src/preprocessing/Pipefile.lock` are used to manage project dependencies and their versions. They are commonly associated with the package manager Pipenv

(4) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.9-slim-bookworm`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - 
make sure ac215-tbd-1.json is downloaded into src/preprocessing/secrets/ to enable GCP authentication
```
cd src/preprocessing/
docker build -t tbd1-preprocess -f Dockerfile .
docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app tbd1-preprocess

# if running the container for the first time, you might need to run:
pipenv install -r requirements.txt
```
Inside Docker container, sample preprocessing steps - 
```
# download first 10 images
python preprocess.py -d -f "raw_images/public_image_set" -m 10
 
# resize images to 100X100
python preprocess.py -p -f "raw_images/public_image_set" -s 100 --suffix "_processed"
# Alternatively, to augment images, producing 5X augmented images of size 128X128
python preprocess.py -p -f "raw_images/public_image_set" -s 128 -a -n 5 --suffix "_augmented"

# upload resized images individually
python preprocess.py -u -f "raw_images/public_image_set_processed"
# Alternatively, upload resized images in batch (as a compressed zip file) (Preferred)
python preprocess.py -u -f "raw_images/public_image_set_processed" -b 10000
```

** Data Versioning**
- This container reads the raw dataset and initiates the versioning using dvc.
  
(1) `src/data_versioning/cli.py`  -This script will simpily download raw data from the bucket

(2) `src/data_versioning/Pipefile` and `src/data_versioning/Pipefile.lock` are used to manage project dependencies and their versions. They are commonly associated with the package manager Pipenv 

(3) `src/data_versioning/Dockerfile` Dockerfile to build the container for data versioning

(4) `src/data_versioning/docker-shell.sh` Run this shell script to build and run the container

To run Dockerfile - 
make sure ac215-tbd-1.json is downloaded into src//secrets/ to enable GCP authentication
```
cd src/data_versioning/
sh docker-shell.sh

# initialize dvc tracking
dvc init
dvc remote add -d image_dataset gs://fakenew_classifier_data_bucket/dvc_store
dvc add public_image_set
dvc push
```

**Notebooks** 
This folder contains code that is not part of container
