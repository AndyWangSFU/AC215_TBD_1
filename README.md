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
      └── src
            ├── preprocessing
            │   ├── Dockerfile
            │   ├── preprocess.py
            │   └── requirements.txt
            └── validation
                  ├── Dockerfile
                  ├── cv_val.py
                  └── requirements.txt


--------
# AC215 - Milestone2 - “Multimodal Fake News Detector”

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1

**Project**
In this project we aim to develop an application that develops and deploys two models, one for detecting fake content when only text is given and another when text and images are inputs. The rationale is that when a user chooses to use our API and, for instance, check news they found online, they might not have an image but will most likely have text. However, if the user can provide an image associated with the text, the image may provide additional context that improves the discriminatory power of our model.

### Milestone2 ###

We gathered dataset of 1M butterflies representing 17K species. Our dataset comes from following sources - (1),(2),(3) with approx 100GB in size. We parked our dataset in a private Google Cloud Bucket. 

**Preprocess container**
- This container reads 100GB of data and resizes the image sizes and stores it back to GCP
- Input to this container is source and destincation GCS location, parameters for resizing, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocessing/preprocess.py`  - Here we do preprocessing on our dataset of 100GB, we reduce the image sizes (a parameter that can be changed later) to 128x128 for faster iteration with our process. Now we have dataset at 10GB and saved on GCS. 

(2) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - `special butterfly package` 

(3) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.9-slim-bookworm`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - 
make sure ac215-tbd-1.json is downloaded into src/preprocessing/secrets/
```
cd src/preprocessing/
docker build -t tbd1-preprocess -f Dockerfile .
docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app tbd1-preprocess
```
Inside Docker container, sample preprocessing steps - 
```
# download first 10 images
python preprocess.py -d -f "raw_images/public_image_set" -m 10 
# resize images to 100X100
python preprocess.py -p -f "raw_images/public_image_set" -s 100
# upload resized images
python preprocess.py -u -f "raw_images/public_image_set_processed"
```

**Cross validation, Data Versioning**
- This container reads preprocessed dataset and creates validation split and uses dvc for versioning.
- Input to this container is source GCS location, parameters if any, secrets needed - via docker
- Output is flat file with cross validation splits
  
(1) `src/validation/cv_val.py` - Since our dataset is quite large we decided to stratify based on species and kept 80% for training and 20% for validation. Our metrics will be monitored on this 20% validation set. 

(2) `requirements.txt` - We used following packages to help us with cross validation here - `iterative-stratification` 

(3) `src/validation/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 

----
You may adjust this template as appropriate for your project.
