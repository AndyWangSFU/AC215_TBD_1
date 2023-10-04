<img width="1260" alt="image" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/d412a056-e1a5-4061-b652-656f3b71a44c">AC215-Template (Milestone3)
==============================

AC215 - Milestone3

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── .gitignore
      ├── .gitattributes
      ├── .dvcignore
      ├── .dvc
      ├── reports
      └── src
            ├── preprocessing
            │   ├── Dockerfile
            │   ├── data_loader.py
            │   ├── process.py
            │   ├── requirements.txt
            │   ├── Pipfile
            │   └── Pipfile.lock
            ├── data_versioning
            │   ├── Dockerfile
            │   ├── cli.py
            │   ├── docker-shell.sh
            │   ├── .dvcignore
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── README.md
            │   ├── .dvc
            │   └── dvc_cli.sh
            ├── models
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── multimodal_binary_training.py
            │   ├── requirements.txt
            │   ├── run_docker.sh
            │   ├── train.py
            │   └── train_cli_example_input.json
            ├── tfrecords
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── requirements.txt
            │   └── tfrecords.py      


--------
# AC215 - Milestone3 - “Multimodal Fake News Detector”

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1

**Project**
In this project we aim to build and deploy a model that can detecting fake content when text and images are provided as inputs. 

### Milestone3

**Objectives for Milestone3**

We address each of the objectives for Milestone 3 in the following ways:

1. Integrate Distributed Computing and Cloud Storage

[add info on distributed computing]. We have used Google Cloud Platform (GCP) to store our training and test images/text as it supports the vast scale of these datasets.

<img width="1268" alt="Screenshot 2023-10-04 at 7 18 31 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/4d2b99d6-e94f-4ef6-8399-5d9ffe41cc46">





2. Utilize TensorFlow for Data Management

We built a TFRecords container and have generated some TFRecords files which we have tested for use in training our model. We also tested TFData for model training. We found that TFRecords did not streamline our pipeline significantly compared to TFData and was fairly slow to generate when implemented. Therefore, for now, we are performing model training with pre-fetched TFData files and it works well. We are keeping the TFRecords container in our repo because if we subsequently find TFRecords indeed provides large performance boosts, we aim to leverage TFRecords in Milestone 4.

3. Develop Advanced Training Workflows

We train our model using both text and image data. We implement experiment tracking using Weights & Biases. Tracking was performed using the `wandb` library we included inside of our `train.py` script. We were able to train our model in several hours using a GCP virtual machine. We therefore did not feel the need to use serverless training. We performed model training using a single machine, single GPU strategy, although the code enables Single Machine, Multiple GPU if multiple GPUs avaliable (we could not get a quota for more than 1 GPU for any region).


#### Code Structure

**Data Pre-Processing Container** [to be adjusted]

- This container downloads data from the Google Cloud Bucket, resizes and processes the data, stores it back to GCP.
- Our inputs for this container depend on whether an image is simply being resized or augmented. For image resizing, the main parameter is size of output image. If augmenting image, parameters is the extent of augmentation (e.g., 5X) and the size of output images.
- In addition, the container downloads the metadata for the images from Google Cloud Bucket, filters and cleans the metadata for images that are uncorrupted, and stores this filtered metadata back into GCP.
- The rationale for not downloading the images themselves and preprocessing them is that the image dataset is very large and has already been preprocessed. We only need to check for whether an image is uncorrupted or not. Similarly, the text data that we use to train and test our model has already been pre-processed. We therefore leave these in the GCP as well and do not download them using our pre-processing container. 
- Output from this container stored at GCS location

(1) `src/datapipeline/dataloader.py`  - This script loads the original immutable data to our compute instance's local `raw` folder for processing.

(2) `src/datapipeline/build_records.py`  - Loads a local copy of the dataset, processes it according to our new transformations, converts it to TFRecords, and pushes it to a GCP bucket sink.

(3) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Data Versioning Container**


**TFRecoed container**


 
**Model Training Container**

- This container contains all our training scripts and modeling components. It will use TFRecords, train, and then output model to a GCP bucket.
- The input for this container is TFRecords corresponding to our training data and the output is a bucket for storing the trained model.
- Output is a saved TF Keras model.

(1) `train.py` - This script pulls in TFRecords for images that have already been augmented and preprocessed and fits the model. It takes the following arguments:

> > --batch_size [int] : the size of batches used to train the model
> > --layer_size [int] : the size of the adjustable dense layer
> > --num_gpus [int] : a parameter to adjust the number of GPUs used to train the model. Default is 1
> > --max_epochs [int] : a parameter to define the max number of epochs for model training
> > --train_path [string] : path to training dataset
> > --num_gpus [string] : path to validation dataset

(3) `src/models/vgg16/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 
