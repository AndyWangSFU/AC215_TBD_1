AC215-Template (Milestone3)
==============================

AC215 - Milestone3

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
# AC215 - Milestone3 - “Multimodal Fake News Detector”

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1

**Project**
In this project we aim to develop an application that develops and deploys two models, one for detecting fake content when only text is given and another when text and images are inputs. The rationale is that when a user chooses to use our API and, for instance, check news they found online, they might not have an image but will most likely have text. However, if the user can provide an image associated with the text, the image may provide additional context that improves the discriminatory power of our model.

### Milestone3

We address each of the objectives for Milestone 3 in the following ways:

1. Integrate Distributed Computing and Cloud Storage

[add info on distributed computing]. We have used Google Cloud Platform (GCP) to store our training and test images and text as it supports the vast scale of these datasets.

2. Utilize TensorFlow for Data Management

We initially tried to train the model using a data pipeline that only included TFData but found that training occured too slowly (hrs/days timeframe). We therefore implemented TFRecord which streamlined our data pipeline.

3. Develop Advanced Training Workflows

We train our model using both text and image data. We implement experiment tracking using Weights & Biases. We were able to train our model in several hours using a GCP virtual machine. We therefore did not feel the need to use serverless training. We performed model training using a single machine, single GPU strategy, although the code enables Single Machine, Multiple GPU if multiple GPUs avaliable (we could not get a quota for more than 1 GPU for any region).


**Key implementation highlights**

*Experiment Tracking*

Below is a screenshot from our Weights & Biases workspace. Tracking was performed using the `wandb` library we included inside of our `train.py` script. 

[add image]

#### Code Structure

**Data Folder**
Don't submit data, but we want to show one possible way of structuring data transformations.

**Data Processing Container**

- This container reads 100GB of data, transforming the images to TFRecords and stores them on a GCP bucket
- Input to this container is source and destination GCS location, parameters for resizing, secrets needed - via docker
- Output from this container stored on GCP bucket

(1) `src/datapipeline/dataloader.py`  - This script loads the original immutable data to our compute instance's local `raw` folder for processing.

(2) `src/datapipeline/build_records.py`  - Loads a local copy of the dataset, processes it according to our new transformations, converts it to TFRecords, and pushes it to a GCP bucket sink.

(3) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**VGG16 Training Container**

- This container contains all our training scripts and modeling components. It will use data from a GCP bucket, train, and then output model artifacts (saved model) to a GCP bucket.
- The input for this container is the source bucket for our training data and the output bucket for storing the trained model.
- Output is a saved TF Keras model.

(1) `src/models/vgg16/train_multi_gpu.py` - This script converts incoming data to TFRecords, applies standard image augmentation, and fits the model. It takes the following arguments:

> > --gpu [int] : the number of GPUs to use for training, default is 1
> > --input [string] : the source of the training data
> > --output [string] : the bucket which to store model artifacts

(3) `src/models/vgg16/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Instructions here`

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 
