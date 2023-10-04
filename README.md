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


- This container contains all our training scripts and modeling components. 
- It currently takes in a `.json` file that has the path to the cleaned metadata, image directory, and several model architectural and training hyperparameters.
- The `multimodal_binary_training.py` script will perform several model training runs given different `layer_sizes` in `.json` file and create multiple W&B runs. The model information and performance metrics are all stored in W&B
- The resulting model checkpoints of each training runs are stored as artifacts in W&B and can be easily reloaded to perform inference. 

(1) `src/models/multimodal_binary_training.py` - This script pulls in cleaned metadata and runs a prefetch-enabled TFData pipeline that resizes and normalizes the images and also turns the text into appropriate BERT inputs (text_input_mask, text_input_type_ids, text_input_word_ids), and fits models with different layer sizes. Single-node-multiGPU training is enabled as the script automatically trains the model on all available GPUs it can find in the host environment. In our case, we were only able to recieve quota for 1 GPU, and we configured a VM that has one GPU to run this.

It takes in a configuration `.json` (here as example `train_cli_example_input.json`) file that has the following arguments:

> > --batch_size [int] : the size of batches used to train the model
> > --layer_size [int] : the size of the adjustable dense layer
> > --max_epochs [int] : a parameter to define the max number of epochs for model training
> > --train_path [string] : path to training metadata
> > --val_path [string] : path to validation metadata
> > --input_mode [string]: mode of input, current it only support TFData

(2) `src/models/Dockerfile` - This dockerfile starts with  `FROM tensorflow/tensorflow:2.13.0-gpu`. This statement uses the tensorflow-gpu as the base image for version 2.13.0.

(3) `src/models/run_docker.sh` Shell script to run the container

To run Dockerfile:
1. docker build -t training -f Dockerfile .
2. sh run_docker.sh


**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 
