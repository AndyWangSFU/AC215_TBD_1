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

![AC215_TBD_1](https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/53b18850-f0d1-47a4-8f78-71724c18faff)
Figure 1: Project data pipeline

### Milestone3

**Objectives for Milestone3**

We address each of the objectives for Milestone 3 in the following ways:

1. Integrate Distributed Computing and Cloud Storage

[add info on distributed computing]. We have used Google Cloud Platform (GCP) to store our training and test images/text as it supports the vast scale of these datasets.

<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/5d9e256a-c711-430f-b6da-b8f8ed4377c3">

Figure 2: Google Cloud Platform being used to store different versions of training and test data


2. Utilize TensorFlow for Data Management

We built a TFRecords container and have generated some TFRecords files which we have tested for use in training our model. We also tested TFData for model training. We found that TFRecords did not streamline our pipeline significantly compared to TFData and was fairly slow to generate when implemented. Therefore, for now, we are performing model training with pre-fetched TFData files and it works well. We are keeping the TFRecords container in our repo because if we subsequently find TFRecords indeed provides large performance boosts, we aim to leverage TFRecords in Milestone 4.

3. Develop Advanced Training Workflows

We train our model using both text and image data. We implement experiment tracking using Weights & Biases. Tracking was performed using the `wandb` library we included inside of our `train.py` script. We were able to train our model in several hours using a GCP virtual machine. We therefore did not feel the need to use serverless training. We performed model training using a single machine, single GPU strategy, although the code enables Single Machine, Multiple GPU if multiple GPUs avaliable (we could not get a quota for more than 1 GPU for any region).

<img width="1271" alt="Screenshot 2023-10-04 at 7 22 14 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/f204bf95-e939-4cd5-91dc-768b726bd692">

Figure 3: screenshot of our Weights & Biases dashboard with model training charts


#### Code Structure

**Data Pre-Processing Container**

- This container downloads data from the Google Cloud Bucket, processes the image metadata, processes/augments the images themselves, and stores the data back to GCP.

(1) `src/preprocessing/data_loader.py`  - This script downloads and uploads data to and from GCP. There are two key functions:

a)	"download" 
Function: Download file from “file_path”
Usage: python data_loader.py -d -f “file_path”
Optional: -m number (max number of images to download)

b)	"upload" 
Function: Upload files in local “file_path” in the form of zipped files
Usage: python data_loader.py -u -f “file_path”
Optional: -b number (number of images in a zipfile to upload)

(2) `src/preprocessing/process.py`  - This script performs the preprocessing steps on the metadata and images. There are three key functions:

a)	"update_metadata"
Function: Process a metadata file. Drop NA values in label. Look for potential corrupted and non-existing images. If found, drop the corresponding row from the metadata. Save the cleaned metadata in.a folder called “cleaned metadata”.
Usage: python process.py -c --inpath “path_to_metadata” -f “path_to_images” –outname “name_of_new_metadata”

b)	"process" 
Function: Process the images inside a folder. Read them if possible and resize it. Optionally, users can choose to augment the images. The processed imaged are stored in a folder called “public_image_set” + args.pro_suf.
Usage: python data_loader.py -p -f “path_to_images” -s image_dimension
Optional: -a (to do augmentation). -- pro_suf “suffix_to_name_the_folder”

c)	"augment" 
Function: Augment the images while processing. When an image is processed, it can also be augmented to create several augmented images. The number of augmented images per image processed can be declared by args.augment_num or -n number. The processed imaged are stored in a folder called “public_image_set” + args.aug_suf.
Usage: python data_loader.py -p -f “path_to_images” -s image_dimension -a
Optional: -- aug_suf “suffix_to_name_the_folder”. -n number.

(3) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - `Numpy`, `opencv-python-headless`, `Pillow`, `albumentations`, `google-cloud-storage`, `pandas`  

(4) `src/preprocessing/Pipefile` and `src/preprocessing/Pipefile.lock` are used to manage project dependencies and their versions. They are commonly associated with the package manager Pipenv

(5) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.9-slim-bookworm`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - 
make sure ac215-tbd-1.json is downloaded into src/preprocessing/secrets/ to enable GCP authentication
```
cd src/preprocessing/
docker build -t tbd1-preprocess -f Dockerfile .
docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app tbd1-preprocess

# if running the container for the first time, you might need to run:
pipenv install -r requirements.txt
```

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
1. `docker build -t training -f Dockerfile .`
2. `sh run_docker.sh`

**Notebooks** 
This folder contains code that is not part of container - for e.g: EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations. 
