Milestone4
==============================

AC215 - Milestone4

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references          <- a list of papers related to our project
      ├── requirements.txt
      ├── .gitignore
      ├── .gitattributes
      ├── reports
      ├── presentations        
      │     └── midterm.pdf     <- our midterm presentation slides
      └── src
            ├── preprocessing
            │   ├── Dockerfile
            │   ├── data_loader.py
            │   ├── process.py
            │   ├── requirements.txt
            │   ├── docker-entrypoint.sh
            │   ├── docker-shell.sh
            │   ├── Pipfile
            │   └── Pipfile.lock
            ├── model_compression
            │   ├── Dockerfile
            │   ├── docker-entrypoint.sh
            │   ├── downsize_model.py
            │   ├── model_helpers.py
            │   ├── multimodal_binary_training.py
            │   ├── requirements.txt
            │   └── run_docker.sh
            ├── training      <- model training and evaluation
            │   │── package (not finally used)
            │   │       ├── trainer
            │   │       │     └── multimodal_binary_training.py
            │   │       ├── PKG-INFO.txt
            │   │       ├── setup.cfg
            │   │       └── setup.py
            │   ├── Dockerfile
            │   ├── cli.py (not finally used)
            │   ├── cli.sh (not finally used)
            │   ├── multimodal_binary_training.py
            │   ├── docker-entrypoint.sh (not finally used)
            │   ├── docker-shell.sh (not finally used)
            │   ├── requirements.txt
            │   ├── train_cli_example_input.json
            │   └── run_docker.sh
            └── workflow      <- scripts for automating data download, preprocess, model training, and compression
                ├── Dockerfile
                ├── Pipfile
                ├── Pipfile.lock
                ├── cli.py
                ├── compress.yaml
                ├── docker-entrypoint.sh
                ├── docker-shell.sh
                ├── download.yaml
                ├── model.py
                ├── pipeline.yaml
                ├── sample-pipeline1.yaml
                └── train.yaml    


--------
# AC215 - Milestone4 - “Multimodal Fake News Detector”

**Team Members**
Kyle Ke, Boshen Yan, Fuchen Li, Zihan Wang, Qassi Gaba

**Group Name**
TBD_1

**Project**
In this project we aim to build and deploy a model that can detecting fake content when text and images are provided as inputs. The rationale is that when a user chooses to use our API and, for instance, check news they found online, they might not have an image but will most likely have text. 

<p align="center">
  <img width="460" height="300" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/53b18850-f0d1-47a4-8f78-71724c18faff">
</p>

<p style="text-align: center;">Figure 1: Project data pipeline</p>


### Milestone4

<ins>**Objectives for Milestone4**</ins>

We address each of the objectives for Milestone 4 in the following ways:

*1. Distillation/Quantization/Compression*



<img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/5d9e256a-c711-430f-b6da-b8f8ed4377c3">

Figure 2:   


*2. Vertex AI Pipelines (Kubeflow) and Cloud Functions Integration*





<ins>**Code Structure**</ins>
















**README from milestone3 (remove later)**



*Data Pre-Processing Container*

- This container downloads data from the Google Cloud Bucket, processes the image metadata, processes/augments the images themselves, and stores the data back to GCP.

(1) `src/preprocessing/data_loader.py`  - This script downloads and uploads data to and from GCP. There are two key functions:

- a)	"download"
-       Function: Download file from “file_path”
-       Usage: python data_loader.py -d -f “file_path”
-       Optional: -m number (max number of images to download)

- b)	"upload"
-       Function: Upload files in local “file_path” in the form of zipped files
-       Usage: python data_loader.py -u -f “file_path”
-       Optional: -b number (number of images in a zipfile to upload)

(2) `src/preprocessing/process.py`  - This script performs the preprocessing steps on the metadata and images. There are three key functions:

- a)	"update_metadata"
-       Function: Process a metadata file. Drop NA values in label. Look for potential corrupted and non-existing images. If found, drop the corresponding row from the metadata. Save the cleaned metadata in.a folder called “cleaned metadata”.
-       Usage: python process.py -c --inpath “path_to_metadata” -f “path_to_images” –outname “name_of_new_metadata”

- b)	"process"
-       Function: Process the images inside a folder. Read them if possible and resize it. Optionally, users can choose to augment the images. The processed imaged are stored in a folder called “public_image_set” + args.pro_suf.
-       Usage: python data_loader.py -p -f “path_to_images” -s image_dimension
-       Optional: -a (to do augmentation). -- pro_suf “suffix_to_name_the_folder”

- c)	"augment"
-       Function: Augment the images while processing. When an image is processed, it can also be augmented to create several augmented images. The number of augmented images per image processed can be declared by args.augment_num or -n number. The processed imaged are stored in a folder called “public_image_set” + args.aug_suf.
-       Usage: python data_loader.py -p -f “path_to_images” -s image_dimension -a
-       Optional: -- aug_suf “suffix_to_name_the_folder”. -n number.

(3) `src/preprocessing/requirements.txt` - We used following packages to help us preprocess here - `Numpy`, `opencv-python-headless`, `Pillow`, `albumentations`, `google-cloud-storage`, `pandas`  

(4) `src/preprocessing/Pipefile` and `src/preprocessing/Pipefile.lock` are used to manage project dependencies and their versions. They are commonly associated with the package manager Pipenv

(5) `src/preprocessing/Dockerfile` - This dockerfile starts with  `python:3.9-slim-bookworm`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - 
make sure **ac215-tbd-1.json** is downloaded into src/preprocessing/secrets/ to enable GCP authentication
```
cd src/preprocessing/
docker build -t tbd1-preprocess -f Dockerfile .
docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app tbd1-preprocess

# if running the container for the first time, you might need to run:
pipenv install -r requirements.txt
```

*Data Versioning Container*

- This container sets up data versioning for data in this project.

(1) `src/data_versioninig/dvc_cli.sh` - This script mounts the GCP bucket, asks the user whether they want to use data versioning, and initializes DVC based on input.
  
(2) `src/data_versioning/cli.py`  -This script downloads data from the bucket

(3) `src/data_versioning/Pipefile` and `src/data_versioning/Pipefile.lock` are used to manage project dependencies and their versions as in the other containers.

(4) `src/data_versioning/Dockerfile` Dockerfile to build the container for data versioning

(5) `src/data_versioning/docker-shell.sh` Run this shell script to build and run the container

To run Dockerfile - 
make sure ac215-tbd-1.json is downloaded into src//secrets/ to enable GCP authentication
```
cd src/data_versioning/
sh docker-shell.sh

# initialize dvc tracking
sh dvc_cli.sh
dvc add data_file/folder
dvc push
```

*TFRecords Container*

- This container converts input data into TFRecords files.

(1) `src/tfrecords/tfrecords.py` - This script is an exerpt from model training script with incorporation of TFRecords instead of TFData.

 
*Model Training Container*

- This container contains all our training scripts and modeling components. 
- It currently takes in a `.json` file that has the path to the cleaned metadata, image directory, and several model architectural and training hyperparameters.
- The `multimodal_binary_training.py` script will perform several model training runs given different `layer_sizes` in `.json` file and create multiple W&B runs. The model information and performance metrics are all stored in W&B
- The resulting model checkpoints of each training runs are stored as artifacts in W&B and can be easily reloaded to perform inference. 

(1) `src/models/multimodal_binary_training.py` - This script pulls in cleaned metadata and runs a prefetch-enabled TFData pipeline that resizes and normalizes the images and also turns the text into appropriate BERT inputs (text_input_mask, text_input_type_ids, text_input_word_ids), and fits models with different layer sizes (as seperate W&B runs). Model artifacts and metrics are all stored in respective W&B runs. Single-node-multiGPU training is enabled as the script automatically trains the model on all available GPUs ( 1 GPU in our case given quota limit).

It takes in a configuration `.json` file (here as example `train_cli_example_input.json`) that has the following arguments:

> > --batch_size [int] : the size of batches used to train the model
> > --layer_sizes list[int] : the size of the adjustable dense layer. * this is a list as the script tries all of the layer sizes specified. 
> > --max_epochs [int] : a parameter to define the max number of epochs for model training
> > --train_path [string] : path to training metadata
> > --val_path [string] : path to validation metadata
> > --input_mode [string]: mode of input, current it only support TFData

(2) `src/models/Dockerfile` - This dockerfile starts with  `FROM tensorflow/tensorflow:2.13.0-gpu`. This statement uses the tensorflow-gpu version 2.13.0. as the base image.

(3) `src/models/run_docker.sh` Shell script to run the container

To run Dockerfile:
1. `docker build -t training -f Dockerfile .`
2. `sh run_docker.sh`

