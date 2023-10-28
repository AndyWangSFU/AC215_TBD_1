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
            │   ├── docker-entrypoint.sh 
            │   ├── docker-shell.sh 
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

<!-- <p align="center">
  <img width="460" height="300" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/53b18850-f0d1-47a4-8f78-71724c18faff">
</p>

<p style="text-align: center;">Figure 1: Project data pipeline</p> -->

****

### Milestone4

<ins>**Objectives for Milestone4**</ins>

We address each of the objectives for Milestone 4 in the following ways:

*1. Compression (Quantization)*

For model compression, we compared two different quantization methods: 
- float16 quanitzation for weights. This method decreased model size by 50% but did not lead to any decrease in out-of-sample model performance.
 
- int16 quantization for activations and int8 quantization activations for weights. This method decreased model size by 75% but also decreased out-of-sample AUC by ~ 5%.

The W&B screenshot below displays the model_size, quantization_method, and out-of-sample inference performance of these three quantization methods:

![WhatsApp Image 2023-10-27 at 8 19 28 PM](https://github.com/AndyWangSFU/AC215_TBD_1/assets/48002686/ca3ff2d6-549f-4082-9ff8-df07cffd4584)


<!-- <img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/5d9e256a-c711-430f-b6da-b8f8ed4377c3"> -->

<!-- Figure 2:    -->


*2. Vertex AI Pipelines (Kubeflow) and Cloud Functions Integration*

**Preprocess Container**
Please refer to Milestone3 regarding the rationale and code structure of the preprocessing container.
(1) `src/preprocessing/data_loader.py` - This script downloads and uploads data to and from GCP. 

(2) `src/preprocessing/process.py` - This script performs the preprocessing steps on the metadata and images. 

(3) `src/preprocessing/requirements.txt` - We used the following packages to help us preprocess here - Numpy, opencv-python-headless, Pillow, albumentations, google-cloud-storage, pandas

(4) `src/preprocessing/Pipefile and src/preprocessing/Pipefile.lock` are used to manage project dependencies and their versions. They are commonly associated with the package manager Pipenv

(5) `src/preprocessing/Dockerfile` - This dockerfile starts with python:3.9-slim-bookworm. This attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - make sure `ac215-tbd-1.json` is downloaded into `src/preprocessing/secrets/` to enable GCP authentication.

```
cd src/preprocessing/
docker build -t tbd1-preprocess -f Dockerfile .
docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app tbd1-preprocess

# if running the container for the first time, you might need to run:
pipenv install -r requirements.txt
```


**Model Compression Container**


Below, we can see our Weights & Biases page while training our compression model:


**Model Training Container**





**Workflow Container**


The workflow container follows a similar structure as the class demo. (Reference: https://github.com/dlops-io/ml-workflow#mushroom-app-ml-workflow-management). To activate the docker environment, run `sh docker-shell.sh`, and it will activate the environment with all necessary environment variables (GCP project name, service account, data bucket, and region). The GCP authentication steps are done in `docker-entrypoint.sh`. Together, this workflow container ships our code off to Vertex AI Pipelines and run each tasks in sequence.

After the docker is running, we can call `python cli.py` to call the pipeline. There are 3 containers callable from `cli.py --- preprocessing` (to download and process the images), `models` (to train the models), and `model_compression` (to do model compression). These individual containers can be called as a single-element pipeline using `-d (download), -p(process), -t (model training), and -c (model compression)`. However, this is mainly for testing purposes. In real usage, use `python cli.py – w` to run the whole pipeline. The resulting pipeline can be visualized below on Vertex AI. 

![Picture1](https://github.com/AndyWangSFU/AC215_TBD_1/assets/48002686/629ddff3-9a8d-47f3-a4a9-f8c17aa3a678)


*Figure 1: Pipeline run analysis*


Behind the scenes, all these containers’ images are pushed to Docker hub. The links to the images are specified in the cli.py file (see below). We used these images and passed preset arguments to call the corresponding functions. 

- `DATA_PREPROCESS_IMAGE` = "kirinlfc/fakenews-detector-data-preprocessor"
- `MODEL_COMPRESSION_IMAGE` = "ksiyang/multimodal_fakenews_detector_model_compression"
- `MODEL_TRAIN_IMAGE` = "ksiyang/model_training"

In case anyone wants to test the connection with Vertex AI, call python cli.py – s1. This command will start a sample pipeline that squares and adds numbers. (The same as the in-class demo.)



<ins>**Code Structure**</ins>



*Model Compression Container*

- This container incorporates the model compression technique. 

(1) `src/model_compression/downsize.py`: This script essentially takes in the quantization method and the path GCS bucket path of the raw model and first turns the model into a TFLite model and then performs the specified quantization (float 16 or int8-int16 quantization).

(2) `src/model_compression/Dockerfile`: The callable container to run the model_compression component either locally or as part of the VertexAI pipeline workflow.

(3) `src/model_compression/docker-entrypoint.sh`: Specifies entry point to the Docker container. 

(4) `src/model_compression/run_docker.sh`: Shell script to call to run the container locally.


*Model Training Container*

- This container contains the training scripts and modeling components, which utilizes data from our GCP bucket. After performing the whole training process, the trained model will be saved back to GCP bucket. Please note that we did not end up completely implementing the serverless training via vertex AI (as suggested tutorial here: https://github.com/dlops-io/model-training/tree/main), but we left the structure here. However, we did make the training container callable and ran the training container like the other containers using the vertex AI pipeline in similar manners.

(1) `src/training/multimodal_binary_training.py` - this script pulls in cleaned metadata and runs a prefetch-enabled TFData pipeline that resizes and normalizes the images and also turns the text into appropriate BERT inputs (text_input_mask, text_input_type_ids, text_input_word_ids), and fits models with specified hyperparameters. Model artifacts and metrics are all stored in W&B. The resulting run model is saved in GCP bucket.

(2) `src/training/package` - this is our attempted model training Python code package, a part of the container structure, but was not ultimately used.




*Workflow Container*

- This container can build the whole sequence pipeline in Vertex AI to achieve serverless training.

(1) `src/workflow/cli.py` - this script creates the VertexAI pipeline and calls different containers or all containers sequentially as a pipeline.

(2) `src/workflow/Pipfile` - this file contains the Python packages, sources, and requirements to run the workflow

(3) `src/workflow/compress.yaml` - this file defines the pipeline structure

(4) `src/workflow/Dockerfile` - the file sets the docker environment

<img width="1204" alt="Screenshot 2023-10-27 at 7 52 46 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/48002686/7edcd404-a8ff-4ff7-9e76-0aab3a4b6359">




To run Dockerfile:
1. `docker build -t training -f Dockerfile .`
2. `sh run_docker.sh`

