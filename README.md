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

<!-- <p align="center">
  <img width="460" height="300" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/53b18850-f0d1-47a4-8f78-71724c18faff">
</p>

<p style="text-align: center;">Figure 1: Project data pipeline</p> -->

****

### Milestone4

<ins>**Objectives for Milestone4**</ins>

We address each of the objectives for Milestone 4 in the following ways:

*1. Distillation/Quantization/Compression*

To enable enable our deployment in a resource-constrained environment, we utilize a compression technique that (kyle continues)



<!-- <img width="1264" alt="Screenshot 2023-10-04 at 7 19 39 PM" src="https://github.com/AndyWangSFU/AC215_TBD_1/assets/112672824/5d9e256a-c711-430f-b6da-b8f8ed4377c3"> -->

<!-- Figure 2:    -->


*2. Vertex AI Pipelines (Kubeflow) and Cloud Functions Integration*

**Preprocess Container**
Please refer to Milestone3 regarding the rationale and code structure of the preprocessing container.


**Model Compression Container**


Below, we can see our Weights & Biases page while training our compression model:


**Model Training Container**





**Workflow Container**


The workflow container follows a similar structure as the class demo. (Reference: https://github.com/dlops-io/ml-workflow#mushroom-app-ml-workflow-management). To activate the docker environment, run `sh docker-shell.sh`, and it will activate the environment with all necessary environment variables (GCP project name, service account, data bucket, and region). The GCP authentication steps are done in `docker-entrypoint.sh`. Together, this workflow container ships our code off to Vertex AI Pipelines and run each tasks in sequence.

After the docker is running, we can call `python cli.py` to call the pipeline. There are 3 containers callable from `cli.py --- preprocessing` (to download and process the images), `models` (to train the models), and `model_compression` (to do model compression). These individual containers can be called as a single-element pipeline using `-d (download), -p(process), -t (model training), and -c (model compression)`. However, this is mainly for testing purposes. In real usage, use `python cli.py – w` to run the whole pipeline. The resulting pipeline can be visualized below on Vertex AI. 

![Picture1](https://github.com/AndyWangSFU/AC215_TBD_1/assets/48002686/629ddff3-9a8d-47f3-a4a9-f8c17aa3a678)


*Figure 1: Pipeline run analysis*


Behind the scene, all these containers’ images are pushed to Docker hub. The links to the images are specified in the cli.py file (see below). We used these images and passed preset arguments to call the corresponding functions. 

- `DATA_PREPROCESS_IMAGE` = "kirinlfc/fakenews-detector-data-preprocessor"
- `MODEL_COMPRESSION_IMAGE` = "ksiyang/multimodal_fakenews_detector_model_compression"
- `MODEL_TRAIN_IMAGE` = "ksiyang/model_training"

In case anyone wants to test the connection with Vertex AI, call python cli.py – s1. This command will start a sample pipeline that squares and adds numbers. (The same as the in-class demo.)



<ins>**Code Structure**</ins>



*Model Compression Container*

- This container incorporates the model compression technique. 

(1) `src/data_versioninig/dvc_cli.sh` 


*Model Training Container*

- This container 

(1) `src/data_versioninig/dvc_cli.sh` 


*Workflow Container*

- This container can build the whole sequence pipeline in Vertex AI to achieve serverless training.

(1) `src/workflow/cli.py` - this script creates the pipeline and execute different parameter arguemnts

(2) `src/workflow/Pipfile` - this file contains the Python packages, sources, and requirments to run whe workflow

(3) `src/workflow/compress.yaml` - this file defines the pipeline structure

(4) `src/workflow/Dockerfile` - the file sets the docker environment



```
cd src/preprocessing/

```




To run Dockerfile:
1. `docker build -t training -f Dockerfile .`
2. `sh run_docker.sh`

