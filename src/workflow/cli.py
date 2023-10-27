"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

# from model import model_training, model_deploy

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
#GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

DATA_PREPROCESS_IMAGE = "kirinlfc/fakenews-detector-data-preprocessor"
MODEL_COMPRESSION_IMAGE = "ksiyang/multimodal_fakenews_detector_model_compression"
MODEL_TRAIN_IMAGE = "ksiyang/model_training"

def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def main(args=None):
    print("CLI Arguments:", args)

    if args.download:
        # Define a Container Component
        @dsl.container_component
        def download():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESS_IMAGE,
                command=[],
                args=[
                    "data_loader.py",
                    "-d ", # download image
                    "-m 10", # 10 images
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def download_pipeline():
            download()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            download_pipeline, package_path="download.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "fakenew-detector-data-download-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="download.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.compress:
        # Define a Container Component
        @dsl.container_component
        def compress():
            container_spec = dsl.ContainerSpec(
                image=MODEL_COMPRESSION_IMAGE,
                command=[],
                args=[
                    "downsize_model.py",
                    "--raw_model_path models/raw_models  --quantization_type float16  --hidden_layer_size 224  --learn_rate 0.001  --epochs 20  --batch_size 16  --metadata_path new_data/new_metadata  --images_path new_data/new_images   --new_data_size 41   --wandb 66f5d9722b4bef56a139f86a08c40f3929f97470"
                ],
            )
            return container_spec

        # # Convert the above component into a custom training job
        # compress_job = create_custom_training_job_from_component(
        #     compress,
        #     machine_type = 'e2-standard-8',
        # )

        # Define a Pipeline
        @dsl.pipeline
        def compress_pipeline():
            compress()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            compress_pipeline, package_path="compress.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "fakenew-detector-model-compression-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="compress.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.train:
        # Define a Container Component
        @dsl.container_component
        def train():
            container_spec = dsl.ContainerSpec(
                image=MODEL_TRAIN_IMAGE,
                command=[],
                args=[
                    "multimodal_binary_training.py",
                    "--hidden_layer_size 224   --learn_rate 0.001   --epochs 20   --batch_size 16   --metadata_path new_data/new_metadata   --images_path new_data/new_images   --new_data_size 41   --wandb 66f5d9722b4bef56a139f86a08c40f3929f97470",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def train_pipeline():
            train()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            train_pipeline, package_path="train.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "fakenew-detector-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="train.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)    

    if args.pipeline:
        # Define a Container Component
        @dsl.container_component
        def download():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESS_IMAGE,
                command=[],
                args=[
                    "data_loader.py",
                    "-d ", # download image
                    "-m 10", # 10 images
                ],
            )
            return container_spec

        @dsl.container_component
        def process():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESS_IMAGE,
                command=[],
                args=[
                    "process.py",
                    "-p ", # process image
                ],
            )
            return container_spec
        
        # Define a Container Component
        @dsl.container_component
        def train():
            container_spec = dsl.ContainerSpec(
                image=MODEL_TRAIN_IMAGE,
                command=[],
                args=[
                    "multimodal_binary_training.py",
                    "--hidden_layer_size 224   --learn_rate 0.001   --epochs 20   --batch_size 16   --metadata_path new_data/new_metadata   --images_path new_data/new_images   --new_data_size 41   --wandb 66f5d9722b4bef56a139f86a08c40f3929f97470",
                    ],
            )
            return container_spec     
           
        # Define a Container Component
        @dsl.container_component
        def compress():
            container_spec = dsl.ContainerSpec(
                image=MODEL_COMPRESSION_IMAGE,
                command=[],
                args=[
                    "downsize_model.py",
                    "--raw_model_path models/raw_models  --quantization_type float16  --hidden_layer_size 224  --learn_rate 0.001  --epochs 20  --batch_size 16  --metadata_path new_data/new_metadata  --images_path new_data/new_images   --new_data_size 41   --wandb 66f5d9722b4bef56a139f86a08c40f3929f97470"
                ],
            )
            return container_spec
        
        # Define a Pipeline
        @dsl.pipeline
        def ml_pipeline():
            # download
            download_task = download().set_display_name("data download")
            # process
            process_task = process().set_display_name("data process").after(download_task)
            # model training
            train_task = train().set_display_name("model training").after(process_task)
            # model compress
            compression_task = compress().set_display_name("model compression").after(train_task)         

        # Build yaml file for pipeline
        compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "fakenews-detector-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.sample1:
        print("Sample Pipeline 1")

        # Define Component
        @dsl.component
        def square(x: float) -> float:
            return x**2

        # Define Component
        @dsl.component
        def add(x: float, y: float) -> float:
            return x + y 

        # Define Component
        @dsl.component
        def square_root(x: float) -> float:
            return x**0.5

        # Define a Pipeline
        @dsl.pipeline
        def sample_pipeline(a: float = 3.0, b: float = 4.0) -> float:
            a_sq_task = square(x=a)
            b_sq_task = square(x=b)
            sum_task = add(x=a_sq_task.output, y=b_sq_task.output)
            return square_root(x=sum_task.output).output

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            sample_pipeline, package_path="sample-pipeline1.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "sample-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="sample-pipeline1.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Run just the download",
    )
    parser.add_argument(
        "-p",
        "--process",
        action="store_true",
        help="Run just the data processor",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the models",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="Run the model compressor",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Fakenews Detector App Pipeline",
    )
    parser.add_argument(
        "-s1",
        "--sample1",
        action="store_true",
        help="Sample Pipeline 1",
    )

    args = parser.parse_args()

    main(args)
