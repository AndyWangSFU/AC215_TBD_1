from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
# from api.tracker import TrackerService
import pandas as pd
import os
from fastapi import File, Form, UploadFile
from tempfile import TemporaryDirectory
from api import model
import random
import string
import asyncio
from google.cloud import storage
# Initialize Tracker Service
# tracker_service = TrackerService()

GCP_PROJECT = "AC215"
BUCKET_NAME = "fakenew_classifier_data_bucket"

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

### A function to generate a length 6 sequence with letters and numbers
def generate_sequence(length=6):
    characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    sequence = ''.join(random.choice(characters) for _ in range(length))
    return sequence


# @app.on_event("startup")
# async def startup():
#     print("Startup tasks")
#     # Start the tracker service
#     asyncio.create_task(tracker_service.track())


async def upload(object_path, destination_blob_name):
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)

    # Destination path in GCS
    print(f"Uploading to {destination_blob_name}")

    # Upload the model file directly to GCS
    object_blob = bucket.blob(destination_blob_name)
    object_blob.upload_from_filename(object_path)
    print(f"Uploaded: {destination_blob_name}")

# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}


@app.get("/test")
async def test():
    print("test")
    return {"message": "dsfsddf"}


@app.post("/predict")
async def predict(
    image: bytes = File(...),  # Expect an image file
    text: str = Form(...)  # Expect a text file
):

    

    # Save the image
    
    image_dir = "./images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    random_sequence = generate_sequence()
    # print(random_sequence)    

    
    image_path = os.path.join(image_dir, f"{random_sequence}.jpg")
    with open(image_path, "wb") as output:
        output.write(image)

    text_dir = "./text" 
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    # print(random_sequence)    

    text_path = os.path.join(text_dir, f"{random_sequence}.txt")

    with open(text_path, "w") as output:
        output.write(text)

    print(image_path)
    preprocessed_data = model.preprocess_data_inference(image_path, text)
    
    # Make prediction
    prediction_results = model.make_predictions(preprocessed_data)
    asyncio.create_task(upload(image_path,"post_deployment_data/images"))
    asyncio.create_task(upload(text_path,"post_deployment_data/text"))

    if prediction_results[0][0] > 0.6:
        fake_likelihood = "High"
    else:
        fake_likelihood = "Low"
    return {'Fake Probability': str(prediction_results[0][0]),
            'Fake Likelihood': fake_likelihood}
