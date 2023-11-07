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

# Initialize Tracker Service
# tracker_service = TrackerService()

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
    print("predict image:", len(image), type(image))
    print("predict text:", len(text), type(text))
    

    # Save the image
    
    image_dir = "./images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    random_sequence = generate_sequence()
    # print(random_sequence)    
    print(image)
    print(text)
    
    image_path = os.path.join(image_dir, f"{random_sequence}.jpg")
    with open(image_path, "wb") as output:
        output.write(image)

    text_dir = "./text" 
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    # print(random_sequence)    

    text_path = os.path.join(text_dir, f"{random_sequence}.txt")
    print(os.getcwd())
    print(image_path)
    print(text_path)
    with open(text_path, "w") as output:
        output.write(text)


    preprocessed_data = model.preprocess_data_inference(image_path, text)

    # Make prediction
    prediction_results = model.make_predictions([preprocessed_data['image'], preprocessed_data['text']])

    print(prediction_results)
    return prediction_results
