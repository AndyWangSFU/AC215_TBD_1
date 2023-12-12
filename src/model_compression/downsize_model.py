import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
from model_helpers import *
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import pandas as pd
import pathlib
import wandb
import pickle
import time
import argparse
import json
import argparse
from google.cloud import storage
import pathlib
import zipfile




print("finished imports")
GCP_PROJECT = "AC215"
BUCKET_NAME = "fakenew_classifier_data_bucket"
# set BUCKET_NAME equal environment variable for bucket name

# Initiate Storage client

def download(filepath, max_num):
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # create a local folder for the downloaded file if not exist
    os.makedirs(filepath, exist_ok=True)

    # Find all content in a bucket
    blobs = bucket.list_blobs(prefix=filepath)
    for blob in blobs:
        if max_num == 0:
            break
        if not blob.name.endswith("/"):
            try:
            # Download the blob to a local file with the same name
                blob.download_to_filename(blob.name)
                print(f"Downloaded: {blob.name}")
            except Exception as e:
                print(f"Error downloading {blob.name}: {str(e)}")
            max_num -= 1


def upload(model_path, destination_blob_name):
    # Initiate Storage client
    storage_client = storage.Client(project=GCP_PROJECT)

    # Get reference to bucket
    bucket = storage_client.bucket(BUCKET_NAME)

    # Destination path in GCS
    print(f"Uploading to {destination_blob_name}")

    # Upload the model file directly to GCS
    model_blob = bucket.blob(destination_blob_name)
    model_blob.upload_from_filename(model_path)
    print(f"Uploaded: {destination_blob_name}")




def convert_to_tflite(raw_model_path, quantization_type='16x8'):
    # Load the raw model from the provided path
    raw_model = tf.keras.models.load_model(raw_model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Create a TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(raw_model)
    
    # Set quantization parameters based on the provided quantization_type
    if quantization_type == '16x8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int16]
    elif quantization_type == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        raise ValueError("Unsupported quantization type. Choose '16x8' or 'float16'.")
    
    # Convert the model to TFLite format
    tflite_model = converter.convert()
    
    # Specify the TFLite model file path
    tflite_models_dir = pathlib.Path("./binary_tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    
    if quantization_type == '16x8':
        tflite_model_file = tflite_models_dir / "binary_clf_16x8.tflite"
    elif quantization_type == 'float16':
        tflite_model_file = tflite_models_dir / "binary_clf_f16.tflite"
    
    # Write the TFLite model to a file
    tflite_model_file.write_bytes(tflite_model)
  
    return str(tflite_model_file)



def perform_inference(interpreter, val_ds, batch_size):
    output_index = interpreter.get_output_details()[0]["index"]
    input_details = interpreter.get_input_details()
    input_indices = [detail["index"] for detail in input_details]
    y_preds = []
 
    input_indices = [detail["index"] for detail in input_details]
    for x_batch, y_batch in val_ds:
        batch_size = len(x_batch['text']['input_word_ids'])
        for i in range(batch_size):
            input_data = {
                input_indices[0]: x_batch['text']['input_word_ids'][i:i+1],
                input_indices[1]: x_batch['text']['input_type_ids'][i:i+1],
                input_indices[2]: x_batch['image'][i:i+1],
                input_indices[3]: x_batch['text']['input_mask'][i:i+1]
            }
            
            for index, data in input_data.items():
                interpreter.set_tensor(index, data)
                
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_index)
            y_preds.append(output[0][0])
    
    return y_preds

def main():
    "entering main"
    parser = argparse.ArgumentParser(description="Model Compression and Inference with W&B Logging")

    # Define command-line arguments
    parser.add_argument("--raw_model_path", type=str, required=True, help="Path to the raw model")
    parser.add_argument("--quantization_type", type=str, default="float16", help="Quantization type")
    parser.add_argument("--hidden_layer_size", type=int, default=224, help="Hidden layer size")
    parser.add_argument("--learn_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--metadata_path", type=str, default="new_data/new_metadata", help="Path to metadata")
    parser.add_argument("--images_path", type=str, default="new_data/new_images", help="Path to images")
    parser.add_argument("--new_data_size", type=int, default=41, help="New data size")
    parser.add_argument("-w", "--wandb", type=str, default="", help="WANDB Key")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Create the config dictionary
    config = {
        "raw_model_path": args.raw_model_path,
        "quantization_type": args.quantization_type,
        "hidden_layer_size": args.hidden_layer_size,
        "learn_rate": args.learn_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "metadata_path": args.metadata_path,
        "images_path": args.images_path,
        "new_data_size": args.new_data_size,
    }
    # login into W&B

   

    wandb.login(key=args.wandb)
    

    download(config['metadata_path'], 2)
    download(config['images_path'], config['new_data_size'])
    download(config['raw_model_path'], 1)
 
    # load in the validation data
    val_df = pd.read_csv(f"{config['metadata_path']}/new_val_metadata", sep="\t")
    batch_size = config["batch_size"]
    """Turn metadata into tf.data.Dataset objects"""
    val_ds = prepare_dataset(val_df,batch_size=batch_size, training=False)
    
    wandb.init(project="model_training")

    # Convert the model to TFLite
    tflite_model_path = convert_to_tflite(f"{config['raw_model_path']}/raw_model", quantization_type=config['quantization_type'])

    # upload the model to GCS
    upload(tflite_model_path, "models/compressed")
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Perform inference with the TFLite model
    y_preds = perform_inference(interpreter, val_ds, batch_size)

    # Calculate and log metrics to WandB
    y_true = val_df['2_way_label'].values
    auc = roc_auc_score(y_true, y_preds)
    recall = recall_score(y_true, [1 if p >= 0.5 else 0 for p in y_preds])
    confusion = confusion_matrix(y_true, [1 if p >= 0.5 else 0 for p in y_preds])
    specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])

    wandb.log({"AUC": auc, "Recall": recall, "Specificity": specificity})
    wandb.run.finish()


# run main
if __name__ == "__main__":
    main()

