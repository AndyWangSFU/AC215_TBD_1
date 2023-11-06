import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from google.cloud import storage
import tensorflow_hub as hub
import tensorflow_text as text
import wandb    


GCP_PROJECT = "ac215-398714"
GCS_MODELS_BUCKET_NAME = "fakenew_classifier_data_bucket"
BEST_MODEL = "models/compressed"
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/mnt/c/Users/kylek/OneDrive - Harvard University/AC215/project/AC215_TBD_1/secrets/ac215-tbd-1.json'
#os.environ["WANDB_KEY"] = "/mnt/c/Users/kylek/OneDrive - Harvard University/AC215/project/AC215_TBD_1/secrets/wandb_key.txt"


with open(os.environ["WANDB_KEY"], 'r') as f:
    wandb_key = f.read().strip() 

AUTOTUNE = tf.data.experimental.AUTOTUNE



def load_prediction_model(model_path=None):

    global prediction_model
    print("Loading Model...")
    """
    # if loading from gcs bucket
    download(model_path, 1)
    prediction_model = tf.keras.models.load_model(f"{model_path}/raw_model",
                                                  custom_objects={'KerasLayer': hub.KerasLayer})
    """
    #loading from wandb
    wandb.login(key=wandb_key)
    run = wandb.init()
    artifact = run.use_artifact('tbd_2/AC215_TBD_1-src_model_deployment/binary_clf_model:v0', type='model')
    artifact_dir = artifact.download()
    wandb.finish()
    prediction_model = tf.keras.models.load_model("artifacts/binary_clf_model:v0/raw_model",
                                                custom_objects={'KerasLayer': hub.KerasLayer})
    print("Model Loaded")


def preprocess_data_inference(image_path, text_input):

    
    def make_bert_preprocessing_model(sentence_feature, seq_length=128):
        """Returns Model mapping string features to BERT inputs."""
        
        bert_preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        preprocessor = hub.KerasLayer(bert_preprocess_path)
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name=sentence_feature)
        model_inputs = preprocessor(text_input)

        return keras.Model(inputs=text_input, outputs=model_inputs)
    
    
    bert_preprocess_model = make_bert_preprocessing_model("text_1")

    def preprocess_image(image_path):
        # Check the file extension
        if image_path.lower().endswith(('.png', '.jpeg', '.jpg')):
            # Read the image file
            image = tf.io.read_file(image_path)
            
            # Decode the image based on the file type
            if image_path.lower().endswith('.png'):
                image = tf.image.decode_png(image, channels=3)
            else:  # Assume JPEG if not PNG
                image = tf.image.decode_jpeg(image, channels=3)
            
            # Resize and normalize the image
            image = tf.image.resize(image, (128, 128))
            image = image / 255.0
            
            return image
        else:
            # Handle unsupported file types or other errors
            raise ValueError("Unsupported image file format: " + image_path)

    def preprocess_text(text_input):

        bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]
        text_input = tf.convert_to_tensor([text_input])
        output = bert_preprocess_model(text_input)
        output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
        return output

    # Preprocess image and text
    image = preprocess_image(image_path)
    text = preprocess_text(text_input)

    return_dict = {
        'image': tf.expand_dims(image,axis=0),
        'text' : {'input_type_ids': tf.expand_dims(text['input_type_ids'],axis=0),
        'input_mask': tf.expand_dims(text['input_mask'],axis=0),
        'input_word_ids': tf.expand_dims(text["input_word_ids"],axis=0)}}
    return return_dict  


def make_predictions(preprocessed_data):
    # Make predictions using the loaded model
    predictions = prediction_model.predict([preprocessed_data['image'], preprocessed_data['text']])
    fake_probability = 1 - predictions[0]
    return fake_probability

"""
How you would test a single prediction
"""
load_prediction_model()
#preprocessed_data = preprocess_data_inference('./test.jpg', "Biden promotes 'DARK BRANDON' MUG IN NEW AD")
#make_predictions([preprocessed_data['image'], preprocessed_data['text']])
