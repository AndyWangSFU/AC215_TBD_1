import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import wandb
import pickle
import time
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process JSON configuration file')
    parser.add_argument('config_file', type=str, help='Path to JSON configuration file')
    args = parser.parse_args()
    return args.config_file

def read_json_config(config_file):
    with open(config_file, 'r') as file:
        config_data = json.load(file)
    return config_data


# Define BERT preprocessing model
bert_model_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1"
bert_preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

def make_bert_preprocessing_model(sentence_feature, seq_length=128):
    """Returns Model mapping string features to BERT inputs."""

    preprocessor = hub.KerasLayer(bert_preprocess_path)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name=sentence_feature)
    model_inputs = preprocessor(text_input)

    return keras.Model(inputs=text_input, outputs=model_inputs)

bert_preprocess_model = make_bert_preprocessing_model("text_1")

# Define the dataframe to dataset function
def dataframe_to_dataset(dataframe):
    columns = ["image_path", "clean_title", "2_way_label"]
    dataframe = dataframe[columns].copy()
    labels = dataframe['2_way_label'].to_list()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    return ds

# Define preprocessing functions
resize = (128, 128)
bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]



def preprocess_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize(image, resize)
    image = image / 255.0
    return image


def preprocess_text(text_1):
    text_1 = tf.convert_to_tensor([text_1])
    output = bert_preprocess_model(text_1)
    output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
    return output


def preprocess_text_and_image(sample):
    image = preprocess_image(sample["image_path"])
    text = preprocess_text(sample["clean_title"])
    return {"image": image, "text": text}






auto = tf.data.AUTOTUNE
def prepare_dataset(dataframe, training=True):
    ds = dataframe_to_dataset(dataframe)
    ds = ds.map(lambda x, y: (preprocess_text_and_image(x), y)).cache()
    ds = ds.batch(batch_size).prefetch(auto)
    return ds







### Project image and text embeddings to the same space

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    
    return projected_embeddings


def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained ResNet50V2 model to be used as the base encoder.
    resnet_v2 = keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in resnet_v2.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    image = keras.Input(shape=(128, 128, 3), name="image")


    # Preprocess the input image.
    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image)

    embeddings = resnet_v2(preprocessed_1)

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(image, outputs, name="restnet_50_vision_encoder")




def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(bert_model_path, name="bert",)
    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(inputs)["pooled_output"]

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="bert_text_encoder")


def create_multimodal_model(num_projection_layers=1, projection_dims=224, dropout_rate=0, 
                     vision_trainable=False, text_trainable=False, attention=False,layer_size=224):
    # Receive the images as inputs.
    image = keras.Input(shape=(128, 128, 3), name="image")

    # Receive the text as inputs.
    bert_input_features = ['input_type_ids', 'input_mask', 'input_word_ids']
    text_inputs = {
        feature: keras.Input(shape=(128, ), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Create the encoders.
    vision_encoder = create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, vision_trainable)
    text_encoder = create_text_encoder(num_projection_layers, projection_dims, dropout_rate, text_trainable)

    # Fetch the embedding projections.
    vision_projections = vision_encoder(image)
    text_projections = text_encoder(text_inputs)

    # Cross-attention.
    if attention:
        query_value_attention_seq = keras.layers.Attention(use_scale=True, dropout=0)(
            [vision_projections, text_projections]
        )

    # Concatenate the projections and pass through the classification layer.
    maximum = keras.layers.Maximum()([vision_projections, text_projections])
    if attention:
        maximum = keras.layers.Maximum()([maximum, query_value_attention_seq])
    head_layer_1 = keras.layers.Dense(layer_size, activation="relu")(maximum) 
    head_layer_2 = keras.layers.Dense(32, activation="relu")(head_layer_1)
    outputs = keras.layers.Dense(1, activation="sigmoid",name="clf_output")(head_layer_2)

    # Create an additional output for the concatenated layer.
    max_layer_output = keras.layers.Lambda(lambda x: x, name="maximum_layer")(head_layer_1)

    # Define the model with both main and concatenated layer outputs.
    model = keras.Model([image, text_inputs], [outputs, max_layer_output])

    return model


if __name__ == "__main__":
        
    """Get the config data from the JSON file"""
    config_file_path = parse_arguments()
    config_data = read_json_config(config_file_path)

    batch_size = config_data.get('batch_size', None)
    layer_sizes = config_data.get('layer_size', None)
    learning_rate = config_data.get('learning_rate', None)
    num_gpus = config_data.get('num_gpus', None)
    max_epochs = config_data.get('max_epochs', None)
    train_path = config_data.get('train_path', None)
    val_path = config_data.get('val_path', None)
    input_mode = config_data.get("input_mode",None)

    if input_mode == "tf_data":
        """ Load metadata which has labels, cleaned_titles, and image IDs """
        
        train_df = pd.read_csv(train_path, sep="\t")
        val_df = pd.read_csv(val_path, sep="\t")


        """Turn metadata into tf.data.Dataset objects"""
        train_ds = prepare_dataset(train_df)
        val_ds = prepare_dataset(val_df, training=False)

    # Automatically use all avaliable GPUs
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))


    for layer_size in layer_sizes:
        with strategy.scope():
            multimodal_model = create_multimodal_model(num_projection_layers=1, projection_dims=layer_size, dropout_rate=0, 
                            vision_trainable=False, text_trainable=False, attention=False,layer_size = layer_size)

            losses = {"clf_output": 'binary_crossentropy'}
            multimodal_model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=losses,
                metrics='accuracy'
            )

        wandb.init(
            project = 'model_training',
            config = {
            "learning_rate": learning_rate,
            "epochs": max_epochs,
            "batch_size": batch_size,
            "model_name": 'Binary_Classifier (BERT + RESNET-50) Layer Size {}'.format(layer_size),
            "layer_size":layer_size ,
            "metrics": "accuracy",
            },
        )

        # Specify the loss function for the main output (outputs).



        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            filepath="binary_clf_{}.h5".format(layer_size),
            save_best_only=True,
            save_weights_only=False,
            monitor='val_clf_output_accuracy',
            mode='max',
            verbose=1,
            save_freq='epoch')

        early_stopping_callback = EarlyStopping(
            monitor='val_clf_output_accuracy',
            mode='max',
            patience=2,
            verbose=1,
            restore_best_weights=True
        )

        start_time = time.time()
        # Train the model with the custom learning rate
        history = multimodal_model.fit(
            train_ds,
            epochs=max_epochs,
            validation_data=val_ds,
            callbacks=[checkpoint_callback, early_stopping_callback,WandbCallback()])

        execution_time = (time.time() - start_time)/60.0
        print("Training execution time (mins)",execution_time)

        # Update W&B
        wandb.config.update({"execution_time": execution_time})
        # Close the W&B run
        wandb.run.finish()




