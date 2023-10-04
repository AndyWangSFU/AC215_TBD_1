import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from tensorflow import keras

bert_model_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1"
bert_preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


def make_bert_preprocessing_model(sentence_feature, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_feature: A list with the names of string-valued features.
    seq_length: An integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """
    preprocessor = hub.KerasLayer(bert_preprocess_path)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name=sentence_feature)
    model_inputs = preprocessor(text_input)

    return keras.Model(inputs=text_input, outputs=model_inputs)

def preprocess_image(image_path, resize=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize(image, resize)
    image = image / 255.0
    return image.numpy().tobytes()

def preprocess_text(text_1, bert_preprocess_model):
    text_1 = tf.convert_to_tensor([text_1])
    output = bert_preprocess_model(text_1)
    return output


def sample_example(sample, bert_preprocess_model, resize):
    image = preprocess_image(sample["image_path"], resize)
    text_output = preprocess_text(sample["clean_title"], bert_preprocess_model)
    text_input_mask = tf.squeeze(text_output["input_mask"])
    text_input_type_ids = tf.squeeze(text_output["input_type_ids"])
    text_input_word_ids = tf.squeeze(text_output["input_word_ids"])
    label = sample["2_way_label"]
    feature_dict = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'text_input_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=text_input_mask)),
        'text_input_type_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=text_input_type_ids)),
        'text_input_word_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=text_input_word_ids)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return tf_example



def prepare_dataset(df, bert_preprocess_model, resize, folder_path, batch_size):
    for i, row in df.iterrows():
        tf_example = sample_example(row, bert_preprocess_model, resize)
        if i > 0 and i % batch_size == 0:
            path = os.path.join(folder_path, f"tfrecords_{int(i/batch_size)}.tfrecord")
            print(f"Writing to {path}")
            # Write the serialized example to a TFRecords file.
            with tf.io.TFRecordWriter(path) as writer:
                # Filter the subset of data to write to tfrecord file
                writer.write(tf_example.SerializeToString())


def main(args):
    batch_size = args.batch_size
    file_path = args.dataset_path
    dest_path = args.dest_path
    # Load metadata, labels, cleaned_titles, and image IDs
    df = pd.read_csv(
        file_path, sep="\t", 
        # some columns have issues - reading only the ones we need
        usecols=['id', 'clean_title', 'image_path', '2_way_label']
    )
    # Remove rows with NA in 'id' and 'clean_title' columns for train_df
    df.dropna(subset=['id', 'clean_title', 'image_path'], inplace=True)
    df = df[['id', 'clean_title', 'image_path']]
    df['2_way_label'] = df['2_way_label'].fillna(1).apply(lambda x: 1 if x == 0 else 0)

    bert_preprocess_model = make_bert_preprocessing_model("text_1")
    resize = (128, 128)
    prepare_dataset(df, bert_preprocess_model, resize, batch_size, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Prepare dataset for training/validation/testing. '
            'Image and text data are matched by ID. '
            'The image data is resized and converted to bytes. '
            'The text data is tokenized using a BERT model. '
            'The combined data is then saved as TFRecord files.'
        ))
    
    parser.add_argument("-b", "--batch_size", type=int, default=10000,
                        help="Number of images to shard into each TFRecord file")
    parser.add_argument("-f", "--dataset_path", type=str,
                        default="./data/cleaned_metadata-multimodal_train_cleaned.tsv",
                        help="Path to data tsv")
    parser.add_argument("-d", "--dest_path", type=str, default="data/tfrecords",
                        help="Path to save TFRecord files")

    args = parser.parse_args()

    main(args)