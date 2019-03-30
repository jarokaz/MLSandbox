
from absl import flags
from absl import app

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, add

tf.enable_eager_execution()


IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

def toy_resnet_model():
    inputs = Input(shape=IMAGE_SHAPE, name='image')
    x = Conv2D(32, 3, activation='relu')(inputs)
    x = Conv2D(64, 3, activation='relu')(x)
    block_1_output = MaxPooling2D(3)(x)
    
    x = Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = add([x, block_1_output])
    
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = add([x, block_2_output])
    
    x = Conv2D(64, 3, activation='relu')(block_3_output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='toy_resnet')
    
    return model


def prepare_datasets():
    def _parse_record(example_proto):
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64, default_value=0)
        }
        
        parsed_features = tf.parse_single_example(example_proto, features)
        image = parsed_features['image']
        label = parsed_features['label']
        
        image = tf.image.decode_png(image, channels=3)
        #image = tf.cast(image, tf.float32)
        #image = image / 255
        #image = tf.reshape(image, IMAGE_SHAPE)
        
        #label = tf.one_hot(label, NUM_CLASSES)
        
        #return image, label
        return image, label

    training_images = [os.path.join(FLAGS.training_images, file) for file in os.listdir(FLAGS.training_images)]
    validation_images = [os.path.join(FLAGS.validation_images, file) for file in os.listdir(FLAGS.validation_images)]
    
    train_dataset = tf.data.TFRecordDataset(training_images)
    val_dataset = tf.data.TFRecordDataset(validation_images)
    
    train_dataset = train_dataset.map(_parse_record)
    val_dataset = val_dataset.map(_parse_record)
    
    train_dataset = train_dataset.shuffle(4096).batch(FLAGS.batch_size).repeat()
    val_dataset = val_dataset.batch(FLAGS.batch_size).repeat()
    
    return train_dataset, val_dataset


def train_evaluate():
    
    train_dataset, val_dataset = prepare_datasets()
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
             loss="sparse_categorical_crossentropy",
             metrics=["sparse_categorical_accuracy"]
             )

    model.fit(train_dataset,
         epochs=20,
         steps_per_epoch=1000,
         validation_data=val_dataset,
         validation_steps=200)
    
    

FLAGS = flags.FLAGS
flags.DEFINE_string("training_images", None, "Training images")
flags.DEFINE_string("validation_images", None, "Validation images")
flags.DEFINE_integer("epochs", 10, "Number of epochs to train")
flags.DEFINE_integer("batch_size", 1, "Batch size")

# Required flags
flags.mark_flag_as_required("training_images")
flags.mark_flag_as_required("validation_images")

def main(argv):
    del argv #Unused
    
    print(FLAGS.training_images)
    print(FLAGS.validation_images)
    
    train, val = prepare_datasets()
    
    for record in train.take(1):
        print(record)
    

if __name__ == '__main__':
    
    app.run(main)
