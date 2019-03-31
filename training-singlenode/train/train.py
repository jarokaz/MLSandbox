from absl import flags
from absl import app

import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, add

#tf.enable_eager_execution()


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
        image = tf.cast(image, tf.float32)
        image = image / 255
        
        label = tf.one_hot(label, NUM_CLASSES)
        
        return image, label

    
    train_dataset = tf.data.TFRecordDataset(FLAGS.train_files)
    eval_dataset = tf.data.TFRecordDataset(FLAGS.eval_files)
    
    train_dataset = train_dataset.map(_parse_record)
    eval_dataset = eval_dataset.map(_parse_record)
    
    train_dataset = train_dataset.shuffle(4096).batch(FLAGS.batch_size).repeat()
    eval_dataset = eval_dataset.batch(FLAGS.batch_size).repeat()
    
    return train_dataset, eval_dataset


def train_evaluate():
    
    train_dataset, eval_dataset = prepare_datasets()
    
    model = toy_resnet_model()
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
             loss="categorical_crossentropy",
             metrics=["accuracy"]
             )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=FLAGS['job-dir'].value, update_freq='epoch')
    ]
    
    model.fit(train_dataset,
         epochs=FLAGS.epochs,
         steps_per_epoch=1000,
         callbacks=callbacks,
         validation_data=eval_dataset,
         validation_steps=200)
    
    

FLAGS = flags.FLAGS
flags.DEFINE_list("train_files", None, "Training TFRecord files")
flags.DEFINE_list("eval_files", None, "Evaluation TFRecord files")

flags.DEFINE_integer("epochs", 5, "Number of epochs to train")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("steps_per_epoch", 1000, "Steps per epoch")
flags.DEFINE_integer("validation_steps", 20, "Batch size")

flags.DEFINE_string("job-dir", None, "Job dir")

# Required flags
flags.mark_flag_as_required("train_files")
flags.mark_flag_as_required("eval_files")


def main(argv):
    del argv #Unused
    
    train_evaluate()
    #print(FLAGS.train_files)
    #print(FLAGS.eval_files)
    

if __name__ == '__main__':
    
    app.run(main)