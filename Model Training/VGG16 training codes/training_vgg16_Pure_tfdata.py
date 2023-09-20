import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import random
# disable_eager_execution()


### set dirs
train_path = "ds/train/"
val_path = "ds/val/"

### preprocessing for tf.data
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = random.randint(256, 512)

resize_and_crop = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.RandomCrop(224,224)
])

def prepare(ds):
  # Resize and rescale all datasets
  ds = ds.map(lambda x, y: (resize_and_crop(x), y), 
              num_parallel_calls=AUTOTUNE)

  # Use buffered prefecting on all datasets
  return ds.prefetch(buffer_size=AUTOTUNE)

### main variables
Batch_size = 64
Learning_rate = 0.01
Momentum = 0.9
Epochs = 100 

### defining tf.data generators
traindata_tfds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=Batch_size, image_size=(256,
    256), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False, smart_resize=True
)
train_ds = prepare(traindata_tfds)

valdata_tfds = tf.keras.preprocessing.image_dataset_from_directory(
    val_path, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=Batch_size, image_size=(256,
    256), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False, smart_resize=True
)
val_ds = valdata_tfds.prefetch(buffer_size=AUTOTUNE)


### setting Xavier initializer
initializer = tf.keras.initializers.GlorotNormal()

# model architecture
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.5))
model.add(Dense(units=4096,activation="relu", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))
model.add(Dropout(0.5))
model.add(Dense(units=1001, activation="softmax", kernel_initializer=initializer, kernel_regularizer=l2(5e-4)))

### defining optimizer and compiler
opt = SGD(learning_rate=Learning_rate, momentum=Momentum)
model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

### setting callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1)
model_checkpoint_callback1 = ModelCheckpoint(
    filepath='./model/best_checkpoint',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model_checkpoint_callback2 = ModelCheckpoint(
    filepath='./model/best_checkpoint.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

filename='log.csv'
history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

### training procedure
tf.config.run_functions_eagerly(True)
history = model.fit(train_ds, validation_data=val_ds, epochs=Epochs,
					callbacks=[reduce_lr, model_checkpoint_callback2, model_checkpoint_callback1, history_logger])