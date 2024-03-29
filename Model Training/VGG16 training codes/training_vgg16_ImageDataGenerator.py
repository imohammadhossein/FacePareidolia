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
disable_eager_execution()

### Gradient accumulation class
class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


### scaling and random_crop def
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    #apply scaling
    size = random.randint(256, 512)
    img = cv2.resize(img, (size, size))

    ### apply random crop
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return img[y:(y+dy), x:(x+dx), :]

### scaling and random_crop generator
def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

### set dirs
train_path = "ds/train/"
val_path = "ds/val/"

### main variables
Batch_size = 64
Learning_rate = 0.01
Momentum = 0.9
Epochs = 100 

### defining generators
trdata = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, channel_shift_range=100)
traindata = trdata.flow_from_directory(directory=train_path,target_size=(256,256), batch_size=Batch_size, shuffle=True)
train_crops = crop_generator(traindata, 224)

val_data = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_data = val_data.flow_from_directory(directory=val_path, target_size=(224,224), batch_size=Batch_size, shuffle=True)

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
### for gradient accumulation
custom_model = CustomTrainStep(n_gradients=4, inputs=[model.input], outputs=[model.output])

### defining optimizer and compiler
opt = SGD(learning_rate=Learning_rate, momentum=Momentum)
custom_model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

### setting callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
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
history = custom_model.fit(train_crops, validation_data=validation_data, epochs=Epochs, steps_per_epoch=int(traindata.samples/Batch_size), validation_steps=int(validation_data.samples/Batch_size), 
					callbacks=[reduce_lr, model_checkpoint_callback2, model_checkpoint_callback1, history_logger])