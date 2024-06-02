#import libairies
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import pathlib
dataset_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)
import PIL
roses=list(data_dir.glob("roses/*"))
print(roses[0])
#define details
image_size=(180,180)
batch_size=32
channels=3
#dataset for training :
train_ds=tf.keras.preprocessing.image_dataset_from_directory(data_dir,seed=123 , validation_split=0.2,
  subset="training",image_size=image_size,batch_size=batch_size)
#dataset for testing :
val_data=tf.keras.preprocessing.image_dataset_from_directory(data_dir,seed=123 , validation_split=0.2,
  subset="validation",image_size=image_size,batch_size=batch_size)
#check number of classes :
class_names=train_ds.class_names
print(class_names)
#modeling
pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(180, 180, 3), pooling='avg',
                                                  classes=5,
                                                  weights='imagenet'

                                                  )
for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax'),

])
resnet_model.summary()
resnet_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history=resnet_model.fit(train_ds,validation_data=val_data,epochs=5)
# print the training and validation loss for each epoch
for epoch in range(len(history.history['loss'])):
  print(f"Epoch {epoch+1}: Train loss: {history.history['loss'][epoch]}, Validation loss: {history.history['val_loss'][epoch]}")

