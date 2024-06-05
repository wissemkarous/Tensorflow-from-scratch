#import necessary libraries from python
import datetime 
import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras import layers 
import PIL as pl
import matplotlib.pyplot as plt
import zipfile
# load the data form link 
#extract dataset  zip file
zip_ref = zipfile.ZipFile('10_food_classes_1_percent.zip', 'r')
zip_ref.extractall('10_food_classes_1_percent')
zip_ref.close()

#load the data  and lets build the model
train_ds=tf.keras.preprocessing.image_dataset_from_directory("10_food_classes_1_percent/10_food_classes_1_percent/train",subset="training",seed=123,validation_split=0.2,image_size=(224,224),batch_size=32)

val_ds=tf.keras.preprocessing.image_dataset_from_directory("10_food_classes_1_percent/10_food_classes_1_percent/train",subset="validation",seed=123,validation_split=0.2,image_size=(224,224),batch_size=32)

test_ds=tf.keras.preprocessing.image_dataset_from_directory("10_food_classes_1_percent/10_food_classes_1_percent/test",subset="validation",seed=123,validation_split=0.2,image_size=(224,224),batch_size=32)
#check the class_names
class_names=train_ds.class_names

print(class_names)
#build the necessary layers (rescale and augmentation)

data_augmentation_Rescale=tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.2,0.2),
    layers.Rescaling(1./255,input_shape=(224, 224, 3))
])
#build model using the cnn 

model = tf.keras.Sequential([
    data_augmentation_Rescale,
    layers.Conv2D(32, 3, activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax")
])
#build the model
# Build the model by specifying the input shape
# Build the model by calling it with a sample input tensor
sample_input = tf.random.uniform([1, 224, 224, 3])
model(sample_input)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=["accuracy"])
# Print the output shape of each layer to debug
for layer in model.layers:
    sample_input = layer(sample_input)
    print(f"After layer {layer.name}, the shape is: {sample_input.shape}")
model.summary()
#fir the model
history=model.fit(train_ds,epochs=10,validation_data=val_ds)
#evaluate the model 



print(model.evaluate(test_ds))

#save the model

model.save("10_food_classes_1_percent_saved_model.keras")
