# check version of tensorflow
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense , Flatten
#load data for our model to do task of classification
print('version:')
print(tf.__version__)
# load data to do task of classifaction of image
from tensorflow.keras.datasets import fashion_mnist
# The data has already been sorted into training and test sets for us
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
#check shapes :
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Check the shape of our data
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape
print(train_data.shape)
#create a model from scratch and test it
tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")# output shape  is   a 10 , activation  is softmax

])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(train_data, train_labels,epochs=10,validation_data=(test_data, test_labels),verbose=2)

print(model.summary())
# lets check after scaling all input data
train_data=train_data/255.0
test_data=test_data/255.0
#build a model after scaling
tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")# output shape  is   a 10 , activation  is softmax

])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(train_data, train_labels,epochs=10,validation_data=(test_data, test_labels),verbose=2)

# creata model with learning rate callbacks

tf.random.set_seed(42)
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")# output shape  is   a 10 , activation  is softmax

])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
lr_scheduler=tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-3*10**(epoch/20))
model.fit(train_data, train_labels,epochs=10,validation_data=(test_data, test_labels),verbose=2,callbacks=[lr_scheduler])
##Our model outputs a list of prediction probabilities, meaning, it outputs a number for how likely it thinks a particular class is to be the label.
y_probs = model.predict(test_data)
##The higher the number in the prediction probabilities list, the more likely the model believes that is the right class.

##To find the highest value we can use the argmax() method.
print(y_probs[0].argmax(), class_names[y_probs[0].argmax()])

# Save the entire model
model.save('my_model.keras')

