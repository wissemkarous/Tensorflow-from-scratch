#load the model and test

import tensorflow as tf
import numpy as np

# Step 3: Load the saved .keras model
loaded_model_saved = tf.keras.models.load_model('my_model.keras')

# Step 4: Test the loaded model with sample data
# Generate some sample test data
x_test = tf.random.normal([10,28, 28])
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Predict using the loaded model
predictions_saved = loaded_model_saved.predict(x_test)

print("Predictions from the loaded .keras model:\n", predictions_saved)
print(predictions_saved[0].argmax(), class_names[predictions_saved[0].argmax()])
