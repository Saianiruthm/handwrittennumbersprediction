import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

class MNISTDigitRecognizer:
    def __init__(self):
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocess the data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Define the neural network model
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

        # Train the model
        self.model.fit(x_train, y_train, epochs=10)

    def predict(self, image):
        # Expand the dimensions of the image to match the input shape of the model
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        prediction = self.model.predict(image)

        # Return the predicted class label
        return prediction[0].argmax()

# Create a new MNIST digit recognizer
recognizer = MNISTDigitRecognizer()

# Make a prediction on a new image
new_image = tf.expand_dims(x_test[0], axis=0)
prediction = recognizer.predict(new_image)

# Print the prediction
print('Prediction:', prediction)
