import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

def activation_sharpening(activations, sharpening_factor=2.0):
    """
    Apply activation sharpening to the given activations.
    """
    mean = np.mean(activations)
    sharpened = np.where(activations > mean,
                         activations ** sharpening_factor,
                         activations)
    return sharpened / np.sum(sharpened)

class ActivationSharpeningCallback(tf.keras.callbacks.Callback):
    def __init__(self, sharpening_factor=2.0):
        super().__init__()
        self.sharpening_factor = sharpening_factor

    def on_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                weights = layer.get_weights()
                sharpened_weights = activation_sharpening(weights[0], self.sharpening_factor)
                layer.set_weights([sharpened_weights, weights[1]])

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a simple neural network
model = Sequential([
    Flatten(input_shape=(0, 784)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with activation sharpening
callback = ActivationSharpeningCallback(sharpening_factor=2.0)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")