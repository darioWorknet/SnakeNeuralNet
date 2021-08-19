import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Model
model = keras.Sequential()
model.add(layers.Dense(7, activation="relu", input_shape=(14,)))
model.add(layers.Dense(4))

model.load_weights('snake_brain.h5')

weights = model.get_weights()

for w in weights:
    np.random.shuffle(w)

model.set_weights(weights)


# This funcion returns the index of the biggest value in the array
def get_biggest_value(array):
    biggest_value = 0
    index = 0
    for i in range(len(array)):
        if array[i] > biggest_value:
            biggest_value = array[i]
            index = i
    return index



def decide(environmental_data):
    desition = model.predict(environmental_data)
    direction = get_biggest_value(desition.reshape(-1))
    return direction


if __name__ == '__main__':
    # Call model on a test input
    x = tf.ones((1,14), dtype=tf.float32)
    x /= 2
    print(x)
    decide(x)


