import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Brain:
    def __init__(self, parent_1=None, parent_2=None):
        self.model = self.create_model(parent_1, parent_2)

    def create_model(self, parent_1=None, parent_2=None):
        model = keras.Sequential()
        model.add(layers.Dense(7, activation="relu", input_shape=(14,)))
        model.add(layers.Dense(4))
        if parent_1 and parent_2:
            # weights = []
            # for w1, w2 in zip(parent_1.brain.model.get_weights(), parent_2.brain.model.get_weights()):
            #     weights.append(self.give_birth(w1, w2))
            weights = self.procreate(parent_1, parent_2)
            model.set_weights(weights)
        return model

    def decide(self, environmental_data):
        desition = self.model.predict(environmental_data)
        direction = self.get_biggest_value(desition.reshape(-1))
        return direction

    # This funcion returns the index of the biggest value in the array
    def get_biggest_value(self, array):
        biggest_value = 0
        index = 0
        for i in range(len(array)):
            if array[i] > biggest_value:
                biggest_value = array[i]
                index = i
        return index



    def procreate(self, parent_1, parent_2):
        w1 = parent_1.brain.model.get_weights()
        w2 = parent_2.brain.model.get_weights()
        weights = []
        for w1, w2 in zip(w1, w2):
            # Do this with a probability of 0.9
            if np.random.rand() < 0.9:
                weights.append(self.mix_layers(w1, w2))
            else:
                weights.append(self.mutate(w1.shape))
        return weights

    def mutate(self, shape):
        return np.random.rand(*shape)

    def mix_layers(self, w1, w2):
        aux = np.concatenate((w1, w2))
        n_items = np.prod(aux.shape)
        # Flatten the array and shuffle
        np.random.shuffle(aux.reshape(n_items,))
        # Split array into half and return first item
        return np.split(aux, 2)[0]



if __name__ == '__main__':
    # Call model on a test input
    x = tf.ones((1,14), dtype=tf.float32)
    x /= 2
    print(x)
    b = Brain()
    b.decide(x)



