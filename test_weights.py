import numpy as np



def mix_arrays(a, b):
    aux = np.concatenate((a, b))
    n_items = np.prod(aux.shape)
    # Flatten the array and shuffle
    np.random.shuffle(aux.reshape(n_items,))
    # Split array into half and return first item
    return np.split(aux, 2)[0]

def print_shape(a):
    s = np.prod(a.shape)
    print(s)

# This functions creates a random array of given shape
def random_array(shape):
    return np.random.rand(*shape)

def mutate(shape):
    return np.random.rand(*shape)


# Numpy array of 2 x 3
a = np.array([[1, 2, 3], [4, 5, 6]])
# Numpy array of 2 x 3
b = np.array([[1, 2, 3], [4, 5, 6]])


x = mutate(a.shape)
print(x)





# print(mix_arrays(a, b))

# Numpy array of 1 x 3
a = np.array([1, 2, 3])
# Numpy array of 1 x 3
b = np.array([1, 2, 3])

# print(a.shape)

print(mix_arrays(a, b))



