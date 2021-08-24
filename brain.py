import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

class Brain(nn.Module):
    '''
        Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        # Create neural net which receives a 14-dimensional input vector.
        # and outputs a 4-dimensional vector.
        self.model = nn.Sequential(
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 4)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.model(x)

    def decide(self, snake_data):
        '''
            Decide which direction to move.
        '''
        # Convert snake_data to torch tensor.
        # snake_data is a numpy array
        x = torch.from_numpy(snake_data).float()
        # Pass the input through the neural net.
        y = self.forward(x)
        # Return the index of the largest value.
        return int(y.argmax())

    def get_weights(self):
        return self.model.parameters()

    def get_flattened_weights(self):
        weights = self.get_weights()
        return torch.cat([w.view(-1) for w in weights])

    # This method recives a 1D tensor and resape it to the shape of the model weights.
    def unflatten(self, tensor):
        unflattened = []
        shapes = [x.shape for x in self.model.parameters()]
        cut_index = 0
        for shape in shapes:
            size = np.product(shape)
            unflattened.append(tensor[cut_index:cut_index+size].view(shape))
            cut_index += size
        return unflattened

    def set_weights(self, weights):
        # Check if the weights are the correct shape.
        if isinstance(weights, torch.Tensor):
            weights = self.unflatten(weights)
        # Set the weights to the moodel.
        for param, w in zip(self.model.parameters(), weights):
            param.data = nn.parameter.Parameter(w)
