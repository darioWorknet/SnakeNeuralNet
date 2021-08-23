import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class Brain(nn.Module):
    '''
        Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        # Create neural net which receives a 14-dimensional input vector.
        # and outputs a 4-dimensional vector.
        self.layers = nn.Sequential(
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 4)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

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



