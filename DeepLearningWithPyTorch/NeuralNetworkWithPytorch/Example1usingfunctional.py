# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:07:43 2019

@author: priya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:17:35 2019

author: priya
"""

import numpy as np
import torch
import helper
import matplotlib.pyplot as plt

'''Now we're going to build a larger network that can solve a (formerly) difficult problem,
identifying text in an image. Here we'll use the MNIST dataset which consists of greyscale handwritten digits.
Each image is 28x28 pixels.'''

from torch import nn
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

'''The batch size is the number of images we get in one iteration from the data loader 
and pass through our network, often called a batch. And shuffle=True 
tells it to shuffle the dataset every time we start going through the data loader again.'''

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)


'''Lets build a multi-layer network with 784 input units, 
256 hidden units, and 10 output units and a softmax output'''

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)   # or self.fc1 = nn.Linear(784, 256)  
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)    # or self.fc2 = nn.Linear(256, 10) 
        
    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))       # x = F.sigmoid(self.fc1(x)) 
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1) # x = F.sigmoid(self.fc2(x))
        
        return x
    
# Create the network and look at it's text representation
model = Network()
print(model)

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:]) # 0th row and all cols

img = images[img_idx]
print(helper.view_classify(img.view(1, 28, 28), ps))