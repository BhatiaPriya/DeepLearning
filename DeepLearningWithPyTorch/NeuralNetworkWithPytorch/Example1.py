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

class Network(nn.Module):    ## this class is a child class which is inheriting from the superclass nn.module
    def __init__(self):
        super().__init__()    ## it is calling init method of the parent class(nn module). Though this line pytorch will
                              ## know to register all the different layers and operations that we are going to be putting 
                              ## into this network. If we don't use this line, then it won't be able to track that you are 
                              ## adding to your network
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
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