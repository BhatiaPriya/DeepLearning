# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:35:50 2019

@author: priya
"""

## exploring how we can use PyTorch to build a simple neural network.

# First, import PyTorch
import torch

## Sigmoid activation function 
def activation(x):
    return 1/(1+torch.exp(-x))

### Generate some data
torch.manual_seed(7)  # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 5))

# True weights for our data, random normal variables again
weights = torch.randn_like(features) 

# and a true bias term
bias = torch.randn((1, 1))

## Calculate the output of this network using the weights and bias tensors
output = activation(torch.sum(features * weights) + bias)
print(output)

# One more approach using the same as above
## Calculate the output of this network using matrix multiplication
output = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(output)