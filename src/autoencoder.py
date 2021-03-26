import numpy as np
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from scipy.optimize import minimize, NonlinearConstraint
import random

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from math import sqrt
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.randn(din.size()) * self.stddev
        return din

class DropoutNoise(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x):
        t = torch.rand(x.size()).to(self.device)
        a = t > self.p
        
        return(x*a)

class BasicBlock(nn.Module):
    def __init__(self, input_shape, n_neurons, activation='relu', noise=None, noise_arg=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.input_shape = input_shape
        
        self.has_noise = False
        
        if noise=='gaussian':
            self.has_noise = True
            self.noise = GaussianNoise(noise_arg)
        elif noise=='dropout':
            self.has_noise = True
            self.noise = DropoutNoise(noise_arg)
            
        self.dense_layer = nn.Linear(self.input_shape, self.n_neurons)
        
        activations_map = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}
        self.activation = activations_map[activation]()

    def forward(self, features):
        x=features
        
        if self.has_noise:
            x = self.noise(x)

        x = self.dense_layer(features)
        x = self.activation(x)
        
        return(x)

class SDAE(nn.Module):
    def __init__(self, features, input_shape, hidden_layers, activation='relu', last_activation='relu', noise_type='dropout', noise_arg=0.2):
        super().__init__()
        self.features = features

        self.inputs = [input_shape] + hidden_layers
        
        n = len(self.inputs)
        encoder_units = [BasicBlock(self.inputs[0], self.inputs[1], activation=activation, noise=noise_type, noise_arg=noise_arg)]
        encoder_units.extend([BasicBlock(self.inputs[i], self.inputs[i+1], activation=activation) for i in range(1, n-1)])
        
        self.encoder = nn.Sequential(*encoder_units)
        
        decoder_units = [BasicBlock(self.inputs[i], self.inputs[i-1], activation=activation) for i in range(n-1,1,-1)]
        decoder_units.append(BasicBlock(self.inputs[1], self.inputs[0], activation=last_activation))
        
        self.decoder = nn.Sequential(*decoder_units)
        
    def forward(self, idx):
        encoded = self.encoder(self.features[idx])
        
        decoded = self.decoder(encoded)
        
        return(decoded)

