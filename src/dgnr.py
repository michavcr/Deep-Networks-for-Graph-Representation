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

from autoencoder import *

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

def normalize(M):
    #Put diagonal elements to 0
    M  = M - np.diag(np.diag(M))
    
    #Normalizing by row
    D_inv = np.diag(np.reciprocal(np.sum(M,axis=0)))
    M = np.dot(D_inv,  M)

    return M

def PCO(A, K, alpha):
    """
    For a graph represented by its adjacency matrix *A*, computes the co-occurence matrix by random 
    surfing on the graph with returns. 1-alpha is the probability to make, at each step, a return 
    to the original step.
    """
    A=np.array(A, dtype=float)
    
    #The adjacency matrix A is first normalized
    A=normalize(A) 
    
    n=A.shape[0]
    
    I=np.eye(n)
    
    P=I
    M=np.zeros((n, n))
    
    for i in range(K):
        P = alpha*np.dot(P,A) + (1-alpha)*I
        M = M+P
    
    return(M)

def PPMI(M):
    """Computes the shifted positive pointwise mutual information (PPMI) matrix
    from the co-occurence matrix (PCO) of a graph."""
    
    M=normalize(M)
    cols = np.sum(M, axis=0)
    rows = np.sum(M, axis=1).reshape((-1,1))
    s = np.sum(rows)
    
    P = s*M
    P /= cols
    P /= rows
    
    #P[np.where(P<0)] = 1.0
    P = np.log(P)

    #To avoid NaN when applying log
    P[np.isnan(P)] = 0.0
    P[np.isinf(P)] = 0.0
    P[np.isneginf(P)] = 0.0
    P[np.where(P<0)] = 0.0
    
    return(P)

def sdae(input_net, input_number, hidden_layers, n_epochs=100, batch_size=1, activation='sigmoid', last_activation='sigmoid'):
    #hidden_layers=[500,200,100]
    #input_numer=784
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SDAE(input_number, hidden_layers, activation=activation, last_activation=last_activation).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    summary(model, (input_number,))
    
    tensor_net = torch.Tensor(input_net).to(device)
    train = torch.utils.data.TensorDataset(tensor_net, tensor_net)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs=inputs.to(device)
            
            inputs=torch.flatten(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
        
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % input_number == input_number-1: 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / input_number))
                running_loss = 0.0

    print('Finished Training')
    
    return(model, train_loader)