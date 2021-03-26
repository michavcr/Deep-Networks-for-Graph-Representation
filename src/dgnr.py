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

from autoencoder import SDAE
from utils import *
from random_graphs import connected_components

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

def zero_one_normalisation(matrix, e=1e-5):
    M = np.max(matrix)
    m = np.min(matrix)
    r = (matrix-m) / (M-m + e)
    return(r)

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

    normalized_input_net = zero_one_normalisation(input_net)
    tensor_net = torch.Tensor(normalized_input_net).to(device)
    N = tensor_net.size()[0]
    idx = torch.arange(N).long()

    model = SDAE(tensor_net, input_number, hidden_layers, activation=activation, last_activation=last_activation).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    #summary(model, (input_number,))
    
    train = torch.utils.data.TensorDataset(idx, idx)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            idx, _ = data
            
            idx=idx.long().to(device)
            inputs=tensor_net[idx]

            #inputs=torch.flatten(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(idx)
        
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / input_number))

    print('Finished Training')
    
    return(model, train_loader, tensor_net)

def get_embeddings(train_loader, N, model, size_encoded=100):
    trainiter = iter(train_loader)
    embeddings = np.zeros((N, size_encoded))

    for i,q in enumerate(trainiter):
        idx = q[0]
        for j in idx:
            embedded = model.encoder(model.features[j]).cpu().detach().numpy()
            embeddings[j,:] = embedded.reshape((size_encoded,))
    
    return(embeddings)
    
@timeit
def dngr_pipeline(network, N, hidden_layers, K=10, alpha=0.2, n_epochs=100, batch_size=1, activation='sigmoid', last_activation='sigmoid'):
    ppmi_net = PPMI(PCO(network, K, alpha))
    model, train_loader, tensor_net = sdae(ppmi_net, N, hidden_layers, n_epochs=n_epochs, batch_size=batch_size, activation=activation, last_activation=last_activation)
    
    print("[*] Visualizing an example's output...")
    trainiter = iter(train_loader)
    idx, _ = trainiter.next()

    print(tensor_net[idx])
    print(model(idx))

    print(mean_squared_error(tensor_net[idx].cpu().detach().numpy(), model(idx).cpu().detach().numpy()))
    
    print("[*] Getting the embeddings and visualizing t-SNE...")
    embeddings=get_embeddings(train_loader, N, model, size_encoded=hidden_layers[-1])
    
    cmps = connected_components(network)
    targets = [0 for i in range(N)]

    for i, cmp in enumerate(cmps):
        for n in cmp:
            targets[n] = i
    
    visualize_TSNE(embeddings, targets)
    
    return(embeddings, model, train_loader)