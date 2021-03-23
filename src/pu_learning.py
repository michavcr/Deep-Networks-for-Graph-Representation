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
from utils import *
from random_graphs import *
from dgnr import *

def get_train_test_masks(P, train_size=0.8):
    P = torch.Tensor(P)
    Ipos = (P == 1.)
    Ineg = (P == 0.)

    pos_train = Ipos * torch.rand(P.size())
    pos_train[Ineg] = 1.
    pos_train = pos_train < train_size
    train_neg_rel_size = torch.sum(pos_train) / torch.sum(Ineg)
    
    neg_train = Ineg * torch.rand(P.size())
    neg_train[Ipos] = 1.
    neg_train = neg_train < train_neg_rel_size

    train = pos_train + neg_train
    test = ~train
    pos_test = torch.logical_and(test, Ipos)
    neg_test = torch.logical_and(test, Ineg)

    return(pos_train, neg_train, pos_test, neg_test)
    
class PU_Learner(nn.Module):
    def __init__(self, k, Fd, Ft, activation='identity'):
        super().__init__()
        self.k = k
        self.Fd = Fd
        self.Ft = Ft

        self.H = torch.nn.Parameter(torch.randn(Fd, k))
        self.W = torch.nn.Parameter(torch.randn(k, Ft))
        
        activ_dict = {'sigmoid': torch.nn.Sigmoid, 'identity': torch.nn.Identity}
        self.activation = activ_dict[activation]()

    def forward(self, x, y):
    	  return(self.activation(torch.einsum('ij,jk,kl,li->i', x, self.H, self.W, torch.transpose(y, 1, 0))))

def pu_learning_new(k, x, y, P, n_epochs=100, batch_size=100, lr=1e-3, print_step=4, train_size=0.8):
    #hidden_layers=[500,200,100]
    #input_numer=784
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Fd = x.shape[1]
    Ft = y.shape[1]
    Nd = x.shape[0]
    Nt = y.shape[0]
    
    #Number of variables
    N_variables = Fd * k + Ft * k
    N_examples = Nd*Nt

    cartesian_product = torch.Tensor([[u, v] for u in x for v in y]).to(device)

    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)

    print("Spliting train and test sets...")
    pos_train, neg_train, pos_test, neg_test = get_train_test_masks(P, train_size=train_size)

    P = torch.Tensor(P).to(device)

    train_mask = torch.logical_or(pos_train, neg_train)

    flat_train_mask = train_mask.flatten()

    print("Building the train loader...")
    train = torch.utils.data.TensorDataset(cartesian_product[flat_train_mask], P[train_mask].flatten())
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    batch_num = len(train_loader)

    print("Number of variables:", N_variables)
    print("Finding positive and negative examples...")    

    print("Number of train examples:", P[train_mask].size()[0])
    print("Number of positive examples in train set:", P[pos_train].size()[0])
    print("Number of negative/unlabelled examples in train set:", P[neg_train].size()[0])

    model = PU_Learner(k, Fd, Ft).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # mean-squared error loss
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs=inputs.to(device)
            
            x_batch, y_batch = inputs[:,0,:], inputs[:,1,:]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x_batch, y_batch)
            
            loss = torch.sqrt(criterion(outputs, labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        S = torch.chain_matmul(x, model.H, model.W, torch.transpose(y,0,1))
        auc = compute_auc(P[train_mask].clone(), S[train_mask].clone())
        acc = compute_accuracy(P[train_mask].clone(), S[train_mask].clone())

        print('[%d] loss: %.3f, auc: %.3f, acc: %.3f' %
                  (epoch + 1, running_loss / (batch_num), auc, acc))
        running_loss = 0.0

    print('Finished Training')
    print("Now computing Z=HW^T, then will compute S...")
    
    print(x.size(), model.H.size(), model.W.size(), y.size())
    S = torch.chain_matmul(x, model.H, model.W, torch.transpose(y,0,1))
    
    return(S, model.H, model.W, train_mask, ~train_mask)  
