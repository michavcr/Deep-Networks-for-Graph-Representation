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

def old_pu_learning(x, y, P, k = 7, alpha = 0.2, gamma = 0.3, maxiter=1000):
    Fd = x.shape[1]
    Ft = y.shape[1]
    Nd = x.shape[0]
    Nt = y.shape[0]
    
    #Number of variables
    N_variables = Fd * k + Ft * k
    
    print("Number of variables:", N_variables)    
    print("Finding positive and negative examples...")
    
    Ipos = np.where(P==1.)
    Ineg = np.where(P==0.)

    print("Number of positive examples:", Ipos[0].shape[0])
    print("Number of negative/unlabelled examples:", Ineg[0].shape[0])
    
    alpha_rac = sqrt(alpha)
    
    @timeit
    def objective(z):
        H = z[:Fd*k].reshape((Fd,k))
        W = z[-Ft*k:].reshape((Ft,k))
        
        M = P - (x @ H @ np.transpose(W) @ np.transpose(y))
        
        M[Ineg] *= alpha_rac
        
        L = torch.sum(M**2) + gamma/2 * (np.sum(H**2, axis=(0,1)) + np.sum(W**2, axis=(0,1)))
        print(L)

        return(L)
    
    def constraint(z):
        H = z[:Fd*k].reshape((Fd,k))
        W = z[-Ft*k:].reshape((Ft,k))
        S = x @ H @ np.transpose(W) @ np.transpose(y)
        S = S.reshape((-1,))
        
        return(S)
    
    nlc = NonlinearConstraint(constraint, np.zeros(Nt*Nd), np.ones(Nt*Nd))

    print("Going to minimize... Maximum number of iterations:", maxiter)
    res=minimize(objective, x0 = np.random.randn(N_variables), options={'maxiter':maxiter, 'disp':'True'}, constraints=[nlc], method='trust-constr')
    
    print("\n\nSolved.")
    
    z=res['x']
    H = z[:Fd*k].reshape((Fd,k))
    W = z[-Ft*k:].reshape((Ft,k))

    print("Now computing Z=HW^T, then will compute S...")
    
    S = x @ H @ np.transpose(W) @ np.transpose(y)
    
    return(S)

def pu_learning(x, y, P, pos_train_mask, neg_train_mask, k = 7, alpha = 0.2, gamma = 0.3, maxiter=1000, lr=0.1):
    Fd = x.shape[1]
    Ft = y.shape[1]
    Nd = x.shape[0]
    Nt = y.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Number of variables
    N_variables = Fd * k + Ft * k
    
    print("Number of variables:", N_variables)    
    print("Finding positive and negative examples...")
    
    P = torch.Tensor(P).to(device)
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    
    x_norm = torch.linalg.norm(x)
    y_norm = torch.linalg.norm(y)

    Ipos = pos_train_mask
    Ineg = neg_train_mask
    train_mask = torch.logical_or(Ipos,Ineg)

    print("Number of positive examples:", P[Ipos].size()[0])
    print("Number of negative/unlabelled examples:", P[Ineg].size()[0])
    
    alpha_rac = sqrt(alpha)
    
    def objective(H,W):
        M = P - torch.chain_matmul(x, H, torch.transpose(W, 0, 1), torch.transpose(y, 0, 1))
        
        M[Ineg] *= alpha_rac
        M[~train_mask] = 0.
        
        L = torch.sum(M**2) + gamma/2 * (torch.sum(H**2) + torch.sum(W**2))

        return(L)
    
    def constraint(z):
        z = torch.Tensor(z).to(device)
        H = z[:Fd*k].resize(Fd,k)
        W = z[-Ft*k:].resize(Ft,k)

        S = torch.chain_matmul(x, H, torch.transpose(W, 0, 1), torch.transpose(y, 0, 1))
        S = S.reshape((-1,)).cpu().detach().numpy()
        
        return(S)
    
    #nlc = NonlinearConstraint(constraint, np.zeros(Nt*Nd), np.ones(Nt*Nd))

    print("Going to minimize... Maximum number of iterations:", maxiter)
    #res=minimize(objective, x0 = np.random.randn(N_variables), options={'maxiter':maxiter, 'disp':'True'}, constraints=[nlc], method='trust-constr')
    W = ag.Variable(torch.rand(Ft,k).to(device)/y_norm, requires_grad=True)
    H = ag.Variable(torch.rand(Fd,k).to(device)/x_norm, requires_grad=True)
    
    opt = torch.optim.Adam([H,W], lr=lr)

    for i in range(maxiter):
        # Zeroing gradients
        opt.zero_grad()

        # Evaluating the objective
        obj = objective(H,W)

        # Calculate gradients
        obj.backward() 
        opt.step()
        if i%1000==0:  
            S = torch.chain_matmul(x, H, torch.transpose(W,0,1), torch.transpose(y,0,1))
            auc = compute_auc(P[train_mask],S[train_mask])
            print("Objective:", obj, "(auc:", auc, ")")

    print("\n\nSolved.")
    
    print("Now computing Z=HW^T, then will compute S...")
    
    S = torch.chain_matmul(x, H, torch.transpose(W,0,1), torch.transpose(y,0,1))
    
    return(S, H, W)

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

def new_pu_learning(x, y, P, k = 7, alpha = 0.2, gamma = 0.3, maxiter=1000, lr=1e-3, batch_size=100):
    Fd = x.shape[1]
    Ft = y.shape[1]
    Nd = x.shape[0]
    Nt = y.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Number of variables
    N_variables = Fd * k + Ft * k
    N_examples = Nd*Nt
    
    P = torch.Tensor(P).to(device)
    cartesian_product = torch.Tensor([[u, v] for u in x for v in y]).to(device)
    
    print("Spliting train and test sets...")
    pos_train, neg_train, pos_test, neg_test = get_train_test_masks(P, train_size=0.8)

    train_mask = torch.logical_or(pos_train, neg_train)

    flat_train_mask = train_mask.flatten()

    print("Building the train loader...")
    train = torch.utils.data.TensorDataset(cartesian_product[flat_train_mask], P[train_mask].flatten())
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    print("Number of variables:", N_variables)
    print("Finding positive and negative examples...")
    
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    
    print("Number of train examples:", P.size()[0])
    print("Number of positive examples in train set:", P[pos_train].size()[0])
    print("Number of negative/unlabelled examples in train set:", P[neg_train].size()[0])
    
    alpha_rac = sqrt(alpha)
    
    def objective(a, H, W, b, P):
        M = P.clone().detach()
        for i in range(P.size()[0]):
            s = torch.chain_matmul(a[i].reshape(1,Fd), H, torch.transpose(W, 0, 1), torch.transpose(b[i].reshape(1,Ft), 0, 1))
            M[i] -= s.item()

        Ineg = (P==0.)

        M[Ineg] *= alpha_rac
        
        L = torch.mean(M**2) + gamma/2 * (torch.sum(H**2) + torch.sum(W**2))

        return(L)
    
    def objective_prime(a, H, W, b, P):
        M = P - torch.einsum('ij,jk,kl,li->i', a, H, torch.transpose(W, 0, 1), torch.transpose(b, 0, 1))
        
        Ineg = (P==0.)

        M[Ineg] *= alpha_rec
        L = torch.mean(M**2) + gamma/2 * (torch.sum(H**2) + torch.sum(W**2))

        return(L)

    def total_loss(x,H,W,y,P):
        M = P - torch.chain_matmul(x, H, torch.transpose(W, 0, 1), torch.transpose(y, 0, 1))

        M[~train_mask] = 0.

        M[neg_train] *= alpha_rac

        L = torch.mean(M**2) + gamma/2 * (torch.sum(H**2) + torch.sum(W**2))

        return(L)

    print("Going to minimize... Maximum number of epochs:", maxiter)
    
    W = torch.zeros(Ft,k).to(device)
    W /= torch.linalg.norm(W)*torch.linalg.norm(y) #for initial W and H not to be too big
    H = torch.zeros(Fd,k).to(device)
    H /= torch.linalg.norm(H)*torch.linalg.norm(x)

    W = ag.Variable(W, requires_grad=True)
    H = ag.Variable(H, requires_grad=True)
    
    opt = torch.optim.Adam([H,W], lr=lr)
    
    batch_num = N_examples//batch_size

    for k in range(maxiter):
        for i, batch in enumerate(train_loader, 0):
            # Zeroing gradients
            opt.zero_grad()

            inputs, labels = batch

            a, b = inputs[:,0,:], inputs[:,1,:]
        
            # Evaluating the objective
            obj = objective(a,H,W,b,labels)

            # Calculate gradients
            obj.backward() 
            opt.step()
            
            if i%(batch_size//4) == 0:
                S = torch.chain_matmul(x, H, torch.transpose(W,0,1), torch.transpose(y,0,1))
                stdm = torch.std(S[train_mask])
                auc = compute_auc(P[train_mask],S[train_mask])
                loss = total_loss(x,H,W,y,P)

                print("[epoch=",k,", batch_num=",i,"]","Objective: ", loss.item(), "(standard deviation: ", stdm.item(), ", auc: ", auc.item(), ")")

    print("\n\nSolved.")
    
    print("Now computing Z=HW^T, then will compute S...")
    
    S = torch.chain_matmul(x, H, torch.transpose(W,0,1), torch.transpose(y,0,1))
    
    return(S, H, W)