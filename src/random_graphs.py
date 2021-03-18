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
from pu_learner import *

def random_graph(p, size=(100,100)):
    return(np.array([[int(random.random() < p) for i in range(size[1])] for j in range(size[0])]))

def random_undirected_graph(p, size=(100,100)):
    graph = random_graph(p, size=size)
    graph[np.arange(size[0]),np.arange(size[1])]=0 #nullify the diagonal
    graph = np.maximum(graph, graph.T) #make it symmetric
    
    return(graph)

def random_graph_with_fixed_components(p, nodes_per_component=[50,50]):
    nodes_per_component = np.array(nodes_per_component)
    n_nodes = nodes_per_component.sum()
    n_cmp = nodes_per_component.shape[0]
    
    graph = np.zeros((n_nodes, n_nodes))
    nodes = np.arange(n_nodes)
    np.random.shuffle(nodes)
    
    cmp_nodes = []
    acc=0
    
    for i in range(n_cmp):
        cmp = nodes[acc:(acc+nodes_per_component[i])]
        cmp_nodes.append(cmp)
        acc += nodes_per_component[i]
        
        size = cmp.shape[0]
        submatrix=np.ix_(cmp,cmp)

        graph[submatrix] = random_undirected_graph(p, (size,size))
        
    return(graph)

def neighbors(adj, i):
    return (np.where(adj[i,:]==1)[0])

def dfs(adj, i):
    n = adj.shape[0] #number of nodes in the graph
    visited = [False for k in range(n)]
    
    stack = [i]
    
    while(len(stack)>0):
        k = stack.pop()
        neighborhood = neighbors(adj, k)
        visited[k] = True
        
        for n in neighborhood:
            if not visited[n]:
                stack.append(n)
    
    return(np.where(visited))

def connected_components(adj):
    n = adj.shape[0]
    
    visited = np.array([0 for k in range(n)])
    s = np.sum(visited)
    
    comp=[]
    
    while s<n:
        i = np.where(1-visited)[0][0]
        
        cmp = dfs(adj, i)
        visited[cmp] = 1
        s = np.sum(visited)
        
        comp.append(list(cmp[0]))
    
    return(np.array(comp))

if __name__ == '__main__':
	N_rand_drugs = 784
    N_rand_targets = 1000

    density=0.2
    #generate random matrices in M({0,1})
    #1. a drug-drug network (drug similarities)
    #2. a protein-protein network (protein similarities)
    #3. a drug-protein network (drug-target known relationships)
    dd_net = random_graph_with_fixed_components(density, [196,196,196,196])
    pp_net = random_graph_with_fixed_components(density, [100 for i in range(10)])
    dp_net = random_graph(density,size=(N_rand_drugs,N_rand_targets))

    embeddings_drugs, _, _ = dngr_pipeline(dd_net, N_rand_drugs, [500, 200, 100])
    embeddings_targets, _, _ = dngr_pipeline(pp_net, N_rand_targets, [500, 200, 100])
    pos_train, neg_train, pos_test, neg_test = train_test_split_pu(dp_net)
    S, H, W =pu_learning(embeddings_drugs, embeddings_targets, dp_net, pos_train, neg_train, k=30, maxiter=500000, alpha=0.8, gamma=50)

    test = torch.logical_or(pos_test, neg_test)
    train = torch.logical_or(pos_train, neg_train)

    print(torch.sum(test), torch.sum(train))
    print(torch.std(S), torch.std(S[test]), torch.std(S[train]))