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
from sklearn.metrics import mean_squared_error

from math import sqrt
import time
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score

from scipy.spatial.distance import pdist, squareform

def compute_auc(P,S):
    y_pred = S.clone().detach()

    y_pred[y_pred < 0.] = 0.
    y_pred[y_pred > 1.] = 1.
	
    y_pred = S.cpu().detach().numpy().reshape((-1,))
    y_true = P.cpu().detach().numpy().reshape((-1,))

    return(roc_auc_score(y_true, y_pred))

def compute_accuracy(y_true, y_pred):
    y_pred = y_pred.clone().detach()

    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.

    y_pred = y_pred.cpu().detach().numpy().reshape((-1,))
    y_true = y_true.cpu().detach().numpy().reshape((-1,))

    return(accuracy_score(y_true, y_pred))


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

def visualize_TSNE(embeddings,target):
    tsne = TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=30)
    data = tsne.fit_transform(embeddings)
    #plt.figure(figsize=(12, 6))
    plt.title("TSNE visualization of the embeddings")
    plt.scatter(data[:,0],data[:,1],c=target)

    return

def compute_similarity(net):
    M = 1 - pdist(net, metric='Jaccard')
    M = squareform(M)
    M = M + np.eye(*M.shape)
    M[np.isnan(M)] = 0.

    return(M)

def readnet(net_path):
	return(np.genfromtxt(net_path,delimiter='\t'))