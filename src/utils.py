import numpy as np
import torch
import time
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score

from scipy.spatial.distance import pdist, squareform

def compute_pr_auc(P,S):
    """
    Compute the prediction-recall AUC.

    Parameters
    ----------
    P : torch Tensor
        Ground truth.
    S : torch Tensor
        Predictions.

    Returns
    -------
    The precision-recall AUC.

    """
    y_pred = S.clone().detach()

    y_pred[y_pred < 0.] = 0.
    y_pred[y_pred > 1.] = 1.
    
    y_pred = S.cpu().detach().numpy().reshape((-1,))
    y_true = P.cpu().detach().numpy().reshape((-1,))

    return(average_precision_score(y_true, y_pred))

def compute_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.

    Parameters
    ----------
    y_true : torch Tensor
        Ground truth.
    y_pred : TYPE
        Predictions.

    Returns
    -------
    The confusion matrix.

    """
    y_pred = y_pred.clone().detach()

    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.

    y_pred = y_pred.cpu().detach().numpy().reshape((-1,))
    y_true = y_true.cpu().detach().numpy().reshape((-1,))

    return(confusion_matrix(y_true, y_pred))

def compute_auc(P,S):
    """
    Compute the ROC AUC.

    Parameters
    ----------
    P : torch Tensor
        Ground truth.
    S : TYPE
        Predictions.

    Returns
    -------
    The ROC AUC.

    """

    y_pred = S.clone().detach()

    y_pred[y_pred < 0.] = 0.
    y_pred[y_pred > 1.] = 1.
	
    y_pred = S.cpu().detach().numpy().reshape((-1,))
    y_true = P.cpu().detach().numpy().reshape((-1,))

    return(roc_auc_score(y_true, y_pred))

def compute_accuracy(y_true, y_pred):
    """
    Compute the accuracy.

    Parameters
    ----------
    y_true : torch Tensor
        Ground truth.
    y_pred : TYPE
        Predictions.

    Returns
    -------
    The accuracy.

    """

    y_pred = y_pred.clone().detach()

    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.

    y_pred = y_pred.cpu().detach().numpy().reshape((-1,))
    y_true = y_true.cpu().detach().numpy().reshape((-1,))

    return(accuracy_score(y_true, y_pred))


def timeit(method):
    """
    Time decorator to print the execution time of a function.

    Use @timeit before your functions.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    None.
    
    """

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

def visualize_TSNE(embeddings, target, labels=None):
    """
    Allows to visualize vectors (e.g. embeddings from a neural network model) 
    in a 2d-space, using the t-SNE algorithm.
    Parameters
    ----------
    embeddings : a numpy ndarray of shape (n_samples, embd_size)
        The vectors to plot in the 2D graph.
    target : a numpy ndarray of shape (n_samples,)
        The targets associated with each embedding vector. Should be integers.

    Returns
    -------
    None.

    """

    tsne = TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=30)
    data = tsne.fit_transform(embeddings)
    #plt.figure(figsize=(12, 6))
    plt.title("TSNE visualization of the embeddings")
    scatter = plt.scatter(data[:,0],data[:,1],c=target)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)

    return

def compute_similarity(net):
    """
    Compute Jaccard similarity matrix from a bipartite graph. It allows to
    get a new similarity network with homogeneous nodes.

    Parameters
    ----------
    net : numpy ndarray of shape (N1,N2)
        The "adjacency matrix" of the bipartite graph. The similarity network
        will be computed with the set of nodes of size N1 (corresponding to
        the lines in this case).

    Returns
    -------
    M : numpy ndarray of shape (N1,N2)
      The similarity network.
      
    """
    M = 1 - pdist(net, metric='Jaccard')
    M = squareform(M)
    M = M + np.eye(*M.shape)
    M[np.isnan(M)] = 0.

    return(M)

def readnet(net_path):
    """
    Read an adjacency matrix in a tsv file.

    """
	return(np.genfromtxt(net_path,delimiter='\t'))

def extract_samples(K, tensor):
    """
    Extract K random samples from tensor.

    Parameters
    ----------
    K : int
        Wanted number of samples
    tensor : tensor of any size
        tensor from which extract the samples 
        (the first dimension will be considered as the sample dim)

    Returns
    -------
    samples : tensor of same type as the tensor parameter of shape (K,...)
        the K samples extracted from tensor

    """
    perm = torch.randperm(tensor.size(0))
    idx = perm[:K]
    samples = tensor[idx]
    
    return (samples)