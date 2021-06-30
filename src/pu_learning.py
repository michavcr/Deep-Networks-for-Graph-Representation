import torch
import torch.nn as nn
import torch.optim as optim

from math import sqrt
from sklearn.model_selection import KFold

from autoencoder import *
from utils import *
from random_graphs import *
from dgnr import *

def get_train_test_masks(P, train_size=0.8, test_balance=True):
    """
    Produce train, test masks. 

    Parameters
    ----------
    P : TYPE
        The drug-target network.
    train_size : TYPE, optional
        Train size ratio. The default is 0.8.
    test_balance : TYPE, optional
        whether test should be balanced between positive and "negative" samples
        or not. The default is True.

    Returns
    -------
    None.

    """
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

    if not test_balance:
        test = ~train
        pos_test = torch.logical_and(test, Ipos)
        neg_test = torch.logical_and(test, Ineg)
    else:
        test = ~train
        pos_test = torch.logical_and(test,Ipos)
        neg_test = torch.logical_and(test, Ineg)

        num_pos_test = torch.sum(pos_test)
        test_neg_rel_size = num_pos_test / torch.sum(neg_test)
        
        neg_test = neg_test * torch.rand(P.size())
        neg_test[train] = 1.
        neg_test[Ipos] = 1.
        
        neg_test = neg_test < test_neg_rel_size

    return(pos_train, neg_train, pos_test, neg_test)

class CustomMSELoss(nn.Module):
    """
    Custom MSE Loss with a weight < 1 on negative samples.
    """
    def __init__(self, alpha=0.2):
        super(CustomMSELoss, self).__init__()
        self.alpha=sqrt(alpha)

    def forward(self, inputs, targets):
        #comment out if your model contains a sigmoid or equivalent activation layer
        neg_mask = (targets == 0.)
        
        M = (targets-inputs)**2
        M[neg_mask] *= self.alpha

        loss_res = torch.mean(M)

        return loss_res

class PU_Learner(nn.Module):
    def __init__(self, k, Fd, Ft, X, Y, Nd, Nt, activation='identity', has_bias=False):
        """
        nn.Module to solve PU Learning problem:
            Given P, X, Y; find H (shape (Fd, k)), W (shape (k,Ft)) so that:
               P is approx by X*H*W*Y^T
            In pratice, we try to solve an optimization problem:
                H, W = argmin activation(||P - X*H*W*Y^T||^2_(positive samples)
                                        +alpha*||P - X*H*W*Y^T||^2_(negative samples))
                
        This class only task is to compute S = activation(X*H*W*Y^T + eventual bias)
        on chosen samples of X and Y.

        """
        super().__init__()
        self.k = k
        self.Fd = Fd
        self.Ft = Ft
        
        self.H = torch.nn.Parameter(torch.randn(Fd, k)*1/sqrt(k))
        self.W = torch.nn.Parameter(torch.randn(k, Ft)*1/sqrt(k))
        if not has_bias:
            self.b_x = torch.zeros(Nd)
            self.b_y = torch.zeros(Nt)
        else:
            self.b_x = torch.nn.Parameter(torch.randn(Nd))
            self.b_y = torch.nn.Parameter(torch.randn(Nt))

        self.X = (X - X.mean(0))/(X.std(0)+1e-7)
        self.Y = (Y - Y.mean(0))/(Y.std(0)+1e-7)

        activ_dict = {'sigmoid': torch.nn.Sigmoid, 'identity': torch.nn.Identity}
        self.activation = activ_dict[activation]()

    def forward(self, id_x, id_y):
    	dot = torch.einsum('ij,jk,kl,li->i', self.X[id_x], self.H, self.W, torch.transpose(self.Y[id_y], 1, 0))

    	return(self.activation(dot + self.b_x[id_x] + self.b_y[id_y]))

def pu_learning_base(k, x, y, P, pos_train, neg_train, pos_test, neg_test, n_epochs=100, batch_size=100, lr=1e-3, alpha=1.0, gamma=0., activation='identity'):
    """
    

    Parameters
    ----------
    k : int
        Common space dimension.
    x : numpy ndarray
        "Drugs" embeddings.
    y : numpy ndarray
        "Targets" embeddings.
    P : numpy ndarray
        "Drug-target" network.
    pos_train : bool torch Tensor
        Positive train samples mask.
    neg_train : bool torch Tensor
        Negative train samples mask.
    pos_test : bool torch Tensor
        Positive test samples mask.
    neg_test : bool torch Tensor
        Negative test samples mask.
    n_epochs : int, optional
        Number of epochs. The default is 100.
    batch_size : int, optional
        DESCRIPTION. The default is 100.
    lr : float, optional
        Learning rate. The default is 1e-5.
    train_size : float, optional
        Train set size ratio. The default is 0.8.
    alpha : float, optional
        Negative/unlabelled samples weight in CustomMSELoss. The default is 1.0.
    gamma : float, optional
        Weight decay (Adam's regularization). The default is 0..
    activation : string, optional
        Activation to apply to get a score matrix, either 'sigmoid' or 'identity'. 
        The default is 'identity'.

    Returns
    -------
    None.

    """
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

    cartesian_product = torch.Tensor([[i, j] for i in range(Nd) for j in range(Nt)]).long().to(device)

    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)

    P = torch.Tensor(P).to(device)

    train_mask = torch.logical_or(pos_train, neg_train)
    test_mask = torch.logical_or(pos_test, neg_test)
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

    model = PU_Learner(k, Fd, Ft, x, y, Nd, Nt, activation=activation).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=gamma)

    # mean-squared error loss
    criterion = CustomMSELoss(alpha=alpha)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs=inputs.to(device)
            
            id_x_batch, id_y_batch = inputs[:,0], inputs[:,1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(id_x_batch, id_y_batch)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        S = model.activation(torch.chain_matmul(model.X, model.H, model.W, torch.transpose(model.Y,0,1)) + model.b_x.unsqueeze(-1).expand(-1,Nt) + model.b_y.expand(Nd,-1))
        """print(S)
        print(model.H)
        print(model.W)
        print(model.b_x)
        print(model.b_y)"""
        auc = compute_auc(P[train_mask].clone(), S[train_mask].clone())
        acc = compute_accuracy(P[train_mask].clone(), S[train_mask].clone())

        print('[%d] loss: %.3f, auc: %.3f, acc: %.3f' %
                  (epoch + 1, running_loss / (batch_num), auc, acc))
        running_loss = 0.0

    print('Finished Training')
    print("Now computing Z=HW^T, then will compute S...")
    
    print(x.size(), model.H.size(), model.W.size(), y.size())
    S = model.activation(torch.chain_matmul(model.X, model.H, model.W, torch.transpose(model.Y,0,1))  + model.b_x.unsqueeze(-1).expand(-1,Nt) + model.b_y.expand(Nd,-1))
    
    return(S, model.H, model.W, model.b_x, model.b_y, train_mask, test_mask)

def pu_learning(k, x, y, P, n_epochs=100, batch_size=100, lr=1e-5, train_size=0.8, alpha=1.0, gamma=0., test_balance=True, activation='identity'):
    """
    PU Learning pipeline.

    Parameters
    ----------
    k : int
        Common space dimension.
    x : numpy ndarray
        "Drugs" embeddings.
    y : numpy ndarray
        "Targets" embeddings.
    P : numpy ndarray
        "Drug-target" network.
    n_epochs : int, optional
        Number of epochs. The default is 100.
    batch_size : int, optional
        DESCRIPTION. The default is 100.
    lr : float, optional
        Learning rate. The default is 1e-5.
    train_size : float, optional
        Train set size ratio. The default is 0.8.
    alpha : float, optional
        Negative/unlabelled samples weight in CustomMSELoss. The default is 1.0.
    gamma : float, optional
        Weight decay (Adam's regularization). The default is 0..
    test_balance : True, optional
        Whether test set should be balanced or not. The default is True.
    activation : string, optional
        Activation to apply to get a score matrix, either 'sigmoid' or 'identity'. 
        The default is 'identity'.

    Returns
    -------
    See pu_learning_base.

    """
    print("Spliting train and test sets...")
    pos_train, neg_train, pos_test, neg_test = get_train_test_masks(P, train_size=train_size, test_balance=test_balance)
    
    return(pu_learning_base(k, x, y, P, pos_train, neg_train, pos_test, neg_test, 
                       n_epochs=n_epochs, batch_size=batch_size, lr=lr, 
                       alpha=alpha, gamma=gamma, activation=activation))

#obsolete 
pu_learning_new = pu_learning

def cross_validate(k, x, y, P, N_folds, n_epochs=100, batch_size=100, lr=1e-5, train_size=0.8, alpha=1.0, gamma=0., activation='identity'):
    """
    Cross validation.

    Parameters
    ----------
    k : int
        Common space dimension.
    x : numpy ndarray
        "Drugs" embeddings.
    y : numpy ndarray
        "Targets" embeddings.
    P : numpy ndarray
         Drug - target network.
    N_folds : int
        Number of folds.
    n_epochs : int, optional
        Number of epochs. The default is 100.
    batch_size : int, optional
        Batch Size. The default is 100.
    lr : float, optional
        Learning rate. The default is 1e-5.
    train_size : float<1, optional
        Training set / total set ratio. The default is 0.8.
    alpha : float, optional
        Weights for negative samples in MSE Loss. The default is 1.0.
    gamma : float, optional
        Weight decay (Adam's regularization). The default is 0..
    activation : string, optional
        Activation to apply to get a score matrix, either 'sigmoid' or 'identity'. 
        The default is 'identity'.

    Returns
    -------
    Predictions weighted by auc, average ROC auc, average pr auc, average 
    accuracy.

    """
    pos_mask = (P==1)
    neg_mask = (P==0)
    
    N_pos = pos_mask.sum()
    N_neg = neg_mask.sum()
    
    N = min(N_pos, N_neg)
    
    pos_idx = pos_mask.nonzero()
    neg_idx = neg_mask.nonzero()
    
    pos_idx = extract_samples(N, pos_idx)
    neg_idx = extract_samples(N, neg_idx)
    
    kfold = KFold(n_splits=N_folds, shuffle=False)
    
    kfold_pos = list(kfold.split(pos_idx))
    kfold_neg = list(kfold.split(neg_idx))
    
    S_auc = 0
    S_acc = 0
    S_pr = 0
    
    S_sol = 0

    for fold in range(N_folds):
        print("Fold %d" % (fold+1))
        print("Preparing the masks...")
        pos_train_idx = pos_idx[kfold_pos[fold][0]]
        #print(pos_train_idx.max(0))
        pos_test_idx = pos_idx[kfold_pos[fold][1]]
        neg_train_idx = neg_idx[kfold_neg[fold][0]]
        neg_test_idx = neg_idx[kfold_neg[fold][1]]
        
        pos_train_mask = torch.zeros(pos_mask.size(),dtype=bool)
        pos_train_mask[pos_train_idx[:,0],pos_train_idx[:,1]] = True
        
        pos_test_mask = torch.zeros(pos_mask.size(),dtype=bool)
        pos_test_mask[pos_test_idx[:,0],pos_test_idx[:,1]] = True
        
        neg_train_mask = torch.zeros(neg_mask.size(),dtype=bool)
        neg_train_mask[neg_train_idx[:,0],neg_train_idx[:,1]] = True
        
        neg_test_mask = torch.zeros(neg_mask.size(),dtype=bool)
        neg_test_mask[neg_test_idx[:,0],neg_test_idx[:,1]] = True
        
        test_mask = torch.logical_or(pos_test_mask, neg_test_mask)
        
        print("Starting to learn...")
        S, H, W, b_x, b_y, _, _ = pu_learning_base(k, x, y, P, 
                                              pos_train_mask, neg_train_mask, pos_test_mask, neg_test_mask, 
                                              n_epochs=n_epochs, batch_size=batch_size, lr=lr, 
                                              alpha=alpha, gamma=gamma, activation=activation)
        
        print("Evaluating on test set...")
        auc, pr, acc, _ = eval_test_set(P, S, test_mask)
        
        S_auc += auc
        S_acc += acc
        S_pr += pr

        S_sol += S * auc

    return(S_sol/S_auc, S_auc/N_folds, S_pr/N_folds, S_acc/N_folds)

def eval_test_set(P, S, test):
    """
    Given a ground truth P and predictions S, compute metrics on test set.
    
    auc : ROC AUC
    pr : Precision recall AUC
    acc : Accuracy
    confusion : confusion matrix
    
    """
    print("Evaluation on the test set...")
    print("Test set statistics:")
    n_pos = int(P[test].sum().item())
    n_neg = int((1-P[test]).sum().item())

    print("Number of positive examples:", n_pos)
    print("Number of negative/unlabelled examples:", n_neg)
    
    auc = compute_auc(P[test],S[test])
    pr = compute_pr_auc(P[test], S[test])
    acc = compute_accuracy(P[test],S[test])
    confusion = compute_confusion_matrix(P[test], S[test])
    
    print("\nROC auc: %f" % auc)
    print("PR auc: %f" % pr)
    print("Accuracy: %f" % acc)
    print("Confusion matrix:")
    print(confusion)
    
    return(auc,pr,acc,confusion)