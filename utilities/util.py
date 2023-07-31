from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.distributions import Normal, Gumbel

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Workspace(nn.Module, ABC):
    def __init__(self, n_site, exp_name = None):
        super(Workspace, self).__init__()
        self.n_site = n_site
        self.nodes = []
        self.model = None
        self.exp_name = exp_name

    def forward(self, X):
        pass

def make_one_hot(labels):
    label_set = {}
    n = -1
    for l in labels:
        if l not in label_set:
            n = n + 1
            label_set[l] = n
    num = len(label_set)
    y = np.zeros((len(labels), num))
    for i,l in enumerate(labels):
        j = label_set[l]
        y[i,j] =1
    return y

def norm1(A):
    deg_inv = torch.sum(A, dim=1).pow(-1.0)
    deg_inv[deg_inv == float('inf')] = 0
    C = torch.matmul(torch.diag(deg_inv),A)
    return C


# A is at least 2D, where the first 2D corresponds to a square
# matrix. Each square matrix is symmetrized.
def symmetrize(A):
    A = A.to(device)
    n = A.shape[0]
    triu_ones = torch.triu(torch.ones(n, n), diagonal=1).to(device)
    ndim = len(A.shape)
    if ndim == 2:
        mask = triu_ones
    else:
        shape = [n] * 2 + [1] * (ndim - 2)
        mask = triu_ones.view(shape).expand_as(A)

    A = A * mask
    A = A + torch.transpose(A, dim0=0, dim1=1)
    return A

def sinkhorn_transform(M, gamma, iter = 10):
    # C = torch.exp(-M/gamma)
    C = torch.pow(M,2)
    for i in range(iter):
        C = norm1(C)
        C = norm1(C.transpose(0,1)).transpose(0,1)
    return C
