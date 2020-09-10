# torch
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
import torch.nn as nn
from torch import optim
# torch geometric
from torch_geometric.datasets import BitcoinOTC, Planetoid
from torch_geometric.data import Data, InMemoryDataset, DataLoader, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler
from torch_geometric.data import GraphSAINTNodeSampler
from torch_geometric.nn import SignedGCN, GCNConv, ChebConv, MessagePassing, TopKPooling, global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, get_laplacian, degree
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
# scikit-learn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.testing import ignore_warnings
# graphs
import networkx as nx
import matplotlib.pyplot as plt
# base
import ssl
import math
import random
from itertools import combinations, permutations
import copy
import os.path as osp
import yaml, pathlib
from tqdm.auto import tqdm, trange
import datetime as dt
# arrays
import numpy as np
import pandas as pd
# testing
from pdb import set_trace
# local
from dataset import *
from linkpred import *
#from db_connect import connect

## set random seeds
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


def load_dataset(dataset_name='NDSSL'):
    if dataset_name == 'NDSSL':
        dataset = NDSSLDataset('data/NDSSL data/')
        # dataset.process()
        data = dataset[0]
    if dataset_name == 'Cora':
        ssl._create_default_https_context = ssl._create_unverified_context
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data = dataset[0]
    return dataset

def train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.
    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)
    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    num_nodes = data.num_nodes
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)), min(n_v + n_t, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def train_test_split_big(data, val_ratio=0.1, test_ratio=0.1):
    row, col = data.edge_index
    # data.edge_index = None
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    
    # Positive edges.
    #perm = torch.randperm(row.size(0))
    #row, col = row[perm], col[perm]
    
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_pos_edge_index = to_undirected(data.val_pos_edge_index)
    
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_pos_edge_index = to_undirected(data.test_pos_edge_index)
    
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    return data


def save_graph(graph,file_name):
    # initialze Figure
    plt.figure(num=None, figsize=(40, 40), dpi=1000)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig
    
    
def model_params(model):
    """simple function to return the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]
