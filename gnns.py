#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch_geometric.data import DataLoader as GraphLoader, Data
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, avg_pool

from data_utils import RegeneratedPGM as PGM
from data_utils import rule_metrics, GraphPGM
from vae import Encoder, Decoder, VAE
from autoencoder import Autoencoder

from tqdm import tqdm
from datetime import datetime
import warnings
import sys, getopt, os


class GraphNeuralNet(torch.nn.Module):
    def __init__(self):
        super(GraphNeuralNet, self).__init__()
        self.conv1 = GraphConv(336, 100)
        self.conv2 = GraphConv(100, 100)
        self.conv3 = GraphConv(100, 100)
        self.conv4 = GraphConv(100, 100)
        self.conv5 = GraphConv(100, 100)
        
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 29)
        
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        #GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear4(x)
        
        return x
    
class RowNet(torch.nn.Module):
    def __init__(self, device):
        super(RowNet, self).__init__()
        self.b1conv1 = GCNConv(336, 100)
        self.b1conv2 = GCNConv(100, 100)
        self.b1conv3 = GCNConv(100, 100)
        self.row_cluster = torch.LongTensor([1,1,1,2,2,2,3,3]).to(device)
        self.b2conv1 = GCNConv(100,100)
        self.b2conv2 = GCNConv(100,100)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 29)
        
    def forward(self, data):
        data.x = data.x.float()
        
        #GNN
        data.x = F.relu(self.b1conv1(data.x, data.edge_index))
        data.x = F.relu(self.b1conv2(data.x, data.edge_index))
        data.x = F.relu(self.b1conv3(data.x, data.edge_index))
        
        cluster = torch.add(3*data.batch, self.row_cluster.repeat(data.batch[-1]+1))
        #print(cluster[:24])
        #print(data.num_nodes)
        data = avg_pool(cluster, data)
        #print(data.edge_index)
        data.x = F.relu(self.b2conv1(data.x, data.edge_index))
        data.x = F.relu(self.b2conv2(data.x, data.edge_index))
        
        x = global_mean_pool(data.x, data.batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x





