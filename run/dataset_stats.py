'''
1. Refactor rule checking code (first readability, then performance)

2. Figure out the correct way to pass x_sample and x_rules to the rule checking method

3. Implement adaptive LR
'''

import sys
import getopt
from datetime import datetime
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

from data_utils import RegeneratedPGM as PGM
from data_utils import NoUnionPGM
from data_utils import rule_metrics
from datagen import to_RPM, render
from architectures.fcns import Generator, SimpleGenerator, SimplestGenerator, GeneratorV3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {'BATCH_SIZE': 64,          # number of data points in each batch
          'N_EPOCHS' : 100,            # times to run the model on complete data
          'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 512,        # hidden dimension
          'LATENT_DIM' : 100,        # latent vector dimension
          'lr' : 1e-2,               # learning rate
          'train_size' : 0.8,
          'val_size' : 0.1,
          'test_size' : 0.1}

data_directory = "/home/ege/Documents/bthesis/data/onehot/neutral_2560_no_union_1_rule_augmented/"
models_directory = "/home/ege/Documents/bthesis/models/"

data = PGM(data_directory)

data_iterator = DataLoader(data, batch_size=params['BATCH_SIZE'])


size_stats = torch.zeros(11)
color_stats = torch.zeros(11)
type_stats = torch.zeros(8)
lcolor_stats = torch.zeros(11)
exist_stats = torch.zeros(9)

for data, rules in data_iterator:
    
    labels = data[:,8*345:]
    
    for i in range(9):
        size_stats = size_stats + torch.sum(labels[:,i*11:(i+1)*11], dim=0)
        color_stats = color_stats + torch.sum(labels[:,99+i*11:99+(i+1)*11], dim=0)
        type_stats = type_stats + torch.sum(labels[:,198+i*8:198+(i+1)*8], dim=0)

    for i in range(6):
        lcolor_stats = lcolor_stats + torch.sum(labels[:,270+i*11:270+(i+1)*11], dim=0)

    exist_stats = exist_stats + torch.sum(labels[:,-9:], dim=0)


size_stats /= torch.sum(size_stats)
color_stats /= torch.sum(color_stats)
type_stats /= torch.sum(type_stats)
lcolor_stats /= torch.sum(lcolor_stats)
exist_stats /= torch.sum(exist_stats)

print(1/size_stats)
print(1/color_stats)
print(1/type_stats)
print(1/lcolor_stats)
print(1/exist_stats)

