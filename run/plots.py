from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader as GraphLoader
import wandb
import pandas as pd
import numpy as np

from data_utils import RegeneratedPGM as PGM
from data_utils import rule_metrics, GraphPGM
from architectures.gnns import GraphNeuralNet, GATNet, GATNetV2, GATNetV3

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'BATCH_SIZE': 2560,          # number of data points in each batch
          'N_EPOCHS' : 0,            # times to run the model on complete data
          'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 512,        # hidden dimension
          'LATENT_DIM' : 100,        # latent vector dimension
          'lr' : 1e-3,               # learning rate
          'train_size' : 0.8,
          'val_size' : 0.1,
          'test_size' : 0.1}

data_directory = "/home/ege/Documents/bthesis/data/onehot/neutral_256000/"
models_directory = "/home/ege/Documents/bthesis/models/"
working_root_dir = "/home/ege/Documents/bthesis/data/proto/"


log = False

rule_names = ['SSP',
              'SSX',
              'SSO',
              'SSA',
              'SSU',
              'SCP',
              'SCX',
              'SCO',
              'SCA',
              'SCU',
              'SNP',
              'SNU',
              'SPX',
              'SPO',
              'SPA',
              'STP',
              'STX',
              'STO',
              'STA',
              'STU',
              'LCP',
              'LCX',
              'LCO',
              'LCA',
              'LCU',
              'LPX',
              'LPO',
              'LPA',
              'LPU',]

baselines = [0.8933, 0.8273, 0.8711, 0.8731, 0.9895, 0.9216, 0.8321, 0.8765, 0.8773,
        0.9862, 0.8917, 0.8493, 0.9123, 0.9373, 0.9443, 0.9019, 0.8274, 0.8757,
        0.8711, 0.9883, 0.9168, 0.8392, 0.8515, 0.8644, 0.9862, 0.8442, 0.8829,
        0.8861, 0.9992]
gats = [0.9212, 0.8408, 0.8956, 0.8900, 0.9817, 0.9224, 0.8353, 0.8983, 0.8802,
        0.9839, 0.9514, 0.9551, 0.9323, 0.9542, 0.9621, 0.9283, 0.8314, 0.8966,
        0.8697, 0.9856, 0.9546, 0.8509, 0.8830, 0.8707, 0.9867, 0.8180, 0.8934,
        0.8914, 0.9982]
        
'''
width=0.35
plt.figure(figsize=(20, 10))
plt.bar(torch.arange(29) - width/2, baselines, width, color='chocolate', label='Baseline')
plt.bar(torch.arange(29) + width/2, gats, width, color='b', label='GATNet w/ 2 heads')
plt.xticks(torch.arange(29), rule_names)
plt.title("Per Rule Accuracy of Proposed Classifiers")
plt.axhline(y=0.95, color='g', linestyle='-')
plt.legend()
plt.ylabel('accuracy')
plt.savefig("per_rule_comp.png", dpi=300)
plt.close()

file = pd.read_csv("heads.csv")
npf = file.to_numpy()[:200].astype(float)


plt.figure(figsize=(8,4))
plt.plot(npf[:,0], npf[:,2], 'IndianRed', label="1 head") #1
plt.plot(npf[:,0], npf[:,3], 'g', label="2 heads") #2
plt.plot(npf[:,0], npf[:,1], 'b', label="3 heads") #3
plt.legend()
plt.ylabel('hard accuracy')
plt.xlabel('epoch')
plt.title('Hard Accuracy over Epoch for GATNet with various #heads')
plt.ylim(0.0, 0.2)
#plt.yticks([0.0, 0.5, 0.1, 0.15, 0.2])
plt.savefig("heads.png", dpi=300)
plt.close()
'''
'''
file = pd.read_csv("one_rule_vs_hard.csv")
npf = file.to_numpy()[:100].astype(float)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

plt.figure(figsize=(8,4))
plt.plot(npf[:,0], npf[:,1], 'ForestGreen', label="D1".translate(SUB)) #1
plt.plot(npf[:,0], npf[:,2], 'Orange', label="D2".translate(SUB)) #2

plt.legend()
plt.ylabel('relaxed accuracy')
plt.xlabel('epoch')
plt.title('Relaxed Accuracy over Epoch for Generator on ' + "D1".translate(SUB) + ' and ' +  "D2".translate(SUB))
#plt.ylim(0.0, 0.2)
#plt.yticks([0.0, 0.5, 0.1, 0.15, 0.2])
plt.savefig("one_vs_hard_acc.png", dpi=300)
plt.close()
'''
file = pd.read_csv("one_rule_vs_hard_loss.csv")
npf = file.to_numpy()[:100].astype(float)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

plt.figure(figsize=(8,4))
plt.plot(npf[:,0], npf[:,1], 'ForestGreen', label="D1".translate(SUB)) #1
plt.plot(npf[:,0], npf[:,2], 'Orange', label="D2".translate(SUB)) #2

plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Loss over Epoch for Generator on ' + "D1".translate(SUB) + ' and ' +  "D2".translate(SUB))
#plt.ylim(0.0, 0.2)
#plt.yticks([0.0, 0.5, 0.1, 0.15, 0.2])
plt.savefig("one_vs_hard_loss.png", dpi=300)
plt.close()