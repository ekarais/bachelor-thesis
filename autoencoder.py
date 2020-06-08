'''This code contains the implementation of simple VAE

Refer to the blog post here: https://graviraja.github.io/vanillavae/

Comments:

1. The reconstruction loss may not be a good loss for this problem. A better loss
   would be something that checks which rules are implemented in the VAE output.
   Of course, differentiability is an issue here. Assuming there is a workaround 
   for differentiability, an even better loss/scheme would be making the VAE also
   generate a one-hot encoding that specifies which rules are present in its output,
   and then compare that with the actual rules that are in its output. This loss 
   would have to be composed with a "pixel"-level reconstruction loss because other-
   wise, a global optimum is the all-zero output.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_utils import RegeneratedPGM as PGM
#from datagen import 
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
#warnings.filterwarnings("error")

class mySoftmax(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.softmax = nn.Softmax(dim=2)
        self.offsets = [0, 99, 198, 270, 336]
        self.dimensions = [[9,11],[9,11],[9,8],[6,11]]
    
    def forward(self, x):
        #print(x.shape)
        tbc = []
        #y = self.softmax(x[:,:50].view(x.shape[0], 5, 10)).view(x.shape[0], 50)
        #z = self.softmax(x[:,50:])
        #tbc.append(y)
        #tbc.append(z)
        
        for i in range(9):
            for j in range(4):
                aux_1 = x[:,i*336+self.offsets[j]:i*336+self.offsets[j+1]]
                aux_2 = aux_1.view(x.shape[0], self.dimensions[j][0], self.dimensions[j][1])
                aux_3 = self.softmax(aux_2)
                tbc.append(aux_3.view(x.shape[0], self.dimensions[j][0]*self.dimensions[j][1]))
        
        return torch.cat(tbc, dim=1)

class Autoencoder(nn.Module):
    def __init__(self, dim):
        ''' Autoencoder without compression to test the custom softmax layer.

        '''
        super().__init__()

        self.layer = nn.Linear(dim, dim)
        self.custom_softmax = mySoftmax()

    def forward(self, x):
        x = self.layer(x)
        x = self.custom_softmax(x)
        return x
        












