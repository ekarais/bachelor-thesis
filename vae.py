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
    
    def __init__(self, dim=2):

        super().__init__()

        self.softmax = nn.Softmax(dim=dim)
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

class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        x = F.relu(self.linear(x))
        x = F.relu(self.hidden(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(x)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(x)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.custom_softmax = mySoftmax()

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        x = F.relu(self.linear(x))
        x = F.relu(self.hidden(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = self.custom_softmax(self.out(x))
        # predicted is of shape [batch_size, output_dim]
        #predicted = torch.sigmoid(self.out(hidden))

        return predicted

class VAE(nn.Module):
    def __init__(self, enc, dec):
        ''' This the VAE, which takes a encoder and decoder.

        '''
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var












