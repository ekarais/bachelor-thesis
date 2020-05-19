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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {'BATCH_SIZE': 64,          # number of data points in each batch
          'N_EPOCHS' : 1000,           # times to run the model on complete data
          'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 256,        # hidden dimension
          'LATENT_DIM' : 50,         # latent vector dimension
          'lr' : 1e-3,               # learning rate
          'train_size' : 0.8,
          'val_size' : 0.1,
          'test_size' : 0.1}

data_directory = "/home/ege/Documents/bthesis/data/onehot/neutral_64000/"
data = PGM(data_directory)

train_size = int(len(data)*params['train_size'])
validation_size = int(len(data)*params['val_size'])
test_size = int(len(data)*params['test_size'])

training_set, validation_set, test_set = random_split(data, [train_size, validation_size, test_size])

train_iterator = DataLoader(training_set, batch_size=params['BATCH_SIZE'], shuffle=True)
validation_iterator = DataLoader(validation_set, batch_size=params['BATCH_SIZE'])
test_iterator = DataLoader(test_set, batch_size=params['BATCH_SIZE'])

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
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
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
        self.out = nn.Linear(hidden_dim, output_dim)
        self.custom_softmax = mySoftmax()

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = self.custom_softmax(self.out(hidden))
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


# encoder
encoder = Encoder(params['INPUT_DIM'], params['HIDDEN_DIM'], params['LATENT_DIM'])

# decoder
decoder = Decoder(params['LATENT_DIM'], params['HIDDEN_DIM'], params['INPUT_DIM'])

# vae
model = VAE(encoder, decoder).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=params['lr'])


def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0
    r_loss = 0
    div_loss = 0

    for i, x in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        x = x.view(-1, 9 * 336).float()
        x = x.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var = model(x)

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')
        r_loss += recon_loss.item()

        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        div_loss += kl_loss.item()

        # total loss
        loss = recon_loss + kl_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return r_loss, div_loss, train_loss


def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    r_loss = 0
    div_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, x in enumerate(test_iterator):
            # reshape the data
            x = x.view(-1, 9 * 336).float()
            x = x.to(device)

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')
            r_loss += recon_loss.item()

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            div_loss += kl_loss.item()

            # total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    return r_loss, div_loss, test_loss

best_test_loss = float('inf')

for e in range(params['N_EPOCHS']):

    train_rloss, train_divloss, train_loss = train()
    test_rloss, test_divloss, test_loss = test()

    train_loss /= len(training_set)
    train_rloss /= len(training_set)
    train_divloss /= len(training_set)
    test_loss /= len(test_set)
    test_rloss /= len(test_set)
    test_divloss /= len(test_set)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
    print(f'Epoch {e}, Train_Recon Loss: {train_rloss:.2f}, Test_Recon Loss: {test_rloss:.2f}')
    print(f'Epoch {e}, Train_KL Loss: {train_divloss:.2f}, Test_KL Loss: {test_divloss:.2f}')
    print('')
    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break


# sample and generate a image
z = torch.randn(1, params['LATENT_DIM']).to(device)
reconstructed_RPM = model.dec(z)
RPM = reconstructed_RPM.view((9,336)).data
RPM = RPM.cpu().numpy()
np.save('/home/ege/Documents/bthesis/data/vae_generated/genesis3.npy', RPM)
print(z.shape)
print(RPM)
#plt.figure()
#plt.imshow(img, cmap='gray')
#plt.show()
