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
import wandb
import warnings
#warnings.filterwarnings("error")
from vae import Encoder, Decoder, VAE

wandb.init(project="bachelor-thesis", entity='ege')
wandb.run.save()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {'BATCH_SIZE': 64,          # number of data points in each batch
          'N_EPOCHS' : 1,            # times to run the model on complete data
          'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 128,        # hidden dimension
          'LATENT_DIM' : 25,         # latent vector dimension
          'lr' : 1e-3,               # learning rate
          'train_size' : 0.8,
          'val_size' : 0.1,
          'test_size' : 0.1}

data_directory = "/home/ege/Documents/bthesis/data/onehot/neutral_64000/"
models_directory = "/home/ege/Documents/bthesis/models/"

data = PGM(data_directory)

train_size = int(len(data)*params['train_size'])
validation_size = int(len(data)*params['val_size'])
test_size = int(len(data)*params['test_size'])

training_set, validation_set, test_set = random_split(data, [train_size, validation_size, test_size])

train_iterator = DataLoader(training_set, batch_size=params['BATCH_SIZE'], shuffle=True)
validation_iterator = DataLoader(validation_set, batch_size=params['BATCH_SIZE'])
test_iterator = DataLoader(test_set, batch_size=params['BATCH_SIZE'])



# encoder
encoder = Encoder(params['INPUT_DIM'], params['HIDDEN_DIM'], params['LATENT_DIM'])

# decoder
decoder = Decoder(params['LATENT_DIM'], params['HIDDEN_DIM'], params['INPUT_DIM'])

# vae
model = VAE(encoder, decoder).to(device)
wandb.watch(model)

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


def test(iterator):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    r_loss = 0
    div_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, x in enumerate(iterator):
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

best_val_loss = float('inf')

for e in range(params['N_EPOCHS']):

    train_rloss, train_divloss, train_loss = train()
    train_loss /= len(training_set)
    train_rloss /= len(training_set)
    train_divloss /= len(training_set)

    val_rloss, val_divloss, val_loss = test(validation_iterator)
    val_loss /= len(test_set)
    val_rloss /= len(test_set)
    val_divloss /= len(test_set)

    test_rloss, test_divloss, test_loss = test(test_iterator)
    test_loss /= len(test_set)
    test_rloss /= len(test_set)
    test_divloss /= len(test_set)
    
    wandb.log({"Training Loss": train_loss,
               "Training Reconstruction Loss": train_rloss,
               "Training KL-Loss": train_divloss,
               "Validation Loss": val_loss,
               "Validation Reconstruction Loss": val_rloss,
               "Validation KL-Loss": val_divloss,
               "Test Loss": test_loss,
               "Test Reconstruction Loss": test_rloss,
               "Test KL-Loss": test_divloss})

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Test Loss: {test_loss:.2f}')
    
    
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break


torch.save(model.state_dict(), models_directory + wandb.run.name + '.pth')









