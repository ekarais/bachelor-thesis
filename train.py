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

2. I am not sure if everything is implemented correctly. When I set HIDDEN_DIM = 
   LATENT_DIM = 3024, instead of minimizing the training loss, all losses explode. 
   This might also be because the lr is too high for that setting.

3. Refactor rule checking code (first readability, then performance)

4. Figure out the correct way to pass x_sample and x_rules to the rule checking method

5. Implement adaptive LR
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_utils import RegeneratedPGM as PGM
from data_utils import rule_metrics
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import warnings

import sys, getopt, os
from vae import Encoder, Decoder, VAE
from autoencoder import Autoencoder
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {'BATCH_SIZE': 64,          # number of data points in each batch
          'N_EPOCHS' : 100,            # times to run the model on complete data
          'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 512,        # hidden dimension
          'LATENT_DIM' : 100,        # latent vector dimension
          'lr' : 1e-3,               # learning rate
          'train_size' : 0.8,
          'val_size' : 0.1,
          'test_size' : 0.1}

data_directory = "/home/ege/Documents/bthesis/data/onehot/neutral_256000/"
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

# optimizer
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

# debugging
torch.autograd.set_detect_anomaly(True)

def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0
    r_loss = 0
    div_loss = 0
    sens = 0
    spec = 0

    for i, (x, x_rules) in enumerate(train_iterator):
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

        # rule metrics
        sensitivity, specificity = rule_metrics(x_sample.clone().detach(), x_rules.clone().detach(), device)
        sens += sensitivity
        spec += specificity

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    #normalization
    sens /= (len(training_set)/params['BATCH_SIZE'])
    spec /= (len(training_set)/params['BATCH_SIZE'])

    return r_loss/len(training_set), div_loss/len(training_set), train_loss/len(training_set), sens, spec

def validate():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    r_loss = 0
    div_loss = 0
    sens = 0
    spec = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, x_rules) in enumerate(validation_iterator):
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

            # rule metrics
            sensitivity, specificity = rule_metrics(x_sample.clone().detach(), x_rules.clone().detach(), device)
            sens += sensitivity
            spec += specificity

    #normalization
    sens /= (len(validation_set)/params['BATCH_SIZE'])
    spec /= (len(validation_set)/params['BATCH_SIZE']) 

    return r_loss/len(validation_set), div_loss/len(validation_set), test_loss/len(validation_set), sens, spec

def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    r_loss = 0
    div_loss = 0
    sens = 0
    spec = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, x_rules) in enumerate(test_iterator):
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

            # rule metrics
            sensitivity, specificity = rule_metrics(x_sample.clone().detach(), x_rules.clone().detach(), device)
            sens += sensitivity
            spec += specificity

    #normalization
    sens /= (len(test_set)/params['BATCH_SIZE'])
    spec /= (len(test_set)/params['BATCH_SIZE'])

    return r_loss/len(test_set), div_loss/len(test_set), test_loss/len(test_set), sens, spec
    

def main(argv):
    #warnings.filterwarnings("error") #W&B ResourceWarning when --log is not set
    log = False
    
    opts, _ = getopt.getopt(argv, "l", ["log"])
    for opt, _ in opts:
        if opt in ('-l', '--log'):
            log = True

    if log:
        wandb.init(project="bachelor-thesis", entity='ege', config=params)
        wandb.run.save()
        wandb.watch(model)

    best_val_loss = float('inf')

    for e in range(params['N_EPOCHS']):
        
        train_rloss, train_divloss, train_loss, train_sens, train_spec = train()
        val_rloss, val_divloss, val_loss, val_sens, val_spec = validate()
        test_rloss, test_divloss, test_loss, test_sens, test_spec = test()
        
        if log:
            wandb.log({"Training Loss": train_loss,
                    "Training Reconstruction Loss": train_rloss,
                    "Training KL-Loss": train_divloss,
                    "Training Sensitivity": train_sens,
                    "Training Specificity": train_spec,
                    "Validation Loss": val_loss,
                    "Validation Reconstruction Loss": val_rloss,
                    "Validation KL-Loss": val_divloss,
                    "Validation Sensitivity": val_sens,
                    "Validation Specificity": val_spec,
                    "Test Loss": test_loss,
                    "Test Reconstruction Loss": test_rloss,
                    "Test KL-Loss": test_divloss,
                    "Test Sensitivity": test_sens,
                    "Test Specificity": test_spec})

        print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Test Loss: {test_loss:.2f}')
        
        #Early Stopping
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            patience_counter = 1
        else:
            patience_counter += 1

        if patience_counter > 3:
            break

    model_name = str(datetime.now().strftime("%d_%m_%Y_%H:%M:%S")) if not log else wandb.run.name    
    torch.save(model.state_dict(), models_directory + model_name + '.pth')

if __name__ == "__main__":
    main(sys.argv[1:])




