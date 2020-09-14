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
from architectures.fcns import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {'BATCH_SIZE': 64,          # number of data points in each batch
          'N_EPOCHS' : 100,            # times to run the model on complete data
          'INPUT_DIM' : 9 * 336,     # size of each input
          'HIDDEN_DIM' : 512,        # hidden dimension
          'LATENT_DIM' : 100,        # latent vector dimension
          'lr' : 1e-1,               # learning rate
          'train_size' : 0.8,
          'val_size' : 0.1,
          'test_size' : 0.1}

data_directory = "/home/ege/Documents/bthesis/data/graded/neutral_2560_no_union/"
models_directory = "/home/ege/Documents/bthesis/models/"

data = PGM(data_directory)

train_size = int(len(data)*params['train_size'])
validation_size = int(len(data)*params['val_size'])
test_size = int(len(data)*params['test_size'])
training_set, validation_set, test_set = random_split(data, [train_size, validation_size, test_size])

train_iterator = DataLoader(training_set, batch_size=params['BATCH_SIZE'], shuffle=True)
validation_iterator = DataLoader(validation_set, batch_size=params['BATCH_SIZE'])
test_iterator = DataLoader(test_set, batch_size=params['BATCH_SIZE'])

# model
model = Generator().to(device)

# optimizer, scheduler, loss
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)
criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("The model has {} trainable parameters.".format(model_trainable_params))

# debugging
torch.autograd.set_detect_anomaly(True)



def train():
    # set the train mode
    model.train()

    # loss of the epoch
    
    train_loss = 0
    correct = 0
    total = 0
    first = True

    for data, rules in train_iterator:
        # reshape the data into [batch_size, 784]
        
        x = torch.from_numpy(np.concatenate((data[:,6*33:8*33], rules), axis=1)).to(device)
        labels = data[:,8*33:].to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        logits = model(x)
        
        loss = 0.0
        #compute loss
        for i in range(18):
            #loss = loss + criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion(logits[:,i*11:(i+1)*11], labels[:,i])

        for i in range(9):
            #loss = loss + criterion(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))
            loss = loss + criterion(logits[:,198+i*8:198+(i+1)*8], labels[:,18+i])
        
        for i in range(6):
            #loss = loss + criterion(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))
            loss = loss + criterion(logits[:,270+i*11:270+(i+1)*11], labels[:,27+i])
        #loss = torch.sum(loss_array)

        # backward pass
        loss.backward()
        
        train_loss += loss.item()
        
        # update the weights
        optimizer.step()

        #complete_RPM = data.clone().to(device)
        #complete_RPM[:,8*336:] = logits

        #corr, tot = rule_metrics(complete_RPM.clone().detach(), rules.clone().detach(), device=device)
        #del complete_RPM
        #correct += corr.item()
        #total += tot

    #accuracy = correct/total
    accuracy = 1.0
    
    return train_loss/len(training_set), accuracy

def validate():
    # set the train mode
    model.eval()

    # loss of the epoch
    val_loss = 0
    correct = 0
    total = 0

    for data, rules in validation_iterator:
        # reshape the data into [batch_size, 784]
        x = torch.from_numpy(np.concatenate((data[:,6*33:8*33], rules), axis=1)).to(device)
        labels = data[:,8*33:].to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        logits = model(x)
        
        loss = 0.0
        #compute loss
        for i in range(18):
            #loss = loss + criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion(logits[:,i*11:(i+1)*11], labels[:,i])

        for i in range(9):
            #loss = loss + criterion(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))
            loss = loss + criterion(logits[:,198+i*8:198+(i+1)*8], labels[:,18+i])
        
        for i in range(6):
            #loss = loss + criterion(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))
            loss = loss + criterion(logits[:,270+i*11:270+(i+1)*11], labels[:,27+i])
        #loss = torch.sum(loss_array)

        val_loss += loss.item()

        #complete_RPM = data.clone().to(device)
        #complete_RPM[:,8*336:] = logits

        #corr, tot = rule_metrics(complete_RPM, rules, device=device)
        #del complete_RPM
        #correct += corr.item()
        #total += tot

    #accuracy = correct/total
    accuracy = 1
    return val_loss/len(validation_set), accuracy

def test():
    # set the train mode
    model.eval()

    # loss of the epoch
    test_loss = 0
    correct = 0
    total = 0

    for data, rules in test_iterator:
        # reshape the data into [batch_size, 784]
        x = torch.from_numpy(np.concatenate((data[:,6*33:8*33], rules), axis=1)).to(device)
        labels = data[:,8*33:].to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        logits = model(x)
        
        loss = 0.0
        #compute loss
        for i in range(18):
            #loss = loss + criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion(logits[:,i*11:(i+1)*11], labels[:,i])

        for i in range(9):
            #loss = loss + criterion(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))
            loss = loss + criterion(logits[:,198+i*8:198+(i+1)*8], labels[:,18+i])
        
        for i in range(6):
            #loss = loss + criterion(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))
            loss = loss + criterion(logits[:,270+i*11:270+(i+1)*11], labels[:,27+i])
        #loss = torch.sum(loss_array)

        test_loss += loss.item()

        #complete_RPM = data.clone().to(device)
        #complete_RPM[:,8*336:] = logits

        #corr, tot = rule_metrics(complete_RPM, rules, device=device)
        #del complete_RPM
        #correct += corr.item()
        #total += tot

    #accuracy = correct/total
    accuracy = 1
    return test_loss/len(test_set), accuracy


log = True

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
              'LPU']

if log:
    wandb.init(project="bachelor-thesis-generate", entity='ege', config=params)
    wandb.run.save()
    wandb.watch(model, log='all')


for epoch in range(params['N_EPOCHS']):
    
    
    train_loss, train_acc = train()
    val_loss, val_acc = validate()
    test_loss, test_acc = test()
    
    scheduler.step(val_loss)
    
    if log:
        
        with warnings.catch_warnings():    
            warnings.simplefilter("ignore")
            wandb.log({
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
            })
        
        
    print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Test Loss: {test_loss:.2f}')
    model_name = "generator-" + str(datetime.now().strftime("%d_%m_%Y_%H:%M:%S")) if not log else "generator-" + wandb.run.name    
    torch.save(model.state_dict(), models_directory + model_name + '.pth')

