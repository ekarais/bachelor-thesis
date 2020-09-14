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

data_directory = "/home/ege/Documents/bthesis/data/onehot/neutral_25600_no_union_1_rule_augmented/"
models_directory = "/home/ege/Documents/bthesis/models/"

data = PGM(data_directory)

train_size = int(len(data)*params['train_size'])
validation_size = int(len(data)*params['val_size'])
test_size = int(len(data)*params['test_size'])
training_set, validation_set, test_set = random_split(data, [train_size, validation_size, test_size])

train_iterator = DataLoader(training_set, batch_size=params['BATCH_SIZE'], shuffle=True)
validation_iterator = DataLoader(validation_set, batch_size=params['BATCH_SIZE'])
test_iterator = DataLoader(test_set, batch_size=params['BATCH_SIZE'])

#class weights
size_weights = torch.from_numpy(np.asarray([1.6802, 25.5716, 22.6549, 22.6326, 25.6856, 25.3744, 26.1818, 24.4327, 26.4220, 24.9081, 23.8509])).float().to(device)
color_weights = torch.from_numpy(np.asarray([1.6802, 22.5440, 25.4305, 27.4613, 28.4094, 28.5502, 25.6285, 22.9482, 22.9711, 20.6082, 25.0980])).float().to(device)
type_weights = torch.from_numpy(np.asarray([1.6802, 17.0667, 16.3404, 18.2567, 17.5878, 17.1429, 17.3233, 17.4413])).float().to(device)
lcolor_weights = torch.from_numpy(np.asarray([2.0240, 20.9264, 21.1570, 19.4430, 19.9222, 20.8130, 20.0261, 19.0099, 21.0123, 17.9230, 18.0919])).float().to(device)
exist_weights = torch.from_numpy(np.asarray([9.1262, 9.2714, 8.9168, 8.8998, 8.9254, 9.0203, 8.8240, 9.0906, 8.9425])).float().to(device)

# model
model = GeneratorV3().to(device)

# optimizer, scheduler, loss
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)
criterion_1 = torch.nn.CrossEntropyLoss(weight=size_weights, reduction='sum').to(device)
criterion_2 = torch.nn.CrossEntropyLoss(weight=color_weights, reduction='sum').to(device)
criterion_3 = torch.nn.CrossEntropyLoss(weight=type_weights, reduction='sum').to(device)
criterion_4 = torch.nn.CrossEntropyLoss(weight=lcolor_weights, reduction='sum').to(device)
criterion_5 = torch.nn.BCEWithLogitsLoss(weight=exist_weights, reduction='sum').to(device)

model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("The model has {} trainable parameters.".format(model_trainable_params))

# debugging
#torch.autograd.set_detect_anomaly(True)




def train():
    # set the train mode
    model.train()

    # loss of the epoch
    loss_array = torch.zeros(33)
    train_loss = 0
    correct = 0
    total = 0
    first = True
    
    for data, rules in train_iterator:
        # reshape the data into [batch_size, 784]
        
        x = torch.from_numpy(np.concatenate((data[:,6*345:8*345], rules), axis=1)).float().to(device)
        labels = data[:,8*345:].float().to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        logits = model(x)
        
        loss = 0.0
        #compute loss
        for i in range(9):
            #loss_array[i] = criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion_1(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))

        for i in range(9):
            #loss_array[i] = criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion_2(logits[:,99+i*11:99+(i+1)*11], torch.argmax(labels[:,99+i*11:99+(i+1)*11], dim=1))

        for i in range(9):
            #loss_array[18+i] = criterion(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))
            loss = loss + criterion_3(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))

        for i in range(6):
            #loss_array[27+i] = criterion(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))
            loss = loss + criterion_4(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))

        loss = loss + criterion_5(logits[:,-9:], labels[:,-9:]) #existence loss

        #loss = torch.sum(loss_array)

        # backward pass
        loss.backward()
        
        train_loss += loss.item()
        
        # update the weights
        optimizer.step()

        complete_RPM = data.clone().to(device)
        complete_RPM[:,8*345:] = logits

        corr, mod_corr, tot, _ = rule_metrics(complete_RPM.clone().detach(), rules.clone().detach(), device=device)
        del complete_RPM
        correct += corr.item()
        
        total += tot

    accuracy = correct/total
    mod_acc = mod_corr/total
    return train_loss/len(training_set), accuracy, mod_acc

def validate():
    # set the train mode
    model.eval()

    # loss of the epoch
    val_loss = 0
    correct = 0
    total = 0

    for data, rules in validation_iterator:
        # reshape the data into [batch_size, 784]
        x = torch.from_numpy(np.concatenate((data[:,6*345:8*345], rules), axis=1)).float().to(device)
        labels = data[:,8*345:].float().to(device)

        # forward pass
        logits = model(x)

        loss = 0.0
        #compute loss
        for i in range(9):
            #loss_array[i] = criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion_1(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))

        for i in range(9):
            #loss_array[i] = criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion_2(logits[:,99+i*11:99+(i+1)*11], torch.argmax(labels[:,99+i*11:99+(i+1)*11], dim=1))

        for i in range(9):
            #loss_array[18+i] = criterion(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))
            loss = loss + criterion_3(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))

        for i in range(6):
            #loss_array[27+i] = criterion(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))
            loss = loss + criterion_4(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))

        loss = loss + criterion_5(logits[:,-9:], labels[:,-9:]) #existence loss

        val_loss += loss.item()

        complete_RPM = data.clone().to(device)
        complete_RPM[:,8*345:] = logits

        corr, mod_corr, tot,_ = rule_metrics(complete_RPM, rules, device=device)
        del complete_RPM
        correct += corr.item()
        total += tot

    accuracy = correct/total
    mod_acc = mod_corr/total
    return val_loss/len(validation_set), accuracy, mod_acc

def test(save_preds=False):
    # set the train mode
    model.eval()

    # loss of the epoch
    test_loss = 0
    correct = 0
    total = 0

    for c, (data, rules) in enumerate(test_iterator):
        # reshape the data into [batch_size, 784]
        x = torch.from_numpy(np.concatenate((data[:,6*345:8*345], rules), axis=1)).float().to(device)
        labels = data[:,8*345:].float().to(device)

        # forward pass
        logits = model(x)

        loss = 0.0

        #compute loss
        for i in range(9):
            #loss_array[i] = criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion_1(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))

        for i in range(9):
            #loss_array[i] = criterion(logits[:,i*11:(i+1)*11], torch.argmax(labels[:,i*11:(i+1)*11], dim=1))
            loss = loss + criterion_2(logits[:,99+i*11:99+(i+1)*11], torch.argmax(labels[:,99+i*11:99+(i+1)*11], dim=1))

        for i in range(9):
            #loss_array[18+i] = criterion(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))
            loss = loss + criterion_3(logits[:,198+i*8:198+(i+1)*8], torch.argmax(labels[:,198+i*8:198+(i+1)*8], dim=1))

        for i in range(6):
            #loss_array[27+i] = criterion(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))
            loss = loss + criterion_4(logits[:,270+i*11:270+(i+1)*11], torch.argmax(labels[:,270+i*11:270+(i+1)*11], dim=1))

        loss = loss + criterion_5(logits[:,-9:], labels[:,-9:]) #existence loss

        test_loss += loss.item()

        complete_RPM = data.clone().to(device)
        complete_RPM[:,8*345:] = logits

        if save_preds and c == 0:
            corr, mod_corr, tot, tbrendered = rule_metrics(complete_RPM, rules, device=device)
            tbrendered[0] = tbrendered[0].reshape(-1,9,99)
            tbrendered[1] = tbrendered[1].reshape(-1,9,99)
            tbrendered[2] = tbrendered[2].reshape(-1,9,72)
            tbrendered[3] = tbrendered[3].reshape(-1,9,66)
            rpm_vectors = torch.cat((tbrendered[0], tbrendered[1], tbrendered[2], tbrendered[3]), dim=2).reshape(-1,9*336).cpu().numpy()
            for i in range(64):
                RPM = to_RPM(rpm_vectors[i], onehot=True)
                render(RPM, name="test"+str(i), directory="./generated/")
            #render stuff

        else:
            corr, mod_corr, tot, _ = rule_metrics(complete_RPM, rules, device=device)
        del complete_RPM
        correct += corr.item()
        total += tot

    accuracy = correct/total
    mod_acc = mod_corr/total
    return test_loss/len(test_set), accuracy, mod_acc


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
    
    
    train_loss, train_acc, train_mod_acc = train()
    val_loss, val_acc, val_mod_acc = validate()
    test_loss, test_acc, test_mod_acc = test()
    
    scheduler.step(val_loss)
    
    if log:
        
        with warnings.catch_warnings():    
            warnings.simplefilter("ignore")
            wandb.log({
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Training Moderate Accuracy": train_mod_acc,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Validation Moderate Accuracy": val_mod_acc,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Test Moderate Accuracy": test_mod_acc
            })
        

    #test(save_preds=True)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Test Loss: {test_loss:.2f}')
    model_name = "generator-" + str(datetime.now().strftime("%d_%m_%Y_%H:%M:%S")) if not log else "generator-" + wandb.run.name    
    torch.save(model.state_dict(), models_directory + model_name + '.pth')

test(save_preds=True)