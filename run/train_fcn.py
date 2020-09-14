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

import sys
import getopt
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from data_utils import RegeneratedPGM as PGM
from data_utils import rule_metrics
from architectures.fcns import FCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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

# FCN
model = FCN().to(device)

# optimizer, scheduler, loss
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.6]).to(device), reduction='sum')

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
    
    

    for x, labels in train_iterator:
        # reshape the data into [batch_size, 784]
        x = x.view(-1, 9 * 336).float()
        x = x.to(device)
        labels = labels.float().to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        logits = model(x)
        
        # compute loss
        loss = criterion(logits, labels)

        # get preds from logits
        preds = torch.argmax(torch.cat(((1-logits).view(params['BATCH_SIZE'], 29, 1), logits.view(params['BATCH_SIZE'], 29, 1)), dim=2), dim=2)


        # metrics
        ## accuracy
        correct += (preds == labels).sum().item()
        total += torch.numel(labels)
        
        ##hard accuracy
        hard_correct += (preds == labels).prod(dim=1).sum().item()
        hard_total += preds.size()[0] #batch size

        ## sensitivity
        true_positives += (torch.mul(preds, labels) == 1).sum().item() 
        positives += (labels == 1).sum().item()
        
        ## specificity
        true_negatives += (torch.mul(1-preds, 1-labels) == 1).sum().item()
        negatives += (labels == 0).sum().item()

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    accuracy = correct/total
    hard_acc = hard_correct/hard_total
    sensitivity = true_positives/positives
    specificity = true_negatives/negatives

    #normalization
    train_loss /= len(training_set)
    
    return train_loss, accuracy, hard_acc, sensitivity, specificity

def validate():

    model.eval()

    # loss of the epoch
    val_loss = 0
    
    correct = 0
    total = 0 
    
    hard_correct = 0
    hard_total = 0

    true_positives = 0
    positives = 0
    
    true_negatives = 0
    negatives = 0

    with torch.no_grad():
        for x, labels in validation_iterator:
            
            x = x.view(-1, 9 * 336).float()
            x = x.to(device)
            labels = labels.float().to(device)

            # forward pass
            logits = model(x)
            
            # reconstruction loss
            loss = criterion(logits, labels)
            
            # get preds from logits
            preds = torch.argmax(torch.cat(((1-logits).view(params['BATCH_SIZE'], 29, 1), logits.view(params['BATCH_SIZE'], 29, 1)), dim=2), dim=2)
            
            # metrics
            ## accuracy
            correct += (preds == labels).sum().item()
            total += torch.numel(labels)

            ##hard accuracy
            hard_correct += (preds == labels).prod(dim=1).sum().item()
            hard_total += preds.size()[0] #batch size

            ## sensitivity
            true_positives += (torch.mul(preds, labels) == 1).sum().item() 
            positives += (labels == 1).sum().item()

            ## specificity
            true_negatives += (torch.mul(1-preds, 1-labels) == 1).sum().item()
            negatives += (labels == 0).sum().item()

            val_loss += loss.item()

            

    accuracy = correct/total
    hard_acc = hard_correct/hard_total
    sensitivity = true_positives/positives
    specificity = true_negatives/negatives

    #normalization
    val_loss /= len(validation_set)
    
    return val_loss, accuracy, hard_acc, sensitivity, specificity

def test():

    model.eval()

    # loss of the epoch
    test_loss = 0
    
    correct_vec = torch.zeros([29], device='cuda')
    correct = 0
    total = 0 
    
    hard_correct = 0
    hard_total = 0

    true_positives = 0
    positives = 0
    
    true_negatives = 0
    negatives = 0

    #diagnostics for the 100% bug
    rule_frequencies = torch.zeros([29], device='cuda')

    with torch.no_grad():
        for x, labels in test_iterator:

            x = x.view(-1, 9 * 336).float()
            x = x.to(device)
            labels = labels.float().to(device)

            # forward pass
            logits = model(x)

            # reconstruction loss
            loss = criterion(logits, labels)
            
            # get preds from logits
            preds = torch.argmax(torch.cat(((1-logits).view(params['BATCH_SIZE'], 29, 1), logits.view(params['BATCH_SIZE'], 29, 1)), dim=2), dim=2)
            
            # metrics
            ## accuracy
            correct_vec += (preds == labels).sum(dim=0)
            correct += (preds == labels).sum().item()
            total += torch.numel(labels)

            ##hard accuracy
            hard_correct += (preds == labels).prod(dim=1).sum().item()
            hard_total += preds.size()[0] #batch size

            ## sensitivity
            true_positives += (torch.mul(preds, labels) == 1).sum().item() 
            positives += (labels == 1).sum().item()

            ## specificity
            true_negatives += (torch.mul(1-preds, 1-labels) == 1).sum().item()
            negatives += (labels == 0).sum().item()

            test_loss += loss.item()

            ## diagnostics
            rule_frequencies += labels.sum(dim=0)

            

    accuracy_per_rule = correct_vec/(total/29)
    accuracy = correct/total
    hard_acc = hard_correct/hard_total
    sensitivity = true_positives/positives
    specificity = true_negatives/negatives
    rule_frequencies /= (total/29)
    
    #normalization
    test_loss /= len(test_set)
    
    return test_loss, accuracy_per_rule, accuracy, hard_acc, sensitivity, specificity

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

if log:
    wandb.init(project="bachelor-thesis-fcn", entity='ege', config=params)
    wandb.run.save()
    wandb.watch(model, log='all')


for epoch in range(params['N_EPOCHS']):
    
    
    train_loss, train_acc, train_hacc, train_sens, train_spec = train()
    val_loss, val_acc, val_hacc, val_sens, val_spec = validate()
    test_loss, test_acc_per_rule, test_acc, test_hacc, test_sens, test_spec = test()
    
    scheduler.step(val_loss)
    
    if log:
        

        plt.figure(figsize=(20, 10))
        plt.bar(torch.arange(29), test_acc_per_rule.cpu())
        plt.xticks(torch.arange(29), rule_names)
        plt.axhline(y=0.95, color='g', linestyle='-')
        plt.ylabel('accuracy')
        with warnings.catch_warnings():    
            warnings.simplefilter("ignore")
            wandb.log({
                "Accuracy per rule": [wandb.Image(plt, caption="Accuracy per rule")],
                "Training Loss": train_loss,
                "Training Accuracy": train_acc,
                "Training Hard Accuracy": train_hacc,
                "Training Sensitivity": train_sens,
                "Training Specificity": train_spec,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Validation Hard Accuracy": val_hacc,
                "Validation Sensitivity": val_sens,
                "Validation Specificity": val_spec,
                "Test Loss": test_loss,
                "Test Accuracy": test_acc,
                "Test Hard Accuracy": test_hacc,
                "Test Sensitivity": test_sens,
                "Test Specificity": test_spec
            })
        
        plt.close()
        
    print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Test Loss: {test_loss:.2f}')
    model_name = "gnn-" + str(datetime.now().strftime("%d_%m_%Y_%H:%M:%S")) if not log else "gnn-" + wandb.run.name    
    torch.save(model.state_dict(), models_directory + model_name + '.pth')
