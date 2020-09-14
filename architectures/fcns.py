import torch 
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        input_dim = 9*336
        latent_dim = 200
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, 29)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

class EmbeddingGenerator(nn.Module):

    def __init__(self):
        #super(EmbeddingGenerator, self).__init__()
        input_dim = 2*33 + 29
        latent_dim = 200
        output_dim = 336
        #self.emb = nn.Embedding(11,2)
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, output_dim)
        self.softmax = mySoftmax()
        
    def forward(self, x):
        #x = self.emb(x)
        #x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #return self.softmax(self.fc5(x))
        x = self.fc5(x)
        return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        input_dim = 2*336 + 29
        latent_dim = 200
        output_dim = 336
        
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, output_dim)
        self.softmax = mySoftmax()
        
    def forward(self, x):
        
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class GeneratorV2(nn.Module):

    def __init__(self):
        super(GeneratorV2, self).__init__()
        input_dim = 2*336 + 29
        latent_dim = 400
        output_dim = 336
        
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, output_dim)
        self.softmax = mySoftmax()
        
    def forward(self, x):
        
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class GeneratorV3(nn.Module):

    def __init__(self):
        super(GeneratorV3, self).__init__()
        input_dim = 2*345 + 29
        latent_dim = 100
        output_dim = 345
        
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, output_dim)
        self.softmax = mySoftmax()
        
    def forward(self, x):
        
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class SimplestGenerator(nn.Module):

    def __init__(self):
        super(SimplestGenerator, self).__init__()
        input_dim = 2*336 + 29
        latent_dim = 200
        output_dim = 336
        
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)
        
        
    def forward(self, x):
    
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class SimpleGenerator(nn.Module):

    def __init__(self):
        super(SimpleGenerator, self).__init__()
        input_dim = 2*336 + 29
        latent_dim = 200
        output_dim = 336
        
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)
        
        
    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class mySoftmax(nn.Module):
    
    def __init__(self, dim=2):

        super().__init__()

        self.softmax = nn.Softmax(dim=dim)
        self.offsets = [0, 99, 198, 270, 336]
        self.dimensions = [[9, 11], [9, 11], [9, 8], [6, 11]]
    
    def forward(self, x):
        
        tbc = []
        
        for j in range(4):
            aux_1 = x[:, self.offsets[j]:self.offsets[j+1]]
            aux_2 = aux_1.view(x.shape[0], self.dimensions[j][0], self.dimensions[j][1])
            aux_3 = self.softmax(aux_2)
            tbc.append(aux_3.view(x.shape[0], self.dimensions[j][0]*self.dimensions[j][1]))
        
        return torch.cat(tbc, dim=1)
