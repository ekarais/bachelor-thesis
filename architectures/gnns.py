import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv, global_mean_pool, avg_pool

class GraphNeuralNet(torch.nn.Module):
    def __init__(self):
        super(GraphNeuralNet, self).__init__()
        self.conv1 = GCNConv(336, 100, improved=True)
        self.conv2 = GCNConv(100, 100, improved=True)
        self.conv3 = GCNConv(100, 100, improved=True)
        self.conv4 = GCNConv(100, 100, improved=True)
        self.conv5 = GCNConv(100, 100, improved=True)
        
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 29)
        
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        #GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

class GraphNeuralNet_Large(torch.nn.Module):
    def __init__(self):
        super(GraphNeuralNet_Large, self).__init__()
        self.conv1 = GCNConv(336, 220, improved=False)
        self.conv2 = GCNConv(220, 220, improved=False)
        self.conv3 = GCNConv(220, 220, improved=False)
        self.conv4 = GCNConv(220, 220, improved=False)
        self.conv5 = GCNConv(220, 100, improved=False)
        
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 29)
        
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        #GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

class GATNet(torch.nn.Module):

    #1 head: (336,200) -> (200,200)
    #2 head: (336,100) -> (200,100)
    #3 head: (336,75) -> (225,75)
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(336, 100, heads=2)
        self.conv2 = GATConv(200, 100, heads=2)
        self.conv3 = GATConv(200, 100, heads=2)
        self.conv4 = GATConv(200, 100, heads=2)
        self.conv5 = GATConv(200, 100, heads=2)
        
        self.linear1 = nn.Linear(200, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 29)
        
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        #GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

class GATNetV2(torch.nn.Module):

    #1 head: (336,200) -> (200,200)
    #2 head: (336,100) -> (200,100)
    #3 head: (336,75) -> (225,75)
    def __init__(self):
        super(GATNetV2, self).__init__()
        self.conv1 = GATConv(336, 125, heads=2)
        self.conv2 = GATConv(250, 125, heads=2)
        self.conv3 = GATConv(250, 125, heads=2)
        self.conv4 = GATConv(250, 125, heads=2)
        
        self.linear1 = nn.Linear(250, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 29)
        
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        #GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_mean_pool(x, batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        
        return x

class GATNetV3(torch.nn.Module):

    #1 head: (336,125) -> (200,200)
    #2 head: (336,125) -> (250,75) -> (150,50) -> (100,30)
    #3 head: (336,75) -> (225,75)
    def __init__(self):
        super(GATNetV3, self).__init__()
        self.conv1 = GATConv(336, 125, heads=2)
        self.conv2 = GATConv(250, 75, heads=2)
        self.conv3 = GATConv(150, 50, heads=2)
        self.conv4 = GATConv(100, 30, heads=2)
        
        self.linear1 = nn.Linear(480, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 29)
        
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        #GNN
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        #x = global_mean_pool(x, batch)
        x = x.view(-1, 480)
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        
        return x
    
    
class RowNet(torch.nn.Module):
    def __init__(self, device):
        super(RowNet, self).__init__()
        self.b1conv1 = GCNConv(336, 100)
        self.b1conv2 = GCNConv(100, 100)
        self.b1conv3 = GCNConv(100, 100)
        self.row_cluster = torch.LongTensor([1, 1, 1, 2, 2, 2, 3, 3]).to(device)
        self.b2conv1 = GCNConv(100,100)
        self.b2conv2 = GCNConv(100,100)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 29)
        
    def forward(self, data):
        data.x = data.x.float()
        
        #GNN
        data.x = F.relu(self.b1conv1(data.x, data.edge_index))
        data.x = F.relu(self.b1conv2(data.x, data.edge_index))
        data.x = F.relu(self.b1conv3(data.x, data.edge_index))
        
        cluster = torch.add(3*data.batch, self.row_cluster.repeat(data.batch[-1]+1))
        #print(cluster[:24])
        #print(data.num_nodes)
        data = avg_pool(cluster, data)
        #print(data.edge_index)
        data.x = F.relu(self.b2conv1(data.x, data.edge_index))
        data.x = F.relu(self.b2conv2(data.x, data.edge_index))
        
        x = global_mean_pool(data.x, data.batch)
        
        #MLP
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x





