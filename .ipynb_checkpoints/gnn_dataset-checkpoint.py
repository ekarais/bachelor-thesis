import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import sys

class GraphPGM(InMemoryDataset):
    
    def __init__(self, root, size, transform=None, pre_transform=None):
        self.size = size
        self.data_dir = "/home/ege/Documents/bthesis/data/onehot/neutral_" + str(self.size) + "/"
        super(GraphPGM, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        assert os.path.isdir(self.data_dir)
        
    @property
    def processed_file_names(self):
        return ['data.pt']
            
    def process(self):
        data_list = []
        
        edge_index = torch.empty(2, 56, dtype=torch.long)
        c = 0
        for i in range(8):
            for j in range(8):
                if i != j:
                    edge_index[:,c] = torch.tensor([i,j], dtype=torch.long)
                    c += 1
        
        #Since I don't initialize data.pos, the GNN does not know the positions of the panels ==> invariance! (might also lead to poor performance)
        for filename in os.listdir(self.data_dir):
            RPM = np.load(self.data_dir + filename)
            x = torch.from_numpy(RPM[:8*336].reshape(8,336))
            y = torch.from_numpy(RPM[-29:].reshape(1,29))
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = MyPGM(root="/home/ege/Documents/bthesis/data/proto/", size=256000)


loader = DataLoader(dataset, batch_size=128, shuffle=True)





