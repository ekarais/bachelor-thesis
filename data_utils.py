import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class RegeneratedPGM(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = os.listdir(self.data_path)
        self.length = len(self.data)
        print("Initialized dataset with ", self.length, " samples.")

    def __getitem__(self, index):
        return np.load(self.data_path + self.data[index]).astype(int)[:-29]

    def __len__(self):
        return self.length