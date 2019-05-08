from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import numpy as np
import os

"""
override the dataset class for custom dataset
"""
f_1 = 'paper/ICLR_accepted.csv'
f_2 = 'paper/ICLR_rejected.csv'

class custom_dataset(Dataset):
    def __init__ (self, f_name, file_cnt):
        self.data = pd.read_csv(f_name)
        self.data = np.array(self.data)
        self.data = np.delete(self.data, 0, 1)# remove the first number column
        print(self.data.shape, self.data)
        if file_cnt == 0:
            self.labels = np.zeros((len(self.data), ), dtype = int)
        else:
            self.labels = np.ones((len(self.data), ), dtype = int)
        #print(self.labels)

    def __getitem__(self, index):
        title = self.data[index]
        label = self.labels[index]
        return title, label

    def __len__(self):
        return len(self.data)

def load_custom_dataset():
    custom_dataset_reject = custom_dataset(f_2, 0)
    custom_dataset_accept = custom_dataset(f_1, 1)
    for title, label in custom_dataset_reject:
        print(title, label)
    return 0

if __name__ == '__main__':
    load_custom_dataset()
