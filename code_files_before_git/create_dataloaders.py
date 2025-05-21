import numpy as np
import torch as nn
import matplotlib.pyplot as plt
import CFG
import random
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

random.seed(CFG.seed0)


class Synthetic_Dataset(Dataset):
    def __init__(self, num_samples, data_dict):
        self.num_samples = num_samples
        self.data_dict

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_dict['Ws'][idx], self.data_dict['Ps'][idx]


num_samples = 4

batch_size = 2

# dataset = Synthetic_Dataset(num_samples, data_dict=d)
#
# indices = list(range(num_samples))
#
# train_indices, val_indices = train_test_split(indices, train_size=0.8, random_state=CFG.seed0)
#
# train_dataset = Subset(dataset, train_indices)
# val_dataset = Subset(dataset, val_indices)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

a=5



