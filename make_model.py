from make_data import get_files_name
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module

import numpy as np
import torch.functional as F
import pandas as pd

class GWL_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(np.array(labels))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class NeuralNetwork(Module):
    def __init__():
        return

def read_files():
    data = {
        "train": [],
        "dev":   [],
        "test":  [],
    }

    labels = {
        "train": [],
        "dev":   [],
        "test":  [],
    }

    for type in ["train", "dev", "test"]:
        # for file in get_files_name():
        for file in ["10001555", "10002941"]:
            file_content = pd.read_csv(f"split_data/{type}_data/{file}.csv")

            # data[type].append(torch.from_numpy(file_content[["P","T", "ET", "NDVI"]].to_numpy()))
            # labels[type].append(torch.from_numpy(file_content[["GWL"]].to_numpy()))
            data[type].append(file_content[["P","T", "ET", "NDVI"]].to_numpy())
            labels[type].append(file_content[["GWL"]].to_numpy())

    train_dataset = GWL_Dataset(data["train"], labels["train"])
    dev_dataset   = GWL_Dataset(data["dev"]  , labels["dev"])
    test_dataset  = GWL_Dataset(data["test"] , labels["test"])

    return train_dataset, dev_dataset, test_dataset

def create_dataloader(batch_size: int = 4, shuffle: bool = True):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    dev_dl   = DataLoader(dev_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

    return train_dl, dev_dl, test_dl 

train_ds, dev_ds, test_ds = read_files()
train_dl, dev_dl, test_dl = create_dataloader(batch_size=1, shuffle=True)
    