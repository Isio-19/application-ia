import torch.utils
from torch.nn import Module, GRU, Softmax, MSELoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import math
import sys


class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.gru1 = GRU(5, 6, 15)
        self.soft1 = Softmax(dim=1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.soft1(x)
        return x

    def my_save(self, path):
        torch.save(self.state_dict(), path)

class DataSpliter():
    def __init__(
        self, 
        file_name, 
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        ignore_nan: bool = True
    ):
        file_content = pd.read_csv(file_name)
        if ignore_nan:
            file_content = file_content.dropna(subset=["GWL"])

        # only keep the month for the model
        file_content["date"] = [int(date) for date in pd.to_datetime(file_content["date"]).dt.strftime("%m")]        

        first_index = math.ceil(len(file_content) * train_ratio)
        second_index = math.ceil(len(file_content) * (train_ratio + dev_ratio))

        self.train_labels = file_content["GWL"][:first_index]
        self.dev_labels   = file_content["GWL"][first_index:second_index]
        self.test_labels  = file_content["GWL"][second_index:]

        self.train_data = file_content[["date", "P", "T", "ET", "NDVI"]][:first_index]
        self.dev_data   = file_content[["date", "P", "T", "ET", "NDVI"]][first_index:second_index]
        self.test_data  = file_content[["date", "P", "T", "ET", "NDVI"]][second_index:]

class GWL_DataSet(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values.tolist())
        self.labels = torch.tensor(labels.values.tolist())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        return self.data[id], self.labels[id]
    
def train_loop(train_data, dev_data, model, loss_fn, opt):
    nb_epochs=50
    best_dev_loss=sys.float_info.max
    best_epoch=0

    for epoch in (bar := tqdm(range(nb_epochs))):
        model.train()
        train_loss = 0
        for x, y in train_data:
            opt.zero_grad(set_to_none=True)
            pred=model.forward(x)
            loss = loss_fn(pred, y)
            train_loss += loss.item() / len(train_data)
            loss.backward()
            opt.step()
        
        model.eval()
        dev_loss = 0
        for x, y in dev_data:
            pred=model.forward(x)
            loss = loss_fn(pred, y)
            dev_loss += loss.item() / len(dev_data)

        if best_dev_loss>dev_loss:
            best_dev_loss=dev_loss
            best_epoch=epoch
            model.my_save('models/best_model_MLP.pt')

        bar.clear()
        bar.set_description(
        f"Epoch: {epoch} - Train Loss {train_loss:.4f} - Dev Loss {dev_loss:.4f} - Best Dev Loss {best_dev_loss:.4f}"
        )
        bar.update()

    print("Best model at epoch", best_epoch, "with a loss of", float(best_dev_loss), "on the dev dataset")

split_data = DataSpliter("Data/normalized/10000078.csv", ignore_nan=True)

train_data = GWL_DataSet(split_data.train_data, split_data.train_labels)
dev_data   = GWL_DataSet(split_data.dev_data, split_data.dev_labels)
test_data  = GWL_DataSet(split_data.test_data, split_data.test_labels)

train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False)
dev_data_loader   = DataLoader(dev_data, batch_size=1, shuffle=False)
test_data_loader  = DataLoader(test_data, batch_size=1, shuffle=False)

model = NeuralNetwork()
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=0.1)

train_loop(train_data_loader, dev_data_loader, model, loss_fn, optimizer)