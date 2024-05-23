from make_data import get_files_name, main
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, LSTM, Linear, L1Loss, MSELoss
from torch.optim import Adam
from tqdm.auto import tqdm
from sys import float_info, argv

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GWL_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(np.array(labels))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# TODO: test with embeddings to model the region
# TODO: test with dropout for increased perfomance
class NeuralNetwork(Module):
    def __init__(
        self,
        input_size = 4,
        output_size = 1,
        nb_layers = 2,
        layer_size = 6,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.nb_layers = nb_layers
        self.layer_size = layer_size
      
        super().__init__()
        self.LSTM = LSTM(input_size, layer_size, nb_layers, batch_first=True)
        self.linear = Linear(layer_size, output_size)

    # TODO: ask what is the use of returning ht 
    def forward(self, x, ht = None):
        if ht is None:
            ht = torch.zeros(self.nb_layers, x.size(0), self.layer_size).to(x.device)
            ct = torch.zeros(self.nb_layers, x.size(0), self.layer_size).to(x.device)

        out, (ht, ct) = self.LSTM(x, (ht, ct))
        out = self.linear(out)

        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

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
        for file in get_files_name():
            file_content = pd.read_csv(f"split_data/{type}_data/{file}.csv")

            data[type].append(file_content[["P","T", "ET", "NDVI"]].to_numpy().astype("float32"))
            labels[type].append(file_content[["GWL"]].to_numpy().astype("float32"))

    train_dataset = GWL_Dataset(data["train"], labels["train"])
    dev_dataset   = GWL_Dataset(data["dev"]  , labels["dev"])
    test_dataset  = GWL_Dataset(data["test"] , labels["test"])

    return train_dataset, dev_dataset, test_dataset

def create_dataloader(batch_size: int = 4, shuffle: bool = True):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    dev_dl   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=shuffle)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=shuffle)

    return train_dl, dev_dl, test_dl 

def train_loop(training_data, dev_data, model, path_to_save, loss_fn, opt_fn, nb_epoch = 50):
    best_dev_loss = float_info.max
    best_epoch = 0

    train_losses = []
    dev_losses = []

    for epoch in (bar := tqdm(range(nb_epoch))):
        model.train()
        train_loss = 0
        for x, y in training_data:
            opt_fn.zero_grad()
            pred = model.forward(x)
            loss = loss_fn(pred, y)
            train_loss += loss.item() / len(training_data)
            loss.backward()
            opt_fn.step()
        train_losses.append(train_loss)

        model.eval()
        dev_loss = 0
        for x, y in dev_data:
            pred = model.forward(x)
            loss = loss_fn(pred, y)
            dev_loss += loss.item() / len(dev_data) 
        dev_losses.append(dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            model.save(path_to_save)

        bar.set_description(f"Current epoch: {epoch} | Best epoch {best_epoch} | Current dev loss {dev_loss:.4f} | Best dev loss {best_dev_loss:.4f}")
        bar.update()

    print(f"Best model found @ epoch {best_epoch} with loss of {best_dev_loss:.4f} on the dev dataset")

    return train_losses, dev_losses

def test_loop(model, path_best_model, test_data, loss_fn):
    test_loss = 0
    model.load(path_best_model)
    model.eval()

    for x, y in test_data:
        pred = model.forward(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item() / len(test_data)
    
    print(f"Loss of {test_loss:.4f} obtained with best model")

def plot(t_loss, d_loss):
    x = [i for i in range(len(t_loss))]
    plt.plot(x, t_loss, "-b", label="Training loss")
    plt.plot(x, d_loss, "-r", label="Validation loss")
    plt.legend(loc="best")
    plt.xlabel("Epoches")
    plt.ylabel("Loss value")
    plt.savefig("plot.png")

# PARAMETERS
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

# DATA
batch_size = 16
shuffle = True
make_data = False
na_threshhold = -1
fill_data = False

args = argv
for i, var in enumerate(args):
    if var == "-md":
        make_data = True

    if var == "na" and (len(args) > i+1) and (args[i+1].replace(".", "", 1).isdigit()):
        na_threshhold = float(args[i+1])

    if var == "-fill":
        fill_data = True

# MODEL
input_size = 4
output_size = 1
nb_layers = 5
layer_size = 2
lr = 0.00005
nb_epoch = 200
path_to_best_model = "model/best_model.pt"

if make_data:
    main(na_threshhold, fill_data)

train_ds, dev_ds, test_ds = read_files()
train_dl, dev_dl, test_dl = create_dataloader(batch_size=batch_size, shuffle=False)

model = NeuralNetwork(
    input_size=input_size,
    output_size=output_size,
    nb_layers=nb_layers,
    layer_size=layer_size
)
loss_fn = MSELoss()
# loss_fn = L1Loss()
opt_fn = Adam(model.parameters(), lr=lr)

t_losses, d_losses = train_loop(train_dl, dev_dl, model, path_to_best_model, loss_fn, opt_fn, nb_epoch)
test_loop(model, path_to_best_model, test_dl, loss_fn)

plot(t_losses, d_losses)