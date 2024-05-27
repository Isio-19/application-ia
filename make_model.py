from utils import script_error_print, get_files_name, is_float, is_int
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, LSTM, Linear, L1Loss, MSELoss
from torch.optim import Adam
from tqdm.auto import tqdm
from sys import argv, exit

import torch
import time
import torch.nn.init as init
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
        dropout = 0.1,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.nb_layers = nb_layers
        self.layer_size = layer_size
      
        super().__init__()
        # self.LSTM = LSTM(input_size=input_size, layer_size, nb_layers, batch_first=True, dropout=dropout)
        self.LSTM = LSTM(
            input_size=input_size, 
            hidden_size=layer_size, 
            num_layers=nb_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.linear = Linear(layer_size, output_size)

        # initialize the LSTM layers
        for name, param in self.LSTM.named_parameters():
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.xavier_normal_(param)

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

    files = get_files_name()
    if FIRST_FILES != -1:
        files = files[:FIRST_FILES]

    for type in ["train", "dev", "test"]:
        for file in files:
            file_content = pd.read_csv(f"split_data/{type}_data/{file}.csv")

            data[type].append(file_content[["P","T", "ET", "NDVI"]].to_numpy().astype("float32"))
            labels[type].append(file_content[["GWL"]].to_numpy().astype("float32"))

    train_dataset = GWL_Dataset(data["train"], labels["train"])
    dev_dataset   = GWL_Dataset(data["dev"]  , labels["dev"])
    test_dataset  = GWL_Dataset(data["test"] , labels["test"])

    return train_dataset, dev_dataset, test_dataset

def create_dataloader():
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    return train_dl, dev_dl, test_dl 

def train_loop(training_data, dev_data, model, path_to_save_model, path_to_save_plot, loss_fn, opt_fn, nb_epoch = 50):
    # init the best_dev_loss
    opt_fn.zero_grad()
    x, y = next(iter(dev_data))
    pred = model.forward(x)
    loss = loss_fn(pred, y)
    best_dev_loss =  loss.item()
    best_epoch = 0
    model.save(path_to_save_model)

    train_losses = []
    dev_losses = []

    iterable = range(nb_epoch)
    if not (QUIET):
        iterable = (bar := tqdm(range(nb_epoch)))

    for epoch in iterable:
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

        if best_dev_loss > dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            model.save(path_to_save_model)

        if not QUIET:
            bar.set_description(f"Current epoch: {epoch} | Best epoch {best_epoch} | Current dev loss {dev_loss:.4f} | Best dev loss {best_dev_loss:.4f}")
            bar.update()

    print(f"Best model found @ epoch {best_epoch} with loss of {best_dev_loss:.4f} on the dev dataset")

    plot(train_losses, dev_losses, path_to_save_plot)

def test_loop(model, path_best_model, test_data, loss_fn):
    test_loss = 0
    model.load(path_best_model)
    model.eval()

    for x, y in test_data:
        pred = model.forward(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item() / len(test_data)
    
    print(f"Loss of {test_loss:.4f} obtained with best model")

def plot(t_loss, d_loss, title):
    x = [i for i in range(len(t_loss))]
    plt.plot(x, t_loss, "-b", label="Training loss")
    plt.plot(x, d_loss, "-r", label="Validation loss")
    plt.legend(loc="upper left")
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title("Looses over time")
    annot_min(t_loss, d_loss)
    plt.savefig(title)
    plt.clf()

def annot_min(train, dev):
    index = np.argmin(dev)
    dev_min = np.min(dev)
    train_val = train[index] 
    
    dev_text =   f"Minimum value at  epoch {index}\ny={dev_min:.4f}"
    train_text = f"y={train_val:.4f}"
    
    plt.scatter([index], [dev_min],   color="red")
    plt.scatter([index], [train_val], color="blue")
    plt.annotate(dev_text,   xy=(index, dev_min),   xytext=(10, -15), textcoords="offset pixels")
    plt.annotate(train_text, xy=(index, train_val), xytext=(10, -10), textcoords="offset pixels")

# PARAMETERS
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

# DATA
FIRST_FILES = 20
MAKE_DATA = False
NA_THRESHHOLD = -1
FILL_DATA = False
SHUFFLE = False
BATCH_SIZE = 5

# MODEL
SEED = -1
QUIET = False

input_size = 4
output_size = 1
nb_layers = 5
layer_size = 6
dropout = 0
learning_rate = 0.001
nb_epoch = 1000

# parse the params
args = argv
for i, var in enumerate(args):
    try:
        match var:
            case "-ff" | "--first_files": 
                if not is_int(args, i+1):
                    raise Exception()
                FIRST_FILES = int(args[i+1])
            case "-s" | "--shuffle":
                SHUFFLE = True
            case "-bs" | "--batch_size":
                if not is_int(args, i+1):
                    raise Exception()
                BATCH_SIZE = int(args[i+1])
            case "-nl" | "--number_layer":
                if not is_int(args, i+1):
                    raise Exception()
                nb_layers = int(args[i+1])
            case "-ls" | "--layer_size":
                if not is_int(args, i+1):
                    raise Exception()
                layer_size = int(args[i+1])
            case "-d" | "--dropout":
                if not is_float(args, i+1):
                    raise Exception()
                dropout = float(args[i+1])
            case "-ne" | "--nb_epochs":
                if not is_int(args, i+1):
                    raise Exception()
                nb_epoch = int(args[i+1])
            case "-lr" | "--learning_rate":
                if not is_float(args, i+1):
                    raise Exception()
                learning_rate = float(args[i+1])
            case "--seed":
                if not is_int(args, i+1):
                    raise Exception()
                SEED = int(args[i+1])
            case "-q" | "--quiet":
                QUIET = True
            case "make_model.py":
                pass
            case _:
                if not(is_float(args, i)) and not(is_int(args, i)):
                    raise Exception()
                pass

    except Exception as e:
        # TODO: make a utils file to hide this mess
        script_error_print()
        exit()
    
path_to_best_model = f"model/seed_{SEED}_bs_{BATCH_SIZE}_nl_{nb_layers}_ls_{layer_size}_d_{dropout}_lr_{learning_rate}_e_{nb_epoch}.pt"
path_to_plot = f"model_test/seed_{SEED}_bs_{BATCH_SIZE}_nl_{nb_layers}_ls_{layer_size}_d_{dropout}_lr_{learning_rate}_e_{nb_epoch}.png"

if SEED != -1:
    torch.manual_seed(SEED)
    
train_ds, dev_ds, test_ds = read_files()
train_dl, dev_dl, test_dl = create_dataloader()

model = NeuralNetwork(
    input_size=input_size,
    output_size=output_size,
    nb_layers=nb_layers,
    layer_size=layer_size,
    dropout=dropout,
)
loss_fn = MSELoss()
# loss_fn = L1Loss()
opt_fn = Adam(model.parameters(), lr=learning_rate)

start_time = time.perf_counter_ns()
train_loop(train_dl, dev_dl, model, path_to_best_model, path_to_plot, loss_fn, opt_fn, nb_epoch)
end_time = time.perf_counter_ns()

print(f"Training time: {end_time-start_time} ns")

test_loop(model, path_to_best_model, test_dl, loss_fn)

