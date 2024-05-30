from utils import script_error_print, get_files_name
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, LSTM, Linear, L1Loss, MSELoss
from torch.optim import Adam
from tqdm.auto import trange
from sys import argv, exit

import torch
import os
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

    data_column = []
    label_column = []
    match MODEL_TYPE:
        case "4-1":
            data_column = ["P", "T", "ET", "NDVI"]
            label_column = ["GWL"]
        case "4-6":
            data_column = ["P", "T", "ET", "NDVI"]
            label_column = ["GWL+1", "GWL+2", "GWL+3", "GWL+4", "GWL+5", "GWL+6"]
        case "5-6":
            data_column = ["GWL", "P", "T", "ET", "NDVI"]
            label_column = ["GWL+1", "GWL+2", "GWL+3", "GWL+4", "GWL+5", "GWL+6"]

    for type in ["train", "dev", "test"]:
        for file in files:
            file_content = pd.read_csv(f"data/split_data/{type}_data/{file}.csv")

            data[type].append(file_content[data_column].to_numpy().astype("float32"))
            labels[type].append(file_content[label_column].to_numpy().astype("float32"))

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

    bar = trange(nb_epoch, leave=False, unit=" epoch")

    for epoch in bar:
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

        bar.set_description(f"Current epoch: {epoch} | Best epoch {best_epoch} | Current dev loss {dev_loss:.4f} | Best dev loss {best_dev_loss:.4f}")

    if not QUIET:
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
        
    if not QUIET: 
        print(f"Loss of {test_loss:.4f} obtained with best model")
        
    return test_loss

def plot(train_loss, dev_loss, test_loss, save_path):
    x = [i for i in range(len(train_loss))]
    index = np.argmin(dev_loss)
    d_min = np.min(dev_loss)
    t_val = train_loss[index]

    plt.plot(x, dev_loss, "-r", label="Validation loss")
    plt.scatter([index], [d_min], color="magenta", label=f"Epoch: {index}, value: {d_min:.4f}")
    plt.plot(x, train_loss, "-b", label="Training loss")
    plt.scatter([index], [t_val], color="cyan", label=f"Value: {t_val:.4f}")
    plt.plot([], [], " ", label=f"Testing loss: {test_loss:.4f}")

    plt.legend(loc="upper left")
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title("Looses over time")
    plt.savefig(save_path)
    plt.clf()

# PARAMETERS
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

# DATA
FIRST_FILES = -1
SHUFFLE = False
BATCH_SIZE = 5

# MODEL
SEED = -1
QUIET = False
NAME = False

input_size = -1
output_size = -1
nb_layers = 5       
layer_size = 6
dropout = 0.1
learning_rate = 0.001
nb_epoch = 1000
LOSS_FUNCTION = "l1"

# parse the params
args = iter(argv)
for var in args:
    try:
        match var:
            case "-ff" | "--first_files": 
                var = next(args)
                if not var.isdigit():
                    raise Exception(f"Expected an integer after -ff arg, not {var}")
                FIRST_FILES = int(var)
            case "-s" | "--shuffle":
                SHUFFLE = True
            case "-bs" | "--batch_size":
                var = next(args)
                if not var.isdigit():
                    raise Exception(f"Expected an integer after -bs arg, not {var}")
                BATCH_SIZE = int(var)
            case "-nl" | "--number_layer":
                var = next(args)
                if not var.isdigit():
                    raise Exception(f"Expected an integer after -nl arg, not {var}")
                nb_layers = int(var)
            case "-ls" | "--layer_size":
                var = next(args)
                if not var.isdigit():
                    raise Exception(f"Expected an integer after -ls arg, not {var}")
                layer_size = int(var)
            case "-d" | "--dropout":
                var = next(args)
                if not var.replace(".", "", 1).isdigit():
                    raise Exception(f"Excepted a float after the -d arg, not {var}")
                dropout = float(var)
            case "-ne" | "--nb_epochs":
                var = next(args)
                if not var.isdigit():
                    raise Exception(f"Expected an integer after -ne arg, not {var}")
                nb_epoch = int(var)
            case "-lr" | "--learning_rate":
                var = next(args)
                if not var.replace(".", "", 1).isdigit():
                    raise Exception(f"Excepted a float after the -lr arg, not {var}")
                learning_rate = float(var)
            case "--seed":
                var = next(args)
                if not var.isdigit():
                    raise Exception(f"Expected an integer after --seed arg, not {var}")
                SEED = int(var)
            case "-q" | "--quiet":
                QUIET = True
            case "--name":
                var = next(args)
                if not var.isascii():
                    raise Exception(f"Excepted a string after --name arg, not: {var}")
                NAME = var
            case "-lf" | "--loss_function":
                var = next(args)
                if not var in ["l1", "mse"]:
                    raise Exception(f"Excepted 'l1' or 'mse' after -lf arg, not: {var}")
                LOSS_FUNCTION = var
            case "-mt" | "--model_type":
                var = next(args)
                if not var in ["4-1", "4-6", "5-6"]:
                    raise Exception(f"Excepted '4-1','4-6' or '5-6' after -mt arg, not: {var}")
                MODEL_TYPE = var
                split_arg = var.split("-")
                input_size = int(split_arg[0])
                output_size = int(split_arg[1])
            case "make_model.py":
                pass
            case _:
                raise Exception(f"Caught unexpected argument: {var}")

    except Exception as e:
        print(e)
        script_error_print("make_model.py")
        exit()
  
path_to_best_model = f"model/test.pt"
path_to_plot = f"model_img/nf_{FIRST_FILES}_bs_{BATCH_SIZE}_d_{dropout}_ne_{nb_epoch}_nl_{nb_layers}_ls_{layer_size}_lf_{LOSS_FUNCTION}.png"

if not NAME == False:
    path_to_best_model = f"model/{NAME}.pt"
    path_to_plot = f"model_img/{NAME}.png"
    
if SEED != -1:
    torch.manual_seed(SEED)
    
train_ds, dev_ds, test_ds = read_files()

print(len(train_ds))

train_dl, dev_dl, test_dl = create_dataloader()

model = NeuralNetwork(
    input_size=input_size,
    output_size=output_size,
    nb_layers=nb_layers,
    layer_size=layer_size,
    dropout=dropout,
)

loss_fn = None
if LOSS_FUNCTION == "l1":
    loss_fn = L1Loss()
elif LOSS_FUNCTION == "mse":
    loss_fn = MSELoss()
    
opt_fn = Adam(model.parameters(), lr=learning_rate)

train_losses, dev_losses = train_loop(train_dl, dev_dl, model, path_to_best_model, path_to_plot, loss_fn, opt_fn, nb_epoch)
test_loss = test_loop(model, path_to_best_model, test_dl, loss_fn)
plot(train_losses, dev_losses, test_loss, path_to_plot)
# TODO: Detect only single date point NAN's (and averaging them)
