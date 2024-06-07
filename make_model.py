from utils import get_files_name
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, LSTM, Linear, L1Loss, MSELoss
from torch.optim import Adam
from tqdm.auto import trange

import torch
import os
import argparse
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Dataset class used to build the Dataloader
"""
class GWL_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(np.array(labels))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

"""
Neural Network class for the model
"""
class NeuralNetwork(Module):
    def __init__(
        self, input_size, output_size, nb_layers, layer_size, dropout,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.nb_layers = nb_layers
        self.layer_size = layer_size
      
        # define the layers
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

    def forward(self, x, ht = None):
        if ht is None:
            ht = torch.zeros(self.nb_layers, x.size(0), self.layer_size).to(x.device)
            ct = torch.zeros(self.nb_layers, x.size(0), self.layer_size).to(x.device)

        out, (ht, ct) = self.LSTM(x, (ht, ct))
        out = self.linear(out)

        return out
    
    def save(self, path):
        # get path without file
        temp = path.split("/")
        temp.remove(temp[len(temp)-1])
        temp = "".join(temp)

        if not os.path.exists(temp):
            os.makedirs(temp)

        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def make_dataset_from_all_files(file = None):
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

    files = None

    if file != None:
        files = file
    else:
        files = get_files_name()
        if FIRST_FILES:
            files = files[:FIRST_FILES]

    data_column = []
    label_column = []
    if (MODEL_TYPE == "6-1"):
        data_column = ["year", "month", "P", "T", "ET", "NDVI"]
        label_column = ["GWL"]
    elif (MODEL_TYPE == "6-6"):
        data_column = ["year", "month", "P", "T", "ET", "NDVI"]
        label_column = ["GWL+1", "GWL+2", "GWL+3", "GWL+4", "GWL+5", "GWL+6"]
    elif (MODEL_TYPE == "7-6"):
        data_column = ["year", "month", "GWL", "P", "T", "ET", "NDVI"]
        label_column = ["GWL+1", "GWL+2", "GWL+3", "GWL+4", "GWL+5", "GWL+6"]

    for folder in ["train", "dev", "test"]:
        for file in files:
            file_content = pd.read_csv(f"data/split_data/{folder}_data/{file}.csv")

            data[folder].append(file_content[data_column].to_numpy().astype("float32"))
            labels[folder].append(file_content[label_column].to_numpy().astype("float32"))

    train_dataset = GWL_Dataset(data["train"], labels["train"])
    dev_dataset   = GWL_Dataset(data["dev"]  , labels["dev"])
    test_dataset  = GWL_Dataset(data["test"] , labels["test"])

    return train_dataset, dev_dataset, test_dataset

def create_dataloader():
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    dev_dl   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    return train_dl, dev_dl, test_dl 

def train_loop(training_data, dev_data, model, loss_fn, opt_fn):
    # init the best_dev_loss
    opt_fn.zero_grad()
    x, y = next(iter(dev_data))
    pred = model.forward(x)
    loss = loss_fn(pred, y)
    best_dev_loss =  loss.item()
    best_epoch = 0
    model.save(PATH_MODEL)

    # to plot later
    train_losses = []
    dev_losses = []

    bar = trange(NB_EPOCH, leave=False, unit=" epoch")

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
            model.save(PATH_MODEL)

        bar.set_description(f"Current epoch: {epoch} | Best epoch {best_epoch} | Current dev loss {dev_loss:.4f} | Best dev loss {best_dev_loss:.4f}")

    if not QUIET:
        print(f"Best model found @ epoch {best_epoch} with loss of {best_dev_loss:.4f} on the dev dataset")

    return train_losses, dev_losses

def test_loop(model, test_data, loss_fn):
    test_loss = 0
    model.load(PATH_MODEL)
    model.eval()

    for x, y in test_data:
        pred = model.forward(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item() / len(test_data)
        
    if not QUIET: 
        print(f"Loss of {test_loss:.4f} obtained with best model")
        
    return test_loss

def plot(train_loss, dev_loss, test_loss):
    x = [i for i in range(len(train_loss))]

    # get the best epoch: where the dev loss is at the lowest
    index = np.argmin(dev_loss)
    d_min = np.min(dev_loss)
    t_val = train_loss[index]

    # plot the training and dev loss curve along side the lowest dev loss point (show the training loss at that point)
    plt.plot(x, dev_loss, "-r", label="Validation loss")
    plt.scatter([index], [d_min], color="magenta", label=f"Epoch: {index}, value: {d_min:.4f}")
    plt.plot(x, train_loss, "-b", label="Training loss")
    plt.scatter([index], [t_val], color="cyan", label=f"Value: {t_val:.4f}")
    plt.plot([], [], " ", label=f"Testing loss: {test_loss:.4f}")

    plt.legend(loc="upper left")
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title("Looses over time")

    # get path without file
    temp = PATH_PLOT.split("/")
    temp.remove(temp[len(temp)-1])
    temp = "".join(temp)

    if not os.path.exists(temp):
        os.makedirs(temp)

    plt.savefig(PATH_PLOT)
    plt.clf()
    
    return index, d_min, t_val

def save_result(best_epoch, dev_min, train_val, test_loss):
    df = pd.DataFrame(columns=[
        "Input_size", 
        "Output_size", 
        "Number_layer", 
        "Size_layer", 
        "Learning_rate", 
        "Dropout", 
        "Number_epoch", 
        "Best_epoch", 
        "Train_loss", 
        "Val_loss", 
        "Gen_loss"
    ])
    
    if os.path.isfile("Results.csv"):
        df = pd.read_csv("Results.csv")

    resultToAppend = []
    resultToAppend.append(INPUT_SIZE)
    resultToAppend.append(OUTPUT_SIZE)
    resultToAppend.append(NB_LAYERS)
    resultToAppend.append(LAYER_SIZE)
    resultToAppend.append(LEARNING_RATE)
    resultToAppend.append(DROPOUT)
    resultToAppend.append(NB_EPOCH)
    resultToAppend.append(best_epoch)
    resultToAppend.append(train_val)
    resultToAppend.append(dev_min)
    resultToAppend.append(test_loss)
        
    df.loc[len(df)] = resultToAppend
    
    df.to_csv("Results.csv", index=False)

if __name__ == "__main__":
    # Automatically chooses between CPU and GPU
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")

    parser = argparse.ArgumentParser()
    # Positional arguments
    parser.add_argument("model_type", type=str, choices=["6-1", "6-6", "7-6"], help="define the type of model")
    parser.add_argument("nb_layer", type=int, help="set the number of hidden layers")
    parser.add_argument("layer_size", type=int, help="set the size of the hidden layers")
    parser.add_argument("batch_size", type=int, help="set the batch_size")
    parser.add_argument("nb_epoch", type=int, help="set the nubmer of training epochs")
    parser.add_argument("learning_rate", type=float, help="set the learning rate")
    parser.add_argument("dropout", type=float, help="set the dropout coefficient")

    # Optional arguments
    parser.add_argument("-s", "--shuffle", help="shuffles of the datasets", action="store_true")
    parser.add_argument("-q", "--quiet", help="suppresses the prints", action="store_true")
    parser.add_argument("-ff", "--first_files", type=int, help="only use the first few files")
    parser.add_argument("--seed", type=int, help="set the seed used during the training and testing processes")
    parser.add_argument("--name", type=str, help="override the naming scheme for the model and plot files")
    args = parser.parse_args()
    
    MODEL_TYPE = args.model_type
    INPUT_SIZE = int(MODEL_TYPE.split("-")[0])
    OUTPUT_SIZE = int(MODEL_TYPE.split("-")[1])
    NB_LAYERS = int(args.nb_layer)
    LAYER_SIZE = int(args.layer_size)
    BATCH_SIZE = int(args.batch_size)
    NB_EPOCH = int(args.nb_epoch)
    LEARNING_RATE = float(args.learning_rate)
    DROPOUT = int(args.dropout)
    
    SHUFFLE = args.shuffle
    QUIET = args.quiet
    FIRST_FILES = None
    if args.first_files: FIRST_FILES = int(args.first_files)
    SEED = None
    if args.seed: SEED = int(args.seed)
    NAME = args.name

    PATH_MODEL = f"model/{NAME}.pt"
    PATH_PLOT    = f"img/{NAME}.png"
    if NAME == None:
        PATH_MODEL = f"model/{MODEL_TYPE}_{NB_LAYERS}_{LAYER_SIZE}_{BATCH_SIZE}_{NB_EPOCH}_{LEARNING_RATE}_{DROPOUT}.pt"
        PATH_PLOT    = f"img/{MODEL_TYPE}_{NB_LAYERS}_{LAYER_SIZE}_{BATCH_SIZE}_{NB_EPOCH}_{LEARNING_RATE}_{DROPOUT}.png"

    if SEED:
        torch.manual_seed(SEED)

    # Create the dataloaders
    train_ds, dev_ds, test_ds = make_dataset_from_all_files()
    train_dl, dev_dl, test_dl = create_dataloader()
        
    # Make the model
    model = NeuralNetwork(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        nb_layers=NB_LAYERS,
        layer_size=LAYER_SIZE,
        dropout=DROPOUT,
    )
    
    loss_fn = L1Loss()
    opt_fn = Adam(model.parameters(), lr=LEARNING_RATE)

    # Train 
    train_losses, dev_losses = train_loop(train_dl, dev_dl, model, loss_fn, opt_fn)
    
    # Test 
    test_loss = test_loop(model, test_dl, loss_fn)
    
    # Save the results
    best_epoch, dev_min, train_val = plot(train_losses, dev_losses, test_loss)
    save_result(best_epoch, dev_min, train_val, test_loss)
