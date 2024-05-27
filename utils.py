import pandas as pd
import numpy as np

def get_files_name():
    file_names = pd.read_csv("OUVRAGES.csv")["Ouvrage"].astype("str")

    return_array = []
    for file in file_names:
        if file[0] == "#":
            continue
        return_array.append(file)

    return return_array

def is_float(list, index):
    return (len(list) > index) and (list[index].replace(".", "", 1).isdigit())

def is_int(list, index):
    return (len(list) > index) and (list[index].isdigit())

def script_error_print():
    print("Wrong usage of the script, should be:\n"
        +"python3 make_model.py [OPTIONS]\n"
        +"\n"
        +"With OPTIONS being:\n"
        +"-ff, --first_files INTEGER:\tSets the number of files that will be taken into account when training the model,\n"
        +"\t\t\t\tshould be followed by an integer\n"
        +"\n"
        +"-md, --make_data:\t(Re)Makes the data depending on the given parameters\n"
        +"\n"
        +"-na, --na_threshhold FLOAT:\tSets the NA threshhold,\n"
        +"\t\t\t\tshould be followed by a float,\n"
        +"\t\t\t\tused when making the data files\n"
        +"\n"
        +"-fd, --fill_data:\tReplaces the NA with the mean of the variable,\n"
        +"\t\t\tused when making the data files\n"
        +"\n"
        +"-s, --shuffle:\tShuffles the data\n"
        +"\n"
        +"--seed INTEGER:\tSets a seed to the script,\n"
        +"\t\tmakes the training and testing process deterministic,\n"
        +"\t\tshould be followed by an integer\n"
        +"\n"
        +"-bs, --batch_size INTEGER:\tSets the size of the batch when training and testing the model,\n"
        +"\t\t\t\tshould be followed by an integer\n"
        +"\n"
        +"-nl, --number_layer INTEGER:\tSets the number of layers in the model,\n"
        +"\t\t\t\tshould be followed by an integer\n"
        +"\n"
        +"-ls, --layer_size INTEGER:\tSets the size of the hidden layers of the LSTM in the model,\n"
        +"\t\t\t\tshould be followed by an integer\n"
        +"\n"
        +"-d, --dropout FLOAT:\tSets the dropout parameter of the LSTM in the model,\n"
        +"\t\t\t\tshould be followed by a float\n"
        +"\n"
        +"-ne, --nb_epochs INTEGER:\tSets the nubmer of epochs used for the training of the model,\n"
        +"\t\t\t\tshould be followed by an integer\n"
        +"\n"
        +"-q, --quiet:\tUsed to suppress the prints\n"
    )